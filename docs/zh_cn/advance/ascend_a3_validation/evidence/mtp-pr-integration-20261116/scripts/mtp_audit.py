"""Runtime audit of greedy MTP dispatch and the acceptance-prefix result."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from lmdeploy.pytorch.engine.executor.ray_executor import RayWorkerWrapper


def digest(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


class MTPAuditWorker(RayWorkerWrapper):
    def build_model(self):
        super().build_model()
        from lmdeploy.pytorch.backends.ascend.rejection_sampling import AscendRejectionSamplingImpl
        from lmdeploy.pytorch.spec_decode.reject_sampler import RejectionSampler
        from lmdeploy.pytorch.engine import logits_process

        root=Path(os.environ['VALIDATION_ROOT'])
        output=root/'results'/f'mtp-runtime-{os.environ["VALIDATION_AUDIT_LABEL"]}-rank{self.rank}.jsonl'
        original=RejectionSampler.forward
        sequence=0
        def forbidden(*args,**kwargs):
            raise AssertionError('A random sampling path was reached during an all-greedy MTP audit')
        AscendRejectionSamplingImpl.forward=forbidden
        logits_process._multinomial_sampling=forbidden

        def audited(sampler,target_logits,draft_token_ids,sampling_inputs,draft_probs=None):
            nonlocal sequence
            sequence+=1
            assert sampling_inputs.max_top_k==1, sampling_inputs
            assert isinstance(sampler._impl, AscendRejectionSamplingImpl), type(sampler._impl)
            before_cpu=digest(torch.random.get_rng_state())
            before_npu=digest(torch.npu.get_rng_state())
            result=original(sampler,target_logits,draft_token_ids,sampling_inputs,draft_probs)
            after_cpu=digest(torch.random.get_rng_state())
            after_npu=digest(torch.npu.get_rng_state())
            assert before_cpu==after_cpu and before_npu==after_npu, 'Verifier consumed RNG'
            top=target_logits.float().topk(2,dim=-1)
            argmax=target_logits.argmax(dim=-1).cpu().tolist()
            draft=draft_token_ids.cpu().tolist()
            expected=[];expected_rejected=[];expected_last=[]
            for targets,guesses in zip(argmax,draft):
                row=[]
                for pos,guess in enumerate(guesses):
                    row.append(targets[pos])
                    if guess!=targets[pos]: break
                else:
                    row.append(targets[-1])
                expected_last.append(row[-1])
                expected_rejected.append(len(targets)-len(row))
                expected.append(row+[-1]*(len(targets)-len(row)))
            actual=[part.cpu().tolist() for part in result]
            assert actual==[expected,expected_rejected,expected_last], (actual,expected)
            row=dict(sequence=sequence,pid=os.getpid(),rank=self.rank,
                max_top_k=sampling_inputs.max_top_k,has_greedy=sampling_inputs.has_greedy,
                top_k_none=sampling_inputs.top_k is None,shape=list(target_logits.shape),
                target_argmax=argmax,draft_token_ids=draft,output_token_ids=actual[0],
                num_rejected_tokens=actual[1],last_token_ids=actual[2],
                target_top2_ids=top.indices.cpu().tolist(),target_top2_values=top.values.cpu().tolist(),
                cpu_rng_unchanged=before_cpu==after_cpu,npu_rng_unchanged=before_npu==after_npu,
                cpu_oracle_equal=True,dlinfer_imported=any(n.split('.')[0]=='dlinfer' for n in sys.modules))
            assert not row['dlinfer_imported']
            with output.open('a') as f: f.write(json.dumps(row)+'\n')
            return result
        RejectionSampler.forward=audited
