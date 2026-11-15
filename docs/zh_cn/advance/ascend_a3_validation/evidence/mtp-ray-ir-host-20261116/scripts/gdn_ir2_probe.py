"""Fixed real-input GDN control with process-local compiler IR capture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch
import torch_npu
import lmdeploy.pytorch.kernels.ascend.triton_ops.fla.fused_recurrent as module


def sha(t):
    return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixture',type=Path,default=Path('/root/workspace/a3-mtp-ray-investigation-20261116/results/gdn-real-fixed-graph-r0.pt'))
    p.add_argument('--options',default='{}')
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--graph',action='store_true')
    p.add_argument('--dump-ir',action='store_true')
    p.add_argument('--extra-compiler-options',default='[]')
    p.add_argument('--binary-override',type=Path)
    a=p.parse_args();assert not a.output.exists(),a.output
    folder=a.output.parent.parent/'artifacts'/a.output.stem
    folder.mkdir(exist_ok=False)
    os.environ['TRITON_CACHE_DIR']=str(folder/'triton-cache')
    torch.set_num_threads(1);torch.manual_seed(42);torch.npu.manual_seed_all(42)
    fixture=torch.load(a.fixture,weights_only=True,map_location='cpu')
    values={k:fixture[k].to('npu:0') for k in ['q','k','v','g','beta','initial']}
    q,k,v,g,beta,initial=[values[k] for k in ['q','k','v','g','beta','initial']]
    state=initial.clone();T=q.shape[1]
    cu=torch.tensor([0,T],device='npu:0',dtype=torch.int32)
    ids=torch.tensor([1],device='npu:0',dtype=torch.int64)
    lengths=torch.tensor([27],device='npu:0',dtype=torch.int32)
    options=json.loads(a.options)
    options['debug']=a.dump_ir
    compiled=[];commands=[]
    original_run=subprocess.run
    def compiler_run(cmd,*args,**kwargs):
        if (isinstance(cmd,(list,tuple)) and len(cmd)>1
                and 'bishengir-compile' in Path(str(cmd[0])).name and Path(str(cmd[1])).is_file()):
            sub=folder/f'compile{len(commands):02d}';sub.mkdir()
            patched=list(cmd)
            patched.extend(json.loads(a.extra_compiler_options))
            if a.dump_ir:
                patched.extend([f'--save-temps={sub.resolve()}/temps','--mlir-print-ir-after-all','--mlir-print-ir-module-scope'])
            commands.append(patched)
            (sub/'command.json').write_text(json.dumps(patched,indent=2))
            shutil.copyfile(cmd[1],sub/'input.ttadapter.mlir')
            try:
                ret=original_run(patched,*args,**kwargs)
            except subprocess.CalledProcessError as e:
                (sub/'stdout.txt').write_bytes(e.stdout or b'')
                (sub/'stderr.txt').write_bytes(e.stderr or b'')
                raise
            (sub/'stdout.txt').write_bytes(ret.stdout or b'')
            (sub/'stderr.txt').write_bytes(ret.stderr or b'')
            if a.binary_override:
                base=Path(cmd[cmd.index('-o')+1]);candidates=[base.with_suffix('.o'),Path(str(base)+'_reloc.o')]
                dest=next((x for x in candidates if x.is_file()),None)
                if dest is None:raise RuntimeError(f'No native output binary found: {candidates}')
                shutil.copyfile(dest,sub/'original.o')
                shutil.copyfile(a.binary_override,dest)
                (sub/'binary-override.json').write_text(json.dumps(dict(source=str(a.binary_override),
                    sha256=hashlib.sha256(a.binary_override.read_bytes()).hexdigest(),target=str(dest)),indent=2))
            return ret
        return original_run(cmd,*args,**kwargs)
    subprocess.run=compiler_run
    original_kernel=module.fused_recurrent_gated_delta_rule_fwd_kernel
    class Options:
        def __getitem__(self,grid):
            launch=original_kernel[grid]
            def run(*args,**kwargs):
                kwargs.update(options)
                result=launch(*args,**kwargs)
                if not compiled:compiled.append(result)
                return result
            return run
    module.fused_recurrent_gated_delta_rule_fwd_kernel=Options()
    def call():
        return module.fused_recurrent_gated_delta_rule(q,k,v,g,beta,initial_state=state,
            inplace_final_state=True,cu_seqlens=cu,cache_seqlens_rb=lengths,state_ids_rb=ids,
            num_state=4,use_qk_l2norm_in_kernel=True)[0]
    call();torch.npu.synchronize()
    if compiled:
        obj=compiled[0]
        meta=obj.metadata._asdict() if hasattr(obj.metadata,'_asdict') else vars(obj.metadata)
        (folder/'compiler-metadata.json').write_text(json.dumps(meta,indent=2,default=str))
        for name,data in obj.asm.items():
            dest=folder/f'kernel.{name}'
            if isinstance(data,bytes):dest.write_bytes(data)
            else:dest.write_text(str(data))
    graph=None
    if a.graph:
        graph=torch.npu.NPUGraph()
        state.copy_(initial)
        with torch.npu.graph(graph,auto_dispatch_capture=True):output=call()
        torch.npu.synchronize()
    results=[];reference=None
    for i in range(a.iterations):
        state.copy_(initial)
        if graph:graph.replay()
        else:output=call()
        torch.npu.synchronize()
        s=state.cpu();o=output.cpu()
        if reference is None:reference=s.clone()
        bad=(s-fixture['state']).abs()
        row=dict(iteration=i,state_exact_reference=bool(torch.equal(s,fixture['state'])),
            state_close_reference=bool(torch.allclose(s,fixture['state'],atol=1e-5,rtol=1e-4)),
            max_abs=float(bad.max()),mismatch=int((s!=fixture['state']).sum()),
            output_exact_reference=bool(torch.equal(o,fixture['output'])),
            state_equal_first=bool(torch.equal(s,reference)),per_slot_bad=(bad>1e-5).flatten(1).sum(1).tolist())
        results.append(row)
        if i==0 or (not row['state_close_reference'] and not (folder/'first-bad.pt').exists()):
            torch.save(dict(state=s,output=o,iteration=i),folder/('first.pt' if i==0 else 'first-bad.pt'))
    data=dict(source=module.__file__,options=options,graph=a.graph,fixture=str(a.fixture),
        runtime_cache=os.environ['TRITON_CACHE_DIR'],extra_compiler_options=json.loads(a.extra_compiler_options),
        binary_override=str(a.binary_override) if a.binary_override else None,
        input_hashes={k:sha(t) for k,t in values.items()},results=results,compiler_commands=commands)
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    print('IR2',a.output.stem,'state_bad',sum(not r['state_close_reference'] for r in results),
          'output_bad',sum(not r['output_exact_reference'] for r in results),
          'max_abs',max(r['max_abs'] for r in results),flush=True)

if __name__=='__main__':main()
