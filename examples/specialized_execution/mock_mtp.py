"""Run an exact CPU oracle for a four-position target verification and its
continuation."""

import json
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.backends.default.rejection_sampling import torch_greedy_rejection_sample
from lmdeploy.pytorch.specialized_execution import DecodePlan, SpecializedExecutor, build_specialized_execution


class PrefixTarget(SpecializedExecutor):
    api_version = 1
    artifact_id = 'prefix-sum-target-v1'

    @classmethod
    def describe(cls, ctx):
        return (DecodePlan('verify-three', batch_size=4, num_spec_tokens=3),)

    def run(self, plan, input_ids, base, slots, state):
        prefix = base[:, None] + input_ids.cumsum(1)
        state[slots] = prefix
        # A toy language model's greedy predictions and auxiliary stream.
        return {'ids': prefix.remainder(31), 'aux': prefix + 100}


def main():
    ctx = SimpleNamespace(
        num_spec_tokens=3, model=None, dist=SimpleNamespace(dist_config=SimpleNamespace(dp=1, world_size=1))
    )
    execution = build_specialized_execution(f'{__name__}:PrefixTarget', ctx)
    base = torch.tensor([11, 8, 3, 17])
    queries = torch.zeros((4, 4), dtype=torch.long)
    queries[:, 0] = torch.tensor([7, 3, 2, 1])
    wanted = torch.arange(4)
    next_state = base.clone()
    for j in range(3):
        next_state += queries[:, j]
        queries[:, j + 1] = (next_state.remainder(31) + (wanted == j)).remainder(31)
    state = torch.full((20,), -1, dtype=torch.long)
    slots = torch.arange(1, 17).view(4, 4)
    inputs = SimpleNamespace(
        is_decoding=True,
        is_dummy=False,
        is_chunk=False,
        enable_microbatch=False,
        vision_inputs=None,
        input_ids=queries.flatten()[None],
        max_q_seqlen=4,
        max_kv_seqlen=17,
        specialized_plan=None,
    )
    plan = execution.select(inputs, 4)
    assert plan is not None and plan.num_spec_tokens == 3
    result = execution.executor.run(plan, queries, base, slots, state)
    emitted, rejected, last = torch_greedy_rejection_sample(result['ids'], queries[:, 1:].contiguous())
    consumed = 4 - rejected
    assert consumed.tolist() == [1, 2, 3, 4]
    retained = state[slots[torch.arange(4), consumed - 1]]
    reference = []
    for row, length in enumerate(consumed.tolist()):
        v = int(base[row])
        for token in queries[row, :length]:
            v += int(token)
        reference.append(v)
    assert retained.tolist() == reference
    assert torch.equal(last, retained.remainder(31))
    assert torch.equal(result['aux'], state[slots] + 100)
    # One AR row after a rejected suffix takes the normal route.
    inputs.input_ids, inputs.max_q_seqlen = last[:1, None], 1
    assert execution.select(inputs, 1) is None
    assert ((retained + last) % 31).tolist() == [(s + s % 31) % 31 for s in reference]
    print(
        json.dumps(
            dict(
                accepted=wanted.tolist(),
                consumed=consumed.tolist(),
                rejected=rejected.tolist(),
                emitted=emitted.tolist(),
                committed_state=reference,
                continuation_equal=True,
            )
        )
    )


if __name__ == '__main__':
    main()
