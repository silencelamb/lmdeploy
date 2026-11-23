# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.engine.model_agent import agent as agent_module
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, BatchedOutputs
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.spec_decode.base import BaseSpecModelAgent
from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent
from lmdeploy.pytorch.strategies.ar_spec.model_agent import ARSpecExtraInputs, ARSpecModelAgentStrategy


def inputs(ids, lengths, history, decoding):
    return ModelInputs(input_ids=torch.tensor([ids]), seq_length=torch.tensor(lengths),
                       history_lengths=torch.tensor(history), block_offsets=torch.ones((len(lengths), 3)).int(),
                       num_ignored_history=torch.zeros(len(lengths)).long(), is_decoding=decoding,
                       max_q_seqlen=max(lengths), max_kv_seqlen=max(h + n for h, n in zip(history, lengths)),
                       sum_kv_seqlen=sum(history) + sum(lengths))


@pytest.mark.parametrize('depth', [0, 3])
def test_target_projects_last_prompt_and_every_verify_position_in_one_forward(monkeypatch, depth):
    width = depth + 1
    prefill = inputs([2, 3, 5, 7, 11], [3, 2], [0, 64], False)
    decode = inputs(list(range(2 * width)), [width, width], [31, 127], True)
    raw = torch.arange(5 + 2 * width).view(1, -1, 1)
    calls = []

    def target(model, prompt, config, cache, states, stream, prompt_plan, verify, verify_plan):
        assert prompt is prefill and verify is decode
        calls.append('target')
        return dict(hidden_states=raw, position_ids=prompt.input_ids, decode_position_ids=verify.input_ids,
                    target_inputs_embeds=raw + 1, model_metas=None)

    def head(hidden):
        calls.append('head')
        return hidden + 100

    monkeypatch.setattr(agent_module, 'model_forward', target)
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.patched_model = agent.model_config = agent.cache_engine = agent.state_cache_engine = agent.stream = None
    agent.rank = agent._mixed_forward_count = 0
    agent.agent_strategy = ARSpecModelAgentStrategy(depth)
    agent.spec_agent = BaseSpecModelAgent.__new__(BaseSpecModelAgent)
    agent.spec_agent._enabled = bool(depth)
    agent.get_logits = head
    prompt, verify = agent._mixed_model_forward(prefill, decode, None, None)
    assert calls == ['target', 'head']
    assert prompt['logits'].flatten().tolist() == [102, 104]
    assert verify['logits'].flatten().tolist() == list(range(105, 105 + 2 * width))
    if depth:
        torch.testing.assert_close(prompt['hidden_states'], raw[:, :5])
        torch.testing.assert_close(verify['hidden_states'], raw[:, 5:])


@pytest.mark.parametrize('nonfinal', [False, True])
def test_mixed_mtp_uses_each_rows_accepted_prefix_and_chunk_lookahead(nonfinal):
    prefill = inputs([10, 11, 12], [3], [32], False)
    prefill.is_chunk = nonfinal
    prefill.prefill_next_token_ids = torch.tensor([13]) if nonfinal else None
    decode = inputs(list(range(20, 28)), [4, 4], [90, 150], True)
    prompt_extra = ARSpecExtraInputs(next_token_ids=torch.tensor([99]), num_rejected_tokens=torch.tensor([0]),
                                     last_token_indices=torch.tensor([2]),
                                     target_hidden_states=prefill.input_ids.unsqueeze(-1),
                                     target_position_ids=torch.arange(32, 35).view(1, -1))
    decode_extra = ARSpecExtraInputs(next_token_ids=torch.tensor([70, 80]), num_rejected_tokens=torch.tensor([3, 0]),
                                     last_token_indices=torch.tensor([0, 7]),
                                     target_hidden_states=decode.input_ids.unsqueeze(-1),
                                     target_position_ids=torch.tensor([[90, 91, 92, 93, 150, 151, 152, 153]]))
    agent = SpecModelAgent.__new__(SpecModelAgent)

    async def draft(batch, extra, sampling, discarded_rows):
        assert batch.input_ids.tolist() == [[11, 12, 13 if nonfinal else 99, 70, 22, 23, 23, 25, 26, 27, 80]]
        assert batch.history_lengths.tolist() == [32, 90, 150]
        assert batch.seq_length.tolist() == [3, 4, 4]
        assert extra.last_token_indices.tolist() == [2, 3, 10]
        assert extra.num_rejected_tokens.tolist() == [0, 3, 0]
        assert batch.sum_kv_seqlen == 283
        assert discarded_rows == int(nonfinal)
        return ARSpecExtraInputs(output_draft_token_ids=torch.arange(9).view(3, 3))

    agent._async_autoregressive_model_forward = draft
    p, d = asyncio.run(agent.async_mixed_model_forward(prefill, decode, prompt_extra, decode_extra))
    assert p.output_draft_token_ids.tolist() == [[0, 1, 2]]
    assert d.output_draft_token_ids.tolist() == [[3, 4, 5], [6, 7, 8]]
    assert d.num_rejected_tokens is decode_extra.num_rejected_tokens
    assert decode.input_ids.tolist() == [list(range(20, 28))]


def test_mixed_completion_preserves_both_tickets_through_worker_transfer():
    batch = BatchedOutputs(torch.tensor([10]), torch.tensor([False]), ssm_state_access_id=102,
                           mixed_decode=BatchedOutputs(torch.tensor([[20, 21, -1, -1]]), torch.tensor([False]),
                                                       ssm_state_access_id=103))
    restored = batch.to_cpu().to_numpy().to_tensor()
    assert restored.ssm_state_access_id == 102
    assert restored.mixed_decode.ssm_state_access_id == 103
    torch.testing.assert_close(restored.mixed_decode.next_token_ids, batch.mixed_decode.next_token_ids)


def test_appending_prefill_keeps_the_dispatched_decode_row_map_immutable():
    from lmdeploy.pytorch.disagg.config import EngineRole
    from lmdeploy.pytorch.engine.inputs_maker import InputsMakerAsync

    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(role=EngineRole.Hybrid)
    maker.long_context_chunker = SimpleNamespace(enabled=lambda: False)
    submitted = [object(), object()]
    prompt = SimpleNamespace(sampling_param=SimpleNamespace(max_new_tokens=8))
    maker.update_running_seqs(submitted, None)
    maker.update_running_seqs([prompt], SimpleNamespace(is_chunk=False))
    assert len(submitted) == 2
    assert maker.running_seqs == [*submitted, prompt]


@pytest.mark.parametrize('wrong_ticket', [False, True])
def test_mixed_event_validates_both_row_groups_before_releasing_either(wrong_ticket):
    from lmdeploy.pytorch.engine.engine_loop import EngineLoop

    loop = EngineLoop.__new__(EngineLoop)
    seen = []
    prompt_rows, decode_rows = ['new'], ['a', 'b']
    prefill = SimpleNamespace(inputs='prompt')
    decode = SimpleNamespace(delta=SimpleNamespace(max_q_seqlen=1),
                             cache_inputs=SimpleNamespace(ssm_state_access_id=13))
    out = BatchedOutputs(torch.tensor([8]), torch.tensor([False]), ssm_state_access_id=22 if wrong_ticket else 12,
                         mixed_decode=BatchedOutputs(torch.tensor([6, 7]), torch.tensor([False, False]),
                                                     ssm_state_access_id=13))

    async def prefetch():
        seen.append('prefetch')
        return None, None

    async def complete():
        seen.append('event')
        return out

    def publish(rows, inputs, delta, cache):
        seen.append(('publish', tuple(rows)))
        return cache.ssm_state_access_id, True, False

    def finish(result, rows, inputs, delta, cache, published):
        seen.append(('finish', tuple(rows), result.ssm_state_access_id, published[0]))

    loop._publish_forward, loop._complete_forward, loop._prefetch_next_inputs = publish, finish, prefetch
    loop.executor = SimpleNamespace(get_output_async=complete)
    payload = dict(inputs=prefill, delta=None, cache_inputs=SimpleNamespace(ssm_state_access_id=12),
                   mixed_decode=decode, mixed_running=decode_rows)
    expected = [('publish', ('a', 'b')), ('publish', ('new',)), 'prefetch', 'event']
    if wrong_ticket:
        with pytest.raises(RuntimeError, match='Mixed target completion'):
            asyncio.run(loop._main_loop_get_outputs(prompt_rows, payload))
    else:
        asyncio.run(loop._main_loop_get_outputs(prompt_rows, payload))
        expected += [('finish', ('a', 'b'), 13, 13), ('finish', ('new',), 12, 12)]
    assert seen == expected


@pytest.mark.parametrize(('suffix', 'expected'), [(2048, 4096), (32767, 4096), (32768, 8128), (65536, 8128)])
def test_adaptive_chunk_uses_post_match_suffix_and_reserves_verify_width(suffix, expected):
    from lmdeploy.pytorch.engine.inputs_maker import InputsMakerAsync, LongContextChunker
    from lmdeploy.pytorch.messages import MessageStatus

    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(mixed_long_threshold=32768, max_prefill_token_num=8192, block_size=64,
                                   use_mrope=False)
    maker.long_context_chunker = LongContextChunker(8192)
    seq = SimpleNamespace(num_token_ids=suffix, num_history_ids=131072, prefill_input_shift=1,
                          get_input_multimodals=lambda: {}, get_chunk_limit_multimodals=lambda: {},
                          return_logits=False, return_ce_loss=False, return_routed_experts=False, adapter_name=None,
                          sampling_param=SimpleNamespace(max_new_tokens=256, response_format=None))
    maker.long_context_chunker.set_seq(seq, 4096)
    # By the time we schedule this continuation the remaining length can be much smaller.
    seq.num_token_ids = min(suffix, 5000)
    maker.running_seqs = [SimpleNamespace(status=MessageStatus.RUNNING) for _ in range(7)]
    maker.engine_strategy = SimpleNamespace(get_num_decode_tokens=lambda: 4)
    maker._refresh_mixed_chunk_budget()
    assert maker.long_context_chunker.max_prefill_num == expected
    assert expected + 7 * 4 <= maker.config.max_prefill_token_num
