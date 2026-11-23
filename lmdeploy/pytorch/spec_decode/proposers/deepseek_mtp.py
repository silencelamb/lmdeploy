# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lmdeploy.pytorch import envs

from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer, logger


@SPEC_PROPOSERS.register_module(name='deepseek_mtp')
class DeepseekMTP(BaseSpecProposer):

    _draft_head_group = None

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        """Build the draft model and bind the target embedding."""
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        draft_model = self.model
        if (getattr(draft_model, 'uses_dsa_topk_buffer', False)
                or getattr(draft_model, 'uses_shared_input_embeddings', False)) and hasattr(
                    draft_model, 'set_input_embeddings'):
            draft_model.set_input_embeddings(target_model.get_input_embeddings())

        if (envs.glm53_dp_draft_head and build_model_ctx is not None
                and build_model_ctx.moe_backend == 'deepgemm_mega_moe' and not build_model_ctx.enable_deterministic):
            head = target_model.lm_head
            cfg = self.specdecode_config.dist_config
            group = head.dist_ctx.ep_gpu_group
            if ((cfg.dp, cfg.ep, cfg.tp) == (8, 8, 1) and group is not None and group.size() == cfg.dp
                    and head.tp == 1 and head.bias is None and head.weight.dtype == torch.bfloat16
                    and head.vocab_size % cfg.dp == 0):
                self._draft_head_group = group
                logger.info('Enabled DP-sharded MTP draft LM head on %d ranks.', cfg.dp)

    def _get_mtp_logits(self, hidden_states: torch.Tensor, model_inputs: ModelInputs) -> torch.Tensor:
        """Project a shared vocabulary shard for every DP row and return each owner's complete logits."""
        group = self._draft_head_group
        if group is None or model_inputs.dp_meta is None or not model_inputs.dp_meta.dp_is_decoding:
            return self.target_model.get_logits(hidden_states)
        meta = model_inputs.dp_meta
        # The seed pass carries a batch count separate from its multi-token
        # query count. Each subsequent AR pass has one query per request.
        sizes = meta.dp_batches or meta.dp_token_counts
        if not sizes or max(sizes) > 16:
            return self.target_model.get_logits(hidden_states)
        batch = max(sizes)
        local = hidden_states[0].contiguous()
        rows, hidden_size = local.shape
        if rows < batch:
            local = F.pad(local, (0, 0, 0, batch - rows))

        world = group.size()
        gathered = local.new_empty((world * batch, hidden_size))
        dist.all_gather_into_tensor(gathered, local, group=group)
        head = self.target_model.lm_head
        width = head.vocab_size // world
        weights = head.weight.narrow(0, group.rank() * width, width)
        shard = F.linear(gathered, weights)
        received = torch.empty_like(shard)
        dist.all_to_all_single(received, shard, group=group)
        # Received layout is [vocabulary shard, local padded row, shard columns].
        return received.view(world, batch, width).transpose(0, 1)[:rows].reshape(1, rows, head.vocab_size)

    async def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ARSpecExtraInputs = None,
                    guided_processors: dict | None = None):
        """Get outputs."""
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        draft_model = self.model.get_model() if hasattr(self.model, 'get_model') else self.model
        uses_topk_buffer = getattr(draft_model, 'uses_dsa_topk_buffer', False)

        if extra_inputs is not None:
            last_token_loc = extra_inputs.last_token_indices
            hidden_states = hidden_states[:, last_token_loc]
            if uses_topk_buffer:
                draft_model.compact_topk_indices(last_token_loc)

        if hasattr(draft_model, 'prepare_hidden_states_for_logits'):
            hidden_states = draft_model.prepare_hidden_states_for_logits(hidden_states)
            logits = self._get_mtp_logits(hidden_states, model_inputs)
        else:
            logits = self.get_logits(hidden_states)
        target_hidden_states = hidden_states

        if uses_topk_buffer:
            model_metas = model_metas or [None] * hidden_states.size(1)
            model_metas = [dict(meta or {}, skip_topk=True) for meta in model_metas]
        logits = logits[0]

        guided_bitmask = await self.guided_helper.prepare_bitmask(logits, guided_processors)
        if guided_bitmask is not None:
            self.guided_helper.apply_bitmask(logits, guided_bitmask)

        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        await self.guided_helper.accept_draft_tokens(draft_token_ids, guided_processors)

        return draft_token_ids, model_metas, target_hidden_states
