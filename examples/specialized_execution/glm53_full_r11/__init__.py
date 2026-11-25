"""Full GLM-5.3-Flash R11 worker-side decode program."""

from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from glm53_r11 import R11Projection

from lmdeploy.pytorch.specialized_execution import DecodePlan

from .native import load_programs


def compatible_weight(tensor, shape, dtype, device):
    return tensor.shape == shape and tensor.dtype == dtype and tensor.device == device and tensor.is_contiguous()


def fp8_linear(module, x):
    from lmdeploy.pytorch.kernels.cuda import blocked_fp8_gemm_gluon as gemm
    from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8

    x = x.reshape(8, -1)
    weight, scale = module.weight, module.weight_scale_inv
    quant, xs = quant_fp8(x, 128, dtype=weight.dtype, trans_scale=True, scale_fmt=None)
    out = x.new_empty((8, weight.size(0)))
    tile = 128 if (x.size(1), weight.size(0)) == (1536, 16384) else 64
    a, b, dst = gemm._make_transposed_small_descriptors(quant, weight, out, tile, 8)
    gemm._fp8_gemm_nt_single_partition_kernel[(1, triton.cdiv(weight.size(0), tile))](
        a,
        xs,
        b,
        scale,
        dst,
        8,
        x.size(1),
        *xs.stride(),
        *scale.stride(),
        transpose_output=True,
        scale_block_n=128,
        num_buffers=4,
        num_warps=4,
    )
    return out[None]


class Layer:
    def __init__(self, layer, device):
        from glm53_r11_artifact import r8

        self.layer = layer
        self.attn = attn = layer.self_attn
        self.hc = [
            F.pad(w, (0, 8)).view(4096, 4, 32).transpose(1, 2).contiguous()
            for w in (layer.hc_attn_projection, layer.hc_ffn_projection)
        ]
        self.work = [
            (
                torch.empty((8, 128, 32), device=device),
                torch.empty((8, 32), device=device),
                torch.zeros(8, device=device, dtype=torch.int32),
            )
            for _ in range(2)
        ]
        self.qkv = None
        self.expert_weights = None
        if layer.is_linear_attention:
            w = attn.qkv_proj.weight
            self.qkv = torch.empty_like(w)
            r8._pack[(w.numel() // 512,)](w.view(torch.int16), self.qkv.view(torch.int16), 512)
            self.gate = torch.cat((attn.b_proj.weight, attn.fg_a_proj.weight), dim=0)
        else:
            self.kpool = torch.cat((attn.indexer.wk.weight, attn.indexer.index_kpool_compress_gate), dim=0)
            self.kc = attn.kc.weight.transpose(1, 2).contiguous()
            self.vc = attn.vc.weight.transpose(1, 2).contiguous()


class FullR11(R11Projection):
    artifact_id = 'glm53-full-r11-paired8-132cta-compact-kmajor-v1'

    @classmethod
    def describe(cls, ctx):
        if not super().describe(ctx):
            return ()
        model, config = ctx.model, ctx.model.config
        if (
            ctx.backend_config.moe_backend != 'default'
            or ctx.model_config.use_mla_fp8_cache
            or ctx.cache_config.quant_policy
            or (ctx.cache_config.block_size, ctx.cache_config.kernel_block_size) != (64, 64)
            or model.enable_return_routed_experts
            or ctx.dist.dist_config.world_size != 8
            or (config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps, config.rms_norm_eps) != (4, 20, 1e-6, 1e-5)
            or (config.kv_lora_rank, config.q_lora_rank, config.index_kpool, config.index_topk) != (512, 1536, 4, 2048)
            or (config.num_attention_heads, config.qk_rope_head_dim, config.qk_nope_head_dim, config.v_head_dim)
            != (64, 0, 256, 256)
            or (config.index_n_heads, config.index_head_dim) != (32, 128)
            or torch.backends.cuda.matmul.allow_tf32
            or torch.cuda.device_count() != 8
        ):
            return ()
        if not all(i == ctx.device.index or torch.cuda.can_device_access_peer(ctx.device.index, i) for i in range(8)):
            return ()
        from lmdeploy.pytorch.models.glm5_next import Glm5NextMoE

        layers = model.model.layers
        for layer in layers:
            a = layer.self_attn
            if layer.is_linear_attention:
                if (
                    (a.local_num_heads, a.head_dim, a.conv_kernel_size) != (64, 128, 4)
                    or not compatible_weight(a.qkv_proj.weight, (24576, 4096), torch.bfloat16, ctx.device)
                    or not compatible_weight(a.qkv_conv1d.weight, (24576, 1, 4), torch.float32, ctx.device)
                ):
                    return ()
            elif a.indexer.cache_dtype != torch.float8_e4m3fn:
                return ()
        moes = [layer.mlp for layer in layers if isinstance(layer.mlp, Glm5NextMoE)]
        owned = list(range(36 * ctx.dist.ep_rank, 36 * (ctx.dist.ep_rank + 1)))
        if len(moes) != 42:
            return ()
        for moe in moes:
            e = moe.experts
            if (
                moe.shared_experts is None
                or moe.mega_moe is not None
                or list(e.expert_list) != owned
                or e.scale_fmt is not None
                or e.gate_up.gate_up_interleave
                or moe.gate.fake_eplb
                or not compatible_weight(e.gate_up.weight, (36, 4096, 4096), torch.float8_e4m3fn, ctx.device)
                or not compatible_weight(e.down.weight, (36, 4096, 2048), torch.float8_e4m3fn, ctx.device)
                or not compatible_weight(e.gate_up.weight_scale_inv, (36, 32, 32), torch.float32, ctx.device)
                or not compatible_weight(e.down.weight_scale_inv, (36, 32, 16), torch.float32, ctx.device)
            ):
                return ()
        # The old 512-step driver reserves two extra host metadata positions.
        return (DecodePlan('full-r11-ep8', 8, 66050, graph_safe=True),)

    def __init__(self, ctx):
        import glm53_r11_artifact
        import lmdeploy_glm53_megagemm_r6 as dg

        from lmdeploy.pytorch.models.glm5_next import Glm5NextMoE

        root = Path(glm53_r11_artifact.__file__).parent
        if not (
            dg.GLM53_PRECISION_ABI == dg.GLM53_R5_RETURN_ABI == dg.GLM53_R6_PIPELINE_ABI == 1
            and dg.GLM53_R6_KMAJOR
            and dg.GLM53_WEIGHT_LAYOUT_ABI == 1
        ):
            raise ValueError('R11 requires the frozen R6 K-major block-FP8 provider')
        super().__init__(ctx, native_dir=root)
        try:
            self.prepare_layers(ctx, root, dg, Glm5NextMoE)
        except Exception:
            super().close()
            raise

    def prepare_layers(self, ctx, root, dg, moe_class):
        self.native = load_programs(root, ctx.device)
        self.shared_stream = torch.cuda.Stream(device=ctx.device)
        self.attn_stream = torch.cuda.Stream(device=ctx.device)
        self.layers = [Layer(layer, ctx.device) for layer in self.model.model.layers]
        self.moes = [p for p in self.layers if isinstance(p.layer.mlp, moe_class)]
        self.dg = dg
        self.ep = None
        for prepared in self.moes:
            prepared.expert_weights = self.pack_local(prepared.layer.mlp.experts)
        torch.cuda.synchronize(ctx.device)

    def pack_local(self, experts):
        from .experts import pack_owned_experts

        return pack_owned_experts(self.dg, experts)

    def prepare_distributed(self, ctx):
        from .experts import ExpertParallel

        self.ep = ExpertParallel(self.dg, self.native, ctx.dist.ep_gpu_group, ctx.device)
        for prepared in self.moes:
            self.ep.exchange(prepared.expert_weights)
        torch.cuda.synchronize(ctx.device)

    @staticmethod
    def reduce_partials(partials, output):
        from glm53_r11_artifact.bf16_output import reduce

        reduce[(128,)](partials, output, output.numel(), 256, num_warps=4)

    def kda(self, p, x, cache, metadata, sink):
        from glm53_r11_artifact.kda_norm import forward as recurrent

        a = p.attn
        main, side = torch.cuda.current_stream(), self.attn_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            beta, low = F.linear(x, p.gate).split((64, 256), dim=-1)
            gate, norm_gate = a.fg_b_proj(low)
        mixed = self.native.qkv(x, p.qkv, sink)
        main.wait_stream(side)
        for tensor in (beta, gate, norm_gate):
            tensor.record_stream(main)
        core = recurrent(
            mixed,
            gate,
            beta,
            a.qkv_conv1d.weight,
            None,
            a.A_log,
            a.dt_bias,
            cache[0],
            cache[1],
            metadata,
            64,
            128,
            a.lower_bound,
            norm_gate,
            a.o_norm.weight,
            a.o_norm.eps,
        )
        return self.project(a.layer_idx, core)

    def sparse(self, p, x, cache, md, state_ids, km):
        from glm53_r11_artifact import bf16_bmm, kpool

        from lmdeploy.pytorch.backends.cuda.kpool import kpool_expand_groups_cuda, kpool_raw_decode_update_cuda
        from lmdeploy.pytorch.kernels.cuda.kpool import _score_pages

        a, indexer = p.attn, p.attn.indexer
        main, side = torch.cuda.current_stream(), self.attn_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            key, score = F.linear(x, p.kpool).split(128, dim=-1)
            key = indexer.k_norm(key.float()).to(key.dtype)[0]
            kc = indexer.get_block_cache()
            kpool_raw_decode_update_cuda(
                key,
                score[0],
                *cache[-2:],
                state_ids,
                md.q_seqlens,
                md.kv_seqlens,
                md.cu_seqlens_q,
                kc,
                md.block_offsets,
                indexer.index_kpool_compress_ape,
                4,
                indexer.scale_fmt is not None,
            )
        low, latent = a.fused_qkv_a_proj(x).split([1536, 512], dim=-1)
        low = a.q_a_layernorm(low)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            q, weights = kpool.query(indexer, x, low)
            pages, lengths = km.block_table, km.group_lengths
            width = pages.size(1) * kc.size(1)
            scores = torch.empty((8, width), dtype=torch.float32, device=x.device)
            _score_pages[(8, 64)](
                q,
                weights,
                kc,
                lengths,
                pages,
                scores,
                q.stride(0),
                q.stride(1),
                weights.stride(0),
                pages.stride(0),
                kc.stride(0),
                kc.size(1),
                32,
                128,
                width,
                32,
                128,
                128,
                WIDTH_BOUND=triton.next_power_of_2(width),
                BF16=False,
            )
            groups = self.native.topk(scores, lengths)
            indices = kpool_expand_groups_cuda(groups, lengths, 4, 2048, seq_lens=km.seq_lens)
        query = fp8_linear(a.q_b_proj, low).view(8, 64, 256)
        key, value, _ = a._kv_proj(latent[0, :, None], 512)
        absorbed = query.new_empty((8, 64, 512))
        bf16_bmm.forward(query, p.kc, absorbed)
        impl, kv = a.decode_attn_fwd, cache[0]
        impl._fill_kv_cache_impl(key, value, kv, kv[..., :512], md, 1, k_scales_zeros=None, v_scales_zeros=None)
        main.wait_stream(side)
        indices.record_stream(main)
        stride = impl._BF16_CACHE_INDEX_STRIDE
        last = (kv.size(0) - 1) * kv.stride(0) + (kv.size(1) - 1) * kv.stride(1)
        storage = kv.as_strided((last // stride + 1, *kv.shape[2:]), (stride, *kv.stride()[2:]))
        physical = impl.index_mapper.map_strided_decode(
            indices, md.block_offsets, 1, kv.size(1), kv.stride(0), kv.stride(1), stride
        ).flatten(0, 1)[:, None]
        physical = F.pad(physical, (0, -physical.size(-1) % 128), value=-1)
        output = self.native.mla(absorbed, storage, physical, impl.scale)
        projected = output.new_empty((8, 64, 256))
        bf16_bmm.forward(output, p.vc, projected)
        return a.o_proj(projected.flatten(-2)[None])

    def moe(self, p, x, sink):
        from glm53_r11_artifact.r8 import select

        from lmdeploy.pytorch.models.glm5_next import _glm_swiglu_impl

        mlp = p.layer.mlp
        main, side = torch.cuda.current_stream(), self.shared_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            shared = mlp.shared_experts
            gate_up = fp8_linear(shared.gate_up_proj, x)
            shared_out = fp8_linear(shared.down_proj, _glm_swiglu_impl(gate_up, shared.swiglu_limit, precise_mul=True))
        hidden = x.view(8, 4096)
        logits = mlp.gate.router_gemm(hidden, mlp.gate.weight)
        weights, ids = select(logits, mlp.gate.e_score_correction_bias)
        out = self.ep.run(hidden, weights, ids, p.expert_weights, sink)
        main.wait_stream(side)
        shared_out.record_stream(main)
        return out.view_as(x), shared_out

    def run(
        self,
        plan,
        input_ids,
        position_ids,
        past_key_values,
        attn_metadata,
        state_ids,
        inputs_embeds=None,
        return_input_embeds=False,
        **kwargs,
    ):
        from glm53_r11_artifact.hc_post import hc_post_expand

        from lmdeploy.pytorch.backends.cuda.kpool import kpool_decode_metadata_cuda

        backbone = self.model.model
        embeds = backbone.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = embeds.unsqueeze(2).repeat(1, 1, 4, 1)
        kda = backbone.gated_delta_meta_builder(8, backbone.config.linear_conv_kernel_dim, state_ids, attn_metadata)
        km = kpool_decode_metadata_cuda(attn_metadata, 8, 4, True, page_size=16)
        for p, cache in zip(self.layers, past_key_values):
            layer = p.layer
            x, post, comb = self.native.hc(
                hidden,
                p.hc[0],
                p.work[0],
                layer.hc_attn_base,
                layer.hc_attn_scale,
                layer.input_layernorm.weight,
                p.qkv,
                p.qkv is not None,
            )
            if p.qkv is None:
                x = self.sparse(p, x, cache, attn_metadata, state_ids, km)
            else:
                x = self.kda(p, x, cache, kda, comb)
            hidden = hc_post_expand(x, hidden, post, comb, 4)
            x, post, comb = self.native.hc(
                hidden,
                p.hc[1],
                p.work[1],
                layer.hc_ffn_base,
                layer.hc_ffn_scale,
                layer.post_attention_layernorm.weight,
                None,
                p.expert_weights is not None,
            )
            shared = None
            if p.expert_weights is None:
                x = layer.mlp(x)
            else:
                x, shared = self.moe(p, x, comb)
            hidden = hc_post_expand(x, hidden, post, comb, 4, shared=shared)
        output = backbone.norm(hidden.mean(dim=2))
        return dict(hidden_states=output, target_inputs_embeds=embeds) if return_input_embeds else output

    def close(self):
        self.moes.clear()
        self.layers.clear()
        if self.ep is not None:
            self.ep.buffer.destroy()
            self.ep = None
        super().close()
