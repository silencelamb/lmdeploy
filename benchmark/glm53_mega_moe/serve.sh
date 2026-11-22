#!/usr/bin/env bash
set -euo pipefail

model=${1:?GLM-5.3-Flash snapshot}
backend=${2:?default, deepgemm_mega_moe, or flashinfer_mega_moe}
shift 2
export LMDEPLOY_DP_MASTER_ADDR=${LMDEPLOY_DP_MASTER_ADDR:-127.0.0.1}
export LMDEPLOY_DP_MASTER_PORT=${LMDEPLOY_DP_MASTER_PORT:-20883}
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
graph_args=()
if [[ -n "${GLM53_GRAPH_BATCH_SIZES:-}" ]]; then
  read -r -a graph_sizes <<< "$GLM53_GRAPH_BATCH_SIZES"
  graph_args=(--cudagraph-capture-batch-sizes "${graph_sizes[@]}")
fi
if [[ -n "${GLM53_PCG_TOKENS:-}" ]]; then
  graph_args+=(--piecewise-cudagraph-max-tokens "$GLM53_PCG_TOKENS")
fi
spec_args=()
spec_tokens=${GLM53_SPEC_TOKENS:-0}
if [[ "$spec_tokens" != 0 ]]; then
  [[ "$spec_tokens" =~ ^[1-9][0-9]*$ ]]
  spec_args=(--speculative-algorithm deepseek_mtp --speculative-num-draft-tokens "$spec_tokens")
fi
exec lmdeploy serve api_server "$model" --backend pytorch --distributed-executor-backend ray \
  --tp 1 --dp 8 --ep 8 --moe-backend "$backend" --trust-remote-code --language-model-only \
  --model-name glm53-flash --server-port "${GLM53_PORT:-20990}" --monitoring-port "${GLM53_MONITORING_PORT:-20980}" \
  --max-batch-size "${GLM53_MAX_BATCH_SIZE:-32}" --max-prefill-token-num "${GLM53_MAX_PREFILL_TOKENS:-8192}" \
  --cache-block-seq-len 64 --session-len "${GLM53_SESSION_LEN:-524288}" \
  --cache-max-entry-count 0.8 --enable-prefix-caching --prefix-cache-state-budget 32 \
  --prefill-save-alignment 64 --prefix-cache-decode-save-first --prefix-cache-decode-state-interval 1024 \
  --reasoning-parser default --tool-call-parser glm47 --log-level INFO "${graph_args[@]}" "${spec_args[@]}" "$@"
