#!/usr/bin/env bash
set -euo pipefail

url=${1:?AgentX endpoint URL}
concurrency=${2:?Total concurrency}
output=${3:?Artifact directory}
shift 3
aiperf=${AIPERF_BIN:-aiperf}
if [[ -n "${AGENTX_ROOT:-}" ]]; then
  aiperf=${AIPERF_BIN:-"$AGENTX_ROOT/.venv-agentx/bin/aiperf"}
  export HF_HOME=${HF_HOME:-"$AGENTX_ROOT/cache/huggingface"}
fi
tokenizer=${AGENTX_TOKENIZER:-zai-org/GLM-5.3}
revision=${AGENTX_TOKENIZER_REVISION:-e0b07fd2751b42d5efa199cc02c2b271deadc516}
duration=${AGENTX_DURATION:-1800}
warmup=${AGENTX_WARMUP_REQUESTS_PER_LANE:-10}
grace=${AGENTX_WARMUP_GRACE:-1800}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_HTTP_KEEPALIVE_TIMEOUT=4
export AIPERF_DATASET_MMAP_CACHE_DIR=${AIPERF_DATASET_MMAP_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/lmdeploy-agentx/mmap-cache}
export AIPERF_DATASET_MMAP_BASE_PATH=${AIPERF_DATASET_MMAP_BASE_PATH:-${XDG_CACHE_HOME:-$HOME/.cache}/lmdeploy-agentx/mmap-runs}
mkdir -p "$AIPERF_DATASET_MMAP_CACHE_DIR" "$AIPERF_DATASET_MMAP_BASE_PATH"
cmd=("$aiperf" profile --url "$url" --model "${AGENTX_MODEL:-glm53-flash}"
  --tokenizer "$tokenizer" --tokenizer-revision "$revision"
  --endpoint /v1/chat/completions --endpoint-type chat --streaming --use-server-token-count
  --concurrency "$concurrency" --random-seed "${AGENTX_SEED:-42}" --ui simple --artifact-dir "$output"
  --scenario inferencex-agentx-mvp --public-dataset semianalysis_cc_traces_weka_062126_256k
  --max-context-length 256000 --benchmark-duration "$duration"
  --goodput 'time_to_first_token:3000 output_token_throughput_per_user:40'
  --extra-inputs ignore_eos:true --cache-bust first_turn_prefix
  --trace-idle-gap-cap-seconds 300 --system-idle-gap-cap-seconds 10
  --trajectory-start-min-ratio 0 --trajectory-start-max-ratio 1
  --warmup-requests-per-lane "$warmup" --agentic-warmup-grace-period "$grace")
if (( duration < 1800 )) || (( warmup < 10 )); then
  cmd+=(--unsafe-override)
fi
cmd+=("$@")
mkdir -p "$output"
test ! -e "$output/profile_export.jsonl"
printf '%q ' "${cmd[@]}" > "$output/command.sh"
printf '\n' >> "$output/command.sh"
exec "${cmd[@]}"
