#!/usr/bin/env bash
set -euo pipefail

backend=${1:?default, deepgemm_mega_moe, or flashinfer_mega_moe}
blocks=${2:?GPU blocks per rank}
run=${3:?new run directory name}
shift 3
[[ "$run" =~ ^[a-zA-Z0-9_.-]+$ && "$run" != . && "$run" != .. ]]

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$source_dir"
root=${MEGA_MOE_ARTIFACTS:?host path of the container /artifacts bind mount}
container=${MEGA_MOE_CONTAINER:-glm53-mega-moe-lmd}
model=${MODEL_SNAPSHOT:?host path to the local GLM-5.3-Flash snapshot}
container_model=${CONTAINER_MODEL_SNAPSHOT:-/model/snapshots/${model##*/}}
python=${BENCHMARK_PYTHON:-python3}
port=${GLM53_PORT:-20990}
monitoring_port=${GLM53_MONITORING_PORT:-20980}
router_port=${GLM53_ROUTER_PORT:-20970}
router_metrics_port=${GLM53_ROUTER_METRICS_PORT:-20969}
concurrencies=${AGENTX_CONCURRENCIES-8 32}

mkdir -p "$root/runs"
exec 9>"$root/gpu.lock"
flock -n 9
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'The 8 GPUs must be available for the serving benchmark.' >&2
    exit 1
fi
output="$root/runs/$run"
mkdir "$output"
container_output="/artifacts/runs/$run"
ip=$(docker inspect -f '{{.NetworkSettings.IPAddress}}' "$container")
test -n "$ip"

case "$backend" in
    default) app_path=/src ;;
    deepgemm_mega_moe) app_path="${MEGA_MOE_DEEPGEMM_PATH:-/artifacts/deps/deepgemm-mega}:/src" ;;
    flashinfer_mega_moe)
        app_path="${MEGA_MOE_FLASHINFER_PATH:-/artifacts/build/flashinfer-0.7.4}:/artifacts/deps/flashinfer"
        app_path+=":/artifacts/deps/flashinfer/nvidia_cutlass_dsl/dsl_packages:/src"
        if [[ "${LMDEPLOY_GLM53_FLASHINFER_WEIGHT_LAYOUT:-dual}" == interleaved* ]]; then
            app_path="${MEGA_MOE_DEEPGEMM_PATH:-/artifacts/deps/deepgemm-mega}:$app_path"
        fi
        ;;
    *) echo "Unknown backend $backend" >&2; exit 2 ;;
esac

log_command() {
    local destination=$1
    shift
    printf '%q ' "$@" > "$destination"
    printf '\n' >> "$destination"
}

stop_container_task() {
    local name=$1 exec_pid=$2
    if [[ -s "$output/$name.pid" ]] && kill -0 "$exec_pid" 2>/dev/null; then
        docker exec "$container" kill -TERM "$(cat "$output/$name.pid")" >/dev/null 2>&1 || true
    fi
    wait "$exec_pid" 2>/dev/null || true
}

server_pid=
router_pid=
gpu_monitor=
cleanup() {
    local status=$?
    if [[ -n "$gpu_monitor" ]]; then
        kill "$gpu_monitor" 2>/dev/null || true
        wait "$gpu_monitor" 2>/dev/null || true
    fi
    if [[ -n "$router_pid" ]]; then stop_container_task router "$router_pid"; fi
    if [[ -n "$server_pid" ]]; then stop_container_task server "$server_pid"; fi
    printf '%s\n' "$status" > "$output/exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_ready() {
    local url=$1 pid=$2 deadline=$((SECONDS + 480))
    while ((SECONDS <= deadline)); do
        kill -0 "$pid" 2>/dev/null || return 1
        if curl -fsS --noproxy '*' --max-time 2 "$url" >/dev/null 2>&1; then return 0; fi
        sleep 2
    done
    echo "Readiness timed out for $url" >&2
    return 1
}

urls=()
metrics=()
for i in {0..7}; do
    urls+=("http://$ip:$((port + i))")
    metrics+=("http://$ip:$((monitoring_port + i))/metrics")
done
server_env=()
if [[ -n "${MEGA_MOE_SERVER_ENV_FILE:-}" ]]; then
    cp -- "$MEGA_MOE_SERVER_ENV_FILE" "$output/server.env"
    server_env=(--env-file "$output/server.env")
fi
server=(docker exec "${server_env[@]}" -w /src -e "PYTHONPATH=$app_path" -e "GLM53_PORT=$port" -e "GLM53_MONITORING_PORT=$monitoring_port"
    -e "GLM53_MAX_BATCH_SIZE=${GLM53_MAX_BATCH_SIZE:-32}"
    -e "GLM53_MAX_PREFILL_TOKENS=${GLM53_MAX_PREFILL_TOKENS:-8192}"
    -e "GLM53_GRAPH_BATCH_SIZES=${GLM53_GRAPH_BATCH_SIZES:-}"
    -e "GLM53_PCG_TOKENS=${GLM53_PCG_TOKENS:-}"
    -e "GLM53_SPEC_TOKENS=${GLM53_SPEC_TOKENS:-0}"
    -e "GLM53_SESSION_LEN=${GLM53_SESSION_LEN:-524288}"
    -e "LMDEPLOY_GLM53_FLASHKDA_LIBRARY=${LMDEPLOY_GLM53_FLASHKDA_LIBRARY:-}"
    -e "LMDEPLOY_DP_SYNC_METADATA=${LMDEPLOY_DP_SYNC_METADATA:-0}"
    -e "LMDEPLOY_GLM53_MEGA_PREFILL_CAPACITY=${LMDEPLOY_GLM53_MEGA_PREFILL_CAPACITY:-0}"
    -e "LMDEPLOY_GLM53_DEEPGEMM_FUSE_SHARED=${LMDEPLOY_GLM53_DEEPGEMM_FUSE_SHARED:-0}"
    -e "LMDEPLOY_GLM53_DEEPGEMM_WEIGHT_LAYOUT=${LMDEPLOY_GLM53_DEEPGEMM_WEIGHT_LAYOUT:-dual}"
    -e "LMDEPLOY_GLM53_FLASHINFER_EXACT_SCALES=${LMDEPLOY_GLM53_FLASHINFER_EXACT_SCALES:-0}"
    -e "LMDEPLOY_GLM53_FLASHINFER_WEIGHT_LAYOUT=${LMDEPLOY_GLM53_FLASHINFER_WEIGHT_LAYOUT:-dual}"
    -e "LMDEPLOY_GLM53_FLASHINFER_KNOBS=${LMDEPLOY_GLM53_FLASHINFER_KNOBS:-}"
    "$container" bash -c 'printf "%s\n" "$$" > "$1"; shift; exec "$@"' _
    "$container_output/server.pid" bash benchmark/glm53_mega_moe/serve.sh
    "$container_model" "$backend" --num-gpu-blocks "$blocks")
log_command "$output/server-command.sh" "${server[@]}"
"${server[@]}" > "$output/server.log" 2>&1 &
server_pid=$!

for url in "${urls[@]}"; do wait_ready "$url/v1/models" "$server_pid"; done
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv > "$output/idle-gpu.csv"

if [[ "${SKIP_FIXED_IO:-0}" != 1 ]]; then
    fixed=("$python" benchmark/glm53_mega_moe/fixed_io.py --urls "${urls[@]}"
           --tokenizer "$model" --output "$output/fixed" "$@")
    log_command "$output/fixed-command.sh" "${fixed[@]}"
    "${fixed[@]}" > "$output/fixed.log" 2>&1
fi
if [[ -z "$concurrencies" ]]; then exit 0; fi

router=(docker exec "$container" bash -c 'printf "%s\n" "$$" > "$1"; shift; exec "$@"' _
    "$container_output/router.pid" lmdeploy-router --host 0.0.0.0 --port "$router_port"
    --prometheus-host 0.0.0.0 --prometheus-port "$router_metrics_port" --policy sticky_least_loaded
    --worker-urls "${urls[@]}" --intra-node-data-parallel-size 1
    --request-timeout-secs 18000 --health-check-interval-secs 5)
log_command "$output/router-command.sh" "${router[@]}"
"${router[@]}" > "$output/router.log" 2>&1 &
router_pid=$!
wait_ready "http://$ip:$router_port/health" "$router_pid"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$ip,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

for concurrency in $concurrencies; do
    phase="$output/agentx-c$concurrency"
    mkdir "$phase"
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -lms 1000 \
        > "$phase/gpu.csv" &
    gpu_monitor=$!
    bash benchmark/glm53_mega_moe/agentx.sh "http://$ip:$router_port" "$concurrency" "$phase" \
        --server-metrics "http://$ip:$router_metrics_port/metrics" "${metrics[@]}" > "$phase.log" 2>&1
    kill "$gpu_monitor" 2>/dev/null || true
    wait "$gpu_monitor" 2>/dev/null || true
    gpu_monitor=
done
