#!/usr/bin/env bash
# Build the optional example out of tree with the CUDA 13 driver headers.
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: build.sh OUTPUT_DIRECTORY}
cuda=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$out"
"$cuda/bin/ptxas" -O3 -arch=sm_90a "$src/kda_output_r11_compact.ptx" -o "$out/kda_output.cubin"
"${CXX:-c++}" -O3 -std=c++17 -shared -fPIC -fvisibility=hidden -I"$cuda/include" \
    "$src/kda_output_r11.cpp" -L"$cuda/lib64/stubs" -lcuda -o "$out/libglm53_projection_v1.so"
