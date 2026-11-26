#!/usr/bin/env bash
# Run the copy under BUNDLE/sources. All build products go to a new directory.
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
bundle=$(dirname "$src")
dest=${1:?usage: bash BUNDLE/sources/rebuild_r11.sh NEW_DIRECTORY}
mkdir "$dest"
dest=$(cd "$dest" && pwd)
work="$dest/build"
pkg="$dest/glm53_r11_artifact"
mkdir -p "$work" "$pkg"
cp "$bundle/glm53_r11_artifact/"*.py "$pkg/"
cuda=${CUDA_HOME:-/usr/local/cuda}
site=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
dg_args=()
mla_args=()
if [[ -n ${R11_DEEPGEMM_SOURCE:-} ]]; then dg_args+=(--source "$R11_DEEPGEMM_SOURCE"); fi
if [[ -n ${R11_FLASHMLA_SOURCE:-} ]]; then mla_args+=(--flashmla-source "$R11_FLASHMLA_SOURCE"); fi
if [[ -n ${R11_CUTLASS_SOURCE:-} ]]; then mla_args+=(--cutlass-source "$R11_CUTLASS_SOURCE"); fi
if [[ -n ${R11_ELFUTILS_INCLUDE:-} ]]; then dg_args+=(--elfutils-include "$R11_ELFUTILS_INCLUDE"); fi
python "$src/build_megagemm.py" --r6 --tiled-weights --expert-tile 32 --build-dir "$work/dg" --output "$dest" "${dg_args[@]}"
python "$src/build_decode_native.py" --r5 --build-dir "$work/mla" --output "$work/r5" "${mla_args[@]}"
python "$src/build_r8.py" --output "$work/r8"
python "$src/build_r09.py" --output "$work/r09"
python "$src/build_r11.py" --output "$work/r11" --r10-include "$src/paired8/include" \
    --provider-include "$dest/lmdeploy_glm53_megagemm_r6/include"
"$cuda/bin/nvcc" -O3 -gencode=arch=compute_90a,code=sm_90a -std=c++17 --shared -Xcompiler=-fPIC,-fvisibility=hidden \
    -I"$src/native/decode" -I"$site/tilelang/src" -I"$site/tilelang/3rdparty/cutlass/include" \
    -DENABLE_BF16 -DGLM53_EMBED_SINK "$src/packet.cu" -o "$pkg/libpacket_sink.so"
cp "$work/r5/libglm53_mla.so" "$work/r8/qkv_sink.cubin" "$work/r8/libqkv_sink.so" "$work/r09/libhc_vector.so" \
    "$work/r09/libglm53_topk_r09.so" "$work/r11/librouted.so" "$pkg/"
python "$src/build_device_loop.py" --output "$pkg/libdevice_loop.so"
bash "$src/projection/build.sh" "$pkg"
