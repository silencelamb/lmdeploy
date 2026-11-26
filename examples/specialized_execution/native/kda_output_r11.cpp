// Copyright (c) OpenMMLab. All rights reserved.
#include <cuda.h>
#include <cstdint>

namespace {
constexpr unsigned kSharedBytes = 55320;
}

extern "C" __attribute__((visibility("default"))) int
glm53_projection_open_v1(const char* path, CUmodule* owner, CUfunction* kernel) {
    CUmodule module;
    auto status = cuModuleLoad(&module, path);
    if (status) return status;
    status = cuModuleGetFunction(kernel, module, "glm53_kda_output_r11");
    if (!status) status = cuFuncSetAttribute(*kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                             kSharedBytes);
    if (status) cuModuleUnload(module);
    else *owner = module;
    return status;
}

extern "C" __attribute__((visibility("default"))) int
glm53_output_launch(CUfunction kernel, void* x, void* weights, void* partials, void* stream) {
    alignas(64) CUtensorMap desc[2];
    void* addresses[] = {x, partials};
    constexpr uint64_t cols[] = {8192, 4096}, rows[] = {8, 24};
    for (int i = 0; i < 2; ++i) {
        const bool fp32 = i == 1;
        uint64_t dims[2] = {cols[i], rows[i]}, strides[1] = {cols[i] * (fp32 ? 4u : 2u)};
        uint32_t box[2] = {fp32 ? 32u : 64u, 8}, elements[2] = {1, 1};
        auto status = cuTensorMapEncodeTiled(desc + i, fp32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, addresses[i], dims, strides, box, elements,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (status) return status;
    }
    alignas(64) uint64_t unused_tensor_map[16] = {};
    uint64_t zero = 0;
    void* args[] = {unused_tensor_map, &zero, &zero, &zero, &zero,
                    &desc[0], &zero, &zero, &zero, &zero,
                    &desc[1], &zero, &zero, &zero, &zero, &weights, &zero};
    return cuLaunchKernel(kernel, 64, 3, 1, 128, 1, 1, kSharedBytes, (CUstream)stream, args, nullptr);
}

extern "C" __attribute__((visibility("default"))) int glm53_output_close(CUmodule module) {
    return cuModuleUnload(module);
}
