#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void sparse_index_topk_byte_radix_kernel__kernel(int* __restrict__ Out, const float* __restrict__ Scores, const int* __restrict__ Seqlens, int num_tokens, int score_stride, int score_width);
extern "C" __global__ void __launch_bounds__(1024, 1) sparse_index_topk_byte_radix_kernel__kernel(int* __restrict__ Out, const float* __restrict__ Scores, const int* __restrict__ Seqlens, int num_tokens, int score_stride, int score_width) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* histogram = ((void*)((char*)buf_dyn_shmem + 0));
  void* shared_state = ((void*)((char*)buf_dyn_shmem + 1024));
  uint prefix_key = (uint)0;
  uint prefix_mask = (uint)0;
  int rank = 0;
  int pos_emit_gt = 0;
  int out_pos_buf[1];
  int pos_emit_eq = 0;
  int pos = 0;
  int above_count = 0;
  int below_count = 0;
  int raw_seqlen = Seqlens[((int64_t)((int)blockIdx.x))];
  int condval;
  if ((raw_seqlen < score_width)) {
    condval = raw_seqlen;
  } else {
    condval = score_width;
  }
  if (condval <= 2048) {
    for (int i_s = 0; i_s < 2; ++i_s) {
      int condval_2;
      if ((raw_seqlen < score_width)) {
        condval_2 = raw_seqlen;
      } else {
        condval_2 = score_width;
      }
      int condval_1;
      if ((((((int)threadIdx.x) * 2) + i_s) < condval_2)) {
        condval_1 = ((((int)threadIdx.x) * 2) + i_s);
      } else {
        condval_1 = -1;
      }
      Out[(((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + (((int64_t)((int)threadIdx.x)) * (int64_t)2)) + ((int64_t)i_s))] = condval_1;
    }
  } else {
    prefix_key = (uint)0;
    prefix_mask = (uint)0;
    rank = 2048;
    #pragma unroll
    for (int round_idx = 0; round_idx < 4; ++round_idx) {
      uint byte_mask = ((uint)255 << ((uint)(24 - (round_idx * 8))));
      if (((int)threadIdx.x) < 256) {
        ((int*)histogram)[((int)threadIdx.x)] = 0;
      }
      pos = ((int)threadIdx.x);
      __syncthreads();
      while (1) {
        int condval_3;
        if ((raw_seqlen < score_width)) {
          condval_3 = raw_seqlen;
        } else {
          condval_3 = score_width;
        }
        if (!((pos < condval_3))) { break; }
        float v_ = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos))];
        uint condval_4;
        if ((((*(uint *)(&(v_))) & (uint)2147483648) == (uint)0)) {
          float v__1 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos))];
          condval_4 = ((*(uint *)(&(v__1))) ^ (uint)2147483648);
        } else {
          float v__2 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos))];
          condval_4 = ((*(uint *)(&(v__2))) ^ (uint)4294967295);
        }
        uint key = condval_4;
        if ((key & prefix_mask) == prefix_key) {
          AtomicAdd((&(((int*)histogram)[((int)((key >> ((uint)(24 - (round_idx * 8)))) & (uint)255))])), 1);
        }
        pos = (pos + 1024);
      }
      __syncthreads();
      tl::CumSum1D<1024, false>::run((&(((int*)histogram)[0])), (&(((int*)histogram)[0])), 256);
      __syncthreads();
      if (((int)threadIdx.x) < 256) {
        above_count = (((int*)histogram)[255] - ((int*)histogram)[((int)threadIdx.x)]);
        below_count = 0;
        if (0 < ((int)threadIdx.x)) {
          below_count = ((int*)histogram)[(((int)threadIdx.x) - 1)];
        }
        if ((above_count < rank) && (rank <= (((int*)histogram)[255] - below_count))) {
          ((int*)shared_state)[0] = ((int)threadIdx.x);
          ((int*)shared_state)[1] = above_count;
        }
      }
      __syncthreads();
      int threshold_bin = ((int*)shared_state)[0];
      prefix_key = (prefix_key | (((uint)threshold_bin) << ((uint)(24 - (round_idx * 8)))));
      prefix_mask = (prefix_mask | ((uint)255 << ((uint)(24 - (round_idx * 8)))));
      rank = (rank - ((int*)shared_state)[1]);
    }
    uint threshold_key = prefix_key;
    __syncthreads();
    if (((int)threadIdx.x) == 0) {
      ((int*)shared_state)[0] = 0;
      ((int*)shared_state)[1] = 0;
    }
    __syncthreads();
    pos_emit_gt = ((int)threadIdx.x);
    while (1) {
      int condval_5;
      if ((raw_seqlen < score_width)) {
        condval_5 = raw_seqlen;
      } else {
        condval_5 = score_width;
      }
      if (!((pos_emit_gt < condval_5))) { break; }
      float v__3 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos_emit_gt))];
      uint condval_6;
      if ((((*(uint *)(&(v__3))) & (uint)2147483648) == (uint)0)) {
        float v__4 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos_emit_gt))];
        condval_6 = ((*(uint *)(&(v__4))) ^ (uint)2147483648);
      } else {
        float v__5 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos_emit_gt))];
        condval_6 = ((*(uint *)(&(v__5))) ^ (uint)4294967295);
      }
      uint key_1 = condval_6;
      if (threshold_key < key_1) {
        out_pos_buf[0] = AtomicAddRet((&(((int*)shared_state)[0])), 1);
        if (out_pos_buf[0] < 2048) {
          Out[((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((int64_t)out_pos_buf[(int64_t)0]))] = pos_emit_gt;
        }
      }
      pos_emit_gt = (pos_emit_gt + 1024);
    }
    __syncthreads();
    int gt_count = ((int*)shared_state)[0];
    pos_emit_eq = ((int)threadIdx.x);
    __syncthreads();
    while (1) {
      int condval_7;
      if ((raw_seqlen < score_width)) {
        condval_7 = raw_seqlen;
      } else {
        condval_7 = score_width;
      }
      if (!((pos_emit_eq < condval_7))) { break; }
      float v__6 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos_emit_eq))];
      uint condval_8;
      if ((((*(uint *)(&(v__6))) & (uint)2147483648) == (uint)0)) {
        float v__7 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos_emit_eq))];
        condval_8 = ((*(uint *)(&(v__7))) ^ (uint)2147483648);
      } else {
        float v__8 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos_emit_eq))];
        condval_8 = ((*(uint *)(&(v__8))) ^ (uint)4294967295);
      }
      uint key_2 = condval_8;
      if (key_2 == threshold_key) {
        out_pos_buf[0] = (gt_count + AtomicAddRet((&(((int*)shared_state)[1])), 1));
        if (out_pos_buf[0] < 2048) {
          Out[((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((int64_t)out_pos_buf[(int64_t)0]))] = pos_emit_eq;
        }
      }
      pos_emit_eq = (pos_emit_eq + 1024);
    }
  }
}

