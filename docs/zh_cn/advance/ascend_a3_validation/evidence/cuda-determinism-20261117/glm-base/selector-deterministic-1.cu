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
  void* counts_scan = ((void*)((char*)buf_dyn_shmem + 0));
  void* histogram = ((void*)((char*)buf_dyn_shmem + 4096));
  void* warp_totals = ((void*)((char*)buf_dyn_shmem + 5120));
  void* shared_state = ((void*)((char*)buf_dyn_shmem + 5248));
  uint prefix_key = (uint)0;
  uint prefix_mask = (uint)0;
  int rank = 0;
  int emitted = 0;
  int equal_seen = 0;
  int pos = 0;
  int above_count = 0;
  int below_count = 0;
  uint stable_key = (uint)0;
  uint prefix = (uint)0;
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
        float condval_5;
        if ((Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos))] == 0x0p+0f/*0.000000e+00*/)) {
          condval_5 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_5 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos))];
        }
        float _reinterpret_tmp = condval_5;
        uint condval_4;
        if ((((*(uint *)(&(_reinterpret_tmp))) & (uint)2147483648) == (uint)0)) {
          float condval_6;
          if ((Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos))] == 0x0p+0f/*0.000000e+00*/)) {
            condval_6 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_6 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos))];
          }
          float _reinterpret_tmp_1 = condval_6;
          condval_4 = ((*(uint *)(&(_reinterpret_tmp_1))) ^ (uint)2147483648);
        } else {
          float condval_7;
          if ((Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos))] == 0x0p+0f/*0.000000e+00*/)) {
            condval_7 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_7 = Scores[((((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride)) + ((int64_t)pos))];
          }
          float _reinterpret_tmp_2 = condval_7;
          condval_4 = ((*(uint *)(&(_reinterpret_tmp_2))) ^ (uint)4294967295);
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
    emitted = 0;
    equal_seen = 0;
    int condval_8;
    if ((raw_seqlen < score_width)) {
      condval_8 = raw_seqlen;
    } else {
      condval_8 = score_width;
    }
    for (int tile = 0; tile < ((condval_8 + 1023) >> 10); ++tile) {
      stable_key = (uint)0;
      int condval_9;
      if ((raw_seqlen < score_width)) {
        condval_9 = raw_seqlen;
      } else {
        condval_9 = score_width;
      }
      if (((tile * 1024) + ((int)threadIdx.x)) < condval_9) {
        float condval_11;
        if ((Scores[(((((int64_t)tile) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride))) + ((int64_t)((int)threadIdx.x)))] == 0x0p+0f/*0.000000e+00*/)) {
          condval_11 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_11 = Scores[(((((int64_t)tile) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride))) + ((int64_t)((int)threadIdx.x)))];
        }
        float _reinterpret_tmp_3 = condval_11;
        uint condval_10;
        if ((((*(uint *)(&(_reinterpret_tmp_3))) & (uint)2147483648) == (uint)0)) {
          float condval_12;
          if ((Scores[(((((int64_t)tile) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride))) + ((int64_t)((int)threadIdx.x)))] == 0x0p+0f/*0.000000e+00*/)) {
            condval_12 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_12 = Scores[(((((int64_t)tile) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride))) + ((int64_t)((int)threadIdx.x)))];
          }
          float _reinterpret_tmp_4 = condval_12;
          condval_10 = ((*(uint *)(&(_reinterpret_tmp_4))) ^ (uint)2147483648);
        } else {
          float condval_13;
          if ((Scores[(((((int64_t)tile) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride))) + ((int64_t)((int)threadIdx.x)))] == 0x0p+0f/*0.000000e+00*/)) {
            condval_13 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_13 = Scores[(((((int64_t)tile) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)score_stride))) + ((int64_t)((int)threadIdx.x)))];
          }
          float _reinterpret_tmp_5 = condval_13;
          condval_10 = ((*(uint *)(&(_reinterpret_tmp_5))) ^ (uint)4294967295);
        }
        stable_key = condval_10;
      }
      int condval_14;
      if ((raw_seqlen < score_width)) {
        condval_14 = raw_seqlen;
      } else {
        condval_14 = score_width;
      }
      int condval_15;
      if ((raw_seqlen < score_width)) {
        condval_15 = raw_seqlen;
      } else {
        condval_15 = score_width;
      }
      ((uint*)counts_scan)[((int)threadIdx.x)] = (((uint)((((tile * 1024) + ((int)threadIdx.x)) < condval_14) && (threshold_key < stable_key))) + (((uint)((((tile * 1024) + ((int)threadIdx.x)) < condval_15) && (stable_key == threshold_key))) << (uint)16));
      __syncthreads();
      tl::CumSum2D<1024, 1, false>::run((&(((uint*)counts_scan)[0])), (&(((uint*)counts_scan)[0])), 32, 32);
      __syncthreads();
      if (((int)threadIdx.x) < 32) {
        ((uint*)warp_totals)[((int)threadIdx.x)] = ((uint*)counts_scan)[((((int)threadIdx.x) * 32) + 31)];
      }
      __syncthreads();
      tl::CumSum1D<1024, false>::run((&(((uint*)warp_totals)[0])), (&(((uint*)warp_totals)[0])), 32);
      __syncthreads();
      prefix = ((uint*)counts_scan)[((int)threadIdx.x)];
      if (32 <= ((int)threadIdx.x)) {
        prefix = (prefix + ((uint*)warp_totals)[((((int)threadIdx.x) >> 5) - 1)]);
      }
      int greater_prefix = ((int)(prefix & (uint)65535));
      int equal_prefix = ((int)(prefix >> (uint)16));
      int condval_16;
      if ((raw_seqlen < score_width)) {
        condval_16 = raw_seqlen;
      } else {
        condval_16 = score_width;
      }
      bool selected = ((((tile * 1024) + ((int)threadIdx.x)) < condval_16) && ((threshold_key < stable_key) || ((stable_key == threshold_key) && ((equal_seen + equal_prefix) <= rank))));
      if (selected) {
        Out[((((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + min((((int64_t)equal_seen) + ((int64_t)equal_prefix)), ((int64_t)rank))) + ((int64_t)emitted)) + ((int64_t)greater_prefix)) - ((int64_t)min(equal_seen, rank))) - (int64_t)1)] = ((tile * 1024) + ((int)threadIdx.x));
      }
      int greater_total = ((int)(((uint*)warp_totals)[31] & (uint)65535));
      int equal_total = ((int)(((uint*)warp_totals)[31] >> (uint)16));
      emitted = (((min((equal_seen + equal_total), rank) + emitted) + greater_total) - min(equal_seen, rank));
      equal_seen = (equal_seen + equal_total);
      __syncthreads();
    }
  }
}

