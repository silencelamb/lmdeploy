#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 + s3 * 128)>
#map2 = affine_map<()[s0] -> (128, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 32, s0)>
#map4 = affine_map<()[s0, s1] -> (s0 - s1)>
#map5 = affine_map<()[s0, s1] -> (s1 + 128, s0)>
#map6 = affine_map<()[s0, s1] -> (0, s0 - s1)>
#map7 = affine_map<()[s0] -> (32, s0)>
#map8 = affine_map<()[s0, s1] -> (s0 + s1)>
#map9 = affine_map<()[s0] -> (s0 + 1024)>
#map10 = affine_map<()[s0] -> (s0 + 2048)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = true, hacc.hivmc_version = #hacc.hivmc_version<"0.0.0">, hacc.target = #hacc.target<"Ascend910_9392">, hivm.aic_bitcode = #hivm.aic_bitcode<"/root/workspace/a3-mtp-ray-investigation-20261116/artifacts/upgrade5-wheel322p2-r0/triton/backends/ascend/bishengir/lib/meta_op.aic.c220.O0.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/root/workspace/a3-mtp-ray-investigation-20261116/artifacts/upgrade5-wheel322p2-r0/triton/backends/ascend/bishengir/lib/meta_op.aiv.c220.O0.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/root/workspace/a3-mtp-ray-investigation-20261116/artifacts/upgrade5-wheel322p2-r0/triton/backends/ascend/bishengir/lib/host.O0.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/root/workspace/a3-mtp-ray-investigation-20261116/artifacts/upgrade5-wheel322p2-r0/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.c220.O0.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/root/workspace/a3-mtp-ray-investigation-20261116/artifacts/upgrade5-wheel322p2-r0/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.c220.O0.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @fused_recurrent_gated_delta_rule_fwd_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_bfloat16_t_to_1d(bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bfloat16_t_to_float_1d_with_mode(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_sum_r_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_sum_ar_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsub_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_bfloat16_t_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_bfloat16_t(memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @fused_recurrent_gated_delta_rule_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: f32, %arg15: i64, %arg16: i64, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<20xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c103648_i64 = arith.constant 103648 : i64
    %c61984_i64 = arith.constant 61984 : i64
    %c94752_i64 = arith.constant 94752 : i64
    %c120032_i64 = arith.constant 120032 : i64
    %c1536_i64 = arith.constant 1536 : i64
    %c103584_i64 = arith.constant 103584 : i64
    %c768_i64 = arith.constant 768 : i64
    %c103328_i64 = arith.constant 103328 : i64
    %c103072_i64 = arith.constant 103072 : i64
    %c2_i64 = arith.constant 2 : i64
    %c94880_i64 = arith.constant 94880 : i64
    %c78368_i64 = arith.constant 78368 : i64
    %c60960_i64 = arith.constant 60960 : i64
    %c44576_i64 = arith.constant 44576 : i64
    %c44544_i64 = arith.constant 44544 : i64
    %c44512_i64 = arith.constant 44512 : i64
    %c36320_i64 = arith.constant 36320 : i64
    %c36192_i64 = arith.constant 36192 : i64
    %c19808_i64 = arith.constant 19808 : i64
    %c3424_i64 = arith.constant 3424 : i64
    %c3392_i64 = arith.constant 3392 : i64
    %c3360_i64 = arith.constant 3360 : i64
    %c3104_i64 = arith.constant 3104 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %c2560_i64 = arith.constant 2560 : i64
    %c2528_i64 = arith.constant 2528 : i64
    %c2272_i64 = arith.constant 2272 : i64
    %c2240_i64 = arith.constant 2240 : i64
    %c1728_i64 = arith.constant 1728 : i64
    %c1600_i64 = arith.constant 1600 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c256_i64 = arith.constant 256 : i64
    %cst = arith.constant 9.99999997E-7 : f32
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i64 = arith.constant 8 : i64
    %c128_i64 = arith.constant 128 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16384_i32 = arith.constant 16384 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %c48_i32 = arith.constant 48 : i32
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca[%c0] : memref<1xi64>
    %alloca_2 = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca_2[%c0] : memref<1xi64>
    %0 = hivm.hir.pointer_cast(%c61984_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c103648_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c94752_i64) : memref<32xbf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c120032_i64) : memref<32xbf16, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c61984_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c103648_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c1536_i64) : memref<32xbf16, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c103584_i64) : memref<32xbf16, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c768_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c103328_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c0_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c103072_i64) : memref<128xbf16, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c94752_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c120032_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %14 = arith.muli %arg17, %arg18 : i32
    %15 = arith.muli %14, %arg19 : i32
    %16 = hivm.hir.get_block_idx -> i64
    %17 = arith.trunci %16 : i64 to i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg20 = %17 to %15 step %c48_i32  : i32 {
      %18 = memref.load %alloca_2[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
      %19 = arith.remui %18, %c2_i64 : i64
      %20 = arith.cmpi eq, %19, %c1_i64 : i64
      %21 = arith.select %20, %1, %0 : memref<32x128xf32, #hivm.address_space<ub>>
      hivm.hir.set_mask_norm
      %22 = arith.remsi %arg20, %arg17 : i32
      %23 = arith.divsi %arg20, %arg17 : i32
      %24 = arith.remsi %23, %arg18 : i32
      %25 = arith.divsi %arg20, %14 : i32
      %26 = arith.remsi %25, %arg19 : i32
      %27 = arith.divsi %26, %c16_i32 : i32
      %28 = arith.remsi %26, %c16_i32 : i32
      %29 = arith.divsi %28, %c2_i32 : i32
      %30 = arith.index_cast %27 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg11 to offset: [%30], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %32 = arith.extsi %31 : i32 to i64
      %33 = affine.apply #map()[%30]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg11 to offset: [%33], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %34 = memref.load %reinterpret_cast_3[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %35 = arith.extsi %34 : i32 to i64
      %36 = arith.subi %35, %32 : i64
      %37 = arith.cmpi eq, %36, %c0_i64 : i64
      scf.if %37 {
      } else {
        %39 = arith.muli %22, %c128_i32 : i32
        %40 = arith.muli %24, %c32_i32 : i32
        %41 = arith.muli %32, %c8_i64 : i64
        %42 = arith.extsi %29 : i32 to i64
        %43 = arith.addi %41, %42 : i64
        %44 = arith.muli %43, %c128_i64 : i64
        %45 = arith.muli %32, %c16_i64 : i64
        %46 = arith.extsi %28 : i32 to i64
        %47 = arith.addi %45, %46 : i64
        %48 = arith.muli %47, %c128_i64 : i64
        %49 = arith.extsi %22 : i32 to i64
        %50 = arith.muli %49, %arg16 : i64
        %51 = arith.addi %50, %32 : i64
        %52 = arith.muli %51, %c16_i64 : i64
        %53 = arith.addi %52, %46 : i64
        %54 = arith.muli %53, %c128_i64 : i64
        %reinterpret_cast_4 = memref.reinterpret_cast %arg12 to offset: [%30], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %55 = memref.load %reinterpret_cast_4[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %56 = arith.extsi %55 : i32 to i64
        %reinterpret_cast_5 = memref.reinterpret_cast %arg13 to offset: [%30], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %57 = memref.load %reinterpret_cast_5[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %58 = arith.cmpi sle, %57, %c0_i64 : i64
        scf.if %58 {
        } else {
          %59 = arith.muli %57, %c4_i64 : i64
          %60 = arith.remsi %56, %c4_i64 : i64
          %61 = arith.addi %59, %60 : i64
          %62 = arith.muli %61, %c262144_i64 : i64
          %63 = arith.muli %28, %c16384_i32 : i32
          %64 = arith.index_cast %62 : i64 to index
          %65 = arith.index_cast %63 : i32 to index
          %66 = arith.index_cast %40 : i32 to index
          %67 = arith.index_cast %39 : i32 to index
          %68 = affine.apply #map1()[%67, %64, %65, %66]
          %reinterpret_cast_6 = memref.reinterpret_cast %arg9 to offset: [%68], sizes: [32, 128], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %69 = affine.max #map2()[%66]
          %70 = affine.min #map3()[%69, %66]
          %71 = affine.apply #map4()[%70, %66]
          %72 = affine.max #map2()[%67]
          %73 = affine.min #map5()[%72, %67]
          %74 = affine.apply #map4()[%73, %67]
          %75 = affine.max #map6()[%70, %66]
          %76 = affine.min #map7()[%75]
          %77 = affine.max #map6()[%73, %67]
          %78 = affine.min #map2()[%77]
          %79 = arith.cmpi slt, %76, %c32 : index
          %80 = arith.cmpi slt, %78, %c128 : index
          %81 = arith.ori %79, %80 : i1
          %subview = memref.subview %reinterpret_cast_6[0, 0] [%76, %78] [1, 1] : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_7 = memref.subview %21[0, 0] [%76, %78] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
          scf.if %81 {
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape = memref.collapse_shape %21 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %cast_9 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_9) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
          %cast = memref.cast %subview : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_8 = memref.cast %subview_7 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_2d_float(%cast, %cast_8, %c2_i32, %cst_0, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          %82 = arith.index_cast %44 : i64 to index
          %83 = arith.index_cast %54 : i64 to index
          %84 = arith.index_cast %48 : i64 to index
          %85 = arith.cmpi slt, %74, %c128 : index
          %86 = arith.cmpi slt, %71, %c32 : index
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
          %87:7 = scf.for %arg21 = %c0_i64 to %36 step %c1_i64 iter_args(%arg22 = %21, %arg23 = %47, %arg24 = %47, %arg25 = %67, %arg26 = %67, %arg27 = %66, %arg28 = %66) -> (memref<32x128xf32, #hivm.address_space<ub>>, i64, i64, index, index, index, index)  : i64 {
            %88 = memref.load %alloca[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
            %89 = arith.remui %88, %c2_i64 : i64
            %90 = arith.cmpi eq, %89, %c1_i64 : i64
            %91 = arith.select %90, %3, %2 : memref<32xbf16, #hivm.address_space<ub>>
            %92 = arith.remui %88, %c2_i64 : i64
            %93 = arith.cmpi eq, %92, %c1_i64 : i64
            %94 = arith.select %93, %5, %4 : memref<32x128xf32, #hivm.address_space<ub>>
            %95 = arith.remui %88, %c2_i64 : i64
            %96 = arith.cmpi eq, %95, %c1_i64 : i64
            %97 = arith.select %96, %7, %6 : memref<32xbf16, #hivm.address_space<ub>>
            %98 = arith.remui %88, %c2_i64 : i64
            %99 = arith.cmpi eq, %98, %c1_i64 : i64
            %100 = arith.select %99, %9, %8 : memref<128xbf16, #hivm.address_space<ub>>
            %101 = arith.remui %88, %c2_i64 : i64
            %102 = arith.cmpi eq, %101, %c1_i64 : i64
            %103 = arith.select %102, %11, %10 : memref<128xbf16, #hivm.address_space<ub>>
            %104 = arith.remui %88, %c2_i64 : i64
            %105 = arith.cmpi eq, %104, %c1_i64 : i64
            %106 = arith.select %105, %13, %12 : memref<32x1xf32, #hivm.address_space<ub>>
            %107 = arith.remui %88, %c2_i64 : i64
            %108 = arith.index_cast %107 : i64 to index
            %109 = arith.index_cast %108 : index to i1
            %110 = arith.select %109, %c1_i64, %c2_i64 : i64
            %111 = arith.index_cast %arg24 : i64 to index
            %reinterpret_cast_9 = memref.reinterpret_cast %arg7 to offset: [%111], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %112 = arith.index_cast %arg23 : i64 to index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%112], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %113 = affine.apply #map8()[%82, %arg25]
            %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%113], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %114 = affine.apply #map8()[%82, %arg26]
            %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%114], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %115 = affine.apply #map8()[%83, %arg27]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg8 to offset: [%115], sizes: [32], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<32xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %116 = affine.apply #map8()[%84, %arg28]
            %reinterpret_cast_14 = memref.reinterpret_cast %arg5 to offset: [%116], sizes: [32], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<32xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_15 = memref.subview %reinterpret_cast_11[0] [%74] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_16 = memref.subview %103[0] [%74] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
            scf.if %85 {
              %cast_111 = memref.cast %103 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
              func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_1, %cast_111) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            } {hivm.unlikely_condition}
            %cast_17 = memref.cast %subview_15 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %cast_18 = memref.cast %subview_16 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast_17, %cast_18, %c2_i32, %cst_1, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %117 = hivm.hir.pointer_cast(%c256_i64) : memref<128xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %cast_19 = memref.cast %103 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_20 = memref.cast %117 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_19, %cast_20, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
            %subview_21 = memref.subview %reinterpret_cast_12[0] [%74] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_22 = memref.subview %100[0] [%74] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
            scf.if %85 {
              %cast_111 = memref.cast %100 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
              func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_1, %cast_111) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            } {hivm.unlikely_condition}
            %cast_23 = memref.cast %subview_21 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %cast_24 = memref.cast %subview_22 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast_23, %cast_24, %c2_i32, %cst_1, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %118 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %cast_25 = memref.cast %100 : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_26 = memref.cast %118 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_25, %cast_26, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
            %subview_27 = memref.subview %reinterpret_cast_14[0] [%71] [1] : memref<32xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_28 = memref.subview %97[0] [%71] [1] : memref<32xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
            scf.if %86 {
              %cast_111 = memref.cast %97 : memref<32xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
              func.call @broadcast_scalar_bfloat16_t_to_1d(%cst_1, %cast_111) : (bf16, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            } {hivm.unlikely_condition}
            %cast_29 = memref.cast %subview_27 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %cast_30 = memref.cast %subview_28 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_1d_bfloat16_t(%cast_29, %cast_30, %c2_i32, %cst_1, %c0) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, bf16, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %119 = hivm.hir.pointer_cast(%c1600_i64) : memref<32xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %cast_31 = memref.cast %97 : memref<32xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_32 = memref.cast %119 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_31, %cast_32, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
            %120 = hivm.hir.pointer_cast(%c1728_i64) : memref<128xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %121 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_33 = memref.cast %117 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_34 = memref.cast %117 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_35 = memref.cast %120 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_36 = memref.cast %121 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmul_1d_float(%cast_33, %cast_34, %cast_35, %cast_36) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %122 = hivm.hir.pointer_cast(%c2240_i64) : memref<1xf32, #hivm.address_space<ub>>
            %123 = hivm.hir.pointer_cast(%c2272_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_37 = memref.cast %120 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_38 = memref.cast %122 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_39 = memref.cast %123 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @enablevc_reduce_sum_r_float(%cast_37, %cast_38, %cast_39, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
            %124 = memref.load %122[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, #hivm.address_space<ub>>
            %125 = arith.addf %124, %cst : f32
            %126 = math.sqrt %125 : f32
            %127 = hivm.hir.pointer_cast(%c256_i64) : memref<128xf32, #hivm.address_space<ub>>
            %128 = hivm.hir.pointer_cast(%c2528_i64) : memref<8xf32, #hivm.address_space<ub>>
            %cast_40 = memref.cast %117 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_41 = memref.cast %127 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_42 = memref.cast %128 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vdiv_vs_1d_float(%cast_40, %126, %cast_41, %cast_42) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %129 = hivm.hir.pointer_cast(%c2560_i64) : memref<128xf32, #hivm.address_space<ub>>
            %130 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_43 = memref.cast %118 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_44 = memref.cast %118 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_45 = memref.cast %129 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_46 = memref.cast %130 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmul_1d_float(%cast_43, %cast_44, %cast_45, %cast_46) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %131 = hivm.hir.pointer_cast(%c3072_i64) : memref<1xf32, #hivm.address_space<ub>>
            %132 = hivm.hir.pointer_cast(%c3104_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_47 = memref.cast %129 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_48 = memref.cast %131 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_49 = memref.cast %132 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @enablevc_reduce_sum_r_float(%cast_47, %cast_48, %cast_49, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
            %133 = memref.load %131[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, #hivm.address_space<ub>>
            %134 = arith.addf %133, %cst : f32
            %135 = math.sqrt %134 : f32
            %136 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xf32, #hivm.address_space<ub>>
            %137 = hivm.hir.pointer_cast(%c3360_i64) : memref<8xf32, #hivm.address_space<ub>>
            %cast_50 = memref.cast %118 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_51 = memref.cast %136 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_52 = memref.cast %137 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vdiv_vs_1d_float(%cast_50, %135, %cast_51, %cast_52) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %138 = hivm.hir.pointer_cast(%c256_i64) : memref<128xf32, #hivm.address_space<ub>>
            %139 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_53 = memref.cast %127 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_54 = memref.cast %138 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_55 = memref.cast %139 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmuls_vs_1d_float(%cast_53, %arg14, %cast_54, %cast_55) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %140 = memref.load %reinterpret_cast_10[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %141 = hivm.hir.pointer_cast(%c3392_i64) : memref<1xf32, #hivm.address_space<ub>>
            memref.store %140, %141[%c0] : memref<1xf32, #hivm.address_space<ub>>
            hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
            %142 = hivm.hir.pointer_cast(%c3392_i64) : memref<1xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
            %143 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_56 = memref.cast %141 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_57 = memref.cast %142 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_58 = memref.cast %143 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vexp_1d_float(%cast_56, %cast_57, %cast_58) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
            %144 = memref.load %142[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, #hivm.address_space<ub>>
            %145 = hivm.hir.pointer_cast(%c3424_i64) : memref<32x128xf32, #hivm.address_space<ub>>
            %collapse_shape = memref.collapse_shape %arg22 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            %collapse_shape_59 = memref.collapse_shape %145 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            %146 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_60 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_61 = memref.cast %collapse_shape_59 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_62 = memref.cast %146 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmuls_vs_1d_float(%cast_60, %144, %cast_61, %cast_62) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %expand_shape = memref.expand_shape %136 [[0, 1]] output_shape [1, 128] : memref<128xf32, #hivm.address_space<ub>> into memref<1x128xf32, #hivm.address_space<ub>>
            %147 = hivm.hir.pointer_cast(%c19808_i64) : memref<32x128xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %148 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_63 = memref.cast %145 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_64 = memref.cast %expand_shape : memref<1x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_65 = memref.cast %147 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_66 = memref.cast %148 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmul_2d_float(%cast_63, %cast_64, %cast_65, %cast_66) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %149 = hivm.hir.pointer_cast(%c36192_i64) : memref<32x1xf32, #hivm.address_space<ub>>
            %150 = hivm.hir.pointer_cast(%c36320_i64) : memref<2048xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_67 = memref.cast %147 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_68 = memref.cast %149 : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_69 = memref.cast %150 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @enablevc_reduce_sum_ar_float(%cast_67, %cast_68, %cast_69, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
            %collapse_shape_70 = memref.collapse_shape %149 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
            %151 = hivm.hir.pointer_cast(%c1600_i64) : memref<32xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %152 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_71 = memref.cast %119 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_72 = memref.cast %collapse_shape_70 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_73 = memref.cast %151 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_74 = memref.cast %152 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vsub_1d_float(%cast_71, %cast_72, %cast_73, %cast_74) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %153 = memref.load %reinterpret_cast_9[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %154 = hivm.hir.pointer_cast(%c44512_i64) : memref<1xbf16, #hivm.address_space<ub>>
            memref.store %153, %154[%c0] : memref<1xbf16, #hivm.address_space<ub>>
            hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
            %155 = hivm.hir.pointer_cast(%c44544_i64) : memref<1xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
            %cast_75 = memref.cast %154 : memref<1xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_76 = memref.cast %155 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vcast_bfloat16_t_to_float_1d_with_mode(%cast_75, %cast_76, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
            %156 = memref.load %155[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, #hivm.address_space<ub>>
            %157 = hivm.hir.pointer_cast(%c1600_i64) : memref<32xf32, #hivm.address_space<ub>>
            %158 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_77 = memref.cast %151 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_78 = memref.cast %157 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_79 = memref.cast %158 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmuls_vs_1d_float(%cast_77, %156, %cast_78, %cast_79) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %expand_shape_80 = memref.expand_shape %157 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
            %159 = hivm.hir.pointer_cast(%c44576_i64) : memref<32x128xf32, #hivm.address_space<ub>>
            %160 = hivm.hir.pointer_cast(%c60960_i64) : memref<256xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_81 = memref.cast %expand_shape_80 : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_82 = memref.cast %expand_shape : memref<1x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_83 = memref.cast %159 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_84 = memref.cast %160 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmul_2d_float(%cast_81, %cast_82, %cast_83, %cast_84) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %collapse_shape_85 = memref.collapse_shape %159 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            %collapse_shape_86 = memref.collapse_shape %94 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %110]
            hivm.hir.pipe_barrier[<PIPE_V>]
            %161 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_87 = memref.cast %collapse_shape_59 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_88 = memref.cast %collapse_shape_85 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_89 = memref.cast %collapse_shape_86 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_90 = memref.cast %161 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vadd_1d_float(%cast_87, %cast_88, %cast_89, %cast_90) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %expand_shape_91 = memref.expand_shape %138 [[0, 1]] output_shape [1, 128] : memref<128xf32, #hivm.address_space<ub>> into memref<1x128xf32, #hivm.address_space<ub>>
            %162 = hivm.hir.pointer_cast(%c78368_i64) : memref<32x128xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %163 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_92 = memref.cast %94 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_93 = memref.cast %expand_shape_91 : memref<1x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_94 = memref.cast %162 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_95 = memref.cast %163 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmul_2d_float(%cast_92, %cast_93, %cast_94, %cast_95) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %164 = hivm.hir.pointer_cast(%c94880_i64) : memref<2048xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_96 = memref.cast %162 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_97 = memref.cast %106 : memref<32x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_98 = memref.cast %164 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @enablevc_reduce_sum_ar_float(%cast_96, %cast_97, %cast_98, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
            %collapse_shape_99 = memref.collapse_shape %106 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_100 = memref.cast %collapse_shape_99 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_101 = memref.cast %91 : memref<32xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vcast_float_to_bfloat16_t_1d_with_mode(%cast_100, %cast_101, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            %subview_102 = memref.subview %91[0] [%71] [1] : memref<32xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
            %subview_103 = memref.subview %reinterpret_cast_13[0] [%71] [1] : memref<32xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.pipe_barrier[<PIPE_MTE3>]
            %cast_104 = memref.cast %subview_102 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_105 = memref.cast %subview_103 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_bfloat16_t(%cast_104, %cast_105, %c0_i32) : (memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xbf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            %165 = arith.addi %56, %arg21 : i64
            %166 = arith.addi %165, %c1_i64 : i64
            %167 = arith.remsi %166, %c4_i64 : i64
            %168 = arith.addi %59, %167 : i64
            %169 = arith.muli %168, %c262144_i64 : i64
            %170 = arith.index_cast %169 : i64 to index
            %171 = affine.apply #map1()[%67, %170, %65, %66]
            %reinterpret_cast_106 = memref.reinterpret_cast %arg10 to offset: [%171], sizes: [32, 128], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            %subview_107 = memref.subview %94[0, 0] [%76, %78] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
            %subview_108 = memref.subview %reinterpret_cast_106[0, 0] [%76, %78] [1, 1] : memref<32x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            %cast_109 = memref.cast %subview_107 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_110 = memref.cast %subview_108 : memref<?x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_2d_float(%cast_109, %cast_110, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %110]
            %172 = arith.addi %arg23, %c16_i64 : i64
            %173 = arith.addi %arg24, %c16_i64 : i64
            %174 = affine.apply #map9()[%arg25]
            %175 = affine.apply #map9()[%arg26]
            %176 = affine.apply #map10()[%arg27]
            %177 = affine.apply #map10()[%arg28]
            %178 = arith.addi %88, %c1_i64 : i64
            memref.store %178, %alloca[%c0] : memref<1xi64>
            scf.yield %94, %172, %173, %174, %175, %176, %177 : memref<32x128xf32, #hivm.address_space<ub>>, i64, i64, index, index, index, index
          }
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        }
      }
      %38 = arith.addi %18, %c1_i64 : i64
      memref.store %38, %alloca_2[%c0] : memref<1xi64>
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
