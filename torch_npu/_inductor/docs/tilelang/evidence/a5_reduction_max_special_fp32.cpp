#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void tilelang_fused_amax_0_prim_fn_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_f32 _red_acc;
  vector_bool _red_full = asc_create_mask_b32(PAT_ALL);
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto _red_identity = simd_inst::vdup<float>(tl::limits::kFloatNInf, v, MODE_ZEROING);
  _red_acc = _red_identity;
  for (int32_t _red_chunk = 0; _red_chunk < 2; ++_red_chunk) {
    auto _red_mask = simd_inst::update_mask_b32(min(64, (65 - (_red_chunk * 64))));
    auto _red_loaded = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(_red_chunk * 64)])), 0);
    auto _red_values = simd_inst::vsel(_red_loaded, _red_identity, _red_mask);
    auto _red_max = simd_inst::vmax(_red_acc, _red_values, _red_full, MODE_ZEROING);
    auto _red_max_1 = simd_inst::vsel(_red_acc, _red_max, simd_inst::vcmp_ne(_red_acc, _red_acc, _red_full));
    _red_acc = simd_inst::vsel(_red_values, _red_max_1, simd_inst::vcmp_ne(_red_values, _red_values, _red_full));
  }
  auto _red_result = simd_inst::vcmax(_red_acc, _red_full, MODE_ZEROING);
  auto _red_zero = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
  auto _red_nan_flags = simd_inst::vsel(simd_inst::vdup<float>(float(0x1p+0f/*1.000000e+00*/), v, MODE_ZEROING), _red_zero, simd_inst::vcmp_ne(_red_acc, _red_acc, _red_full));
  auto _red_nan_count = simd_inst::vdupv(simd_inst::vcadd(_red_nan_flags, _red_full, MODE_ZEROING), v, POS_LOWEST, MODE_ZEROING);
  auto _red_result_1 = simd_inst::vsel(simd_inst::vdup<float>(tl::limits::kFloatNaN, v, MODE_ZEROING), _red_result, simd_inst::vcmp_ne(_red_nan_count, _red_zero, _red_full));
  simd_inst::vsts_1st(_red_result_1, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[128])), 0, _red_full);
}

extern "C" __global__ __vector__ void tilelang_fused_amax_0_prim_fn_kernel(__gm__ float* in_ptr0, __gm__ float* out_ptr0) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(in_ptr0[(((int32_t)block_idx) * 65)]))), 1, 260, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 260, 260);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  tilelang_fused_amax_0_prim_fn_kernel_simd_vf_0(buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out_ptr0[((int32_t)block_idx)]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[128]))), 1, 4, static_cast<asc_store_l2_cache_mode>(4), 4, 4);
}

