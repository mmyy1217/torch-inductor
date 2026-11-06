#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void tilelang_fused_add_exp_mul_0_prim_fn_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool _vf_mask = asc_create_mask_b32(PAT_ALL);
  for (int32_t _vf_chunk = 0; _vf_chunk < 2; ++_vf_chunk) {
    auto _vf_v0 = simd_inst::vadd(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(_vf_chunk * 64)])), 0), simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_vf_chunk * 64) + 128)])), 0), _vf_mask, MODE_ZEROING);
    vector_bool v = asc_create_mask_b32(PAT_ALL);
    auto _vf_v1 = simd_inst::vmul(_vf_v0, simd_inst::vdup<float>(float(0x1p-2f/*2.500000e-01*/), v, MODE_ZEROING), _vf_mask, MODE_ZEROING);
    auto _vf_v2 = simd_inst::vexp(_vf_v1, _vf_mask, MODE_ZEROING);
    simd_inst::vsts_norm(_vf_v2, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_vf_chunk * 64) + 256)])), 0, _vf_mask);
  }
}

extern "C" __global__ __vector__ void tilelang_fused_add_exp_mul_0_prim_fn_kernel(__gm__ float* in_ptr0, __gm__ float* in_ptr1, __gm__ float* out_ptr0) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(in_ptr0[(((int32_t)block_idx) * 128)]))), 1, (min(128, (2051 - (((int32_t)block_idx) * 128))) * 4), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (min(128, (2051 - (((int32_t)block_idx) * 128))) * 4), (min(128, (2051 - (((int32_t)block_idx) * 128))) * 4));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[128]))), (__gm__ uint8_t*)((&(in_ptr1[(((int32_t)block_idx) * 128)]))), 1, (min(128, (2051 - (((int32_t)block_idx) * 128))) * 4), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (min(128, (2051 - (((int32_t)block_idx) * 128))) * 4), (min(128, (2051 - (((int32_t)block_idx) * 128))) * 4));
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  tilelang_fused_add_exp_mul_0_prim_fn_kernel_simd_vf_0(buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out_ptr0[(((int32_t)block_idx) * 128)]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[256]))), 1, (min(128, (2051 - (((int32_t)block_idx) * 128))) * 4), static_cast<asc_store_l2_cache_mode>(4), (min(128, (2051 - (((int32_t)block_idx) * 128))) * 4), (min(128, (2051 - (((int32_t)block_idx) * 128))) * 4));
}
