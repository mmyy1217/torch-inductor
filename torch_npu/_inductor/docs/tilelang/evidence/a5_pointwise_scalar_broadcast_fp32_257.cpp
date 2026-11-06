#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void tilelang_fused_add_exp_mul_0_prim_fn_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  float _vf_v0 = (((__ubuf__ float*)buf_dyn_shmem)[((int32_t)threadIdx.x)] + ((__ubuf__ float*)buf_dyn_shmem)[256]);
  ((__ubuf__ float*)buf_dyn_shmem)[(((int32_t)threadIdx.x) + 128)] = expf((_vf_v0 * float(0x1p-2f/*2.500000e-01*/)));
}

extern "C" __global__ __vector__ void tilelang_fused_add_exp_mul_0_prim_fn_kernel(__gm__ float* in_ptr0, __gm__ float* in_ptr1, __gm__ float* out_ptr0) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(in_ptr0[(((int32_t)block_idx) * 128)]))), 1, (min(128, (257 - (((int32_t)block_idx) * 128))) * 4), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (min(128, (257 - (((int32_t)block_idx) * 128))) * 4), (min(128, (257 - (((int32_t)block_idx) * 128))) * 4));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[256]))), (__gm__ uint8_t*)((&(in_ptr1[0]))), 1, 4, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4, 4);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_vf_call<tilelang_fused_add_exp_mul_0_prim_fn_kernel_simt_vf_0>(cce::dim3(128), buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out_ptr0[(((int32_t)block_idx) * 128)]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[128]))), 1, (min(128, (257 - (((int32_t)block_idx) * 128))) * 4), static_cast<asc_store_l2_cache_mode>(4), (min(128, (257 - (((int32_t)block_idx) * 128))) * 4), (min(128, (257 - (((int32_t)block_idx) * 128))) * 4));
}
