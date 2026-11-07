#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void tilelang_fused_sum_0_prim_fn_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_f32 _red_acc;
  vector_bool _red_full = asc_create_mask_b32(PAT_ALL);
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto _red_identity = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
  _red_acc = _red_identity;
  for (int32_t _red_chunk = 0; _red_chunk < 64; ++_red_chunk) {
    auto _red_mask = simd_inst::update_mask_b32(64);
    auto _red_loaded = simd_inst::vlds_unpack<half>((__ubuf__ half*)(&(((__ubuf__ half*)buf_dyn_shmem)[(_red_chunk * 64)])), 0);
    auto _red_values = simd_inst::vsel(simd_inst::vcvt<float>(_red_loaded, _red_full, PART_EVEN, MODE_ZEROING), _red_identity, _red_mask);
    _red_acc = simd_inst::vadd(_red_acc, _red_values, _red_full, MODE_ZEROING);
  }
  auto _red_result = simd_inst::vcadd(_red_acc, _red_full, MODE_ZEROING);
  simd_inst::vsts_1st(simd_inst::vcvt<half>(_red_result, _red_full, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING), (__ubuf__ half*)(&(((__ubuf__ half*)buf_dyn_shmem)[4096])), 0, _red_full);
}

extern "C" __global__ __vector__ void tilelang_fused_sum_0_prim_fn_kernel(__gm__ half* in_ptr0, __gm__ half* out_ptr0) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(in_ptr0[(((int32_t)block_idx) * 4096)]))), 1, 8192, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 8192, 8192);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  tilelang_fused_sum_0_prim_fn_kernel_simd_vf_0(buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out_ptr0[((int32_t)block_idx)]))), (__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[4096]))), 1, 2, static_cast<asc_store_l2_cache_mode>(4), 2, 2);
}

