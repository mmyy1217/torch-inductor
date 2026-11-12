"""DN-only FlashAttention forward derived from PR16 core.py.

Shared FlashAttention forward kernel builder for Ascend NPU (Cube + Vector).

Both MHA and GQA reduce to the same per-tile computation::

    Q:   [q_len, D]
    K/V: [kv_len, D]
    O:   [q_len, D]

MHA uses ``q_len == kv_len``; GQA flattens the query groups into
``q_len = S1 * G`` while K/V stay ``[S2, D]``. Group structure only affects how
the caller reshapes Q/O — the row-wise online-softmax attention is identical.

The softmax writes probabilities directly in NZ layout so the second GEMM can
consume them without an intermediate ND-to-NZ conversion. That packing path is
hardwired to ``D == 128`` (128-column ``vsstb`` stride, ``uint16x128`` merge),
so the builder pins ``head_dim`` to 128 until the path is generalized.
"""

import math
from dataclasses import dataclass

import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout
from tilelang.ascend.language import simd as S


@dataclass(frozen=True)
class FwdTiling:
    block_q: int = 128
    block_kv: int = 128
    num_stages: int = 3


def flash_attention_fwd(
    q_len,
    kv_len,
    head_dim=128,
    *,
    out_dtype="bfloat16",
    softmax_scale=None,
    num_blocks=None,
    tiling=None,
    manual_schedule=False,
    return_lse=False,
):
    """Keep each Q tile resident for the complete KV loop.

    DN layout stores K on L0A and the resident Q tile on L0B, so the first
    GEMM is K @ Q^T. PV transposes the joined P panels back to query-major.
    Both GEMMs and their L0C-to-UB drains keep paired UnitFlag control 3.
    """
    tiling = tiling or FwdTiling()

    # Expand config into compile-time local scalars so the hygienic T.macro
    # definitions below capture plain values, not dataclass attribute reads.
    BR = tiling.block_q
    BC = tiling.block_kv
    PIPELINE_STAGES = tiling.num_stages
    ROWS = BR // 2
    VL = 64  # Ascend fp32 SIMD lane count
    D = head_dim
    dtype = "bfloat16"
    accum_dtype = "float32"
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    SCALE_AFTER_MAX = math.isfinite(scale) and scale > 0

    # Implementation constraints of the SIMD softmax-packing path. These are
    # real limits of the current kernel, not tunables — validate explicitly
    # rather than advertise flexibility the code does not have.
    assert head_dim == 128, "The SIMD packing path requires head_dim == 128."
    assert BR % 2 == 0
    assert BC == 2 * VL
    assert q_len % BR == 0
    assert kv_len % BC == 0

    NUM_M_TILES = q_len // BR
    NUM_KV_BLOCKS = kv_len // BC
    NUM_BLOCKS = NUM_M_TILES if num_blocks is None else num_blocks
    assert NUM_BLOCKS > 0
    assert NUM_KV_BLOCKS > 0
    assert NUM_M_TILES % NUM_BLOCKS == 0, "M tiles must divide evenly across cores."
    M_TILES_PER_BLOCK = NUM_M_TILES // NUM_BLOCKS

    is_cast_out = out_dtype != accum_dtype
    assert out_dtype in (accum_dtype, "bfloat16", "float16"), (
        f"out_dtype must be {accum_dtype} (no cast) or a 16-bit type (bfloat16/float16); got {out_dtype!r}."
    )

    @T.macro
    def qk(kv, K, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub):
        T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_shared[0:BC, 0:D])
        T.copy(K_shared[0:BC, 0:D], qk_a_l0)
        T.gemm(
            qk_a_l0,
            qk_b_l0,
            qk_acc_l0c,
            transpose_B=True,
            clear_accum=True,
            unit_flag_ctrl=3,
        )
        T.dual_copy(qk_acc_l0c, S_ub[0:BC, 0:ROWS], unit_flag_ctrl=3)

    @T.macro
    def softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        # Lanes are the 64 queries owned by this AIV. The key axis is the loop.
        with T.SimdVF(latency=735):
            full = S.pset(32, "PAT_ALL")
            packed_mask = S.pset(16, "PAT_VL64")
            max0 = S.alloc_var("float32")
            max1 = S.alloc_var("float32")
            max2 = S.alloc_var("float32")
            max3 = S.alloc_var("float32")
            max0 = S.vdup(-T.infinity("float32"), "float32", full)
            max1 = S.vdup(-T.infinity("float32"), "float32", full)
            max2 = S.vdup(-T.infinity("float32"), "float32", full)
            max3 = S.vdup(-T.infinity("float32"), "float32", full)
            for r in range(0, BC, 4):
                x0 = S.vld(S_ub[r, 0], "NORM")
                x1 = S.vld(S_ub[r + 1, 0], "NORM")
                x2 = S.vld(S_ub[r + 2, 0], "NORM")
                x3 = S.vld(S_ub[r + 3, 0], "NORM")
                if not SCALE_AFTER_MAX:
                    x0 = S.vmuls(x0, T.float32(scale), full)
                    x1 = S.vmuls(x1, T.float32(scale), full)
                    x2 = S.vmuls(x2, T.float32(scale), full)
                    x3 = S.vmuls(x3, T.float32(scale), full)
                max0 = S.vmax(max0, x0, full)
                max1 = S.vmax(max1, x1, full)
                max2 = S.vmax(max2, x2, full)
                max3 = S.vmax(max3, x3, full)
            block_max = S.vmax(S.vmax(max0, max1, full), S.vmax(max2, max3, full), full)
            if SCALE_AFTER_MAX:
                block_max = S.vmuls(block_max, T.float32(scale), full)
            previous_max = S.vld(m_ub[0], "NORM")
            merged_max = S.vmax(previous_max, block_max, full)
            alpha = S.vexpdif(previous_max, merged_max, full)
            S.vsts(m_ub[0], merged_max, full, "NORM_B32")
            S.vsts(alpha_ub[0], alpha, full, "NORM_B32")
            sum0 = S.alloc_var("float32")
            sum1 = S.alloc_var("float32")
            sum2 = S.alloc_var("float32")
            sum3 = S.alloc_var("float32")
            sum0 = S.vdup(T.float32(0), "float32", full)
            sum1 = S.vdup(T.float32(0), "float32", full)
            sum2 = S.vdup(T.float32(0), "float32", full)
            sum3 = S.vdup(T.float32(0), "float32", full)
            dst = S.make_ubuf_ptr(T.access_ptr(P_nz_ub[0, 0], "w", BC + 1, ROWS), "bfloat16")
            for r in range(0, BC, 4):
                e0 = S.vexpdif(S.vmuls(S.vld(S_ub[r + 0, 0], "NORM"), T.float32(scale), full), merged_max, full)
                sum0 = S.vadd(sum0, e0, full)
                packed0 = S.vpack(T.reinterpret(S.vcvt(e0, "bfloat16", full, sat=False, part=0), "uint32x64"), 0)
                dst = S.vsstb(T.reinterpret(packed0, "bfloat16x128"), dst, T.int32((BC + 1) << 16 | 1), packed_mask, update=True)
                e1 = S.vexpdif(S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), full), merged_max, full)
                sum1 = S.vadd(sum1, e1, full)
                packed1 = S.vpack(T.reinterpret(S.vcvt(e1, "bfloat16", full, sat=False, part=0), "uint32x64"), 0)
                dst = S.vsstb(T.reinterpret(packed1, "bfloat16x128"), dst, T.int32((BC + 1) << 16 | 1), packed_mask, update=True)
                e2 = S.vexpdif(S.vmuls(S.vld(S_ub[r + 2, 0], "NORM"), T.float32(scale), full), merged_max, full)
                sum2 = S.vadd(sum2, e2, full)
                packed2 = S.vpack(T.reinterpret(S.vcvt(e2, "bfloat16", full, sat=False, part=0), "uint32x64"), 0)
                dst = S.vsstb(T.reinterpret(packed2, "bfloat16x128"), dst, T.int32((BC + 1) << 16 | 1), packed_mask, update=True)
                e3 = S.vexpdif(S.vmuls(S.vld(S_ub[r + 3, 0], "NORM"), T.float32(scale), full), merged_max, full)
                sum3 = S.vadd(sum3, e3, full)
                packed3 = S.vpack(T.reinterpret(S.vcvt(e3, "bfloat16", full, sat=False, part=0), "uint32x64"), 0)
                dst = S.vsstb(T.reinterpret(packed3, "bfloat16x128"), dst, T.int32((BC + 1) << 16 | 1), packed_mask, update=True)
            block_sum = S.vadd(S.vadd(sum0, sum1, full), S.vadd(sum2, sum3, full), full)
            updated_sum = S.alloc_var(accum_dtype)
            updated_sum = S.vld(l_ub[0], "NORM")
            S.vmadd(updated_sum, alpha, block_sum, full)
            S.vsts(l_ub[0], updated_sum, full, "NORM_B32")

    @T.macro
    def pack_p(P_nz_ub, P_shared):
        # Each AIV writes its KxQ panel; subcore 1 lands 8192 bf16 elements later.
        T.dual_copy(P_nz_ub[0:BC, 0:ROWS], P_shared[0:BC, 0:BR])

    @T.macro
    def pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub):
        T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_shared[0:D, 0:BC], transpose=True)
        T.copy(P_shared[0:BC, 0:BR], pv_a_l0)
        T.copy(V_shared[0:D, 0:BC], pv_b_l0)
        T.gemm(
            pv_a_l0,
            pv_b_l0,
            pv_acc_l0c,
            transpose_A=True,
            transpose_B=True,
            clear_accum=True,
            unit_flag_ctrl=3,
        )
        T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D], unit_flag_ctrl=3)

    @T.macro
    def accumulate_output(O_tmp_ub, O_ub, alpha_ub):
        with T.SimdVF(latency=318):
            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    updated = S.alloc_var(accum_dtype)
                    updated = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    S.vmadd(updated, alpha, cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")

    @T.macro
    def kv_auto(
        kv,
        K,
        V,
        K_shared,
        V_shared,
        P_shared,
        qk_a_l0,
        qk_b_l0,
        pv_a_l0,
        pv_b_l0,
        qk_acc_l0c,
        pv_acc_l0c,
        S_ub,
        P_nz_ub,
        O_tmp_ub,
        O_ub,
        m_ub,
        l_ub,
        alpha_ub,
        old_m_ub,
        row_sum_ub,
    ):
        qk(kv, K, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)
        softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        pack_p(P_nz_ub, P_shared)
        pv(kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)
        accumulate_output(O_tmp_ub, O_ub, alpha_ub)

    @T.macro
    def kv_manual(
        kv,
        K,
        V,
        K_shared,
        V_shared,
        P_shared,
        qk_a_l0,
        qk_b_l0,
        pv_a_l0,
        pv_b_l0,
        qk_acc_l0c,
        pv_acc_l0c,
        S_ub,
        P_nz_ub,
        O_tmp_ub,
        O_ub,
        m_ub,
        l_ub,
        alpha_ub,
        old_m_ub,
        row_sum_ub,
    ):
        # Fixed-stage solution under manual per-pipe source ordering. The
        # stage vector in source task order is:
        # [0,1,1,3,5,6,3,6,6,6,7,7].
        with T.Stage(0):
            T.copy(K[kv * BC : (kv + 1) * BC, 0:D], K_shared[0:BC, 0:D])
        with T.Stage(1):
            T.copy(K_shared[0:BC, 0:D], qk_a_l0)
            T.gemm(
                qk_a_l0,
                qk_b_l0,
                qk_acc_l0c,
                transpose_B=True,
                clear_accum=True,
                unit_flag_ctrl=3,
            )
        with T.Stage(3):
            T.dual_copy(qk_acc_l0c, S_ub[0:BC, 0:ROWS], unit_flag_ctrl=3)
        with T.Stage(5):
            softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        with T.Stage(6):
            pack_p(P_nz_ub, P_shared)
        with T.Stage(3):
            T.copy(V[kv * BC : (kv + 1) * BC, 0:D], V_shared[0:D, 0:BC], transpose=True)
        with T.Stage(6):
            T.copy(P_shared[0:BC, 0:BR], pv_a_l0)
            T.copy(V_shared[0:D, 0:BC], pv_b_l0)
            T.gemm(
                pv_a_l0,
                pv_b_l0,
                pv_acc_l0c,
                transpose_A=True,
                transpose_B=True,
                clear_accum=True,
                unit_flag_ctrl=3,
            )
        with T.Stage(7):
            T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D], unit_flag_ctrl=3)
            accumulate_output(O_tmp_ub, O_ub, alpha_ub)

    @T.macro
    def init_m_l(m_ub, l_ub):
        with T.SimdVF(latency=400):
            init_mask = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), accum_dtype, init_mask)
            initial_m = S.vdup(-T.infinity(accum_dtype), accum_dtype, init_mask)
            S.vsts(l_ub[0], zero, init_mask, "NORM_B32")
            S.vsts(m_ub[0], initial_m, init_mask, "NORM_B32")

    @T.macro
    def clear_O_ub(O_ub):
        with T.SimdVF(latency=142):
            clear_mask = S.pset(32, "PAT_ALL")
            output_zero = S.vdup(T.float32(0.0), accum_dtype, clear_mask)
            for i in range(ROWS):
                for c in range(0, D, VL):
                    S.vsts(O_ub[i, c], output_zero, clear_mask, "NORM_B32")

    @T.macro
    def finalize_fp32(mt, O, O_ub, l_ub):
        with T.SimdVF(latency=1780):
            normalize_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                denominator = S.vld(l_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)
                    S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")
        T.dual_copy(O_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])

    @T.macro
    def finalize_cast(mt, O, O_ub, O_cast_ub, l_ub):
        with T.SimdVF(latency=1780):
            normalize_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                denominator = S.vld(l_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)
                    S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")
        with T.SimdVF(latency=193):
            cast_full = S.pset(32, "PAT_ALL")
            for r in range(ROWS):
                for c in range(0, D, VL):
                    ob = S.vcvt(S.vld(O_ub[r, c], "NORM"), out_dtype, cast_full)
                    S.vsts(O_cast_ub[r, c], ob, cast_full, "PK_B32")
        T.dual_copy(O_cast_ub[0:ROWS, 0:D], O[mt * BR : (mt + 1) * BR, 0:D])

    @T.macro
    def store_lse(mt, LSE, m_ub, l_ub, lse_ub):
        with T.SimdVF(latency=256):
            lse_mask = S.pset(32, "PAT_ALL")
            lse = S.vadd(S.vln(S.vld(l_ub[0], "NORM"), lse_mask), S.vld(m_ub[0], "NORM"), lse_mask)
            S.vsts(lse_ub[0], lse, lse_mask, "NORM_B32")
        T.dual_copy(lse_ub[0:ROWS], LSE[mt * BR : (mt + 1) * BR])

    kv_step = kv_manual if manual_schedule else kv_auto

    if not return_lse:

        @T.prim_func
        def main(
            Q: T.Buffer((q_len, D), dtype),
            K: T.Buffer((kv_len, D), dtype),
            V: T.Buffer((kv_len, D), dtype),
            O: T.Buffer((q_len, D), out_dtype),
        ):
            with T.Kernel(NUM_BLOCKS) as bx:
                Q_shared = T.alloc_l1((BR, D), dtype)
                K_shared = T.alloc_l1((BC, D), dtype)
                V_shared = T.alloc_l1((D, BC), dtype)
                P_shared = T.alloc_l1((BR, BC), dtype)

                qk_a_l0 = T.alloc_l0a((BR, D), dtype)
                qk_b_l0 = T.alloc_l0b((BC, D), dtype)
                pv_a_l0 = T.alloc_l0a((BR, BC), dtype)
                pv_b_l0 = T.alloc_l0b((D, BC), dtype)
                qk_acc_l0c = T.alloc_l0c((BR, BC), accum_dtype)
                pv_acc_l0c = T.alloc_l0c((BR, D), accum_dtype)

                S_ub = T.alloc_shared((BC, ROWS), accum_dtype)
                P_nz_ub = T.alloc_shared((BC + 1, ROWS), dtype)
                T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
                O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
                O_ub = T.alloc_shared((ROWS, D), accum_dtype)
                m_ub = T.alloc_shared((ROWS,), accum_dtype)
                l_ub = T.alloc_shared((ROWS,), accum_dtype)
                alpha_ub = T.alloc_shared((ROWS,), accum_dtype)
                old_m_ub = T.alloc_shared((ROWS,), accum_dtype)
                row_sum_ub = T.alloc_shared((ROWS,), accum_dtype)
                T.annotate_buffer_versions(
                    {
                        # Resident Q now lives in L0B for K @ Q^T.
                        qk_b_l0: 1,
                        K_shared: 3,
                        V_shared: 3,
                        qk_acc_l0c: 2,
                        pv_acc_l0c: 2,
                        S_ub: 3,
                        P_nz_ub: 2,
                        alpha_ub: 3,
                        O_tmp_ub: 2,
                    }
                )

                for local_mt in T.serial(M_TILES_PER_BLOCK):
                    mt = bx * M_TILES_PER_BLOCK + local_mt
                    T.copy(Q[mt * BR : (mt + 1) * BR, 0:D], Q_shared[0:BR, 0:D])
                    # Load Q into L0B once; qk() does not overwrite it in the KV loop.
                    T.copy(Q_shared[0:BR, 0:D], qk_b_l0)

                    init_m_l(m_ub, l_ub)
                    clear_O_ub(O_ub)

                    for kv in T.Pipelined(
                        NUM_KV_BLOCKS,
                        num_stages=PIPELINE_STAGES,
                        annotations={"enable_offset": True},
                    ):
                        kv_step(
                            kv,
                            K,
                            V,
                            K_shared,
                            V_shared,
                            P_shared,
                            qk_a_l0,
                            qk_b_l0,
                            pv_a_l0,
                            pv_b_l0,
                            qk_acc_l0c,
                            pv_acc_l0c,
                            S_ub,
                            P_nz_ub,
                            O_tmp_ub,
                            O_ub,
                            m_ub,
                            l_ub,
                            alpha_ub,
                            old_m_ub,
                            row_sum_ub,
                        )

                    if is_cast_out:
                        # Byte reinterpretation only: aliases O_ub, consumes no extra
                        # UB capacity or buffer versions.
                        O_cast_ub = T.view(O_ub, (ROWS * 2, D), dtype=out_dtype)
                        finalize_cast(mt, O, O_ub, O_cast_ub, l_ub)
                    else:
                        finalize_fp32(mt, O, O_ub, l_ub)

        return main

    @T.prim_func
    def main_with_lse(
        Q: T.Buffer((q_len, D), dtype),
        K: T.Buffer((kv_len, D), dtype),
        V: T.Buffer((kv_len, D), dtype),
        O: T.Buffer((q_len, D), out_dtype),
        LSE: T.Buffer((q_len,), accum_dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            Q_shared = T.alloc_l1((BR, D), dtype)
            K_shared = T.alloc_l1((BC, D), dtype)
            V_shared = T.alloc_l1((D, BC), dtype)
            P_shared = T.alloc_l1((BR, BC), dtype)

            qk_a_l0 = T.alloc_l0a((BR, D), dtype)
            qk_b_l0 = T.alloc_l0b((BC, D), dtype)
            pv_a_l0 = T.alloc_l0a((BR, BC), dtype)
            pv_b_l0 = T.alloc_l0b((D, BC), dtype)
            qk_acc_l0c = T.alloc_l0c((BR, BC), accum_dtype)
            pv_acc_l0c = T.alloc_l0c((BR, D), accum_dtype)

            S_ub = T.alloc_shared((BC, ROWS), accum_dtype)
            P_nz_ub = T.alloc_shared((BC + 1, ROWS), dtype)
            T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
            O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_ub = T.alloc_shared((ROWS, D), accum_dtype)
            m_ub = T.alloc_shared((ROWS,), accum_dtype)
            l_ub = T.alloc_shared((ROWS,), accum_dtype)
            alpha_ub = T.alloc_shared((ROWS,), accum_dtype)
            old_m_ub = T.alloc_shared((ROWS,), accum_dtype)
            row_sum_ub = T.alloc_shared((ROWS,), accum_dtype)
            T.annotate_buffer_versions(
                {
                    # Resident Q now lives in L0B for K @ Q^T.
                    qk_b_l0: 1,
                    K_shared: 3,
                    V_shared: 3,
                    qk_acc_l0c: 2,
                    pv_acc_l0c: 2,
                    S_ub: 3,
                    P_nz_ub: 2,
                    alpha_ub: 3,
                    O_tmp_ub: 2,
                }
            )

            for local_mt in T.serial(M_TILES_PER_BLOCK):
                mt = bx * M_TILES_PER_BLOCK + local_mt
                T.copy(Q[mt * BR : (mt + 1) * BR, 0:D], Q_shared[0:BR, 0:D])
                # Load Q into L0B once; qk() does not overwrite it in the KV loop.
                T.copy(Q_shared[0:BR, 0:D], qk_b_l0)

                init_m_l(m_ub, l_ub)
                clear_O_ub(O_ub)

                for kv in T.Pipelined(
                    NUM_KV_BLOCKS,
                    num_stages=PIPELINE_STAGES,
                    annotations={"enable_offset": True},
                ):
                    kv_step(
                        kv,
                        K,
                        V,
                        K_shared,
                        V_shared,
                        P_shared,
                        qk_a_l0,
                        qk_b_l0,
                        pv_a_l0,
                        pv_b_l0,
                        qk_acc_l0c,
                        pv_acc_l0c,
                        S_ub,
                        P_nz_ub,
                        O_tmp_ub,
                        O_ub,
                        m_ub,
                        l_ub,
                        alpha_ub,
                        old_m_ub,
                        row_sum_ub,
                    )

                store_lse(mt, LSE, m_ub, l_ub, row_sum_ub)
                if is_cast_out:
                    O_cast_ub = T.view(O_ub, (ROWS * 2, D), dtype=out_dtype)
                    finalize_cast(mt, O, O_ub, O_cast_ub, l_ub)
                else:
                    finalize_fp32(mt, O, O_ub, l_ub)

    return main_with_lse
