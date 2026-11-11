# Adapted from Tile-AI TileLang; the original license is retained below.
# Source: tile-ai/tilelang core.py, eb9c5444db4f6ea1db2361a776df2bc6c502ed66.
"""Dense BF16 attention: Cube QK/PV, SIMD online softmax and a three-stage pipeline.

MIT License

Copyright (c) Tile-AI.
**During the period from December 1, 2024, to Mar 14, 2025, this project is
subject to additional collaboration terms with Microsoft Corporation.**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
"""

import math

import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout
from tilelang.ascend.language import simd as S


def dense_attention_prim_func(shape, scale):
    # D128 SIMD/Q-residency adapted from HelloWorld07-7 PR16, commit 77ced9c.
    B, H, q_len, kv_len, D, HKV = shape
    BH = B * H
    SCALE_AFTER_MAX = math.isfinite(scale) and scale > 0
    # D256 uses the validated DK64 inner pipeline; other D paths are unchanged.
    BR = 64 if D == 512 else 128
    BC, ROWS, VL = 128, BR // 2, 64
    DK = 64 if D == 256 else 128
    SOFTMAX_LATENCY = 735
    ACCUMULATE_LATENCY = 318
    STAGED_VERSIONS = 2 if D == 256 else 3
    GROUP_SIZE = H // HKV
    NUM_KV_BLOCKS = kv_len // BC
    NUM_M_TILES = q_len // BR
    NUM_TILES = BH * NUM_M_TILES
    NUM_BLOCKS = min(28, NUM_TILES) if D == 128 else NUM_TILES
    dtype = out_dtype = "bfloat16"
    accum_dtype = "float32"

    @T.macro
    def qk(bh, kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub):
        if D == DK:
            T.copy(K[bh, kv * BC : (kv + 1) * BC, 0:D], K_shared)
            T.copy(K_shared, qk_a_l0)
            T.gemm(qk_a_l0, qk_b_l0, qk_acc_l0c, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
        else:
            if D == 256:
                for dc in T.Pipelined(
                    D // DK, num_stages=2, annotations={"enable_offset": True}
                ):
                    T.copy(
                        K[bh, kv * BC : (kv + 1) * BC, dc * DK : (dc + 1) * DK],
                        K_shared,
                    )
                    T.copy(Q_shared[0:BR, dc * DK : (dc + 1) * DK], qk_a_l0)
                    T.copy(K_shared, qk_b_l0)
                    T.gemm(
                        qk_a_l0,
                        qk_b_l0,
                        qk_acc_l0c,
                        transpose_B=True,
                        clear_accum=(dc == 0),
                    )
            else:
                for dc in T.serial(D // DK):
                    T.copy(
                        K[bh, kv * BC : (kv + 1) * BC, dc * DK : (dc + 1) * DK],
                        K_shared,
                    )
                    T.copy(Q_shared[0:BR, dc * DK : (dc + 1) * DK], qk_a_l0)
                    T.copy(K_shared, qk_b_l0)
                    T.gemm(
                        qk_a_l0,
                        qk_b_l0,
                        qk_acc_l0c,
                        transpose_B=True,
                        clear_accum=(dc == 0),
                    )
        if D == 128:
            T.dual_copy(qk_acc_l0c, S_ub[0:BC, 0:ROWS], unit_flag_ctrl=3)
        else:
            T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC])

    @T.macro
    def softmax_pr16(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
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
            safe_max = S.vsel(S.vdup(T.float32(0), "float32", full), merged_max,
                              S.vcmps(merged_max, -T.infinity("float32"), full, "eq"))
            alpha = S.vexpdif(previous_max, safe_max, full)
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
                e0 = S.vexpdif(S.vmuls(S.vld(S_ub[r + 0, 0], "NORM"), T.float32(scale), full), safe_max, full)
                sum0 = S.vadd(sum0, e0, full)
                packed0 = S.vpack(T.reinterpret(S.vcvt(e0, "bfloat16", full, sat=False, part=0), "uint32x64"), 0)
                dst = S.vsstb(T.reinterpret(packed0, "bfloat16x128"), dst, T.int32((BC + 1) << 16 | 1), packed_mask, update=True)
                e1 = S.vexpdif(S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), full), safe_max, full)
                sum1 = S.vadd(sum1, e1, full)
                packed1 = S.vpack(T.reinterpret(S.vcvt(e1, "bfloat16", full, sat=False, part=0), "uint32x64"), 0)
                dst = S.vsstb(T.reinterpret(packed1, "bfloat16x128"), dst, T.int32((BC + 1) << 16 | 1), packed_mask, update=True)
                e2 = S.vexpdif(S.vmuls(S.vld(S_ub[r + 2, 0], "NORM"), T.float32(scale), full), safe_max, full)
                sum2 = S.vadd(sum2, e2, full)
                packed2 = S.vpack(T.reinterpret(S.vcvt(e2, "bfloat16", full, sat=False, part=0), "uint32x64"), 0)
                dst = S.vsstb(T.reinterpret(packed2, "bfloat16x128"), dst, T.int32((BC + 1) << 16 | 1), packed_mask, update=True)
                e3 = S.vexpdif(S.vmuls(S.vld(S_ub[r + 3, 0], "NORM"), T.float32(scale), full), safe_max, full)
                sum3 = S.vadd(sum3, e3, full)
                packed3 = S.vpack(T.reinterpret(S.vcvt(e3, "bfloat16", full, sat=False, part=0), "uint32x64"), 0)
                dst = S.vsstb(T.reinterpret(packed3, "bfloat16x128"), dst, T.int32((BC + 1) << 16 | 1), packed_mask, update=True)
            block_sum = S.vadd(S.vadd(sum0, sum1, full), S.vadd(sum2, sum3, full), full)
            updated_sum = S.alloc_var("float32")
            updated_sum = S.vld(l_ub[0], "NORM")
            S.vmadd(updated_sum, alpha, block_sum, full)
            S.vsts(l_ub[0], updated_sum, full, "NORM_B32")

    @T.macro
    def softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        with T.SimdVF(latency=SOFTMAX_LATENCY):
            softmax_full = S.pset(32, "PAT_ALL")
            softmax_one = S.pset(32, "PAT_VL1")
            # One bf16 register is 128 lanes: 8 C0 groups at BC=128, 4 at BC=64.
            softmax_b16 = S.pset(16, "PAT_VL128" if BC == 128 else "PAT_VL64")
            dst_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC), "bfloat16"
            )
            S.vsts(old_m_ub[0], S.vld(m_ub[0], "NORM"), softmax_full, "NORM_B32")
            if BC == 64:
                # A 64-column row is exactly one fp32 vector register.
                for r in range(ROWS):
                    s0 = S.vmuls(
                        S.vld(S_ub[r, 0], "NORM"), T.float32(scale), softmax_full
                    )
                    S.vsts(S_ub[r, 0], s0, softmax_full, "NORM_B32")
                    S.vsts(m_ub[r], S.vcmax(s0, softmax_full), softmax_one, "ONEPT_B32")
            else:
                for r in range(0, ROWS, 2):
                    s00 = S.vmuls(
                        S.vld(S_ub[r, 0], "NORM"), T.float32(scale), softmax_full
                    )
                    s01 = S.vmuls(
                        S.vld(S_ub[r, VL], "NORM"), T.float32(scale), softmax_full
                    )
                    s10 = S.vmuls(
                        S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), softmax_full
                    )
                    s11 = S.vmuls(
                        S.vld(S_ub[r + 1, VL], "NORM"), T.float32(scale), softmax_full
                    )
                    S.vsts(S_ub[r, 0], s00, softmax_full, "NORM_B32")
                    S.vsts(S_ub[r, VL], s01, softmax_full, "NORM_B32")
                    S.vsts(S_ub[r + 1, 0], s10, softmax_full, "NORM_B32")
                    S.vsts(S_ub[r + 1, VL], s11, softmax_full, "NORM_B32")
                    max0 = S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full)
                    max1 = S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full)
                    S.vsts(m_ub[r], max0, softmax_one, "ONEPT_B32")
                    S.vsts(m_ub[r + 1], max1, softmax_one, "ONEPT_B32")
            S.mem_bar("VST_VLD")
            merged_max = S.vmax(
                S.vld(m_ub[0], "NORM"), S.vld(old_m_ub[0], "NORM"), softmax_full
            )
            S.vsts(m_ub[0], merged_max, softmax_full, "NORM_B32")
            S.mem_bar("VST_VLD")
            if BC == 64:
                for r in range(ROWS):
                    max0 = S.vld(m_ub[r], "BRC_B32")
                    max0 = S.vsel(
                        S.vdup(T.float32(0), "float32", softmax_full),
                        max0,
                        S.vcmps(max0, -T.infinity("float32"), softmax_full, "eq"),
                    )
                    e0 = S.vexpdif(S.vld(S_ub[r, 0], "NORM"), max0, softmax_full)
                    S.vsts(
                        row_sum_ub[r],
                        S.vcadd(e0, softmax_full),
                        softmax_one,
                        "ONEPT_B32",
                    )
                    # Narrow into 32-bit lanes, pack, then scatter four 32B blocks.
                    packed0 = S.vpack(
                        T.reinterpret(
                            S.vcvt(e0, "bfloat16", softmax_full, sat=False, part=0),
                            "uint32x64",
                        ),
                        0,
                    )
                    dst_nz_ptr = S.vsstb(
                        T.reinterpret(packed0, "bfloat16x128"),
                        dst_nz_ptr,
                        T.int32(ROWS + 1 << 16 | 1),
                        softmax_b16,
                        update=True,
                    )
            else:
                for r in range(0, ROWS, 2):
                    max0 = S.vld(m_ub[r], "BRC_B32")
                    max0 = S.vsel(
                        S.vdup(T.float32(0), "float32", softmax_full),
                        max0,
                        S.vcmps(max0, -T.infinity("float32"), softmax_full, "eq"),
                    )
                    max1 = S.vld(m_ub[r + 1], "BRC_B32")
                    max1 = S.vsel(
                        S.vdup(T.float32(0), "float32", softmax_full),
                        max1,
                        S.vcmps(max1, -T.infinity("float32"), softmax_full, "eq"),
                    )
                    p00, p01 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                    p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")
                    e00 = S.vexpdif(p00, max0, softmax_full)
                    e10 = S.vexpdif(p10, max1, softmax_full)
                    e01 = S.vexpdif(p01, max0, softmax_full)
                    e11 = S.vexpdif(p11, max1, softmax_full)
                    row_sum0 = S.vcadd(S.vadd(e00, e01, softmax_full), softmax_full)
                    row_sum1 = S.vcadd(S.vadd(e10, e11, softmax_full), softmax_full)
                    S.vsts(row_sum_ub[r], row_sum0, softmax_one, "ONEPT_B32")
                    S.vsts(row_sum_ub[r + 1], row_sum1, softmax_one, "ONEPT_B32")
                    even_bf16_0 = S.vcvt(
                        e00, "bfloat16", softmax_full, sat=False, part=0
                    )
                    even_bf16_1 = S.vcvt(
                        e10, "bfloat16", softmax_full, sat=False, part=0
                    )
                    odd_bf16_0 = S.vcvt(
                        e01, "bfloat16", softmax_full, sat=False, part=1
                    )
                    odd_bf16_1 = S.vcvt(
                        e11, "bfloat16", softmax_full, sat=False, part=1
                    )
                    merged_bits0 = S.vor(
                        T.reinterpret(even_bf16_0, "uint16x128"),
                        T.reinterpret(odd_bf16_0, "uint16x128"),
                        softmax_b16,
                    )
                    merged_bits1 = S.vor(
                        T.reinterpret(even_bf16_1, "uint16x128"),
                        T.reinterpret(odd_bf16_1, "uint16x128"),
                        softmax_b16,
                    )
                    dst_nz_ptr = S.vsstb(
                        T.reinterpret(merged_bits0, "bfloat16x128"),
                        dst_nz_ptr,
                        T.int32(ROWS + 1 << 16 | 1),
                        softmax_b16,
                        update=True,
                    )
                    dst_nz_ptr = S.vsstb(
                        T.reinterpret(merged_bits1, "bfloat16x128"),
                        dst_nz_ptr,
                        T.int32(ROWS + 1 << 16 | 1),
                        softmax_b16,
                        update=True,
                    )
            S.mem_bar("VST_VLD")
            current_max = S.vld(m_ub[0], "NORM")
            safe_max = S.vsel(
                S.vdup(T.float32(0), "float32", softmax_full),
                current_max,
                S.vcmps(current_max, -T.infinity("float32"), softmax_full, "eq"),
            )
            alpha_reg = S.vexpdif(S.vld(old_m_ub[0], "NORM"), safe_max, softmax_full)
            S.vsts(alpha_ub[0], alpha_reg, softmax_full, "NORM_B32")
            l_scaled = S.vmul(S.vld(l_ub[0], "NORM"), alpha_reg, softmax_full)
            l_updated = S.vadd(l_scaled, S.vld(row_sum_ub[0], "NORM"), softmax_full)
            S.vsts(l_ub[0], l_updated, softmax_full, "NORM_B32")

    @T.macro
    def pack_p(P_nz_ub, P_shared):
        if D == 128:
            T.dual_copy(P_nz_ub[0:BC, 0:ROWS], P_shared[0:BC, 0:BR])
        else:
            T.dual_copy(P_nz_ub[0:ROWS, 0:BC], P_shared[0:BR, 0:BC])

    @T.macro
    def pv(bh, kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub):
        if D == DK:
            T.copy(V[bh, kv * BC : (kv + 1) * BC, 0:D], V_shared, transpose=True)
            T.copy(P_shared, pv_a_l0)
            T.copy(V_shared, pv_b_l0)
            T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_A=True, transpose_B=True, clear_accum=True, unit_flag_ctrl=3)
            T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D], unit_flag_ctrl=3)
        else:
            if D == 256:
                T.copy(P_shared, pv_a_l0)
                for dc in T.Pipelined(
                    D // DK, num_stages=2, annotations={"enable_offset": True}
                ):
                    T.copy(
                        V[bh, kv * BC : (kv + 1) * BC, dc * DK : (dc + 1) * DK],
                        V_shared,
                        transpose=True,
                    )
                    T.copy(V_shared, pv_b_l0)
                    T.gemm(
                        pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True
                    )
                    T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, dc * DK : (dc + 1) * DK])
            else:
                for dc in T.serial(D // DK):
                    T.copy(
                        V[bh, kv * BC : (kv + 1) * BC, dc * DK : (dc + 1) * DK],
                        V_shared,
                        transpose=True,
                    )
                    T.copy(P_shared, pv_a_l0)
                    T.copy(V_shared, pv_b_l0)
                    T.gemm(
                        pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True
                    )
                    T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, dc * DK : (dc + 1) * DK])

    @T.macro
    def accumulate_output(O_tmp_ub, O_ub, alpha_ub):
        with T.SimdVF(latency=ACCUMULATE_LATENCY):
            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    prev = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    if D == 128:
                        updated = S.alloc_var(accum_dtype)
                        updated = prev
                        S.vmadd(updated, alpha, cur, rescale_mask)
                    else:
                        updated = S.vadd(S.vmul(alpha, prev, rescale_mask), cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")

    @T.macro
    def kv_auto(
        bh,
        kv,
        K,
        V,
        Q_shared,
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
        qk(bh, kv, K, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)
        if D == 128:
            softmax_pr16(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        else:
            softmax(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        pack_p(P_nz_ub, P_shared)
        pv(bh, kv, V, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)
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
    def finalize_cast(bh, mt, O, O_ub, O_cast_ub, l_ub):
        with T.SimdVF(latency=1780):
            normalize_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                raw_denominator = S.vld(l_ub[i], "BRC_B32")
                denominator = S.vsel(
                    S.vdup(T.float32(1), "float32", normalize_mask),
                    raw_denominator,
                    S.vcmps(raw_denominator, T.float32(0), normalize_mask, "eq"),
                )
                for c in range(0, D, VL):
                    normalized = S.vdiv(
                        S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask
                    )
                    if D == 256:
                        ob = S.vcvt(normalized, out_dtype, normalize_mask)
                        S.vsts(O_cast_ub[i, c], ob, normalize_mask, "PK_B32")
                    else:
                        S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")
        if D != 256:
            with T.SimdVF(latency=193):
                cast_full = S.pset(32, "PAT_ALL")
                for r in range(ROWS):
                    for c in range(0, D, VL):
                        ob = S.vcvt(S.vld(O_ub[r, c], "NORM"), out_dtype, cast_full)
                        S.vsts(O_cast_ub[r, c], ob, cast_full, "PK_B32")
        T.dual_copy(O_cast_ub[0:ROWS, 0:D], O[bh, mt * BR : (mt + 1) * BR, 0:D])

    @T.prim_func
    def main(
        Q4: T.Buffer((B, H, q_len, D), dtype),
        K4: T.Buffer((B, HKV, kv_len, D), dtype),
        V4: T.Buffer((B, HKV, kv_len, D), dtype),
        O4: T.Buffer((B, H, q_len, D), out_dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            # Rank-4 ABI: alias the parameter storage as rank-3, leaving the
            # scheduled body (and its device code) untouched.
            Q = T.view(Q4, (BH, q_len, D))
            K = T.view(K4, (B * HKV, kv_len, D))
            V = T.view(V4, (B * HKV, kv_len, D))
            O = T.view(O4, (BH, q_len, D))
            Q_shared = T.alloc_l1((BR, D), dtype)
            K_shared = T.alloc_l1((BC, DK), dtype)
            V_shared = T.alloc_l1((DK, BC), dtype)
            P_shared = T.alloc_l1((BR, BC), dtype)
            qk_a_l0 = T.alloc_l0a((BR, DK), dtype)
            qk_b_l0 = T.alloc_l0b((BC, DK), dtype)
            pv_a_l0 = T.alloc_l0a((BR, BC), dtype)
            pv_b_l0 = T.alloc_l0b((DK, BC), dtype)
            qk_acc_l0c = T.alloc_l0c((BR, BC), accum_dtype)
            pv_acc_l0c = T.alloc_l0c((BR, DK), accum_dtype)
            S_ub = T.alloc_shared((BC, ROWS) if D == 128 else (ROWS, BC), accum_dtype)
            P_nz_ub = T.alloc_shared((BC + 1, ROWS) if D == 128 else (ROWS + 1, BC), dtype)
            T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
            O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_ub = T.alloc_shared((ROWS, D), accum_dtype)
            # Row statistics use full vector loads, including for ROWS < VL.
            m_ub = T.alloc_shared((VL,), accum_dtype)
            l_ub = T.alloc_shared((VL,), accum_dtype)
            alpha_ub = T.alloc_shared((VL,), accum_dtype)
            old_m_ub = T.alloc_shared((VL,), accum_dtype)
            row_sum_ub = T.alloc_shared((VL,), accum_dtype)
            if D == 256:
                T.annotate_buffer_versions(
                    {
                        K_shared: 3,
                        V_shared: 3,
                        qk_a_l0: 2,
                        qk_b_l0: 2,
                        pv_a_l0: 1,
                        pv_b_l0: 2,
                        qk_acc_l0c: 2,
                        pv_acc_l0c: 2,
                        S_ub: STAGED_VERSIONS,
                        P_nz_ub: 2,
                        alpha_ub: STAGED_VERSIONS,
                        O_tmp_ub: 1,
                    }
                )
            else:
                T.annotate_buffer_versions(
                    {
                        K_shared: 3,
                        V_shared: 3,
                        qk_acc_l0c: 2,
                        pv_acc_l0c: 2,
                        S_ub: 3,
                        P_nz_ub: 2,
                        alpha_ub: 3,
                        # A single D=512 output temporary keeps UB within its capacity.
                        O_tmp_ub: 1 if D == 512 else 2,
                    }
                )
            for local_mt in T.serial((NUM_TILES + NUM_BLOCKS - 1) // NUM_BLOCKS):
                tile = bx + local_mt * NUM_BLOCKS
                if tile < NUM_TILES:
                    bh = tile // NUM_M_TILES
                    mt = tile % NUM_M_TILES
                    T.copy(Q[bh, mt * BR : (mt + 1) * BR, 0:D], Q_shared[0:BR, 0:D])
                    if D == 128:
                        T.copy(Q_shared, qk_b_l0)
                    init_m_l(m_ub, l_ub)
                    clear_O_ub(O_ub)
                    for kv in T.Pipelined(
                        NUM_KV_BLOCKS, num_stages=3, annotations={"enable_offset": True}
                    ):
                        kv_auto(
                            bh // H * HKV + (bh % H) // GROUP_SIZE,
                            kv,
                            K,
                            V,
                            Q_shared,
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
                    O_cast_ub = T.view(O_ub, (ROWS * 2, D), dtype=out_dtype)
                    finalize_cast(bh, mt, O, O_ub, O_cast_ub, l_ub)

    return main
