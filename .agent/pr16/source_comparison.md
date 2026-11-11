# Source comparison: TileLang persistent vs installed FAS/FIA (D128, BF16, MQA)

Scope: `.agent/pr16/SOURCE_GUIDED_OPTIMIZATION.md`. One bounded optimization experiment.
Baseline parent: `persistent.py` SHA256 `b800d9f5054731e04c558d82b7967b4ef8c8bafd56404d72154155300753c301`.

## Source identity

| Item | Value | Status |
| --- | --- | --- |
| Installed OPP tree | `/usr/local/Ascend/cann/opp/.../ops_transformer/ascendc/` | confirmed (read at runtime path) |
| OPP version | `9.2.0`, `timestamp=20260930_000324083` | confirmed |
| FAS entry | `flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_vec_base.h` | confirmed |
| FAS cube | `flash_attention_score/op_kernel/arch35/flash_attention_noquant_block_cube.h` | confirmed |
| FIA entry | `fused_infer_attention_score/op_kernel/arch35/flash_attention_noquant_block_vec_base_fused_infer.h` | confirmed |
| Shared softmax VF | `common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h` | confirmed |
| Our generated code | `work/pr16-fia/persistent/artifacts/<sha>/device_code.txt` | confirmed |

Note: the coordinator's local leads referenced a 9.1.0-beta.3 tree. This server has **9.2.0**;
line numbers below are from 9.2.0 and differ from the 9.1 numbers quoted in the task doc.

## Confirmed runtime facts (not hypotheses)

| Fact | Evidence |
| --- | --- |
| All three implementations launch 28 logical blocks / 56 mix blocks on PR shape | `msprof op_summary` `Block Num`=28, `Mix Block Num`=56 for all arms |
| FAS resolves to `aclnnFlashAttentionScoreV4_...`; FIA to `FusedInferAttentionScore` | `op_summary` `Op Name` |
| Our kernel is **fixpipe-bound**; FAS is not | see pipe table below |
| D128 aligned path selects UB output, not GM | `GetC2Position` returns `VECCALC` when `dTemplateType <= Aligned128` (`flash_attention_score_common_regbase_arch35.h:220-223`) |

## Pipe times (per-task median µs, same device, same inputs)

| Shape | Impl | duration | mac | mte1 | mte2 | fixpipe | vec | fix/dur |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short 1×64×1024×1024 | TileLang | 151.1 | 102.6 | 55.0 | 94.0 | **149.2** | 122.0 | **98.7%** |
| short | FAS | 109.1 | 99.1 | 66.8 | 93.0 | 94.4 | 83.0 | 86.5% |
| long 1×8×4096×4096 | TileLang | 246.0 | 212.9 | 122.0 | 204.6 | **244.1** | 203.9 | **99.2%** |
| long | FAS | 218.1 | 207.3 | 138.4 | 188.7 | 199.2 | 163.2 | 91.3% |
| pr 1×32×8192×8192 | TileLang | 3521.1 | 3133.0 | 1794.2 | 2721.0 | **3519.3** | 3211.7 | **99.9%** |
| pr | FAS | 3235.9 | 3197.8 | 2139.2 | 2825.2 | 3103.7 | 2740.2 | 95.9% |

MAC/MTE1/MTE2 are comparable or better on our side. The gap is concentrated in **fixpipe**
and secondarily `vec`. Our fixpipe is saturated against duration in every shape; FAS keeps
5–13% of headroom.

Fixpipe gap scales with KV block count, i.e. it is a **per-KV-block fixed cost**:

| Shape | KV blocks (N/128) | fixpipe gap | gap / block |
| --- | ---: | ---: | ---: |
| short | 8 | 54.8 µs | 6.85 µs |
| long | 32 | 44.9 µs | 1.40 µs |
| pr | 64 | 415.6 µs | 6.49 µs |

(short and pr agree at ~6.5–6.9 µs/block; long is an outlier, likely because its 32 blocks
map differently onto 28 cores — reported as measured, not explained.)

## Difference table

| Dimension | FAS (9.2.0) | Our persistent | Consequence |
| --- | --- | --- | --- |
| Work assignment | `CalcS1Coord`/`CalcS2Coord` split S1 across cores; `s2LoopCount` drives a per-iteration pipeline | persistent loops `local_mt` over tiles, `kv` over KV blocks | Both 28 blocks; internal tile shape not proven identical |
| KV reuse | K loaded to L1 NZ once per KV block (`LoadKeyToL1Nz`), V via `GetValueGm` | K/V copied per KV block into L1, Q held in L0A across the loop | Similar intent; our Q residency is already the PR change |
| QK tiling | `s1BaseSize`×`s2BaseSize` from template types, `l1BaseD=128` for BF16 | BR=128, BC=128, D=128 | Same nominal tile |
| PV tiling | `IterateBmm2` with `BuffersPolicy3buff` L1 ring; `bmm2OutPos` UB for D128 | `pv` with L1 V_shared, L0C→UB `dual_copy` | Both UB output for D128 |
| **Softmax first KV** | **`s2LoopCount == 0` instantiates `ProcessVec1VfDn<..., isUpdate=false, ...>`** (`flash_attention_noquant_block_vec_base.h:479-530`) → `ProcessVec1DnNoUpdateVF` | **every KV block runs the generic online update** (`softmax_pr16` always reads `old_m_ub`, computes `alpha`, updates `l`) | **This is the lead the task doc flagged. Confirmed present in 9.2.0.** |
| **Output update first KV** | **`s2LoopCount == 0` → `DataCopy(vec2ResUb, mmRes, vec2CalcSize)` (plain copy, no rescale)** (`:1372-1374`); later blocks use `FlashUpdateNew` | **every KV block runs `accumulate_output`** (`O = alpha*O + O_tmp`) | Our first block multiplies by `alpha=0` instead of copying |
| Output finalize | `LastDivNew` at `s2LoopCount == s2LoopLimit`, single divide | `finalize_cast` divides by `l` | Equivalent |
| Buffer lifetimes | `taskIdMod2`/`taskIdMod3`/`multiCoreIdxMod3` ring buffers | `kv % 3` / `kv & 1` indexed buffers in generated code | Both ring-buffered |
| Sync | `SetCrossCore`/`WaitFlag<MTE3_V>` pairs | `T.Pipelined` + `dual_copy` unit_flag_ctrl=3 | Different mechanism, not compared byte-wise |
| Auxiliary outputs | LSE optional | none | Not a factor |

## Candidate chosen and outcome

**Candidate**: D128-only combination of (a) drop per-Q-tile `clear_O_ub` and (b) overwrite
`O_ub` from `O_tmp_ub` on `kv == 0` instead of the generic rescale. Rationale: matches FAS's
`s2LoopCount == 0` plain-copy branch, and the overwrite makes (a) safe (O_ub's prior contents
are never read, so uninitialised/non-finite UB bytes cannot leak in).

**Numerical result**: correct on every gate case (see JSON). The removal of `clear_O_ub` alone
is *also* numerically correct, but only because `alpha == 0` masks garbage; that variant was
rejected as unsafe and is kept only as an attribution probe.

**Performance result — candidate rejected**:

| Shape | parent (µs) | candidate (µs) | speedup |
| --- | ---: | ---: | ---: |
| short | 193.0 / 194.7 | 195.5 / 196.3 | 0.987 / 0.992 |
| long | 288.0 / 292.0 | 353.0 / 361.3 | **0.816 / 0.808** |
| pr | 3573.6 / 3559.0 | 4634.3 / 4634.1 | **0.771 / 0.768** |

**Why it failed**: the `if kv == 0` inside `T.Pipelined` is rewritten by software pipelining
into prologue predicates (`kv==1`, `kv==3`, `kv==64`, …) in the generated code, adding 5 extra
predicate nests and breaking the steady-state pipeline. The intended first-KV specialization
materialized in source but **not** as the intended device-code shape.

**Attribution probe (not a candidate)**: dropping `clear_O_ub` alone gives only
long 283.7 µs (vs 288.0), pr 3540.9 µs (vs 3573.6) — about 1–2%. So the removable work is
small; the real gap is fixpipe, which this change does not touch.

## Recommendation

**Do not integrate.** The candidate is numerically correct but 19–23% slower on the two large
shapes. The first-KV lead is real in FAS source but is the wrong lever here: our kernel is
fixpipe-bound (98.7–99.9% of duration), and the removable vector work is only 1–2%.

The actionable direction is the fixpipe path (~6.5 µs per KV block), not softmax/output
update. That is a larger change than this task's budget allows and is left for a future round.

## Caveats

- FAS/FIA source is the **installed 9.2.0**, not the PR author's custom build.
- Equal 28/56 block counts do **not** prove identical internal scheduling.
- The long-shape fixpipe gap (1.40 µs/block) does not follow the short/pr pattern; left
  unexplained rather than fitted.
- Native numbers in this document are context from the same session, not a simultaneous
  controlled native-win measurement.
