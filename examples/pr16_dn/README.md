# PR16 DN layout experiment

Measured source files from run129, unchanged. This is an isolated experimental implementation, not an integrated Inductor template.

Upstream: https://github.com/HelloWorld07-7/tilelang-asc/blob/77ced9c5889e1ff4b22d1f19cb77a2829195c53b/examples/ascend/flash_attention/core.py
Review: https://github.com/yangsichan/tilelang-asc/pull/16
The original code remains subject to the upstream repository license. Preserve upstream attribution when reusing this derived implementation.

Files: `parent_core.py` is the PR baseline; `candidate_core.py` is the measured DN variant; `dn.patch` is the exact source diff. SHA256SUMS pins their identities.

## Implementation

Attention mathematics is unchanged: softmax(scale * Q K^T) V. Keep flattened query indexing s*G+g, BF16, D=128, BR=BC=128, three stages and original launch geometry.

1. Place resident Q in L0B once per query tile; load each K tile into L0A. QK GEMM computes K Q^T, producing a [key,query] score tile instead of [query,key]. Keep paired unit_flag_ctrl=3 for GEMMs and their L0C drains.
2. Split score output along the query dimension between the two AIVs. Each receives [128 keys,64 queries]. A SIMD lane now represents a query. Reduce over keys using four independent elementwise maximum chains, then combine them; do not horizontally reduce across query lanes. For positive scale, scale the maximum afterward; for nonpositive scale, scale before finding the maximum.
3. Apply the same online recurrence: m_new=max(m_old,block_max), alpha=exp(m_old-m_new), p=exp(scale*score-m_new), l_new=alpha*l_old+sum_keys(p). Use four elementwise sum chains. Convert probabilities to BF16 and pack directly to compact NZ using vpack and vsstb, PAT_VL64 and stride ((BC+1)<<16)|1.
4. Join both AIV [key,query] panels into P_shared [128,128]. Use transpose_A=True and transpose_B=True in PV (V is staged transposed) so the output is query-major P^T V. Preserve FP32 output rescaling/FMA and final normalization.

No extra guard changes are claimed; this preserves the PR numerical scope and does not inherit the broader production template's guards. The checked random/scale cases are not proof for all extreme finite BF16 inputs.

## Using the builder

In an already configured compatible Ascend TileLang environment:

```python
import tilelang
from candidate_core import flash_attention_fwd
prim = flash_attention_fwd(8192 * 32, 8192, 128,
    num_blocks=32, softmax_scale=128**-0.5, manual_schedule=False)
# Compile using the same target/pass configuration as the PR harness.
```

Q/O shapes are [S1*G,128], K/V [S2,128], contiguous BF16. Preserve s*G+g flattening. Do not pass BNSD storage as if it were this flattened BSND view. Use the PR's existing compiler/launch harness with only the builder import changed. The source files are self-contained builders, not a standalone benchmark or environment installer.

Measured stack: Ascend950PR_9579, 28 physical AIC / 56 AIV, TileLang 79eaaae3 with the coordinator's existing SIMD patches, torch/torch_npu 2.14, CANN runtime 9.2.0-beta.2. The commit alone does not recreate the modified compiler environment. Compile options included TL_ENABLE_FAST_MATH, manual_schedule=False. Validate generated layout and numerics on a different toolchain before timing.

## Results

10 warmups; 50 profiler device tasks per side per round; 5 alternating rounds; no explicit L2 flush. Each final time is the median of five per-round means. Use main_kernel/MIX_AIC task durations, not a sum of overlapping AIC/AIV records. Contention-interrupted attempts excluded. No Inductor wrapper in this experiment.

| S1=S2 | Logical blocks (both) | PR baseline us | DN us | Baseline/DN |
|---|---:|---:|---:|---:|
|8192|32|5745.97|5355.50|1.0729|
|7168|28|2661.17|2469.79|1.0775|

Both rows use G=32, D=128, BF16, scale=1/sqrt(128), physical 28 AIC. The second shape satisfies the baseline divisibility condition with 28 blocks; it is not a controlled block-count-only comparison.

| Round | 8192 parent us | 8192 DN us | 7168 parent us | 7168 DN us |
|---|---:|---:|---:|---:|
|0|5747.96|5363.96|2622.14|2472.57|
|1|5757.09|5356.49|2661.17|2469.79|
|2|5745.97|5355.50|2690.80|2475.53|
|3|5734.42|5352.14|2609.75|2453.36|
|4|5735.96|5348.45|2669.64|2450.88|

Full-output FP32 reference checks passed with atol=rtol=0.016, finite BF16 outputs; both kernels had the same reported maximum errors. Main shapes: 0.000341 (8192), 0.000350 (7168). Additional single/multi-tile seeds, zero/negative scale and larger-magnitude input passed; the larger-input max absolute error 0.0696 passed the combined absolute+relative criterion, not an absolute-only 0.016 bound.

This establishes a measured improvement over the PR kernel on this host, not a win over FIA V5 or a reproduction on the author's 32-AIC hardware. Raw profiler archives remain on the measurement host; this package contains source and summary, not the complete raw evidence archive.
