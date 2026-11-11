# Step 1: compare installed native attention implementations

This is the active remote-agent task. Do not rerun the upstream32 study or the full 26-case matrix. Do not optimize kernels in this task.

## Question and scope

On one idle device and one unchanged environment, compare the current TileLang persistent candidate with installed FlashAttentionScore and installed FusedInferAttentionScore V5. Determine the fastest supported implementation for each of three shapes. This is not a reproduction of the PR author's custom-built FIA V5 baseline.

| Case | B | Hq | Hkv | M | N | D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 64 | 1 | 1024 | 1024 | 128 |
| long | 1 | 8 | 1 | 4096 | 4096 | 128 |
| pr | 1 | 32 | 1 | 8192 | 8192 | 128 |

All inputs are contiguous BF16 BNSD; default scale is 1/sqrt(128); seed 2026; no mask, dropout, quantization, positional bias, or paged KV. Keep the same logical inputs across implementations.

## First establish FIA availability

Inspect the installed torch_npu Python entry points, schemas and their op-plugin dispatch before writing a call. The `npu_fused_infer_attention_score` / `npu_fused_infer_attention_score_v2` names are discovery candidates, not guaranteed APIs or proof of backend V5. Record the actual callable, exact arguments, tensor/list conventions, layouts, head counts and resolved ACLNN/operator name. Do not guess optional defaults that change attention semantics. Set inference-only options consistently; disable optional LSE when supported and report mandatory auxiliary outputs.

Start with a small aligned BF16 MQA correctness smoke test using the installed interface. Confirm that it resolves to FIA V5 via available dispatch source and runtime/profiler evidence. Distinguish a missing Python wrapper, missing library symbol, unsupported shape and failed invocation. If installed FIA V5 is unavailable, save the exact evidence and stop the FIA arm: no custom build, shared-package reinstall, environment upgrade or substitution with a different version. Report unavailable rather than a performance loss. Do not repeat the existing two-arm study just to fill a table.

## Reuse the existing benchmark

Create one small `bench_fia.py` adaptation of this package's `bench.py`, preserving the original. Add only the installed FIA callable and identity metadata necessary for this comparison. Keep the existing TileLang path (`torch.compile` using the persistent overlay) and eager FlashAttentionScore path. Do not replace TileLang with a direct kernel call in the main table.

Use `prepare_overlay.py --variant persistent --cube-cores <actual count>` for TileLang. Use an isolated baseline environment for the native arms; no TileLang overlay leakage. Record original and generated candidate hashes, driver hash, device model/core counts, Torch/torch_npu, TileLang commit/diff, CANN runtime/compiler versions and operator identities. The expected unmodified persistent source SHA256 is b800d9f5054731e04c558d82b7967b4ef8c8bafd56404d72154155300753c301; a core-count substitution must be recorded.

The main metric is **user API-path NPU-event latency**: compiled FlexAttention for TileLang versus the two eager native operator APIs. This measures the available user paths, not identical host wrappers or isolated kernels. Exclude compilation, reference computation and explicit layout preparation; include conversions inside an operation if required, and disclose them. Prefer BNSD for all arms to avoid conversions. Do not subtract event and profiler durations to claim host overhead.

## Correctness and timing

1. Verify each arm against the same FP32 reference before timing and after repeated buffer reuse. Check all outputs for finiteness and retain atol=rtol=0.03. Reuse the existing full reference where practical; if the large case needs a chunked reference, document it. A sampled reference must be labelled sampled and must not be reported as full-output coverage.
2. Run 10 warmups and 30 positive finite NPU-event samples, compilation excluded, in a fresh process per row.
3. Round 1 order: TileLang persistent, FlashAttentionScore, FIA V5. Round 2: reverse order. With all arms available this is **18 timing runs**, not a broader sweep.
4. Recheck device health and foreign processes before and during each run. Only use the assigned idle device; terminate only your own process group on contention. Retain completed rows and archive interrupted attempts; resume missing rows only. Stop on numerical or compilation failures and report the issue rather than silently changing the kernel.
5. Save all samples, median/min/max, correctness errors/coverage, exit status and source identity per row. A shape rejected by FIA is unsupported, not a zero or infinite timing.

After the main table passes, profile **only the PR shape**, once per available implementation using the already validated msprof workflow. Report attention device-task durations separately, with operator names, block/mix counts and exact collection protocol. Preserve raw exports on the server; submit a compact row export/JSON. Do not add concurrent AIC/AIV records or infer launches from raw record count. If an operation has multiple tasks, report that explicitly rather than selecting only its fastest task. Profiling is diagnostic and cannot replace unprofiled event measurements.

## Return and stop

Append a concise Chinese reply to CODEAGENT.md and add `compare_fia_results.json` with:

- API availability, exact calls and resolved backend identities;
- environment/source hashes and numerical evidence;
- three-shape event table with both round medians and all raw samples;
- clearly named speedups `flash_ms / tilelang_ms` and `fia_ms / tilelang_ms` (>1 means TileLang is faster), plus FIA versus FlashAttentionScore;
- separate PR-shape device-task table and any unsupported cases;
- whether native baselines differ materially by shape; no extrapolation to D256/512 or all GQA.

Recompute ratios from samples. The preceding compare_upstream_results.json contains inverse/misnamed ratio keys; do not copy those labels into this result. Preserve previous evidence unchanged and explain any correction in the new reply.

No production integration, no full-matrix expansion, no kernel tuning, no group-repository push, no PR creation. Fetch before amending the existing one communication commit, preserve both parties' additions and use an explicit expected-SHA force-with-lease. Stop after returning this bounded result; the user will decide step 2.
