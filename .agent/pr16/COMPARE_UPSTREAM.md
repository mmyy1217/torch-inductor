# Same-environment comparison with PR16

Purpose: separate adaptation overhead, logical-block scheduling, and measurement scope. This is not yet a reproduction of the reported 1.03x against FIA V5.

## Prepare an isolated directory

Use the already working TileLang/CANN environment; do not reinstall shared packages. Select an idle device with `ASCEND_RT_VISIBLE_DEVICES`; device 0 inside Python is the selected logical device. Record actual Cube/Vector counts, versions, compiler identity and diffs. Monitor occupancy during each bounded run and interrupt only your own process group if another user arrives. Preserve partial results.

```bash
work=/absolute/path/to/new/pr16-comparison
mkdir -p "$work"
cp .agent/pr16/compare_upstream.py "$work/compare.py"
cp .agent/pr16/persistent.py "$work/adapted28.py"
curl --fail --location 'https://raw.githubusercontent.com/HelloWorld07-7/tilelang-asc/77ced9c5889e1ff4b22d1f19cb77a2829195c53b/examples/ascend/flash_attention/core.py' -o "$work/upstream_core.py"
python3 - "$work" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
s = (root/'adapted28.py').read_text()
assert s.count('min(28, NUM_TILES)') == 1
(root/'adapted32.py').write_text(s.replace('min(28, NUM_TILES)', 'min(32, NUM_TILES)'))
PY
sha256sum "$work"/*.py
```

The 28-block arm is specifically for the previously measured 28-AIC server. Do not treat 28 as a universal constant. The original PR's 2048 query tiles are not divisible by 28, and its builder rejects that setting. Run the unmodified upstream kernel with **32 logical blocks on the same 28-AIC device**, which is valid scheduling and does not pretend that the device has 32 physical cores. Compare adapted32 first, then adapted28. Do not modify upstream remainder handling in this experiment.

## Run a bounded two-round comparison

First inspect the driver and verify its four-tensor ABI against the installed compiler. The coordinator only syntax-checked this new driver; its first device attempt was interrupted by another user before results. Fix any interface issue explicitly, retaining the diff. Do not silently change numerical thresholds or sampling.

```bash
for round in 1 2; do
  order='upstream32 adapted32 adapted28 native'
  if [ "$round" = 2 ]; then order='native adapted28 adapted32 upstream32'; fi
  for impl in $order; do
    # Recheck device occupancy before each command; keep an occupancy monitor active.
    timeout 300 python -u -P "$work/compare.py" --impl "$impl" --out "$work/$round-$impl"
  done
done
```

An output directory must be new. Archive interrupted attempts rather than overwriting them. Compile and correctness failures stop the experiment; do not continue to timing on a failed gate.

- Shape: B1/Hq32/Hkv1/M8192/N8192/D128, BF16, default scale, seed 2026.
- Identical logical BNSD input values across implementations; upstream views the contiguous H,M rows as a two-dimensional tensor. Every query head shares the same K/V, so flattening is mathematically valid. This is **head-major flattening**, not the original PR's BSND storage order; report that difference.
- Both TileLang paths use direct calls and preallocated outputs, without Inductor wrappers. Ten warmups, 30 NPU-event samples, reversed implementation order, fresh process per row.
- The driver checks all output elements for finiteness and six deterministic query rows per head against an independent FP32 reference before and after timing. This is a **sampled numerical check**, not full-output correctness coverage.
- Native uses eager `npu_fusion_attention`, including its returned output allocations. Keep this scope distinct; do not label its ratio as pure kernel speedup or FIA V5 reproduction.
- Save source hashes, generated device source, all samples, pre/post errors and environment. Aggregate two round medians only after validating both complete rows.

## Resolve the remaining measurement differences

After the direct comparison passes, use one existing supported profiler workflow on each of upstream32, adapted32, adapted28 and native at the same shape/environment. Export a **copy** of raw traces, identify the actual attention task, and report device-task durations separately from NPU-event intervals. Never sum concurrent AIC/AIV records to obtain latency or count them as separate launches. Keep this bounded; do not build another review/driver hierarchy.

If the original PR's FIA V5 baseline and build instructions are available, run them in an isolated build with the same input semantics, flags and profiling scope. Otherwise explicitly report **FIA V5 unavailable; original 1.03x not reproduced**. Do not substitute `npu_fusion_attention` and call it the same baseline.

Interpretation:

1. upstream32 vs adapted32: evidence about adaptation, with layout-order/guard differences still disclosed.
2. adapted32 vs adapted28: evidence about logical-block scheduling on this actual device.
3. direct vs existing Inductor API timings: descriptive unless measured together with the same inputs/environment; do not subtract different sessions to claim wrapper overhead.
4. Original PR vs FIA V5: only claim reproduction after both baseline and measurement scope match.

Return a concise table, raw samples JSON, generated-source hashes, numerical coverage, versions, actual native operator name, and an explanation supported by the measurements. Do not integrate kernels into the production branch, push to the group repository, or create a PR. Amend the existing single communication commit after fetching and preserving both parties' updates, with an explicit force-with-lease.
