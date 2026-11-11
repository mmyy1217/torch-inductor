# PR16 experiment package

This package is experimental. Production source remains identical to the base commit. Start with [CODEAGENT.md](../../CODEAGENT.md) for the task and coordination rules.

## Files

| File | Purpose |
| --- | --- |
| `compute.py` | PR16 arithmetic, Q residency and paired UnitFlag changes for D128 |
| `persistent.py` | Same computation plus multiple query tiles per block |
| `prepare_overlay.py` | Copies checkout Inductor into a fresh directory; links the installed package without modifying it |
| `bench.py` | Existing API-path benchmark plus a post-timing correctness check outside the measured interval |
| `matrix.json` | Existing 26-shape matrix; shape order is B, HQ, HKV, M, N, D |
| `source_hashes.json` | Candidate, benchmark and matrix identities |

## Environment

Use an already configured Huawei Ascend 950 environment. The coordinator used Torch/torch_npu 2.14, TileLang `79eaaae37014abdd2ab7ee381732a1e906286e51` with existing local SIMD patches (diff SHA256 `aae5d77024255a0f95d6133d97af338c03390ad8dc40757bb3b3f7f96b029d63`), CANN runtime 9.2.0-beta.2 and compiler 9.3.0 weekly 20261009.01. The patch is **not bundled**; that commit alone is not an exact reproduction environment. Record a different toolchain honestly and validate API availability before testing; do not silently change a shared installation.

Use the allocated physical device via `ASCEND_RT_VISIBLE_DEVICES` and `ASCEND_VISIBLE_DEVICES`; the benchmark selects logical device 0. Query the real hardware count in that environment:

```bash
python -c 'import torch, torch_npu; torch.npu.set_device(0); p=torch.npu.get_device_properties(0); print(p); print(p.cube_core_num, p.vector_core_num)'
```

The example below uses `28` only as a placeholder for the measured count. All work directories must be new. Run from the repository root after activating the device environment.

```bash
repo="$PWD"
work="$HOME/flex-pr16-codeagent"
python .agent/pr16/prepare_overlay.py --variant baseline --out "$work/baseline"
python .agent/pr16/prepare_overlay.py --variant compute --out "$work/compute"
python .agent/pr16/prepare_overlay.py --variant persistent --cube-cores 28 --out "$work/persistent"
```

## Correctness gate

Use subshells so overlays do not leak between implementations. Repeat the command for `compute` and `persistent`. Start with the small case, then the matrix and zero/negative scale. Keep the normal reference tolerance unchanged.

```bash
(
  source "$work/compute/env.sh"
  python -u -P "$repo/.agent/pr16/bench.py" --impl tilelang \
    --shape 1 8 1024 1024 --kv-heads 2 --head-dim 128 --check-only
)
```

Additional checks use `--scale 0` and `--scale -0.125`. D256/D512 control cases use `--head-dim 256` / `512`. Source and output artifacts remain in the overlay work directory. The helper prints its generated `env.sh`; `source.json` records the actual generated kernel hash, including the selected core count.

## Performance

After correctness passes, omit `--check-only`. Use the same command for all three overlays; select `--impl torch_npu` under the baseline overlay for the native reference. Each command is a fresh process. Redirect to a distinct log and record its exit code; never overwrite interrupted attempts.

| Case | Arguments after `--impl` selection |
| --- | --- |
| Short MQA | `--shape 1 64 1024 1024 --kv-heads 1 --head-dim 128` |
| Long MQA | `--shape 1 8 4096 4096 --kv-heads 1 --head-dim 128` |
| PR shape | `--shape 1 32 8192 8192 --kv-heads 1 --head-dim 128` |

Round 1: baseline, compute, persistent, native. Round 2: reverse order. Ten warmups, 30 NPU-event samples, compilation/reference excluded. Preserve the sample arrays; speedup is baseline latency divided by candidate latency (>1 means the candidate is faster). These are API-path timings, not isolated kernel durations. The large PR case needs more CPU reference time and memory.

D128 is the only optimized dimension in this package. No new BSND layout, explicit cross-head staging or D256/D512 optimization is claimed. Keep performance results from different servers separate.

## Returning results

Append the reply to `CODEAGENT.md`. Add only a concise environment/measurement JSON if useful. Do not commit caches, full profiler archives, tensors, private machine paths or credentials. Amend the one handoff commit after fetching the latest remote branch; use an explicit expected SHA with `--force-with-lease`. A rejected lease requires reconciling the other agent's update, not overriding it.
