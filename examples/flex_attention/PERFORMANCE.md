# Dense attention performance

Huawei Ascend 950PR; 26 static BF16 shapes, measured 2026-11-11 UTC.

- **Correctness:** 29 Inductor checks passed, including zero and negative scale; all 156 benchmark invocations passed their reference checks.
- **Performance:** TileLang is faster than Inductor Triton on all 26 measured shapes (1.33–4.59x). Native torch_npu remains faster on every shape.
- **Implementation:** rank-four launch ABI; D=256 uses DK=64 inner L0 double buffering and register-forwarded final BF16 conversion. D=128/512 retain their existing computation paths.

## Method

- Same Q/K/V, seed 2026, default scale; tiled CPU FP32 reference rounded to BF16, `atol=rtol=0.03`.
- Ten warmups and 30 NPU-event samples per invocation; each shape/backend/round runs in a fresh process. Round two reverses backend order.
- Table values are the mean of the two round medians. TileLang and Triton use compiled FlexAttention; torch_npu uses eager `npu_fusion_attention`.
- API-path latency includes wrapper/launch and output allocation work; compilation and CPU reference are excluded. These are not isolated device-kernel timings. Reference checks precede timing.
- Shared host and unlocked clocks; device occupancy snapshots do not establish exclusive resources throughout measurement. Small differences can reflect noise.
- A device initialization timeout interrupted round two before kernel execution. Only missing invocations were resumed; completed measurements were retained.
- FLOPs = `4*B*HQ*M*N*D`; softmax FLOPs are excluded. No unverified theoretical peak or MFU is claimed.

## Results

Speedup = baseline latency / TileLang latency. Values above 1 mean TileLang is faster; values below 1 mean it is slower.

| D | B | HQ/HKV | M/N | TileLang (µs) | Triton (µs) | torch_npu (µs) | Triton / TileLang | torch_npu / TileLang |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 4/4 | 256/256 | 49.22 | 76.42 | 35.26 | 1.55x | 0.72x |
| 128 | 1 | 8/8 | 1024/1024 | 62.24 | 151.40 | 48.97 | 2.43x | 0.79x |
| 128 | 1 | 8/2 | 1024/1024 | 62.09 | 149.60 | 48.34 | 2.41x | 0.78x |
| 128 | 1 | 64/1 | 1024/1024 | 222.28 | 635.61 | 132.56 | 2.86x | 0.60x |
| 128 | 2 | 8/2 | 512/2048 | 80.75 | 240.86 | 63.16 | 2.98x | 0.78x |
| 128 | 4 | 4/4 | 1024/512 | 68.42 | 142.53 | 45.99 | 2.08x | 0.67x |
| 128 | 1 | 8/1 | 4096/4096 | 308.42 | 1263.56 | 240.83 | 4.10x | 0.78x |
| 128 | 2 | 4/2 | 256/128 | 52.80 | 72.94 | 30.61 | 1.38x | 0.58x |
| 128 | 1 | 8/2 | 8192/8192 | 996.50 | 4569.99 | 827.23 | 4.59x | 0.83x |
| 256 | 1 | 4/4 | 256/256 | 53.91 | 73.24 | 33.73 | 1.36x | 0.63x |
| 256 | 1 | 8/8 | 1024/1024 | 89.34 | 194.21 | 73.83 | 2.17x | 0.83x |
| 256 | 1 | 8/2 | 1024/1024 | 95.21 | 192.40 | 72.44 | 2.02x | 0.76x |
| 256 | 1 | 64/1 | 1024/1024 | 385.54 | 890.00 | 288.62 | 2.31x | 0.75x |
| 256 | 2 | 8/2 | 512/2048 | 131.30 | 322.68 | 110.31 | 2.46x | 0.84x |
| 256 | 4 | 4/4 | 1024/512 | 92.37 | 182.90 | 67.87 | 1.98x | 0.73x |
| 256 | 1 | 8/1 | 4096/4096 | 625.94 | 1797.21 | 557.96 | 2.87x | 0.89x |
| 256 | 2 | 4/2 | 256/128 | 51.73 | 75.51 | 48.02 | 1.46x | 0.93x |
| 512 | 1 | 4/4 | 256/256 | 54.56 | 72.64 | 45.14 | 1.33x | 0.83x |
| 512 | 1 | 8/8 | 1024/1024 | 173.65 | 264.23 | 160.56 | 1.52x | 0.92x |
| 512 | 1 | 8/2 | 1024/1024 | 173.88 | 262.66 | 159.36 | 1.51x | 0.92x |
| 512 | 1 | 64/1 | 1024/1024 | 1087.63 | 1540.50 | 808.57 | 1.42x | 0.74x |
| 512 | 2 | 8/2 | 512/2048 | 292.62 | 461.74 | 276.47 | 1.58x | 0.94x |
| 512 | 4 | 4/4 | 1024/512 | 196.20 | 261.59 | 143.99 | 1.33x | 0.73x |
| 512 | 1 | 8/1 | 4096/4096 | 1935.44 | 3056.12 | 1624.61 | 1.58x | 0.84x |
| 512 | 2 | 4/2 | 256/128 | 54.81 | 75.03 | 42.59 | 1.37x | 0.78x |
| 512 | 1 | 4/4 | 4096/4096 | 1031.07 | 1631.76 | 833.06 | 1.58x | 0.81x |

## Validation

- Ten CPU contract/template tests pass with device compilation mocked.
- The 26-shape matrix covers MHA/GQA/MQA, batch 1/2/4, unequal sequence lengths, single KV tiles and D=128/256/512. This is not exhaustive coverage.
- Additional checks cover zero-scale D256/D512 and negative-scale D256.
- Generated artifacts and source identities were checked for the tested implementation.

## Environment


| Component | Measured version |
| --- | --- |
| Hardware / driver | Huawei Ascend 950PR / `25.7.rc1` |
| PyTorch / torch_npu | `2.14.0+cpu` / `2.14.0` |
| Triton-Ascend | Module reports `3.5.0`; package `3.6.0.dev0+gitf3733306` |
| TileLang | `79eaaae37014abdd2ab7ee381732a1e906286e51`, with existing SIMD patches |
| TileLang patch SHA256 | `aae5d77024255a0f95d6133d97af338c03390ad8dc40757bb3b3f7f96b029d63` |
| CANN runtime / compiler | `9.2.0-beta.2` / `9.3.0` (`V100R001C12B140`, weekly `20261009.01`) |
| Host / concurrency | Python 3.11; OMP threads 2; compilation threads 1 |

The compiler includes pre-existing local patches; checking out its commit alone is not a complete reproduction recipe. Reuse a matching built environment or obtain the corresponding patch and dependency set. Compiler code was not changed or rebuilt for this experiment.
