# TileLang Dense FlexAttention

Dense attention for Huawei Ascend 950PR, integrated with the Inductor TileLang backend.

## Quick start

- Install matching PyTorch / torch_npu 2.14, Huawei Ascend TileLang and compatible CANN.
- Use this checkout's Python sources with the matching torch_npu binaries.
- Select an allocated device with `ASCEND_RT_VISIBLE_DEVICES`.
- Import `torch_npu._inductor` before importing `flex_attention`, as in the examples.

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 python examples/flex_attention/dense.py
TORCHINDUCTOR_COMPILE_THREADS=1 python examples/flex_attention/mixed.py
```

| Entry | Purpose |
| --- | --- |
| `dense.py` | Minimal dense attention example |
| `mixed.py` | Pointwise operations before and after attention |
| `bench.py` | Correctness and latency: TileLang, Inductor Triton, torch_npu |
| [PERFORMANCE.md](PERFORMANCE.md) | Measured shapes, methodology, results and environment |

Select `config.flex_attention.implementation = "tilelang"` and
`torch.compile(..., dynamic=False, options={"npu_backend": "tilelang"})`.
The template forms a fusion boundary; compatible pointwise operations can fuse with each other.

## Supported inputs

| Property | Requirement |
| --- | --- |
| Operation | Forward dense attention; omit `score_mod` and `block_mask` |
| Q | `[B, HQ, M, D]` |
| K / V | `[B, HKV, N, D]`, identical shapes |
| D | **128, 256 or 512**, equal across Q/K/V |
| Heads | MHA (`HQ=HKV`), GQA (`HQ % HKV=0`), MQA (`HKV=1`) |
| Dimensions | Positive and static; matching B; M and N independently divisible by 128 |
| Layout / dtype | Contiguous BF16 tensors on the same NPU |
| Scale | Finite static value; default `1/sqrt(D)` |
| Size | Fewer than `2**31` elements per input; device memory must suffice |
| Execution | Inference, Python wrapper; default grad mode works without input gradients |

- Use `flex_attention(q, k, v, enable_gqa=True)` for GQA/MQA. KV heads are indexed directly, without expanding KV tensors.
- B and head counts have no additional fixed cap. The measured matrix is a subset of the supported contract.
- M may differ from N. This covers aligned prefill/chunked-query shapes, not single-token decode.
- D is the per-head dimension, not the model hidden size. D=384 is not supported.
- Shape changes can recompile. Templates compile during wrapper loading.
- Dense means all keys are visible; it does not implement causal attention.

### Deferred

| Area | Not supported in this version |
| --- | --- |
| Customization | score/mask callbacks, BlockMask, captured tensors |
| Shapes | Other D values, unequal QK/V dimensions, batch broadcast, sequence tails, decode M=1, symbolic shapes |
| Types / layouts | FP16/FP32 inputs, noncontiguous inputs |
| Interfaces | Backward, LSE/max outputs, C++ wrapper, AOTInductor |
| Optimization | Template prologue/epilogue fusion, multi-candidate autotuning |

Unsupported inputs fail explicitly; there is no automatic Triton fallback.
No manual compiler revision is required. After rebuilding a compiler locally,
restart Python and use fresh Inductor / TileLang cache directories.

## Benchmark

Run each implementation on the same allocated device; change only `--impl`:

```bash
python examples/flex_attention/bench.py --impl tilelang --shape 2 8 512 2048 --kv-heads 2 --head-dim 256
python examples/flex_attention/bench.py --impl triton --shape 2 8 512 2048 --kv-heads 2 --head-dim 256
python examples/flex_attention/bench.py --impl torch_npu --shape 2 8 512 2048 --kv-heads 2 --head-dim 256
```

- `--shape B HQ M N`; `--kv-heads` defaults to HQ; `--head-dim` defaults to 128.
- `--check-only` omits timing; `--scale` exercises a non-default scale.
- Independent tiled CPU FP32 reference, BF16 output, `atol=rtol=0.03`.
- Ten warmups, median of 30 NPU-event samples; compilation and reference excluded.
- TileLang/Triton measure compiled graphs; torch_npu calls `npu_fusion_attention` directly. These are API-path latencies, not isolated-kernel timings.
- FLOPs: `4*B*HQ*M*N*D`. `--peak-tflops` optionally computes MFU from a confirmed non-sparse BF16 peak; it is not inferred from device names.

## Validation and diagnostics

| Test | Coverage |
| --- | --- |
| `test/_inductor/test_tilelang_flex_attention.py` | CPU contracts, static shape/head/scale specialization, template generation, reuse and cache isolation; device compilation mocked |
| `test/_inductor/tilelang/test_flex_integration.py` | Backend dispatch; NPU mixed graph, MHA/GQA/MQA, D=128/256/512, single KV tile, unequal lengths, custom scale, repeated calls and auxiliary-output rejection |

- Set `TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=/path/to/debug` for FX, pre/post-fusion IR and wrapper code.
- Set `FLEX_TILELANG_ARTIFACT_DIR=/path/to/artifacts` for rendered TileLang source, lowered IR and device code.
- Disable artifact collection during timing.
