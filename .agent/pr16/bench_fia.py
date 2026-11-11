# ruff: noqa: E402
"""Compare TileLang persistent FlexAttention with installed FlashAttentionScore and FIA V5.

Minimal adaptation of .agent/pr16/bench.py: adds the installed FusedInferAttentionScore
arm and identity metadata. The original bench.py is preserved unchanged.
"""

import argparse
import json
import statistics
import time

print("MONITOR_STAGE: importing", flush=True)

import torch
import torch_npu
import torch_npu._inductor  # noqa: F401 - Patch before binding FlexAttention.
from torch_npu._inductor import config
from torch.nn.attention.flex_attention import flex_attention

print("MONITOR_STAGE: ready", flush=True)

FIA_CALLABLE = "torch_npu.npu_fused_infer_attention_score"
FIA_KWARGS = dict(input_layout="BNSD", softmax_lse_flag=False)


def reference(q, k, v, scale):
    """Group Q heads onto KV heads; tile scores to bound CPU reference memory."""
    b, h, m, _ = q.shape
    group = h // k.shape[1]
    result = torch.empty_like(q, device="cpu")
    q, k, v = (x.cpu().float() for x in (q, k, v))
    for bi in range(b):
        for hi in range(h):
            ki = hi // group
            for start in range(0, m, 128):
                scores = (q[bi, hi, start : start + 128] @ k[bi, ki].T) * scale
                result[bi, hi, start : start + 128] = scores.softmax(-1) @ v[bi, ki]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("tilelang", "flash", "fia"), required=True)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        default=(1, 4, 1024, 1024),
        metavar=("B", "HQ", "M", "N"),
    )
    parser.add_argument("--kv-heads", type=int, help="Defaults to HQ (MHA); 1 selects MQA")
    parser.add_argument("--head-dim", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    b, h, m, n = args.shape
    hk = args.kv_heads if args.kv_heads is not None else h
    d = args.head_dim
    if min(b, h, hk, m, n) <= 0 or h % hk or m % 128 or n % 128:
        parser.error("positive dimensions, HQ divisible by HKV, and M/N divisible by 128 required")
    scale = d**-0.5 if args.scale is None else args.scale
    if not -float("inf") < scale < float("inf"):
        parser.error("finite scale required")

    torch.npu.set_device(0)
    torch.manual_seed(2026)
    q = torch.randn(b, h, m, d, device="npu", dtype=torch.bfloat16)
    k, v = [torch.randn(b, hk, n, d, device="npu", dtype=q.dtype) for _ in range(2)]

    meta = dict(
        impl=args.impl,
        shape=[b, h, hk, m, n, d],
        scale=scale,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        samples=30,
        warmup=10,
        input_layout="BNSD",
    )

    if args.impl == "tilelang":
        config.flex_attention.implementation = "tilelang"

        def attention(q, k, v):
            return flex_attention(q, k, v, scale=scale, enable_gqa=(hk != h))

        operation = torch.compile(
            attention,
            fullgraph=True,
            dynamic=False,
            options={"npu_backend": "tilelang"},
        )
        meta["scope"] = "compiled_flex_attention_user_api"
        meta["backend"] = "npu_backend=tilelang"
    elif args.impl == "flash":
        def operation(q, k, v):
            return torch_npu.npu_fusion_attention(
                q, k, v, h, "BNSD", scale=scale, keep_prob=1.0, sparse_mode=0
            )[0]
        meta["scope"] = "eager_operator_with_allocation"
        meta["callable"] = "torch_npu.npu_fusion_attention"
        meta["call_args"] = dict(head_num=h, input_layout="BNSD", scale=scale,
                                 keep_prob=1.0, sparse_mode=0)
    else:
        def operation(q, k, v):
            return torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=h, num_key_value_heads=hk,
                scale=scale, **FIA_KWARGS
            )[0]
        meta["scope"] = "eager_operator_with_allocation"
        meta["callable"] = FIA_CALLABLE
        meta["call_args"] = dict(num_heads=h, num_key_value_heads=hk, scale=scale, **FIA_KWARGS)

    with torch.no_grad():
        actual = operation(q, k, v).cpu()
        expected = reference(q, k, v, scale)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
        result = dict(
            meta,
            correct=True,
            reference_coverage="full_output_chunked_rows",
            max_abs_error=(actual.float() - expected.float()).abs().max().item(),
            finite=bool(torch.isfinite(actual).all()),
        )
        if not args.check_only:
            for _ in range(10):
                operation(q, k, v)
            torch.npu.synchronize()
            times, wall = [], []
            for _ in range(30):
                start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
                begin = time.perf_counter()
                start.record()
                operation(q, k, v)
                end.record()
                end.synchronize()
                wall.append((time.perf_counter() - begin) * 1000)
                times.append(start.elapsed_time(end))
            ms = statistics.median(times)
            result.update(
                median_ms=ms,
                min_ms=min(times),
                max_ms=max(times),
                samples_ms=times,
                wall_median_ms=statistics.median(wall),
                tflops=4 * b * h * m * n * d / (ms * 1e9),
            )
            checked = operation(q, k, v).cpu()
            torch.testing.assert_close(checked, expected, atol=0.03, rtol=0.03)
            result["post_correct"] = True
            result["post_max_abs_error"] = (checked.float() - expected.float()).abs().max().item()
            result["post_finite"] = bool(torch.isfinite(checked).all())
        print(json.dumps(result))
        return result


if __name__ == "__main__":
    main()
