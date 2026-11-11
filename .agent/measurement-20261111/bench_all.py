# ruff: noqa: E402
"""One measurement task: 27-shape x 4-backend x 2-round protocol.

Run exactly one (shape, impl, round) per fresh process. Emits one JSON object on
stdout. Protocol: full chunked FP32 oracle, explicit finite check, atol=rtol=0.03,
pre and post correctness, 10 warmup + 30 NPU-event samples.

Arms:
  tl     torch.compile(flex_attention, npu_backend="tilelang")  [overlay kernel]
  triton torch.compile(flex_attention, npu_backend="default")   [config impl triton]
  fas    torch_npu.npu_fusion_attention (FlashAttentionScore V4)
  fia    torch_npu.npu_fused_infer_attention_score (installed V5)
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
    parser.add_argument("--impl", choices=("tl", "triton", "fas", "fia"), required=True)
    parser.add_argument("--shape", type=int, nargs=6, required=True,
                        metavar=("B", "HQ", "HKV", "M", "N", "D"))
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    b, h, hk, m, n, d = args.shape
    if min(b, h, hk, m, n, d) <= 0 or h % hk or m % 128 or n % 128:
        parser.error("positive dims, HQ divisible by HKV, M/N divisible by 128 required")
    scale = d**-0.5
    torch.npu.set_device(args.device)
    torch.manual_seed(2026)
    q = torch.randn(b, h, m, d, device="npu", dtype=torch.bfloat16)
    k, v = [torch.randn(b, hk, n, d, device="npu", dtype=q.dtype) for _ in range(2)]

    meta = dict(
        impl=args.impl,
        round=args.round,
        shape=[b, h, hk, m, n, d],
        scale=scale,
        device=args.device,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        samples=30,
        warmup=10,
        input_layout="BNSD",
    )

    if args.impl in ("tl", "triton"):
        config.flex_attention.implementation = "tilelang" if args.impl == "tl" else "triton"

        def attention(q, k, v):
            return flex_attention(q, k, v, scale=scale, enable_gqa=(hk != h))

        operation = torch.compile(
            attention,
            fullgraph=True,
            dynamic=False,
            options={"npu_backend": "tilelang" if args.impl == "tl" else "default"},
        )
        meta["scope"] = "compiled_flex_attention_user_api"
        meta["backend"] = "npu_backend=" + ("tilelang" if args.impl == "tl" else "default")
        meta["flex_impl"] = config.flex_attention.implementation
    elif args.impl == "fas":
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
                scale=scale, input_layout="BNSD", softmax_lse_flag=False
            )[0]
        meta["scope"] = "eager_operator_with_allocation"
        meta["callable"] = "torch_npu.npu_fused_infer_attention_score"
        meta["call_args"] = dict(num_heads=h, num_key_value_heads=hk, scale=scale,
                                 input_layout="BNSD", softmax_lse_flag=False)

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
