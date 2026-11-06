"""
Benchmark adaptive LayerNorm using torch_npu profiler.

Run:
    DEMO_BACKEND=tilelang \
        python test/_inductor/tilelang/benchmark_tilelang_layernorm.py
    DEMO_BACKEND=triton \
        python test/_inductor/tilelang/benchmark_tilelang_layernorm.py
"""
import os
import shutil
import time

import torch
import torch.nn.functional as F
import torch_npu


backend = os.environ.get("DEMO_BACKEND", "tilelang").lower()
if backend == "tilelang":
    compile_options = {"npu_backend": "tilelang"}
elif backend == "triton":
    compile_options = None
else:
    raise ValueError(f"unknown DEMO_BACKEND={backend!r}")

os.environ.setdefault("TILELANG_CACHE_DIR", f"/tmp/tilelang_layernorm_{backend}")

from tilelang.profiler.bench import do_bench_npu


DEVICE = "npu"
rtol, atol = 1e-2, 1e-2
_PROF_DIR = os.path.join(os.environ["TILELANG_CACHE_DIR"], "bench_torch_tmp")


def do_bench_torch(fn, warmup=5, rep=30, prof_dir=_PROF_DIR, timeout=120, keep_res=False):
    """
    Benchmark a callable using torch_npu.profiler.profile.

    All NPU kernel durations produced by one fn() call are summed, then the
    active iterations are averaged. Returns average latency in milliseconds.
    """
    import pandas as pd

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    shutil.rmtree(prof_dir, ignore_errors=True)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ):
        for _ in range(warmup + rep):
            fn()
        torch.npu.synchronize()

    deadline = time.time() + timeout
    kernel_details_file = None
    while time.time() < deadline:
        for root, _, files in os.walk(prof_dir):
            for name in files:
                if name == "kernel_details.csv":
                    kernel_details_file = os.path.join(root, name)
                    break
        if kernel_details_file:
            break
        time.sleep(1)

    if kernel_details_file is None:
        print(f"[do_bench_torch] WARNING: kernel_details.csv not found in {prof_dir} after {timeout}s")
        shutil.rmtree(prof_dir, ignore_errors=True)
        return float("inf")

    df = pd.read_csv(kernel_details_file)
    total_calls = warmup + rep
    kernels_per_call = max(1, len(df) // total_calls)
    active_start = warmup * kernels_per_call
    active_df = df.iloc[active_start: active_start + rep * kernels_per_call]
    avg_us = active_df["Duration(us)"].sum() / rep

    if not keep_res:
        shutil.rmtree(prof_dir, ignore_errors=True)
    return avg_us / 1e3


def do_bench_event(fn, warmup=20, rep=100):
    fn()
    torch.npu.synchronize()

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(rep):
        fn()
    end_event.record()
    torch.npu.synchronize()
    return start_event.elapsed_time(end_event) / rep


def ada_layer_norm_ref(x, scale, shift, eps):
    n = x.shape[-1]
    y = F.layer_norm(
        x.float(),
        (n,),
        weight=None,
        bias=None,
        eps=eps,
    )
    y = y * scale.float() + shift.float()
    return y.to(x.dtype)


def bench_eager_per_op(x, scale, shift, eps):
    n = x.shape[-1]
    x_float = x.float()
    y = F.layer_norm(x_float, (n,), weight=None, bias=None, eps=eps)
    scaled = y * scale.float()

    ops = [
        ("cast x.float", lambda: x.float()),
        ("layer_norm fp32", lambda: F.layer_norm(x_float, (n,), weight=None, bias=None, eps=eps)),
        ("cast scale.float", lambda: scale.float()),
        ("mul scale", lambda: y.mul(scale.float())),
        ("cast shift.float", lambda: shift.float()),
        ("add shift", lambda: scaled.add(shift.float())),
        ("cast output", lambda: scaled.add(shift.float()).to(x.dtype)),
    ]

    print("\n--- Eager per-op latency (torch_npu profiler) ---")
    total = 0.0
    for name, fn in ops:
        latency = do_bench_torch(fn)
        total += latency
        print(f"  {name:<25} {latency:.4f} ms")
    print(f"  {'total (sum of parts)':<25} {total:.4f} ms")


def main():
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    torch._dynamo.reset()

    m = int(os.environ.get("DEMO_M", "1024"))
    n = int(os.environ.get("DEMO_N", "10240"))
    eps = float(os.environ.get("DEMO_EPS", "1e-5"))
    dtype_name = os.environ.get("DEMO_DTYPE", "float32")
    dtype = getattr(torch, dtype_name)

    x = torch.randn(m, n, device=DEVICE, dtype=dtype)
    scale = torch.randn(m, n, device=DEVICE, dtype=dtype)
    shift = torch.randn(m, n, device=DEVICE, dtype=dtype)

    compiled = torch.compile(
        ada_layer_norm_ref,
        backend="inductor",
        options=compile_options,
    )

    ref = ada_layer_norm_ref(x, scale, shift, eps)
    out = compiled(x, scale, shift, eps)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    max_err = (out - ref).abs().max().item()
    print(
        f"Correctness PASS  backend={backend}  shape={m}x{n} "
        f"dtype={dtype_name}  max_err={max_err:.6f}"
    )

    if os.environ.get("DEMO_SKIP_EAGER_PER_OP", "0") != "1":
        bench_eager_per_op(x, scale, shift, eps)

    if os.environ.get("DEMO_RESULT_BENCH_METHOD", "profiler").lower() == "event":
        latency_eager = do_bench_event(lambda: ada_layer_norm_ref(x, scale, shift, eps))
        latency_compiled = do_bench_event(lambda: compiled(x, scale, shift, eps))
    else:
        latency_eager = do_bench_torch(lambda: ada_layer_norm_ref(x, scale, shift, eps))
        latency_compiled = do_bench_npu([lambda: compiled(x, scale, shift, eps)])

    print("\n--- End-to-end latency ---")
    print(f"  Eager    (end-to-end): {latency_eager:.4f} ms")
    print(f"  Compiled (end-to-end): {latency_compiled:.4f} ms")
    print(f"  Speedup: {latency_eager / latency_compiled:.2f}x")
    print(
        "BENCH_RESULT "
        f"backend={backend} "
        f"case=ada_layernorm "
        f"shape={m}x{n} "
        f"dtype={dtype_name} "
        f"eager_ms={latency_eager:.6f} "
        f"compiled_ms={latency_compiled:.6f} "
        f"speedup={latency_eager / latency_compiled:.4f} "
        f"max_err={max_err:.6f}"
    )


if __name__ == "__main__":
    main()
