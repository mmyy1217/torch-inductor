"""Measure an already validated dense Inductor wrapper with NPU Graph replay."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder", type=Path, help="The runner's <output>/<impl>/dense directory"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--peak-source")
    args = parser.parse_args()
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("set ASCEND_RT_VISIBLE_DEVICES to your allocated device")
    if (args.peak_tflops is None) != (args.peak_source is None):
        parser.error("provide both --peak-tflops and --peak-source, or neither")
    if args.peak_tflops is not None and (
        not math.isfinite(args.peak_tflops)
        or args.peak_tflops <= 0
        or not args.peak_source.strip()
    ):
        parser.error("peak must be finite and positive, with a nonempty source")
    folder = args.folder.resolve()
    case = json.loads((folder / "case.json").read_text())
    checked = json.loads((folder / "result.json").read_text())
    if case["id"] != "dense" or checked["status"] != "PASS":
        parser.error("requires a successfully validated dense case from run.py")
    output = (args.output or folder / "graph_benchmark.json").resolve()
    if output.exists():
        parser.error("output already exists; preserve previous samples")
    os.environ.pop("FLEX_TILELANG_ARTIFACT_DIR", None)
    for key, name in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor_cache"),
        ("TRITON_CACHE_DIR", "triton_cache"),
        ("TILELANG_CACHE_DIR", "tilelang_cache"),
    ):
        os.environ[key] = str(folder / name)
    import torch
    import torch_npu._inductor  # noqa: F401 - Load NPU patches before the wrapper.

    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location(
        "measured_wrapper", folder / "output_code.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(2026)
    inputs = module.get_args()
    expected_shapes = [
        [case["b"], case["hq"], case["sq"], case["d"]],
        [case["bkv"], case["hkv"], case["sk"], case["d"]],
        [case["bkv"], case["hkv"], case["sk"], case["dv"]],
    ]
    if len(inputs) != 3 or [list(x.shape) for x in inputs] != expected_shapes:
        parser.error("wrapper input signature does not match the dense case")
    with torch.no_grad():
        for _ in range(10):
            expected = module.call(list(inputs))[0]
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(20):
                actual = module.call(list(inputs))[0]
        for _ in range(10):
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected)
        inputs[0].add_(0.125)
        expected = module.call(list(inputs))[0]
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected)
        samples = []
        for _ in range(20):
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / 200)
    median = statistics.median(samples)
    flops = (
        2 * case["b"] * case["hq"] * case["sq"] * case["sk"] * (case["d"] + case["dv"])
    )
    tflops = flops / median / 1e6
    result = dict(
        mode="NPU graph replay of 20 compiled forwards",
        implementation=checked["impl"],
        case=case,
        median_us=median,
        samples_us=samples,
        effective_flops=flops,
        effective_tflops=tflops,
        operator_mfu_pct=None
        if args.peak_tflops is None
        else tflops / args.peak_tflops * 100,
        peak_tflops=args.peak_tflops,
        peak_source=args.peak_source,
        correctness="matches uncaptured output, including changed Q",
        includes="attention and wrapper device kernels; excludes Python launch per forward",
        input_sha256_after_q_update=[
            hashlib.sha256(
                x.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for x in inputs
        ],
        wrapper_sha256=hashlib.sha256(
            (folder / "output_code.py").read_bytes()
        ).hexdigest(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
