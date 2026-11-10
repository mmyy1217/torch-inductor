#!/usr/bin/env python3
"""Correctness-only native TileLang acceptance matrix; each NPU case is isolated.

--list requires no torch. build_case() is shared with the device-free integration
suite, whose mocked launch does not validate the actual sparse device loop.
"""

import argparse
import importlib.util
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
CASE_NAMES = (
    "full_only",
    "partial_only",
    "mixed",
    "zero_count_rows",
    "partial_zero_rows",
    "noncontiguous_kv",
    "block256",
    "scaled_softcap",
    "batch_head",
    "signed_remainder",
    "signed_floor_divide",
)


def build_case(name, device="cpu"):
    import torch

    if device != "cpu":
        import torch_npu._inductor  # noqa: F401 - Apply NPU patches before binding callbacks.
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    if name not in CASE_NAMES:
        raise ValueError(name)
    b, h, m, n = 1, 2, 256, 512
    block_size, scale, score = 128, 128**-0.5, None
    if name == "full_only":

        def mask_fn(b, h, m, n):
            return n >= 128

        def score_fn(s, b, h, m, n):
            return s + (n - m) * 0.01

        score = score_fn
    elif name == "partial_only":

        def mask_fn(b, h, m, n):
            return (m % 2 == 0) & (n % 2 == 0)
    elif name == "mixed":

        def mask_fn(b, h, m, n):
            return m >= n
    elif name == "zero_count_rows":

        def mask_fn(b, h, m, n):
            return (m >= 128) & (n >= 128)
    elif name == "partial_zero_rows":

        def mask_fn(b, h, m, n):
            return (m % 2 == 0) & (n % 2 == 0) & (n >= 128)
    elif name == "noncontiguous_kv":

        def mask_fn(b, h, m, n):
            return (n < 128) | (n >= 384)
    elif name == "block256":
        m, block_size = 512, 256

        def mask_fn(b, h, m, n):
            return m >= n
    elif name in ("signed_remainder", "signed_floor_divide"):

        def mask_fn(b, h, m, n):
            return ((m - n) % 7 < 3) & ((m - n) % -11 > -8)

        if name == "signed_remainder":

            def score_fn(s, b, h, m, n):
                return s + ((m - n) % 7) * 0.2 + ((m - n) % -11) * 0.1
        else:

            def score_fn(s, b, h, m, n):
                return s + ((m - n) // 7) * 0.03 + ((m - n) // -11) * 0.02

        score = score_fn
    elif name == "scaled_softcap":
        scale = 0.25

        def score_fn(s, b, h, m, n):
            return torch.tanh(s / 2) * 2

        score = score_fn

        def mask_fn(b, h, m, n):
            return (m >= n) & (m - n < 192)
    else:
        b, h = 2, 3

        def mask_fn(b, h, m, n):
            return (b % 2 == h % 2) & (m >= n)

        def score_fn(s, b, h, m, n):
            return s + (b + h) * 0.01 * (n - m)

        score = score_fn

    generator = torch.Generator().manual_seed(2026)
    inputs = tuple(
        torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
        for shape in ((b, h, m, 128), (b, h, n, 128), (b, h, n, 128))
    )
    mask = create_block_mask(
        mask_fn,
        B=b if name == "batch_head" else None,
        H=h if name == "batch_head" else None,
        Q_LEN=m,
        KV_LEN=n,
        BLOCK_SIZE=block_size,
        device="cpu",
    ).to(device)

    def attention(q, k, v):
        return flex_attention(q, k, v, score_mod=score, block_mask=mask, scale=scale)

    return dict(
        name=name,
        inputs=inputs,
        attention=attention,
        mask=mask,
        mask_mod=mask_fn,
        score_mod=score,
        scale=scale,
        block_size=block_size,
    )


def build_cases(device="cpu"):
    return [build_case(name, device) for name in CASE_NAMES]


def reference(case):
    """Independent FP32 dense attention, including zero output for empty rows."""
    import torch

    q, k, v = (x.detach().cpu().float() for x in case["inputs"])
    b = torch.arange(q.shape[0])[:, None, None, None]
    h = torch.arange(q.shape[1])[None, :, None, None]
    m = torch.arange(q.shape[2])[None, None, :, None]
    n = torch.arange(k.shape[2])[None, None, None, :]
    scores = (q @ k.transpose(-2, -1)) * case["scale"]
    if case["score_mod"] is not None:
        scores = case["score_mod"](scores, b, h, m, n)
    visible = case["mask_mod"](b, h, m, n)
    scores = torch.where(visible, scores, -torch.inf)
    probs = torch.softmax(scores, -1).nan_to_num(0)
    return probs @ v


def metadata(case):
    mask = case["mask"]
    return {
        name: None
        if getattr(mask, name) is None
        else getattr(mask, name).detach().cpu()
        for name in (
            "kv_num_blocks",
            "kv_indices",
            "full_kv_num_blocks",
            "full_kv_indices",
        )
    }


def run_worker(name, folder, impl):
    import torch
    import torch_npu
    from torch._inductor.utils import run_and_get_code

    module_spec = importlib.util.spec_from_file_location(
        "matrix_implementations", HERE / "implementations.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    torch.npu.set_device(0)
    case = build_case(name, "npu")
    torch.save(tuple(x.cpu() for x in case["inputs"]), folder / "inputs.pt")
    tensors = metadata(case)
    torch.save(tensors, folder / "metadata.pt")
    (folder / "metadata.json").write_text(
        json.dumps(
            {k: None if v is None else v.tolist() for k, v in tensors.items()}, indent=2
        )
    )
    (folder / "case.json").write_text(
        json.dumps(
            dict(
                name=name,
                implementation=impl,
                shapes=[list(x.shape) for x in case["inputs"]],
                scale=case["scale"],
                block_size=case["block_size"],
                dtype="bfloat16",
            ),
            indent=2,
        )
    )
    (folder / "environment.json").write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                torch_path=torch.__file__,
                torch_npu_path=torch_npu.__file__,
                device=torch.npu.get_device_name(0),
                python=sys.version,
            ),
            indent=2,
        )
    )
    expected = reference(case)
    torch.save(expected, folder / "reference.pt")
    with torch.no_grad():
        compiled = module.compile_attention(case["attention"], impl)
        actual, codes = run_and_get_code(compiled, *case["inputs"])
        torch.npu.synchronize()
    if impl == "tilelang":
        from unittest.mock import patch
        import tilelang
        from torch_npu._inductor.runtime import tilelang as runtime

        with (
            torch.no_grad(),
            patch.object(
                tilelang,
                "compile",
                side_effect=AssertionError("compile during repeat launch"),
            ),
            patch.object(
                runtime,
                "compile_tilelang",
                side_effect=AssertionError("registration during repeat launch"),
            ),
        ):
            repeated = compiled(*case["inputs"])
            torch.npu.synchronize()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    code = "\n\n".join(codes)
    (folder / "output_code.py").write_text(code)
    actual = actual.detach().cpu().float()
    torch.save(actual, folder / "actual.pt")
    result = dict(
        case=name,
        status="FAIL",
        max_abs_error=float((actual - expected).abs().max()),
        finite=bool(torch.isfinite(actual).all()),
        atol=0.03,
        rtol=0.03,
    )
    (folder / "correctness.json").write_text(json.dumps(result, indent=2))
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    if impl == "tilelang":
        if (
            "compile_tilelang(" not in code
            or ".launch(" not in code
            or "out=buf" not in code
            or "flex_tilelang.dense" in code
        ):
            raise AssertionError("missing native TileLang wrapper/output ABI")
        for ir_name in ("ir_pre_fusion.txt", "ir_post_fusion.txt"):
            ir_files = list((folder / "debug").rglob(ir_name))
            if not ir_files or not any(
                "TileLangTemplateBuffer" in p.read_text() for p in ir_files
            ):
                raise AssertionError(
                    "missing native TileLangTemplateBuffer scheduling evidence: "
                    + ir_name
                )
        for artifact in (
            "runtime.json",
            "rendered_kernel.py",
            "prim_func.py",
            "lowered_device_ir.py",
            "device_code.txt",
            "compiler.json",
        ):
            if not list((folder / "tilelang").rglob(artifact)):
                raise AssertionError("missing TileLang execution artifact: " + artifact)
        for compiler_path in (folder / "tilelang").rglob("compiler.json"):
            if (
                json.loads(compiler_path.read_text()).get("compilation_phase")
                != "wrapper_initialization"
            ):
                raise AssertionError(
                    "kernel compilation was not registered during wrapper initialization"
                )
        result["repeat_launch_without_compilation"] = True
    result["status"] = "PASS"
    (folder / "correctness.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=("all", *CASE_NAMES), default="all")
    parser.add_argument("--impl", choices=("tilelang", "triton"), default="tilelang")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.list:
        print("\n".join(CASE_NAMES))
        return
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("set ASCEND_RT_VISIBLE_DEVICES to your allocated device")
    if args.output is None:
        parser.error("--output required")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    folder = args.output.resolve()
    if args.worker:
        try:
            result = run_worker(args.case, folder, args.impl)
        except Exception:
            result = dict(case=args.case, status="FAIL", error=traceback.format_exc())
            (folder / "result.json").write_text(json.dumps(result, indent=2))
            print(result["error"], file=sys.stderr)
            raise SystemExit(1) from None
        (folder / "result.json").write_text(json.dumps(result, indent=2))
        return
    if folder.exists() and any(folder.iterdir()):
        parser.error("output must be empty; preserve prior evidence")
    folder.mkdir(parents=True, exist_ok=True)
    selected = CASE_NAMES if args.case == "all" else (args.case,)
    results = []
    for name in selected:
        dest = folder / name
        dest.mkdir()
        env = dict(
            os.environ,
            TORCH_COMPILE_DEBUG="1",
            TORCH_COMPILE_DEBUG_DIR=str(dest / "debug"),
            TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
            TORCHINDUCTOR_CACHE_DIR=str(dest / "inductor_cache"),
            TRITON_CACHE_DIR=str(dest / "triton_cache"),
            TILELANG_CACHE_DIR=str(dest / "tilelang_cache"),
            FLEX_TILELANG_ARTIFACT_DIR=str(dest / "tilelang"),
        )
        command = [
            sys.executable,
            "-P",
            str(Path(__file__).resolve()),
            "--worker",
            "--case",
            name,
            "--impl",
            args.impl,
            "--output",
            str(dest),
        ]
        (dest / "command.json").write_text(json.dumps(command, indent=2))
        with (dest / "run.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = -1
        result_path = dest / "result.json"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else dict(
                case=name,
                status="FAIL",
                error="worker timed out or exited before result",
                returncode=code,
            )
        )
        if code != 0 and result.get("status") == "PASS":
            result.update(
                status="FAIL",
                error="worker returned nonzero after writing result",
                returncode=code,
            )
        results.append(result)
        print(name + ": " + result["status"], flush=True)
    (folder / "summary.json").write_text(json.dumps(results, indent=2))
    raise SystemExit(0 if all(r["status"] == "PASS" for r in results) else 1)


if __name__ == "__main__":
    main()
