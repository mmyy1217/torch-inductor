#!/usr/bin/env python3
"""批量验收入口：选用例 → 子进程执行 → 保存正确性、IR 和可选计时结果。

学习单次调用先看 first_dense.py；此文件承担批量测试，不是最小示例。
阅读顺序：main() → run_worker() → module.build() → compile_attention()。
"""

import ast
import importlib.util
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # Explicit task-local imports, also under python -P.


def case_paths():
    """扫描 DONE/TODO，建立用例名到 Python 文件的映射。"""
    paths = {}
    for group in ("DONE", "TODO"):
        for path in sorted((HERE / "cases" / group).glob("*.py")):
            if path.stem in paths:
                raise ValueError(f"Duplicate case name: {path.stem}")
            paths[path.stem] = path
    return paths


def cases():
    """合并默认 shape 与各文件 CONFIG；只解析文本，--list 不加载 NPU。"""
    result = []
    for path in sorted(case_paths().values()):
        tree = ast.parse(path.read_text())
        overrides = next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets)
        )
        c = dict(
            b=1,
            bkv=1,
            hq=4,
            hkv=4,
            sq=256,
            sk=256,
            d=128,
            dv=128,
            dtype="bfloat16",
            window=128,
        )
        c.update(overrides)
        c["id"] = path.stem
        c["tilelang_status"] = path.parent.name
        result.append(c)
    return result


def load_case(name):
    """例如 dense → cases/DONE/dense.py，返回包含 build() 的模块。"""
    spec = importlib.util.spec_from_file_location(
        "flex_case_" + name, case_paths()[name]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_worker(a, c, folder):
    """执行一个用例。a 是命令行参数，c 是 shape 配置，folder 是产物目录。"""
    # 1. 初始化设备并记录环境，便于追查不同机器的结果差异。
    import torch
    import torch_npu
    import triton
    from torch._inductor.utils import run_and_get_code

    torch.npu.set_device(0)
    torch.manual_seed(2026)
    env = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        torch_path=torch.__file__,
        torch_npu_path=torch_npu.__file__,
        triton_path=triton.__file__,
        device=torch.npu.get_device_name(0),
        python=sys.version,
        packages={
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        variables={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("ASCEND", "CANN", "TRITON", "TORCH", "OMP_"))
        },
    )
    (folder / "environment.json").write_text(json.dumps(env, indent=2))
    dtype = getattr(torch, c["dtype"])

    def tensor(b, h, s, d):
        x = torch.randn(
            (b, s, h, d) if c.get("noncontiguous") else (b, h, s, d),
            dtype=dtype,
            device="cpu",
        )
        if c.get("noncontiguous"):
            x = x.transpose(1, 2)
        return x.to("npu").requires_grad_(c.get("backward", False))

    # 2. 创建 Q/K/V；形状依次为 [batch, heads, sequence, head_dim]。
    q = tensor(c["b"], c["hq"], c["sq"], c["d"])
    k = tensor(c["bkv"], c["hkv"], c["sk"], c["d"])
    v = tensor(c["bkv"], c["hkv"], c["sk"], c["dv"])
    (folder / "inputs.json").write_text(
        json.dumps(
            [
                dict(
                    shape=list(x.shape),
                    stride=list(x.stride()),
                    sha256=hashlib.sha256(
                        x.detach()
                        .cpu()
                        .contiguous()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    ).hexdigest(),
                )
                for x in (q, k, v)
            ],
            indent=2,
        )
    )
    # 3. 这里接上 cases/DONE/dense.py；build 返回函数，此时还没有执行 attention。
    module = load_case(c["id"])
    torch.npu.synchronize()
    start = time.perf_counter()
    fn, _, _ = module.build(c, "npu")  # fn 就是用例中定义的 attention(q, k, v)。
    torch.npu.synchronize()
    mask_ms = (time.perf_counter() - start) * 1000
    from implementations import compile_attention

    # 4. 两种实现都用 backend="inductor"；辅助函数只切换内部模板配置。
    compiled = compile_attention(fn, a.impl)
    is_backward = c.get("backward", False)
    # Fixed upstream gradient; use exactly the same values for reference and timing.
    grad = torch.randn(c["b"], c["hq"], c["sq"], c["dv"], dtype=dtype).to("npu")

    def execute():
        if is_backward:
            out = compiled(q, k, v)
            grads = torch.autograd.grad(out, (q, k, v), grad_outputs=grad)
            return out, grads
        with torch.no_grad():
            return compiled(q, k, v)

    # 5. 首次调用才触发编译并运行；同时拿到 Inductor 生成的 wrapper 源码。
    start = time.perf_counter()
    actual, codes = run_and_get_code(execute)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - start) * 1000
    code = "\n".join(codes)
    (folder / "output_code.py").write_text(code)
    if a.impl == "triton" and (
        "async_compile.triton(" not in code or "@triton.jit" not in code
    ):
        raise RuntimeError("No generated Triton JIT code; inspect output_code.py")

    if a.impl == "tilelang":
        # Inductor may use Triton pointwise kernels to construct BlockMask metadata.
        # The attention itself is checked through its native call, IR and artifacts.
        if (
            "compile_tilelang(" not in code
            or ".launch(" not in code
            or "flex_tilelang.dense" in code
        ):
            raise RuntimeError(
                "Expected rendered TileLang compilation and launch; inspect output_code.py"
            )
        for name in (
            "rendered_kernel.py",
            "prim_func.py",
            "lowered_device_ir.py",
            "device_code.txt",
            "compiler.json",
            "runtime.json",
        ):
            if not list((folder / "tilelang").rglob(name)):
                raise RuntimeError(f"Missing TileLang execution artifact: {name}")

    # 6. 第二次 build 只取 CPU 参考所需的 mask/score 回调，不执行返回的 attention。
    # 用独立 FP32 数学公式分块核对输出，避免创建完整 S×S 矩阵。
    _, mask, score = module.build(c, "cpu")
    qc, kc, vc = [
        x.detach().cpu().float().requires_grad_(is_backward) for x in (q, k, v)
    ]
    out_actual = actual[0] if is_backward or c.get("aux") else actual
    ac = out_actual.detach().cpu().float()
    valid_pairs, max_error = 0, 0.0
    ref_batches = []
    for b in range(c["b"]):
        ref_heads = []
        for h in range(c["hq"]):
            ref_chunks = []
            for lo in range(0, c["sq"], 128):
                hi = min(lo + 128, c["sq"])
                m, n = torch.arange(lo, hi)[:, None], torch.arange(c["sk"])[None, :]
                visible = (
                    mask(b, h, m, n)
                    if mask
                    else torch.ones(hi - lo, c["sk"], dtype=torch.bool)
                ).expand(hi - lo, c["sk"])
                valid_pairs += int(visible.sum())
                kh = h // (c["hq"] // c["hkv"])
                kb = b if c["bkv"] != 1 else 0
                s = qc[b, h, lo:hi] @ kc[kb, kh].T / c["d"] ** 0.5
                if score:
                    s = score(s, b, h, m, n)
                s = s.masked_fill(~visible, -float("inf"))
                ref = torch.nan_to_num(torch.softmax(s, -1), nan=0.0) @ vc[kb, kh]
                torch.testing.assert_close(
                    ac[b, h, lo:hi], ref, atol=a.atol, rtol=a.rtol
                )
                max_error = max(
                    max_error, float((ac[b, h, lo:hi] - ref).abs().max().detach())
                )
                if c.get("aux"):
                    expected = (
                        torch.logsumexp(s, -1)
                        if c["aux"] == "lse"
                        else s.max(-1).values
                    )
                    got = (
                        getattr(actual[1], c["aux"])[b, h, lo:hi].detach().cpu().float()
                    )
                    torch.testing.assert_close(got, expected, atol=a.atol, rtol=a.rtol)
                if is_backward:
                    ref_chunks.append(ref)
            if is_backward:
                ref_heads.append(torch.cat(ref_chunks))
        if is_backward:
            ref_batches.append(torch.stack(ref_heads))
    grad_errors = []
    if is_backward:
        ref_out = torch.stack(ref_batches)
        expected_grads = torch.autograd.grad(
            ref_out, (qc, kc, vc), grad_outputs=grad.cpu().float()
        )
        for got, expected in zip(actual[1], expected_grads):
            got = got.detach().cpu().float()
            torch.testing.assert_close(got, expected, atol=a.atol, rtol=a.rtol)
            grad_errors.append(float((got - expected).abs().max()))
    (folder / "correctness.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                max_abs_error=max_error,
                gradient_max_abs_errors=grad_errors,
                atol=a.atol,
                rtol=a.rtol,
            ),
            indent=2,
        )
    )
    # 7. 检查 FX、融合前后 IR 等产物，确认确实经过目标模板。
    artifacts = {
        name: [str(p.relative_to(folder)) for p in (folder / "debug").rglob(name)]
        for name in (
            "fx_graph_readable.py",
            "fx_graph_transformed.py",
            "ir_pre_fusion.txt",
            "ir_post_fusion.txt",
            "output_code.py",
        )
    }
    (folder / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
    missing = [name for name, paths in artifacts.items() if not paths]
    if missing:
        raise RuntimeError(
            f"Missing required Inductor debug artifacts: {missing}; inspect run.log"
        )
    if a.impl == "tilelang":
        for name in ("ir_pre_fusion.txt", "ir_post_fusion.txt"):
            text = "\n".join((folder / p).read_text() for p in artifacts[name])
            if (
                "SchedulerNode(TileLangTemplateBuffer)" not in text
                or "FallbackKernel" in text
            ):
                raise RuntimeError(f"Native template evidence missing in {name}")
    # 8. check 到此已完成正确性验证；bench 才继续预热和计时。
    samples = []
    if a.mode == "bench":
        for _ in range(a.warmup):
            execute()
        torch.npu.synchronize()
        for _ in range(a.repeats):
            begin, end = (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(a.inner):
                execute()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / a.inner)
    median = statistics.median(samples) if samples else None
    # Do not use forward FLOPs to label forward+backward utilization.
    effective = None if is_backward else 2 * valid_pairs * (c["d"] + c["dv"])
    tflops = effective / (median * 1e-6) / 1e12 if median and effective else None
    result = dict(
        status="PASS",
        case=c,
        impl=a.impl,
        mode=a.mode,
        median_us=median,
        samples_us=samples,
        effective_flops=effective,
        effective_tflops=tflops,
        utilization_pct=100 * tflops / a.peak_tflops
        if tflops and a.peak_tflops
        else None,
        peak_tflops=a.peak_tflops,
        peak_source=a.peak_source,
        mask_build_ms=mask_ms,
        compile_and_first_run_ms=compile_ms,
        max_abs_error=max_error,
        timed_region="forward+backward" if is_backward else "forward",
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
    )
    (folder / "benchmark.json").write_text(json.dumps(result, indent=2))
    return result


def main(default_case=None):
    """父进程选用例并启动 worker；--worker 子进程只运行一个用例。"""
    # A. 解析参数、合并 shape、过滤尚未支持的 TileLang 用例。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--impl", choices=["triton", "tilelang"], default="triton")
    p.add_argument("--mode", choices=["check", "bench"], default="check")
    p.add_argument(
        "--seq", type=int, help="Override both Q/KV lengths; decode Q remains 1"
    )
    p.add_argument("--q-seq", type=int, help="Override Q length independently")
    p.add_argument("--kv-seq", type=int, help="Override KV length independently")
    p.add_argument("--batch", type=int, help="Override matching Q/KV batch size")
    p.add_argument("--heads", type=int, help="Override matching Q/KV head count")
    p.add_argument(
        "--case",
        choices=["all"] + [c["id"] for c in cases()],
        default=default_case or "causal",
    )
    p.add_argument("--output", default="results/" + time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--peak-tflops", type=float)
    p.add_argument("--peak-source", default="")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--atol", type=float, default=0.03)
    p.add_argument("--rtol", type=float, default=0.03)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    selected = [c for c in cases() if a.case in ("all", c["id"])]
    if a.seq is not None:
        if a.seq <= 0:
            p.error("--seq must be positive")
        for c in selected:
            c.update(sq=1 if c["id"] == "decode" else a.seq, sk=a.seq)
    for arg, keys in (
        (a.q_seq, ("sq",)),
        (a.kv_seq, ("sk",)),
        (a.batch, ("b", "bkv")),
        (a.heads, ("hq", "hkv")),
    ):
        if arg is not None:
            if arg <= 0:
                p.error("shape overrides must be positive")
            for c in selected:
                c.update({key: arg for key in keys})
    if a.impl == "tilelang":
        supported = {c["id"] for c in cases() if c["tilelang_status"] == "DONE"}
        if a.case == "all":
            selected = [c for c in selected if c["id"] in supported]
        elif a.case not in supported and not a.list:
            p.error(
                "TileLang v2 unsupported case; use --case all for the supported forward subset"
            )
        if not a.list and any(c["sq"] % 128 or c["sk"] % 128 for c in selected):
            p.error("TileLang v2 requires sequence lengths divisible by 128")
    if a.list:
        print(json.dumps(selected, indent=2))
        return 0
    if min(a.warmup, a.repeats, a.inner, a.timeout) <= 0:
        p.error("warmup/repeats/inner/timeout must be positive")
    if a.peak_tflops is not None and (a.peak_tflops <= 0 or not a.peak_source):
        p.error(
            "positive --peak-tflops requires --peak-source (hardware/spec/precision)"
        )
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # B. 子进程入口：实际计算在 run_worker()，结果写入 result.json。
    if a.worker:
        (out / "case.json").write_text(json.dumps(selected[0], indent=2))
        try:
            result = run_worker(a, selected[0], out)
        except Exception:
            result = dict(status="FAIL", case=selected[0], error=traceback.format_exc())
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return int(result["status"] != "PASS")
    if any(out.iterdir()):
        p.error("output directory must be empty; preserve earlier measurements")
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                cases=selected,
                arguments=vars(a),
                source_sha256={
                    str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in HERE.rglob("*.py")
                },
            ),
            indent=2,
        )
    )
    # C. 父进程为每个用例建独立目录和进程，隔离缓存与编译环境变量。
    results = []
    for c in selected:
        folder = out / a.impl / c["id"]
        folder.mkdir(parents=True)
        env = os.environ.copy()
        env["TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_IMPL"] = a.impl
        env["TORCH_COMPILE_DEBUG"] = "1"
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        env["TORCH_COMPILE_DEBUG_DIR"] = str(folder / "debug")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(folder / "inductor_cache")
        env["TRITON_CACHE_DIR"] = str(folder / "triton_cache")
        env["TILELANG_CACHE_DIR"] = str(folder / "tilelang_cache")
        env["FLEX_TILELANG_ARTIFACT_DIR"] = str(folder / "tilelang")
        # 再次启动本文件，带 --worker 后走上面的 B 分支，不会递归启动子进程。
        command = [
            sys.executable,
            "-P",
            str(Path(__file__).resolve()),
            "--worker",
            "--case",
            c["id"],
            "--output",
            str(folder),
            "--impl",
            a.impl,
            "--mode",
            a.mode,
            "--warmup",
            str(a.warmup),
            "--repeats",
            str(a.repeats),
            "--inner",
            str(a.inner),
            "--atol",
            str(a.atol),
            "--rtol",
            str(a.rtol),
        ]
        if a.seq is not None:
            command += ["--seq", str(a.seq)]
        for flag in ("q_seq", "kv_seq", "batch", "heads"):
            if getattr(a, flag) is not None:
                command += ["--" + flag.replace("_", "-"), str(getattr(a, flag))]
        if a.peak_tflops:
            command += [
                "--peak-tflops",
                str(a.peak_tflops),
                "--peak-source",
                a.peak_source,
            ]
        with (folder / "run.log").open("w") as log:
            try:
                proc = subprocess.Popen(
                    command,
                    cwd=folder,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                proc.wait(timeout=a.timeout)
                status = f"EXIT_{proc.returncode}"
            except subprocess.TimeoutExpired:
                os.killpg(
                    proc.pid, signal.SIGKILL
                )  # only this case's own process group
                proc.wait()
                status = "TIMEOUT"
        path = folder / "result.json"
        result = (
            json.loads(path.read_text())
            if path.exists()
            else dict(status=status, case=c)
        )
        if status != "EXIT_0" and result.get("status") == "PASS":
            result["status"] = status
        results.append(result)
        print(c["id"], result["status"], flush=True)
        (out / "summary.json").write_text(json.dumps(results, indent=2))
    # D. 汇总所有用例；任一失败则以非零状态退出。
    with (out / "summary.csv").open("w", newline="") as f:
        fields = [
            "case",
            "status",
            "median_us",
            "effective_tflops",
            "utilization_pct",
            "max_abs_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    key: r["case"]["id"] if key == "case" else r.get(key)
                    for key in fields
                }
            )
    return int(any(r["status"] != "PASS" for r in results))


if __name__ == "__main__":
    sys.exit(main())
