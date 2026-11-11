"""Prepare an isolated candidate overlay without changing the checkout or wheel."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("baseline", "compute", "persistent"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cube-cores", type=int)
    args = parser.parse_args()
    if args.variant == "persistent" and (
        args.cube_cores is None or args.cube_cores <= 0
    ):
        parser.error("persistent requires the measured positive --cube-cores")
    if args.variant != "persistent" and args.cube_cores is not None:
        parser.error("--cube-cores applies only to persistent")
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    hashes = json.loads((package / "source_hashes.json").read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("handoff source changed: " + name)
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("run under the target torch/torch_npu environment")
    installed = Path(spec.origin).resolve().parent
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source/torch_npu/_inductor"
    shutil.copytree(
        repo / "torch_npu/_inductor",
        source,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    kernel = source / "kernel/tilelang_flex_kernel.py"
    if args.variant != "baseline":
        text = (package / (args.variant + ".py")).read_text()
        if args.variant == "persistent":
            old = "NUM_BLOCKS = min(28, NUM_TILES)"
            if text.count(old) != 1:
                raise RuntimeError("unexpected persistent source")
            text = text.replace(old, f"NUM_BLOCKS = min({args.cube_cores}, NUM_TILES)")
        kernel.write_text(text)
    overlay = out / "overlay/torch_npu"
    overlay.mkdir(parents=True)
    for child in installed.iterdir():
        if child.name not in ("_inductor", "__pycache__"):
            (overlay / child.name).symlink_to(child)
    (overlay / "_inductor").symlink_to(source)
    metadata = {
        "variant": args.variant,
        "cube_cores": args.cube_cores,
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "source_repo": str(repo),
        "installed_package": str(installed),
    }
    (out / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    exports = {
        "TILELANG_CACHE_DIR": out / "cache/tilelang",
        "TORCHINDUCTOR_CACHE_DIR": out / "cache/inductor",
        "TRITON_CACHE_DIR": out / "cache/triton",
        "FLEX_TILELANG_ARTIFACT_DIR": out / "artifacts",
    }
    lines = [
        f"export {key}={shlex.quote(str(value))}" for key, value in exports.items()
    ]
    lines += [
        f'export PYTHONPATH={shlex.quote(str(out / "overlay"))}${{PYTHONPATH:+:$PYTHONPATH}}',
        "export OMP_NUM_THREADS=2 MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1",
        "export TORCHINDUCTOR_COMPILE_THREADS=1 TORCHNPU_PRECOMPILE_THREADS=1",
        "export INDUCTOR_ASCEND_TILELANG_AUTOTUNE=0",
        "unset PYTHONEXECUTABLE __PYVENV_LAUNCHER__",
    ]
    (out / "env.sh").write_text("\n".join(lines) + "\n")
    print(out / "env.sh")


if __name__ == "__main__":
    main()
