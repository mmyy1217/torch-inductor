"""Build an isolated overlay whose TileLang kernel is the shipped current_dn_kernel.py.

Mirrors .agent/pr16/prepare_overlay.py but takes an explicit kernel file instead of a
named variant, so the measured source identity is exactly the handoff file's SHA256.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("run under the target torch/torch_npu environment")
    installed = Path(spec.origin).resolve().parent
    repo = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source/torch_npu/_inductor"
    shutil.copytree(
        repo / "torch_npu/_inductor",
        source,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    kernel = source / "kernel/tilelang_flex_kernel.py"
    text = args.kernel.read_text()
    if "'" * 3 in text or "\\" in text:
        raise RuntimeError("kernel source is unsafe for wrapper string embedding")
    kernel.write_text(text)
    overlay = out / "overlay/torch_npu"
    overlay.mkdir(parents=True)
    for child in installed.iterdir():
        if child.name not in ("_inductor", "__pycache__"):
            (overlay / child.name).symlink_to(child)
    (overlay / "_inductor").symlink_to(source)
    metadata = {
        "variant": "dn",
        "kernel_source": str(args.kernel),
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
    lines = [f"export {key}={shlex.quote(str(value))}" for key, value in exports.items()]
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
