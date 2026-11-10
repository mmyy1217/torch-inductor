"""Copy an installed torch_npu wheel and overlay an explicitly selected source tree."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    repo = args.source_repo.resolve()
    dest = args.output.resolve()
    if dest.exists():
        parser.error("output must not exist; use a new personal directory")
    if dest == repo or repo in dest.parents:
        parser.error("output must be outside the source checkout")
    implementation = repo / "torch_npu/_inductor"
    if not all(
        (implementation / name).is_file()
        for name in (
            "kernel/tilelang_flex_attention.py",
            "tilelang/codegen/tilelang_scheduling.py",
        )
    ):
        parser.error(
            "source-repo must contain both TileLang codegen and FlexAttention templates"
        )
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=repo, text=True
        ).strip()
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        parser.error("source-repo must be a Git checkout with a commit")
    if Path(top).resolve() != repo:
        parser.error("source-repo must point to the checkout root")
    source_hashes = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(implementation.rglob("*.py"))
    }
    sys.path[:] = [
        entry
        for entry in sys.path
        if entry
        and Path(entry).resolve() != repo
        and not (Path(entry) / "overlay.json").exists()
    ]
    import torch
    import torch_npu

    if not torch.__version__.startswith(
        "2.14."
    ) or not torch_npu.__version__.startswith("2.14."):
        parser.error("requires matching torch / torch_npu 2.14")
    source = Path(torch_npu.__file__).resolve().parent
    if source == repo / "torch_npu" or repo in source.parents:
        parser.error(
            "expected an installed wheel, not extensions from the source checkout"
        )
    if dest == source or source in dest.parents:
        parser.error("output must be outside the installed package")
    dest.mkdir(parents=True)
    shutil.copytree(
        source,
        dest / "torch_npu",
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.rmtree(dest / "torch_npu/_inductor", ignore_errors=True)
    shutil.copytree(
        implementation,
        dest / "torch_npu/_inductor",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    copied_hashes = {
        str(path.relative_to(dest)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((dest / "torch_npu/_inductor").rglob("*.py"))
    }
    if copied_hashes != source_hashes:
        raise RuntimeError(
            "Source files changed while copying; discard this overlay and retry"
        )
    (dest / "overlay.json").write_text(
        json.dumps(
            dict(
                source_repo=str(repo),
                source_package=str(source),
                revision=revision,
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                source_sha256=source_hashes,
            ),
            indent=2,
        )
    )
    print(f"Overlay ready: {dest}")
    print(
        "Prepend this directory to PYTHONPATH in a fresh process; keep PYTHONSAFEPATH=1."
    )


if __name__ == "__main__":
    main()
