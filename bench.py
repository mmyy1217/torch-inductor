"""Run the selected checkout's benchmark; optionally repeat a shape matrix."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args, remaining = parser.parse_known_args()
    script = args.source_repo.resolve() / "examples/flex_attention/bench.py"
    if not script.is_file():
        parser.error("source-repo must contain examples/flex_attention/bench.py")
    # Prepare the matching overlay first; this only selects the benchmark source.
    if args.matrix is None:
        return subprocess.call([sys.executable, "-P", str(script), *remaining])
    if args.output is None:
        parser.error("--matrix requires a new --output directory")
    args.output.mkdir(parents=True, exist_ok=False)
    failed = False
    for case in json.loads(args.matrix.read_text()):
        b, h, hk, m, n, d = case["shape"]
        for impl in ("tilelang", "triton", "torch_npu"):
            command = [
                sys.executable,
                "-P",
                str(script),
                "--impl",
                impl,
                "--shape",
                str(b),
                str(h),
                str(m),
                str(n),
                "--kv-heads",
                str(hk),
                "--head-dim",
                str(d),
                *remaining,
            ]
            name = case["id"] + "-" + impl
            print(name, flush=True)
            with (args.output / (name + ".log")).open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
