"""Resumable driver for the 27-shape x 4-backend x 2-round measurement.

Each task = one fresh python process running bench_all.py. Results land in
<out>/raw/<shape_id>.<impl>.r<round>.json. Completed tasks are skipped, so an
interrupted run can be resumed without redoing work.

Round order is reversed: round 1 runs tl,triton,fas,fia; round 2 runs
fia,fas,triton,tl.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

IMPLS = ("tl", "triton", "fas", "fia")
PR_SHAPE = {"id": "pr16-8192-mqa-d128", "shape": [1, 32, 1, 8192, 8192, 128]}


def load_shapes(repo):
    matrix = json.loads(
        (repo / ".agent/measurement-20261111/matrix26.json").read_text()
    )
    return list(matrix) + [PR_SHAPE]


def _utilization(device):
    out = subprocess.run(
        ["npu-smi", "info", "-t", "usages", "-i", str(device)],
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if "NPU Utilization(%)" in line:
            try:
                return float(line.split(":")[1].strip())
            except (IndexError, ValueError):
                return None
    return None


def _processes(device):
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True).stdout
    section = None
    count = 0
    for line in out.splitlines():
        if line.startswith("| NPU ID"):
            section = None
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) > 2 and parts[1].isdigit() and len(parts) >= 5 and "/" in parts[4]:
            section = parts[1]
            continue
        if section == str(device) and len(parts) > 2 and parts[1].isdigit():
            count += 1
    return count


def device_free(device, seconds=10):
    """Require the device to show no processes and 0% util for `seconds`."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        util = _utilization(device)
        if util is None or util > 0 or _processes(device) > 0:
            return False
        time.sleep(2)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--device", type=int, required=True,
                        help="logical device id passed to torch.npu.set_device")
    parser.add_argument("--check-device", type=int,
                        help="physical id for npu-smi occupancy checks; defaults to --device")
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--impls", nargs="+", default=list(IMPLS))
    parser.add_argument("--shapes", nargs="*", help="shape ids; default all")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-occupancy-check", action="store_true")
    args = parser.parse_args()

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    shapes = load_shapes(args.repo)
    if args.shapes:
        wanted = set(args.shapes)
        shapes = [s for s in shapes if s["id"] in wanted]
        missing = wanted - {s["id"] for s in shapes}
        if missing:
            parser.error("unknown shape ids: " + ", ".join(sorted(missing)))

    check_device = args.device if args.check_device is None else args.check_device
    env = dict(os.environ)
    env.setdefault("PYTHONSAFEPATH", "1")
    failures = []
    for rnd in args.rounds:
        order = list(args.impls) if rnd % 2 == 1 else list(reversed(args.impls))
        for shape in shapes:
            for impl in order:
                tag = f"{shape['id']}.{impl}.r{rnd}"
                target = raw / f"{tag}.json"
                if target.exists() and target.stat().st_size > 0:
                    print(f"SKIP {tag}", flush=True)
                    continue
                if not args.no_occupancy_check and not device_free(check_device):
                    print(f"BUSY device {check_device} before {tag}", flush=True)
                    failures.append((tag, "device busy"))
                    continue
                cmd = [
                    args.python, "-P", "-u",
                    str(args.repo / ".agent/measurement-20261111/bench_all.py"),
                    "--impl", impl,
                    "--shape", *[str(x) for x in shape["shape"]],
                    "--round", str(rnd),
                    "--device", str(args.device),
                ]
                if args.check_only:
                    cmd.append("--check-only")
                print(f"RUN {tag}", flush=True)
                begin = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                elapsed = time.time() - begin
                (logs / f"{tag}.log").write_text(
                    f"# exit={proc.returncode} elapsed={elapsed:.1f}s\n"
                    f"# cmd={' '.join(cmd)}\n"
                    f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
                )
                if proc.returncode != 0:
                    print(f"FAIL {tag} exit={proc.returncode}", flush=True)
                    failures.append((tag, f"exit {proc.returncode}"))
                    continue
                payload = None
                for line in reversed(proc.stdout.strip().splitlines()):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            payload = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                if payload is None:
                    print(f"FAIL {tag} no json", flush=True)
                    failures.append((tag, "no json"))
                    continue
                payload["elapsed_s"] = elapsed
                target.write_text(json.dumps(payload, indent=2) + "\n")
                print(
                    f"DONE {tag} median={payload.get('median_ms', float('nan')):.3f}ms"
                    f" max_abs={payload.get('max_abs_error', float('nan')):.5f}",
                    flush=True,
                )

    if failures:
        print("\nFAILURES:", flush=True)
        for tag, why in failures:
            print(f"  {tag}: {why}", flush=True)
        return 1
    print("\nALL TASKS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
