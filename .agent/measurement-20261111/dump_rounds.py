"""Dump every per-round median (main run + rerun) so spread can be judged."""

import json
import statistics
from pathlib import Path

ROOT = Path("/home/f00959234/measure-20261111")
REPO = Path("/home/f00959234/wt/flex-pr16-measure")
IMPLS = ("tl", "triton", "fas", "fia")


def medians(base, sid, impl):
    out = []
    for rnd in (1, 2):
        f = ROOT / base / "raw" / f"{sid}.{impl}.r{rnd}.json"
        if not f.exists() or f.stat().st_size == 0:
            out.append(None)
        else:
            out.append(json.loads(f.read_text())["median_ms"] * 1000)
    return out


def main():
    shapes = json.loads(
        (REPO / ".agent/measurement-20261111/matrix26.json").read_text()
    )
    ids = [s["id"] for s in shapes] + ["pr16-8192-mqa-d128"]
    print("shape,impl,main_r1,main_r2,rerun_r1,rerun_r2,spread_pct,chosen_us")
    for sid in ids:
        for impl in IMPLS:
            vals = medians("out", sid, impl) + medians("rerun", sid, impl)
            ok = [v for v in vals if v]
            spread = (max(ok) - min(ok)) / statistics.median(ok) * 100 if ok else 0.0
            # Chosen value: median of all available rounds (main + rerun).
            chosen = statistics.median(ok) if ok else None
            cells = [f"{v:.1f}" if v else "NA" for v in vals]
            print(f"{sid},{impl},{','.join(cells)},{spread:.1f},"
                  f"{chosen:.1f}" if chosen else f"{sid},{impl},{','.join(cells)},{spread:.1f},NA")


if __name__ == "__main__":
    main()
