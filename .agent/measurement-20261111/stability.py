"""Classify timing receipts as DVFS-ramp contaminated or stable.

A receipt is flagged when its 30 samples show a sustained downward drift: the
median of the first 10 samples exceeds the median of the last 10 by more than
5%. That is the signature of clock ramp-up, not of a slower kernel, so its
median must not be compared against a receipt measured at a different clock.
"""

import json
import statistics
from pathlib import Path

ROOT = Path("/home/f00959234/measure-20261111")
REPO = Path("/home/f00959234/wt/flex-pr16-measure")
IMPLS = ("tl", "triton", "fas", "fia")
DRIFT_PCT = 5.0


def receipts():
    shapes = json.loads(
        (REPO / ".agent/measurement-20261111/matrix26.json").read_text()
    )
    ids = [s["id"] for s in shapes] + ["pr16-8192-mqa-d128"]
    for sid in ids:
        for impl in IMPLS:
            for base in ("out", "rerun"):
                for rnd in (1, 2):
                    f = ROOT / base / "raw" / f"{sid}.{impl}.r{rnd}.json"
                    if f.exists() and f.stat().st_size > 0:
                        yield sid, impl, base, rnd, json.loads(f.read_text())


def main():
    flagged, total = [], 0
    for sid, impl, base, rnd, d in receipts():
        s = d["samples_ms"]
        total += 1
        head = statistics.median(s[:10])
        tail = statistics.median(s[-10:])
        drift = (head - tail) / statistics.median(s) * 100
        if drift > DRIFT_PCT:
            flagged.append((drift, sid, impl, base, rnd,
                            round(d["median_ms"] * 1000, 1),
                            round(head * 1000, 1), round(tail * 1000, 1)))
    flagged.sort(reverse=True)
    print(f"receipts={total} flagged={len(flagged)} drift_threshold={DRIFT_PCT}%")
    print("drift%,shape,impl,base,round,median_us,head10_us,tail10_us")
    for row in flagged:
        print("%.1f,%s,%s,%s,%d,%.1f,%.1f,%.1f" % row)


if __name__ == "__main__":
    main()
