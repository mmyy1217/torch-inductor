"""Build the small committed JSON: per-shape medians, ratios, stability flags."""

import json
import statistics
from pathlib import Path

ROOT = Path("/home/f00959234/measure-20261111")
REPO = Path("/home/f00959234/wt/flex-pr16-measure")
IMPLS = ("tl", "triton", "fas", "fia")
DRIFT_PCT = 5.0


def receipts(sid, impl):
    out = []
    for base in ("out", "rerun"):
        for rnd in (1, 2):
            f = ROOT / base / "raw" / f"{sid}.{impl}.r{rnd}.json"
            if f.exists() and f.stat().st_size > 0:
                d = json.loads(f.read_text())
                s = d["samples_ms"]
                head = statistics.median(s[:10])
                tail = statistics.median(s[-10:])
                out.append({
                    "source": base,
                    "round": rnd,
                    "median_us": d["median_ms"] * 1000,
                    "head10_us": head * 1000,
                    "tail10_us": tail * 1000,
                    "drift_pct": (head - tail) / statistics.median(s) * 100,
                    "max_abs_error": d["max_abs_error"],
                    "post_max_abs_error": d["post_max_abs_error"],
                    "correct": bool(d["correct"] and d["post_correct"]
                                    and d["finite"] and d["post_finite"]),
                })
    return out


def main():
    shapes = json.loads(
        (REPO / ".agent/measurement-20261111/matrix26.json").read_text()
    )
    ids = [(s["id"], s["shape"]) for s in shapes]
    ids.append(("pr16-8192-mqa-d128", [1, 32, 1, 8192, 8192, 128]))

    rows = []
    for sid, shape in ids:
        row = {"id": sid, "shape": shape}
        for impl in IMPLS:
            rs = receipts(sid, impl)
            if not rs:
                row[impl] = {"status": "missing"}
                continue
            meds = [r["median_us"] for r in rs]
            row[impl] = {
                "status": "ok",
                "receipts": rs,
                "median_of_receipts_us": statistics.median(meds),
                "spread_pct": (max(meds) - min(meds)) / statistics.median(meds) * 100,
                "max_abs_error": max(r["max_abs_error"] for r in rs),
                "all_correct": all(r["correct"] for r in rs),
                "dvfs_flagged": any(r["drift_pct"] > DRIFT_PCT for r in rs),
            }
        tl = row["tl"].get("median_of_receipts_us")
        for impl, key in (("fas", "fas_over_tl"), ("fia", "fia_over_tl"),
                          ("triton", "triton_over_tl")):
            v = row[impl].get("median_of_receipts_us")
            row[key] = round(v / tl, 4) if (tl and v) else None
        rows.append(row)

    payload = {
        "protocol": {
            "warmup": 10,
            "samples": 30,
            "timer": "torch.npu.Event",
            "rounds": 2,
            "round_order": "round1 tl,triton,fas,fia; round2 reversed",
            "fresh_process_per_task": True,
            "atol": 0.03,
            "rtol": 0.03,
            "reference": "chunked FP32 CPU oracle over all output rows",
            "dvfs_flag_rule": f"median(first10)-median(last10) > {DRIFT_PCT}% of median",
        },
        "rows": rows,
    }
    text = json.dumps(payload, indent=2)
    (ROOT / "out/summary_compact.json").write_text(text + "\n")
    print("rows", len(rows), "bytes", len(text))
    bad = [(r["id"], i) for r in rows for i in IMPLS
           if not r[i].get("all_correct", False)]
    print("not all correct:", bad)
    flagged = [(r["id"], i) for r in rows for i in IMPLS if r[i].get("dvfs_flagged")]
    print("dvfs flagged:", flagged)


if __name__ == "__main__":
    main()
