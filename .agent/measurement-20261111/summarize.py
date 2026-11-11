"""Collapse raw per-task JSON into the 27-row table and a summary JSON."""

import argparse
import json
import statistics
from pathlib import Path

IMPLS = ("tl", "triton", "fas", "fia")
LABEL = {"tl": "TileLang", "triton": "Triton", "fas": "FAS V4", "fia": "FIA V5"}


def load_shapes(repo):
    matrix = json.loads((repo / ".agent/measurement-20261111/matrix26.json").read_text())
    return list(matrix) + [{"id": "pr16-8192-mqa-d128",
                            "shape": [1, 32, 1, 8192, 8192, 128]}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = args.out / "raw"
    rerun = args.out.parent / "rerun/raw"
    rows, problems = [], []
    for shape in load_shapes(args.repo):
        row = {"id": shape["id"], "shape": shape["shape"]}
        for impl in IMPLS:
            per_round, errors = [], []
            for rnd in (1, 2):
                f = raw / f"{shape['id']}.{impl}.r{rnd}.json"
                if not f.exists() or f.stat().st_size == 0:
                    errors.append(f"round{rnd} missing")
                    continue
                d = json.loads(f.read_text())
                per_round.append(d)
                if not (d.get("correct") and d.get("post_correct") and d.get("finite")
                        and d.get("post_finite")):
                    errors.append(f"round{rnd} correctness")
            if not per_round:
                row[impl] = {"status": "missing", "why": "; ".join(errors)}
                problems.append((shape["id"], impl, "; ".join(errors)))
                continue
            medians = [d["median_ms"] for d in per_round]
            row[impl] = {
                "status": "ok" if not errors else "ok_with_issue",
                "round_medians_ms": medians,
                "n_receipts": len(medians),
                "mean_of_round_medians_ms": statistics.mean(medians),
                "spread_pct": ((max(medians) - min(medians))
                               / statistics.median(medians) * 100),
                "max_abs_error": max(d["max_abs_error"] for d in per_round),
                "post_max_abs_error": max(d["post_max_abs_error"] for d in per_round),
                "samples_ms": [d["samples_ms"] for d in per_round],
            }
            if errors:
                problems.append((shape["id"], impl, "; ".join(errors)))
        tl = row.get("tl", {}).get("mean_of_round_medians_ms")
        for impl, key in (("fas", "fas_over_tl"), ("fia", "fia_over_tl"),
                          ("triton", "triton_over_tl")):
            v = row.get(impl, {}).get("mean_of_round_medians_ms")
            row[key] = (v / tl) if (tl and v) else None
        rows.append(row)

    payload = {"rows": rows, "problems": problems}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    hdr = ("| Case | B,HQ,HKV,M,N,D | TL us | Triton us | FAS V4 us | FIA V5 us "
           "| V4/TL | V5/TL | Triton/TL |")
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [hdr, sep]
    for r in rows:
        def us(impl):
            v = r.get(impl, {}).get("mean_of_round_medians_ms")
            return f"{v * 1000:.3f}" if v else "NA"
        def ratio(key):
            v = r.get(key)
            return f"{v:.3f}" if v else "NA"
        lines.append(
            f"| {r['id']} | {','.join(str(x) for x in r['shape'])} "
            f"| {us('tl')} | {us('triton')} | {us('fas')} | {us('fia')} "
            f"| {ratio('fas_over_tl')} | {ratio('fia_over_tl')} | {ratio('triton_over_tl')} |"
        )
    table = "\n".join(lines)
    (args.out / "table.md").write_text(table + "\n")
    print(table)
    if problems:
        print("\nProblems:")
        for p in problems:
            print("  ", p)


if __name__ == "__main__":
    main()
