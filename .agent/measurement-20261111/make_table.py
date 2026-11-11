"""Render the 27-row table from summary_compact.json."""

import json
import statistics
from pathlib import Path

ROOT = Path("/home/f00959234/measure-20261111")
IMPLS = ("tl", "triton", "fas", "fia")


def main():
    data = json.loads((ROOT / "out/summary_compact.json").read_text())
    rows = data["rows"]
    lines = [
        "| Case | B,HQ,HKV,M,N,D | TL us | Triton us | FAS V4 us | FIA V5 us "
        "| V4/TL | V5/TL | Triton/TL | TL spread |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def us(i):
            v = r[i].get("median_of_receipts_us")
            return f"{v:.3f}" if v else "NA"
        def ratio(k):
            v = r.get(k)
            return f"{v:.3f}" if v else "NA"
        spread = r["tl"].get("spread_pct")
        lines.append(
            f"| {r['id']} | {','.join(str(x) for x in r['shape'])} "
            f"| {us('tl')} | {us('triton')} | {us('fas')} | {us('fia')} "
            f"| {ratio('fas_over_tl')} | {ratio('fia_over_tl')} "
            f"| {ratio('triton_over_tl')} "
            f"| {spread:.1f}% |" if spread is not None else ""
        )
    table = "\n".join(x for x in lines if x)
    (ROOT / "out/table_final.md").write_text(table + "\n")
    print(table)

    for impl, key in (("fas", "fas_over_tl"), ("fia", "fia_over_tl"),
                      ("triton", "triton_over_tl")):
        vals = [r[key] for r in rows if r.get(key)]
        wins = sum(1 for v in vals if v > 1)
        print(f"\n{impl}: TL faster {wins}/{len(vals)} "
              f"range {min(vals):.3f}-{max(vals):.3f} median {statistics.median(vals):.3f}")


if __name__ == "__main__":
    main()
