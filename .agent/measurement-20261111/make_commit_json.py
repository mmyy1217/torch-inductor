"""Build the small JSON that goes into the handoff commit.

Full per-sample receipts stay on the measurement server; this file carries the
per-shape medians, ratios, correctness maxima and stability flags.
"""

import json
import statistics
from pathlib import Path

ROOT = Path("/home/f00959234/measure-20261111")
COMPACT = ROOT / "out/summary_compact.json"
IMPLS = ("tl", "triton", "fas", "fia")


def main():
    data = json.loads(COMPACT.read_text())
    rows = []
    for r in data["rows"]:
        row = {"id": r["id"], "shape": r["shape"]}
        for impl in IMPLS:
            d = r[impl]
            row[impl] = {
                "median_us": round(d["median_of_receipts_us"], 3),
                "receipt_medians_us": [round(x["median_us"], 3)
                                       for x in d["receipts"]],
                "spread_pct": round(d["spread_pct"], 1),
                "max_abs_error": d["max_abs_error"],
                "all_correct": d["all_correct"],
                "dvfs_flagged": d["dvfs_flagged"],
            }
        for key in ("fas_over_tl", "fia_over_tl", "triton_over_tl"):
            row[key] = r[key]
        rows.append(row)

    payload = {
        "environment": {
            "host": "Ascend950PR_9579 (A5-YELLOW, 141.61.15.162)",
            "aicore": 28,
            "aivector": 56,
            "hbm_gb": 123.2,
            "driver": "26.2.rc1 (ascendhal 7.35.23)",
            "cann_runtime": "9.2.0",
            "cann_toolkit": "9.3.0",
            "opp": "9.2.0",
            "torch": "2.14.0+cpu",
            "torch_npu": "2.14.0",
            "triton_ascend": "3.6.0",
            "tilelang": "0.1.15+cuda.git47976f21",
            "tilelang_commit": "47976f210de3598444003e9b05de8204dfa7ff88",
            "tilelang_dirty": False,
            "pytorch_commit": "fbc1a66a1595365d1432b09fb32008df9a16e38c",
            "device_used": "physical NPU 2 (exclusive; 0% util and no processes before each task)",
        },
        "sources": {
            "kernel": "current_dn_kernel.py",
            "kernel_sha256": "eb1f7f07da0f3ff527141c99aad34a35cf431f660ca6983da9f6b8d7f1843ef7",
            "matrix26_sha256": "d726d95d91b3ed63e7d8ccce5e58396df1f3af3cbe1145f73dfb14ffc904bdca",
            "bench_sha256": "a780ee4bb7a87a8f68a5730f7f12b18b560ba3bc190d88d08be2202797d8536e",
            "driver_sha256": "faacfa1439252fefb4c02f3dcbbc2df9e506f3f8fe161fd11bf9f1a16a8c605f",
            "overlay": "isolated copy of torch_npu/_inductor with kernel file replaced; installed wheel untouched",
            "overlay_artifacts": 27,
            "dn_markers_verified": "rendered_kernel.py contains NUM_BLOCKS = min(28, NUM_TILES) if D == 128 else NUM_TILES",
        },
        "protocol": data["protocol"],
        "notes": {
            "triton_cxx": "TRITON_ASCEND_CXX_STD=-std=c++20 required; triton-ascend hardcodes -std=c++17, which cannot compile torch 2.14 ATen headers",
            "fia_dispatch": "LD_DEBUG=bindings shows libopapi_transformer.so binding aclnnFusedInferAttentionScoreV5 and aclnnFusedInferAttentionScoreV5GetWorkspaceSize at call time",
            "rounds": "216 primary receipts (27x4x2) plus 48 rerun receipts for 6 high-spread shapes",
            "raw_samples": "kept on server at /home/f00959234/measure-20261111/{out,rerun}/raw",
        },
        "rows": rows,
    }
    text = json.dumps(payload, indent=2)
    (ROOT / "out/commit_results.json").write_text(text + "\n")
    print("bytes", len(text))


if __name__ == "__main__":
    main()
