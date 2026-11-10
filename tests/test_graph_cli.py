"""Reject unverified or sparse wrappers before importing device dependencies."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GraphCLITests(unittest.TestCase):
    def test_only_validated_dense_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for case, status in (("causal", "PASS"), ("dense", "FAIL")):
                with self.subTest(case=case, status=status):
                    (folder / "case.json").write_text(json.dumps({"id": case}))
                    (folder / "result.json").write_text(json.dumps({"status": status}))
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-P",
                            str(ROOT / "graph_bench.py"),
                            str(folder),
                        ],
                        env=dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0"),
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(
                        "requires a successfully validated dense", result.stderr
                    )
                    self.assertFalse((folder / "graph_benchmark.json").exists())


if __name__ == "__main__":
    unittest.main()
