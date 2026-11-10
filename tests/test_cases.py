"""Check standalone case discovery without importing the NPU backend."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExampleDiscoveryTests(unittest.TestCase):
    def test_done_and_todo_cases_load_from_grouped_paths(self):
        runner = load_source("flex_example_runner", "run.py")
        configs = runner.cases()
        self.assertEqual(len(configs), 21)
        self.assertEqual(
            {c["id"] for c in configs if c["tilelang_status"] == "DONE"},
            {
                "dense",
                "causal",
                "sliding_window",
                "document_mask",
                "relative_bias",
                "softcap",
                "causal_with_bias",
            },
        )
        for config in configs:
            module = runner.load_case(config["id"])
            self.assertTrue(callable(module.build))
            self.assertEqual(
                Path(module.__file__).parent.name, config["tilelang_status"]
            )

    def test_example_cli_can_inspect_todo_without_running(self):
        script = ROOT / "run.py"
        result = subprocess.run(
            [
                sys.executable,
                "-P",
                str(script),
                "--impl",
                "tilelang",
                "--case",
                "gqa",
                "--list",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout)[0]["tilelang_status"], "TODO")
        for path in (ROOT / "cases").glob("*/*.py"):
            tree = ast.parse(path.read_text())
            imports = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            patch_line = next(
                node.lineno
                for node in imports
                if isinstance(node, ast.Import)
                and any(alias.name == "torch_npu._inductor" for alias in node.names)
            )
            bind_line = next(
                node.lineno
                for node in imports
                if isinstance(node, ast.ImportFrom)
                and node.module == "torch.nn.attention.flex_attention"
            )
            self.assertLess(patch_line, bind_line)


if __name__ == "__main__":
    unittest.main()
