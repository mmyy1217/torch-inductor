"""Both routing switches must agree when selecting a comparison backend."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ImplementationTests(unittest.TestCase):
    def test_both_backend_switches(self):
        path = Path(__file__).resolve().parents[1] / "implementations.py"
        spec = importlib.util.spec_from_file_location("implementations", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = SimpleNamespace(
            flex_attention=SimpleNamespace(implementation="triton")
        )
        compile_fn = Mock()
        fake_torch = SimpleNamespace(compile=compile_fn)
        with patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "torch_npu": SimpleNamespace(_inductor=SimpleNamespace(config=config)),
                "torch_npu._inductor": SimpleNamespace(config=config),
                "torch_npu._inductor.config": config,
            },
        ):
            for implementation, backend in (
                ("tilelang", "tilelang"),
                ("triton", "default"),
            ):
                module.compile_attention("fn", implementation)
                self.assertEqual(config.flex_attention.implementation, implementation)
                compile_fn.assert_called_with(
                    "fn",
                    backend="inductor",
                    fullgraph=True,
                    dynamic=False,
                    options={"npu_backend": backend},
                )
