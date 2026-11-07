"""Preserve explicit casts when CSE exposes the promoted computation dtype."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


_PATH = Path(__file__).resolve().parents[3] / "torch_npu/_inductor/tilelang/codegen/tilelang_kernel.py"
_FUNCTION = next(
    node for node in ast.parse(_PATH.read_text(encoding="utf-8")).body
    if isinstance(node, ast.FunctionDef) and node.name == "_build_vec_ops"
)
_MODULE = ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    _FUNCTION,
], type_ignores=[])
_GLOBALS = {"VecOp": SimpleNamespace(create=lambda name, operands, out, dtype: SimpleNamespace(
    name=name, operands=operands, out=out, dtype=dtype,
))}
exec(compile(ast.fix_missing_locations(_MODULE), str(_PATH), "exec"), _GLOBALS)


def cast_value(source, dtype):
    return SimpleNamespace(dtype="float32", buffer=None, const=None,
                           op=SimpleNamespace(name="to_dtype", operands=(source,), dtype=dtype))


class TestCastLowering(unittest.TestCase):
    def test_narrow_cast_survives_promoted_cse_dtype(self):
        for dtype in ("float16", "bfloat16"):
            values = {
                "input": SimpleNamespace(dtype="float32", buffer="input_ub", const=None, op=None),
                "narrow": cast_value("input", dtype),
            }
            operations = []
            result = _GLOBALS["_build_vec_ops"]("narrow", "output", operations, values)
            self.assertEqual(result, "output")
            self.assertEqual([(op.name, op.dtype) for op in operations], [("copy", dtype)])

    def test_narrow_then_widen_keeps_rounding_boundary(self):
        for dtype in ("float16", "bfloat16"):
            values = {
                "input": SimpleNamespace(dtype="float32", buffer="input_ub", const=None, op=None),
                "narrow": cast_value("input", dtype),
                "wide": cast_value("narrow", "float32"),
            }
            operations = []
            result = _GLOBALS["_build_vec_ops"]("wide", "output", operations, values)
            self.assertEqual(result, "output")
            self.assertEqual([op.dtype for op in operations], [dtype, "float32"])


if __name__ == "__main__":
    unittest.main()
