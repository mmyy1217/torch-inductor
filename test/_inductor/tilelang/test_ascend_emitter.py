"""Device-independent checks of the A5 pointwise plan/emitter contract."""

import contextlib
import importlib.util
import itertools
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest


_PATH = Path(__file__).resolve().parents[3] / "torch_npu/_inductor/tilelang/codegen/ascend.py"
_SPEC = importlib.util.spec_from_file_location("ascend_pointwise_emitter", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class Code:
    def __init__(self):
        self.lines = []
        self.depth = 0

    def writeline(self, line):
        self.lines.append("    " * self.depth + line)

    @contextlib.contextmanager
    def indent(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1


class Tensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value


def op(name, operands, out, dtype="float32"):
    return SimpleNamespace(name=name, operands=operands, out=out, dtype=dtype, execution_dtype=dtype)


def emit(operations, shapes, outputs=("y",)):
    plan = SimpleNamespace(outputs=outputs, compute=SimpleNamespace(operations=operations))
    allocations = {name: SimpleNamespace(shape=shape) for name, shape in shapes.items()}
    code = Code()
    _MODULE.AscendPointwiseEmitter(code, allocations).emit(plan)
    return "\n".join(code.lines)


def execute(source, tensors):
    def cast(value, dtype):
        fmt = {"float32": "f", "float16": "e"}[dtype]
        return struct.unpack(fmt, struct.pack(fmt, value))[0]

    language = SimpleNamespace(
        SimtVF=lambda **kwargs: contextlib.nullcontext(),
        Parallel=lambda *shape: range(shape[0]) if len(shape) == 1 else itertools.product(*(range(x) for x in shape)),
        cast=cast, exp=math.exp, log=math.log, sqrt=math.sqrt, rsqrt=lambda x: 1 / math.sqrt(x), abs=abs,
    )
    exec(compile(source, "<generated-vf>", "exec"), {"T": language, **tensors})


class TestAscendEmitter(unittest.TestCase):
    def test_fused_chain_keeps_intermediates_local(self):
        source = emit(
            [op("add", ("x", 1.0), "tmp"), op("mul", ("tmp", 0.5), "tmp"), op("exp", ("tmp",), "y")],
            {"x": "(4,)", "tmp": "(4,)", "y": "(4,)"},
        )
        x = Tensor({i: float(i) for i in range(4)})
        y = Tensor({})
        # There is intentionally no runtime storage for tmp.
        execute(source, {"x": x, "y": y})
        for i in range(4):
            self.assertAlmostEqual(y[i], math.exp((i + 1) * 0.5), places=6)

    def test_broadcast_and_multiple_boundary_outputs(self):
        source = emit(
            [op("add", ("x", "bias"), "y"), op("mul", ("y", 2.0), "z")],
            {"x": "(2, 3)", "bias": "(1, 3)", "y": "(2, 3)", "z": "(2, 3)"},
            ("y", "z"),
        )
        x = Tensor({(i, j): float(i + j) for i in range(2) for j in range(3)})
        bias = Tensor({(0, j): float(j) for j in range(3)})
        y, z = Tensor({}), Tensor({})
        execute(source, {"x": x, "bias": bias, "y": y, "z": z})
        self.assertEqual(y.values, {(i, j): float(i + 2*j) for i in range(2) for j in range(3)})
        self.assertEqual(z.values, {key: 2 * value for key, value in y.values.items()})

    def test_incompatible_layout_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "layout conversion"):
            emit([op("copy", ("x",), "y")], {"x": "(3,)", "y": "(4,)"})

    def test_scalar_intermediate_broadcasts_inside_vf(self):
        source = emit(
            [op("add", ("x", 1.0), "tmp"), op("exp", ("tmp",), "y")],
            {"x": "(1,)", "tmp": "(1,)", "y": "(4,)"},
        )
        y = Tensor({})
        execute(source, {"x": Tensor({0: 0.5}), "y": y})
        for i in range(4):
            self.assertAlmostEqual(y[i], math.exp(1.5), places=6)

    def test_unvalidated_operation_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "not enabled"):
            emit([op("maximum", ("x", 0), "y")], {"x": "(4,)", "y": "(4,)"})

    def test_integer_execution_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "floating-point"):
            emit([op("mul", ("x", 2), "y", "int32")], {"x": "(4,)", "y": "(4,)"})


if __name__ == "__main__":
    unittest.main()
