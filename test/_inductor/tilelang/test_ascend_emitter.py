"""Device-independent checks of the A5 pointwise plan/emitter contract."""

import contextlib
import importlib.util
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
        return self, index

    def __setitem__(self, index, value):
        self.values[index] = value


def op(name, operands, out, dtype="float32"):
    return SimpleNamespace(name=name, operands=operands, out=out, dtype=dtype, execution_dtype=dtype)


def emit(operations, shapes, outputs=("y",)):
    plan = SimpleNamespace(outputs=outputs, compute=SimpleNamespace(operations=operations))
    dtypes = {op.out: op.dtype for op in operations}
    allocations = {name: SimpleNamespace(shape=shape, dtype=dtypes.get(name, "float32")) for name, shape in shapes.items()}
    code = Code()
    _MODULE.AscendPointwiseEmitter(code, allocations).emit(plan)
    return "\n".join(code.lines)


def execute(source, tensors):
    def cast(value, dtype):
        if dtype == "bfloat16":
            bits = struct.unpack("I", struct.pack("f", value))[0]
            bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
            return struct.unpack("f", struct.pack("I", bits))[0]
        fmt = {"float32": "f", "float16": "e"}[dtype]
        try:
            return struct.unpack(fmt, struct.pack(fmt, value))[0]
        except OverflowError:
            return math.copysign(math.inf, value)

    def load(address, dist="NORM"):
        tensor, index = address
        indices = index if isinstance(index, tuple) else (index,)
        def read(offset):
            key = (*indices[:-1], indices[-1] + offset)
            return tensor.values[key if isinstance(index, tuple) else key[0]]
        if dist.startswith("BRC"):
            return [read(0)] * (64 if dist == "BRC_B32" else 128)
        if dist == "UNPK_B16":
            return [item for i in range(64) for item in (read(i), 0.0)]
        assert dist == "NORM", dist
        return [read(i) for i in range(64)]

    def convert(value, dtype, mask=None, sat=False):
        assert not sat
        if dtype == "float32":
            return [cast(x, dtype) for x in (value[::2] if len(value) == 128 else value)]
        return [item for x in value for item in (cast(x, dtype), 0.0)]

    def store(address, value, mask, dist):
        tensor, index = address
        indices = index if isinstance(index, tuple) else (index,)
        if dist == "PK_B32":
            value = value[::2]
        else:
            assert dist == "NORM_B32"
        assert len(value) == 64
        for i, x in enumerate(value):
            key = (*indices[:-1], indices[-1] + i)
            tensor.values[key if isinstance(index, tuple) else key[0]] = x

    def binary(fn):
        return lambda a, b, mask: [cast(fn(x, y), "float32") for x, y in zip(a, b)]

    def unary(fn):
        return lambda a, mask: [cast(fn(x), "float32") for x in a]

    language = SimpleNamespace(
        SimdVF=contextlib.nullcontext, serial=range, cast=cast,
        simd=SimpleNamespace(
            pset=lambda bits: bits, vld=load, vcvt=convert, vsts=store,
            vdup=lambda x, dtype: [cast(x, dtype)] * 64,
            vadd=binary(lambda a, b: a+b), vsub=binary(lambda a, b: a-b),
            vmul=binary(lambda a, b: a*b), vdiv=binary(lambda a, b: a/b),
            vexp=unary(math.exp), vln=unary(math.log), vsqrt=unary(math.sqrt), vabs=unary(abs),
        ),
    )
    exec(compile(source, "<generated-vf>", "exec"), {"T": language, **tensors})


class TestAscendEmitter(unittest.TestCase):
    def test_fused_chain_keeps_intermediates_local(self):
        source = emit(
            [op("add", ("x", 1.0), "tmp"), op("mul", ("tmp", 0.5), "tmp"), op("exp", ("tmp",), "y")],
            {"x": "(64,)", "tmp": "(64,)", "y": "(64,)"},
        )
        x = Tensor({i: float(i) / 64 for i in range(64)})
        y = Tensor({})
        # There is intentionally no runtime storage for tmp.
        execute(source, {"x": x, "y": y})
        for i in range(64):
            self.assertAlmostEqual(y.values[i], math.exp((i / 64 + 1) * 0.5), places=6)

    def test_broadcast_and_multiple_boundary_outputs(self):
        source = emit(
            [op("add", ("x", "bias"), "y"), op("mul", ("y", 2.0), "z")],
            {"x": "(2, 64)", "bias": "(1, 64)", "y": "(2, 64)", "z": "(2, 64)"},
            ("y", "z"),
        )
        x = Tensor({(i, j): float(i + j) for i in range(2) for j in range(64)})
        bias = Tensor({(0, j): float(j) for j in range(64)})
        y, z = Tensor({}), Tensor({})
        execute(source, {"x": x, "bias": bias, "y": y, "z": z})
        self.assertEqual(y.values, {(i, j): float(i + 2*j) for i in range(2) for j in range(64)})
        self.assertEqual(z.values, {key: 2 * value for key, value in y.values.items()})

    def test_incompatible_layout_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "layout conversion"):
            emit([op("copy", ("x",), "y")], {"x": "(3,)", "y": "(64,)"})

    def test_scalar_intermediate_broadcasts_inside_vf(self):
        source = emit(
            [op("add", ("x", 1.0), "tmp"), op("exp", ("tmp",), "y")],
            {"x": "(1,)", "tmp": "(1,)", "y": "(64,)"},
        )
        y = Tensor({})
        execute(source, {"x": Tensor({0: 0.5}), "y": y})
        for i in range(64):
            self.assertAlmostEqual(y.values[i], math.exp(1.5), places=6)

    def test_low_precision_rounding_before_widening(self):
        for dtype in ("float16", "bfloat16"):
            operations = [op("copy", ("x",), "low", dtype), op("mul", ("low", 1.003), "y")]
            operations[1].execution_dtype = dtype
            source = emit(operations, {"x": "(128,)", "low": "(128,)", "y": "(128,)"})
            y = Tensor({})
            execute(source, {"x": Tensor({i: 1.0001 for i in range(128)}), "y": y})
            expected = 1.0029296875 if dtype == "float16" else 1.0
            self.assertEqual(y.values, {i: expected for i in range(128)})
            self.assertIn("T.SimdVF", source)
            self.assertNotIn("T.SimtVF", source)

    def test_runtime_tile_width_contract(self):
        for width in (64, 128, 256):
            _MODULE.validate_simd_tile_width(width)
        for width in (-64, 0, 1, 32, 65):
            with self.assertRaises(ValueError):
                _MODULE.validate_simd_tile_width(width)

    def test_unaligned_physical_width_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "physical tile width"):
            emit([op("copy", ("x",), "y")], {"x": "(3,)", "y": "(3,)"})

    def test_unvalidated_operation_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "not enabled"):
            emit([op("maximum", ("x", 0), "y")], {"x": "(64,)", "y": "(64,)"})

    def test_integer_execution_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "floating-point"):
            emit([op("mul", ("x", 2), "y", "int32")], {"x": "(64,)", "y": "(64,)"})


if __name__ == "__main__":
    unittest.main()
