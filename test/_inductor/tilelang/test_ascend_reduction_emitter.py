"""Reduction semantics with poisoned physical tails and scalar output bounds."""

import math
from types import SimpleNamespace
import unittest

from test_ascend_emitter import Code, Tensor, execute, _MODULE


class TestAscendReductionEmitter(unittest.TestCase):
    def reduce(self, values, kind, dtype="float32"):
        n = len(values)
        width = (n + 63) // 64 * 64
        allocations = {
            "x": SimpleNamespace(shape=f"(1, {width})", dtype=dtype),
            "y": SimpleNamespace(shape="(1, 1)", dtype=dtype),
        }
        operation = SimpleNamespace(
            source="x", accumulator="y", reduction_type=kind,
            source_tensor=SimpleNamespace(dims=("x", "r")), reduce_dims=("r",),
        )
        code = Code()
        _MODULE.AscendReductionEmitter(code, allocations).emit(operation, clear=True)
        x = Tensor({(0, i): values[i] if i < n else math.nan for i in range(width)})
        y = Tensor({})
        execute("\n".join(code.lines), {"x": x, "y": y}, _remain_R=n)
        self.assertEqual(set(y.values), {(0, 0)})
        return y.values[(0, 0)]

    def test_sum_masks_poisoned_tail(self):
        for n in (1, 63, 64, 65, 257, 2051, 4096):
            values = [float(i % 7 - 3) for i in range(n)]
            self.assertEqual(self.reduce(values, "sum"), sum(values))

    def test_max_negative_values_and_tail(self):
        for n in (1, 63, 64, 65, 257, 2051):
            self.assertEqual(self.reduce([-float(i + 1) for i in range(n)], "max"), -1.0)

    def test_nan_and_infinity(self):
        for kind in ("sum", "max"):
            for position in (0, 63, 64, 128):
                values = [-1.0] * 129
                values[position] = math.nan
                self.assertTrue(math.isnan(self.reduce(values, kind)))
        self.assertEqual(self.reduce([-math.inf] * 65, "max"), -math.inf)
        self.assertEqual(self.reduce([-math.inf, 1.0, math.inf], "max"), math.inf)

    def test_low_precision_load_and_one_element_store(self):
        for dtype in ("float16", "bfloat16"):
            self.assertEqual(self.reduce([0.5] * 65, "sum", dtype), 32.5)
            self.assertEqual(self.reduce([-0.5] * 65, "max", dtype), -0.5)


if __name__ == "__main__":
    unittest.main()
