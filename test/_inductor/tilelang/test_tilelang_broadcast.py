import os
import unittest

import sympy

# Keep the broad correctness matrix fast; autotune is covered separately.
os.environ["INDUCTOR_ASCEND_TILELANG_AUTOTUNE"] = "0"

import torch
from torch._inductor.utils import run_and_get_code

from torch_npu._inductor.tilelang.codegen.tilelang_scheduling import (
    TileLangScheduling,
)

from tilelang_test_utils import TileLangInductorTest, requires_tilelang


@requires_tilelang
class TestTileLangBroadcast(TileLangInductorTest):
    @staticmethod
    def _add(lhs, rhs):
        return lhs + rhs

    def _run_cases(self, cases):
        for name, lhs_shape, rhs_shape in cases:
            with self.subTest(name=name):
                lhs = torch.randn(lhs_shape, device="npu", dtype=torch.float32)
                rhs = torch.randn(rhs_shape, device="npu", dtype=torch.float32)
                self.assert_compiled_close(self._add, lhs, rhs)

    def test_pointwise_tiling_keeps_empty_reduction_tree(self):
        tiling = TileLangScheduling.select_tiling(
            (), sympy.Tuple(33, 17), sympy.S.One
        )
        self.assertEqual(tiling, {"x": sympy.Integer(561), "r": sympy.S.One})

    def test_2d_broadcast(self):
        self._run_cases((
            ("row", (32, 16), (1, 16)),
            ("column", (32, 16), (32, 1)),
            ("row_tail", (33, 17), (1, 17)),
            ("column_tail", (33, 17), (33, 1)),
            ("wide_row", (32, 128), (1, 128)),
            ("scalar_2d", (32, 16), (1, 1)),
        ))

    def test_2d_broadcast_autotune(self):
        autotune_env = {
            "INDUCTOR_ASCEND_TILELANG_AUTOTUNE": "1",
            "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_WARMUP": "0",
            "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_REP": "1",
            "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_TIMEOUT": "60",
        }
        previous = {name: os.environ.get(name) for name in autotune_env}
        os.environ.update(autotune_env)
        try:
            lhs = torch.randn((32, 16), device="npu", dtype=torch.float32)
            rhs = torch.randn((1, 16), device="npu", dtype=torch.float32)
            self.assert_compiled_close(self._add, lhs, rhs)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_3d_single_axis_broadcast(self):
        self._run_cases((
            ("axis0", (4, 8, 16), (1, 8, 16)),
            ("axis1", (4, 8, 16), (4, 1, 16)),
            ("axis2", (4, 8, 16), (4, 8, 1)),
            ("implicit_axis0", (4, 8, 16), (8, 16)),
        ))

    def test_3d_multi_axis_broadcast(self):
        self._run_cases((
            ("axis0_axis1", (4, 8, 16), (1, 1, 16)),
            ("axis0_axis2", (4, 8, 16), (1, 8, 1)),
            ("axis1_axis2", (4, 8, 16), (4, 1, 1)),
            ("scalar_3d", (4, 8, 16), (1, 1, 1)),
        ))

    def test_dynamic_2d_broadcast(self):
        torch._dynamo.reset()
        compiled = torch.compile(
            self._add,
            backend="inductor",
            dynamic=True,
            options={"npu_backend": "tilelang"},
        )
        generated_codes = []
        for m, n in ((32, 16), (33, 17), (8, 128)):
            with self.subTest(shape=(m, n)):
                lhs = torch.randn((m, n), device="npu", dtype=torch.float32)
                rhs = torch.randn((1, n), device="npu", dtype=torch.float32)
                expected = self._add(lhs, rhs)
                actual, shape_codes = run_and_get_code(compiled, lhs, rhs)
                generated_codes.extend(shape_codes)
                torch.npu.synchronize()
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-3,
                    atol=1e-3,
                )
        self.assert_all_kernels_are_tilelang(generated_codes)


if __name__ == "__main__":
    unittest.main()
