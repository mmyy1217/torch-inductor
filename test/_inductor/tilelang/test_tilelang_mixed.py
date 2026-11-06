import os
import unittest
from itertools import permutations

os.environ["INDUCTOR_ASCEND_TILELANG_AUTOTUNE"] = "0"

import torch

from tilelang_test_utils import TileLangInductorTest, requires_tilelang


def _softmax(x, temperature=1.0):
    scaled = x / temperature
    maximum = scaled.max(dim=-1, keepdim=True).values
    numerator = torch.exp(scaled - maximum)
    return numerator / numerator.sum(dim=-1, keepdim=True)


@requires_tilelang
class TestTileLangMixedKernels(TileLangInductorTest):
    force_disable_caches = True

    def _assert_reduction(self, shape, fn):
        x = torch.randn(shape, device="npu", dtype=torch.float32)
        self.assert_compiled_close_and_get_code(fn, x)

    def test_softmax(self):
        x = torch.randn((128, 512), device="npu", dtype=torch.float32)
        self.assert_compiled_close(_softmax, x)

    def test_softmax_temperature(self):
        x = torch.randn((64, 256), device="npu", dtype=torch.float32)
        self.assert_compiled_close(_softmax, x, 0.5)

    def test_layer_norm_affine(self):
        def layer_norm_affine(x, weight, bias):
            mean = x.mean(dim=-1, keepdim=True)
            variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            normalized = (x - mean) / torch.sqrt(variance + 1e-5)
            return normalized * weight + bias

        x = torch.randn((32, 256), device="npu", dtype=torch.float32)
        weight = torch.ones((256,), device="npu", dtype=torch.float32)
        bias = torch.zeros((256,), device="npu", dtype=torch.float32)
        self.assert_compiled_close(layer_norm_affine, x, weight, bias)

    def test_rms_norm(self):
        def rms_norm(x, weight):
            rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6)
            return x / rms * weight

        x = torch.randn((32, 256), device="npu", dtype=torch.float32)
        weight = torch.ones((256,), device="npu", dtype=torch.float32)
        self.assert_compiled_close(rms_norm, x, weight)

    def test_sum_reduction(self):
        self._assert_reduction(
            (32, 256), lambda x: x.sum(dim=-1, keepdim=True)
        )
        self._assert_reduction(
            (32, 8192), lambda x: x.sum(dim=-1, keepdim=True)
        )
        self._assert_reduction(
            (256, 32), lambda x: x.sum(dim=0, keepdim=True)
        )
        self._assert_reduction(
            (8192, 32), lambda x: x.sum(dim=0, keepdim=True)
        )

    def test_column_broadcast_then_reduce_rows(self):
        m, n = 33, 17
        matrix = torch.randn((m, n), device="npu", dtype=torch.float32)
        column = torch.randn((m, 1), device="npu", dtype=torch.float32)
        actual = self.assert_compiled_close(
            lambda x, y: (x + y).sum(dim=0, keepdim=True), matrix, column
        )
        self.assertEqual(tuple(actual.shape), (1, n))

    def test_high_dim_broadcast_then_reduce_last_dim(self):
        def add_sum(matrix, broadcast_input):
            return (matrix + broadcast_input).sum(dim=-1, keepdim=True)

        for r in (17, 4097):
            with self.subTest(r=r):
                matrix = torch.randn((2, 3, r), device="npu", dtype=torch.float32)
                broadcast_input = torch.randn(
                    (2, 1, r), device="npu", dtype=torch.float32
                )
                actual, generated_codes = self.assert_compiled_close_and_get_code(
                    add_sum,
                    matrix,
                    broadcast_input,
                )
                self.assertEqual(tuple(actual.shape), (2, 3, 1))
                generated_code = "\n".join(generated_codes)
                self.assertIn("X1BLOCK", generated_code)
                self.assertIn("X0BLOCK", generated_code)
                self.assertIn("R2BLOCK", generated_code)

    def test_column_broadcast_then_reduce_rows_supported_ops(self):
        m, n = 33, 17
        matrix = torch.randn((m, n), device="npu", dtype=torch.float32)
        column = torch.randn((m, 1), device="npu", dtype=torch.float32)
        positive_column = torch.rand((m, 1), device="npu", dtype=torch.float32) + 0.5

        cases = (
            ("sub_max", lambda x, y: (x - y).max(dim=0, keepdim=True).values, column),
            ("mul_min", lambda x, y: (x * y).min(dim=0, keepdim=True).values, column),
            (
                "div_mean",
                lambda x, y: (x / y).mean(dim=0, keepdim=True),
                positive_column,
            ),
            (
                "maximum_sum",
                lambda x, y: torch.maximum(x, y).sum(dim=0, keepdim=True),
                column,
            ),
            (
                "minimum_max",
                lambda x, y: torch.minimum(x, y).max(dim=0, keepdim=True).values,
                column,
            ),
            (
                "tanh_add_sum",
                lambda x, y: torch.tanh(x + y).sum(dim=0, keepdim=True),
                column,
            ),
        )

        for name, fn, broadcast_input in cases:
            with self.subTest(name=name):
                actual = self.assert_compiled_close(fn, matrix, broadcast_input)
                self.assertEqual(tuple(actual.shape), (1, n))

    def test_dynamic_column_broadcast_then_reduce_rows(self):
        def add_sum(matrix, column):
            return (matrix + column).sum(dim=0, keepdim=True)

        m, n = 33, 17
        matrix = torch.randn((m, n), device="npu", dtype=torch.float32)
        column = torch.randn((m, 1), device="npu", dtype=torch.float32)
        actual, generated_codes = self.assert_compiled_close_and_get_code(
            add_sum,
            matrix,
            column,
            dynamic=True,
        )
        self.assertEqual(tuple(actual.shape), (1, n))
        generated_code = "\n".join(generated_codes)
        self.assertIn(
            "in_ptr0: T.Tensor((_rnumel, _xnumel)",
            generated_code,
        )
        self.assertIn("dims=[0]", generated_code)

    def test_broadcast_reduction_axis_permutations(self):
        role_sizes = {"G": 2, "R": 33, "B": 17}

        for axis_roles in permutations(("G", "R", "B")):
            reduction_dim = axis_roles.index("R")
            matrix_shape = tuple(role_sizes[role] for role in axis_roles)
            broadcast_shape = tuple(
                1 if role == "B" else role_sizes[role]
                for role in axis_roles
            )
            expected_shape = tuple(
                1 if role == "R" else role_sizes[role]
                for role in axis_roles
            )

            def add_sum(matrix, broadcast_input):
                return (matrix + broadcast_input).sum(
                    dim=reduction_dim,
                    keepdim=True,
                )

            with self.subTest(axis_roles="".join(axis_roles)):
                matrix = torch.randn(
                    matrix_shape,
                    device="npu",
                    dtype=torch.float32,
                )
                broadcast_input = torch.randn(
                    broadcast_shape,
                    device="npu",
                    dtype=torch.float32,
                )
                actual, generated_codes = (
                    self.assert_compiled_close_and_get_code(
                        add_sum,
                        matrix,
                        broadcast_input,
                    )
                )
                self.assertEqual(tuple(actual.shape), expected_shape)
                generated_code = "\n".join(generated_codes)
                self.assertIn("T.reduce(", generated_code)
                self.assertIn("dims=[", generated_code)

    def test_dynamic_broadcast_reduction_axis_permutations_2d(self):
        role_sizes = {"R": 33, "B": 17}

        for axis_roles in permutations(("R", "B")):
            reduction_dim = axis_roles.index("R")
            matrix_shape = tuple(role_sizes[role] for role in axis_roles)
            broadcast_shape = tuple(
                1 if role == "B" else role_sizes[role]
                for role in axis_roles
            )
            expected_shape = tuple(
                1 if role == "R" else role_sizes[role]
                for role in axis_roles
            )

            def add_sum(matrix, broadcast_input):
                return (matrix + broadcast_input).sum(
                    dim=reduction_dim,
                    keepdim=True,
                )

            with self.subTest(axis_roles="".join(axis_roles)):
                matrix = torch.randn(
                    matrix_shape,
                    device="npu",
                    dtype=torch.float32,
                )
                broadcast_input = torch.randn(
                    broadcast_shape,
                    device="npu",
                    dtype=torch.float32,
                )
                actual, generated_codes = (
                    self.assert_compiled_close_and_get_code(
                        add_sum,
                        matrix,
                        broadcast_input,
                        dynamic=True,
                    )
                )
                self.assertEqual(tuple(actual.shape), expected_shape)
                generated_code = "\n".join(generated_codes)
                self.assertIn("T.reduce(", generated_code)
                self.assertIn("dims=[", generated_code)


if __name__ == "__main__":
    unittest.main()
