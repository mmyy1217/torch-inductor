import os
import unittest

os.environ["INDUCTOR_ASCEND_TILELANG_AUTOTUNE"] = "0"

import torch

from tilelang_test_utils import TileLangInductorTest, requires_tilelang


@requires_tilelang
class TestTileLangOps(TileLangInductorTest):
    def _run_cases(self, cases):
        for name, fn, inputs in cases:
            with self.subTest(name=name):
                self.assert_compiled_close(fn, *inputs)

    def test_binary_tensor_ops(self):
        n = 1024
        x = torch.randn(n, device="npu", dtype=torch.float32)
        y = torch.randn(n, device="npu", dtype=torch.float32)
        positive = torch.rand(n, device="npu", dtype=torch.float32) + 0.1
        x16 = torch.randn(n, device="npu", dtype=torch.float16)
        y16 = torch.randn(n, device="npu", dtype=torch.float16)
        positive16 = torch.rand(n, device="npu", dtype=torch.float16) + 0.1
        x32 = torch.randint(-100, 100, (n,), device="npu", dtype=torch.int32)
        y32 = torch.randint(1, 10, (n,), device="npu", dtype=torch.int32)
        self._run_cases((
            ("add_fp32", lambda a, b: a + b, (x, y)),
            ("add_fp16", lambda a, b: a + b, (x16, y16)),
            ("sub_fp32", lambda a, b: a - b, (x, y)),
            ("sub_fp16", lambda a, b: a - b, (x16, y16)),
            ("mul_fp32", lambda a, b: a * b, (x, y)),
            ("mul_fp16", lambda a, b: a * b, (x16, y16)),
            ("mul_int32", lambda a, b: a * b, (x32, y32)),
            ("div_fp32", lambda a, b: a / b, (x, positive)),
            ("div_fp16", lambda a, b: a / b, (x16, positive16)),
            ("maximum_fp32", torch.maximum, (x, y)),
            ("maximum_fp16", torch.maximum, (x16, y16)),
            ("minimum_fp32", torch.minimum, (x, y)),
            ("minimum_fp16", torch.minimum, (x16, y16)),
        ))

    def test_binary_scalar_ops(self):
        x = torch.randn(1024, device="npu", dtype=torch.float32)
        x16 = torch.randn(1024, device="npu", dtype=torch.float16)
        x32 = torch.randint(-100, 100, (1024,), device="npu", dtype=torch.int32)
        self._run_cases((
            ("add_scalar_fp32", lambda a: a + 2.0, (x,)),
            ("add_scalar_fp16", lambda a: a + 2.0, (x16,)),
            ("sub_scalar_fp32", lambda a: a - 0.5, (x,)),
            ("mul_scalar_fp32", lambda a: a * 3.0, (x,)),
            ("mul_scalar_int32", lambda a: a * 3, (x32,)),
            ("div_scalar_fp32", lambda a: a / 4.0, (x,)),
            ("pow_scalar_int32", lambda a: torch.pow(a, 2), (x32,)),
        ))

    def test_unary_ops(self):
        n = 1024
        x = torch.randn(n, device="npu", dtype=torch.float32)
        x16 = torch.randn(n, device="npu", dtype=torch.float16)
        positive = torch.rand(n, device="npu", dtype=torch.float32) + 0.1
        positive16 = torch.rand(n, device="npu", dtype=torch.float16) + 0.1
        trig = torch.rand(n, device="npu", dtype=torch.float32) * 4.0 - 2.0
        erf = torch.rand(n, device="npu", dtype=torch.float32) * 2.0 - 1.0
        x32 = torch.randint(-100, 100, (n,), device="npu", dtype=torch.int32)
        self._run_cases((
            ("exp_fp32", torch.exp, (x,)),
            ("exp_fp16", torch.exp, (x16,)),
            ("log_fp32", torch.log, (positive,)),
            ("log_fp16", torch.log, (positive16,)),
            ("relu_fp32", torch.relu, (x,)),
            ("relu_fp16", torch.relu, (x16,)),
            ("sigmoid_fp32", torch.sigmoid, (x,)),
            ("sigmoid_fp16", torch.sigmoid, (x16,)),
            ("sqrt_fp32", torch.sqrt, (positive,)),
            ("rsqrt_fp32", torch.rsqrt, (positive,)),
            ("abs_fp32", torch.abs, (x,)),
            ("abs_fp16", torch.abs, (x16,)),
            ("abs_int32", torch.abs, (x32,)),
            ("cos_fp32", torch.cos, (trig,)),
            ("sin_fp32", torch.sin, (trig,)),
            ("erf_fp32", torch.erf, (erf,)),
            ("tanh_fp32", torch.tanh, (x,)),
            ("neg_fp32", torch.neg, (x,)),
            ("neg_fp16", torch.neg, (x16,)),
            ("neg_int32", torch.neg, (x32,)),
        ))

    def test_fused_pointwise_chains(self):
        n = 1024
        x = torch.randn(n, device="npu", dtype=torch.float32)
        y = torch.randn(n, device="npu", dtype=torch.float32)
        positive = torch.rand(n, device="npu", dtype=torch.float32) + 0.1
        trig = torch.rand(n, device="npu", dtype=torch.float32) * 4.0 - 2.0
        self._run_cases((
            ("silu", lambda a: a * torch.sigmoid(a), (x,)),
            (
                "gelu_tanh",
                lambda a: a * 0.5 * (
                    1.0 + torch.tanh(0.7978845608 * (
                        a + 0.044715 * a * a * a
                    ))
                ),
                (x,),
            ),
            ("scale_bias", lambda a, b, c: a * b + c, (x, y, positive)),
            ("exp_scale", lambda a: torch.exp(a) * 2.0 - 1.0, (x,)),
            ("sqrt_relu", lambda a: torch.relu(torch.sqrt(a)), (positive,)),
            ("log_sigmoid", lambda a: torch.log(torch.sigmoid(a)), (x,)),
            ("abs_sqrt", lambda a: torch.sqrt(torch.abs(a) + 1e-6), (x,)),
            ("neg_exp", lambda a: torch.exp(-a), (x,)),
            ("cos_sin_add", lambda a: torch.cos(a) + torch.sin(a), (trig,)),
            ("tanh_scale", lambda a: torch.tanh(a * 0.5) * 2.0, (x,)),
        ))

    @unittest.skip(
        "fused fp16 intermediates are currently materialized as float32"
    )
    def test_fused_fp16_chains(self):
        x = torch.randn(1024, device="npu", dtype=torch.float16)
        self._run_cases((
            ("silu_fp16", lambda a: a * torch.sigmoid(a), (x,)),
            ("scale_shift_fp16", lambda a: a * 3.0 + 1.0, (x,)),
            ("exp_add_fp16", lambda a: torch.exp(a) + a, (x,)),
        ))

    @unittest.skip(
        "fused int32 intermediates are currently materialized as float32"
    )
    def test_fused_int32_chains(self):
        x = torch.randint(-100, 100, (1024,), device="npu", dtype=torch.int32)
        y = torch.randint(1, 10, (1024,), device="npu", dtype=torch.int32)
        self._run_cases((
            ("abs_mul_int32", lambda a: torch.abs(a) * 2, (x,)),
            ("neg_mul_int32", lambda a, b: (-a) * b, (x, y)),
            ("pow_mul_int32", lambda a, b: torch.pow(a, 2) * b, (x, y)),
            (
                "abs_min_int32",
                lambda a, b: torch.minimum(torch.abs(a), b),
                (x, y),
            ),
            ("mul_mul_int32", lambda a, b: a * b * 2, (x, y)),
        ))

    def test_contiguous_tensor_ranks(self):
        for shape in ((32, 128), (4, 32, 128), (2, 4, 32, 64)):
            with self.subTest(shape=shape, dtype=torch.float32):
                x = torch.randn(shape, device="npu", dtype=torch.float32)
                y = torch.randn(shape, device="npu", dtype=torch.float32)
                positive = torch.rand(shape, device="npu", dtype=torch.float32) + 0.1
                self._run_cases((
                    ("add", lambda a, b: a + b, (x, y)),
                    ("mul", lambda a, b: a * b, (x, y)),
                    ("relu", torch.relu, (x,)),
                    ("sigmoid", torch.sigmoid, (x,)),
                    ("tanh", torch.tanh, (x,)),
                    ("exp", torch.exp, (x,)),
                    ("sqrt", torch.sqrt, (positive,)),
                    ("silu", lambda a: a * torch.sigmoid(a), (x,)),
                    ("neg_exp", lambda a: torch.exp(-a), (x,)),
                    ("scale_bias", lambda a, b, c: a * b + c, (x, y, positive)),
                ))
            with self.subTest(shape=shape, dtype=torch.float16):
                x16 = torch.randn(shape, device="npu", dtype=torch.float16)
                self._run_cases((
                    ("add_fp16", lambda a, b: a + b, (x16, x16)),
                    ("relu_fp16", torch.relu, (x16,)),
                ))

    def test_reductions(self):
        x = torch.randn((64, 128), device="npu", dtype=torch.float32)
        x16 = torch.randn((64, 128), device="npu", dtype=torch.float16)
        large16 = torch.randn((32, 256), device="npu", dtype=torch.float16)
        self._run_cases((
            ("sum_keepdim_fp32", lambda a: a.sum(dim=1, keepdim=True), (x,)),
            ("sum_keepdim_fp16", lambda a: a.sum(dim=1, keepdim=True), (x16,)),
            ("sum_fp32", lambda a: a.sum(dim=1), (x,)),
            (
                "max_keepdim_fp32",
                lambda a: a.max(dim=1, keepdim=True).values,
                (x,),
            ),
            (
                "min_keepdim_fp32",
                lambda a: a.min(dim=1, keepdim=True).values,
                (x,),
            ),
            ("mean_keepdim_fp32", lambda a: a.mean(dim=1, keepdim=True), (x,)),
            (
                "sum_larger_fp16",
                lambda a: a.sum(dim=1, keepdim=True),
                (large16,),
            ),
        ))


if __name__ == "__main__":
    unittest.main()
