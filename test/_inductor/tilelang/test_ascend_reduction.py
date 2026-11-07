"""Run inside an A5 npu-guard process with verified 32/64 core limits."""

import torch

from tilelang_test_utils import TileLangInductorTest, requires_tilelang


@requires_tilelang
class TestAscendReduction(TileLangInductorTest):
    def check_reduction(self, fn, x, *, atol=1e-4, rtol=1e-4):
        from torch._inductor.utils import run_and_get_code

        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "tilelang"})
        actual, codes = run_and_get_code(compiled, x)
        self.assert_all_kernels_are_tilelang(codes)
        source = "\n".join(codes)
        self.assertIn("T.SimdVF", source)
        self.assertNotIn("T.SimtVF", source)
        torch.testing.assert_close(actual, fn(x), atol=atol, rtol=rtol, equal_nan=True)
        other = x * 0.5
        torch.testing.assert_close(compiled(other), fn(other), atol=atol, rtol=rtol, equal_nan=True)
        return source

    def test_sum_and_max(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            tolerance = 1e-4 if dtype == torch.float32 else 2e-2
            for n in (63, 64, 65, 257, 2051, 4096):
                x = torch.randn((3, n), device="npu", dtype=dtype) * 0.1 - 0.5
                for kind, fn in (("sum", lambda a: a.sum(-1)), ("max", lambda a: a.amax(-1))):
                    with self.subTest(dtype=dtype, n=n, kind=kind):
                        self.check_reduction(fn, x, atol=tolerance, rtol=tolerance)

    def test_softmax(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for n in (63, 64, 65, 257, 2051):
                with self.subTest(dtype=dtype, n=n):
                    x = torch.randn((3, n), device="npu", dtype=dtype) * 3 + 20
                    source = self.check_reduction(lambda a: torch.softmax(a, -1), x, atol=2e-3, rtol=2e-2)
                    self.assertIn("T.simd.vcmax", source)
                    self.assertIn("T.simd.vcadd", source)

    def test_special_values(self):
        values = torch.full((4, 65), -1.0, device="npu")
        values[0, 64] = float("nan")
        values[1] = -float("inf")
        values[2, 1] = float("inf")
        for fn in (lambda a: a.sum(-1), lambda a: a.amax(-1), lambda a: torch.softmax(a, -1)):
            self.check_reduction(fn, values)


if __name__ == "__main__":
    import unittest
    unittest.main()
