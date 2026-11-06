"""A5 end-to-end cases; run inside the server's guarded, capped device process."""

import torch

from tilelang_test_utils import TileLangInductorTest, requires_tilelang


@requires_tilelang
class TestAscendPointwise(TileLangInductorTest):
    def test_backend_switch_restores_foreach(self):
        import os
        from unittest.mock import patch

        from torch._inductor.scheduler import Scheduler
        from torch_npu._inductor import _load_backend

        try:
            with patch.dict(os.environ, {"TORCHINDUCTOR_NPU_BACKEND": "default"}):
                _load_backend()
                baseline = Scheduler.create_foreach_nodes
            with patch.dict(os.environ, {"TORCHINDUCTOR_NPU_BACKEND": "tilelang"}):
                _load_backend()
                self.assertIsNot(Scheduler.create_foreach_nodes, baseline)
            with patch.dict(os.environ, {"TORCHINDUCTOR_NPU_BACKEND": "default"}):
                _load_backend()
                self.assertIs(Scheduler.create_foreach_nodes, baseline)
        finally:
            _load_backend()

    def test_fused_chain_and_tails(self):
        def fn(x, y):
            return torch.exp((x + y) * 0.25)

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for size in (1, 64, 257, 2051):
                with self.subTest(dtype=dtype, size=size):
                    x = torch.randn(size, device="npu", dtype=dtype) * 0.1
                    y = torch.randn_like(x) * 0.1
                    self.check_a5(fn, (x, y))

    def test_multiple_outputs(self):
        def fn(x, y):
            z = x + y
            return z, torch.exp(z * 0.25)

        self.check_a5(fn, (torch.randn(257, device="npu"), torch.randn(257, device="npu")))

    def test_scalar_broadcast(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check_a5(
                    lambda x, y: torch.exp((x + y) * 0.25),
                    (torch.randn(257, device="npu", dtype=dtype), torch.randn(1, device="npu", dtype=dtype)),
                )

    def check_a5(self, fn, args):
        from torch._inductor.utils import run_and_get_code

        torch._dynamo.reset()
        expected = fn(*args)
        compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "tilelang"})
        actual, codes = run_and_get_code(compiled, *args)
        self.assert_all_kernels_are_tilelang(codes)
        source = "\n".join(codes)
        self.assertIn("tilelang.ascend.language", source)
        self.assertIn("T.SimdVF", source)
        self.assertNotIn("T.SimtVF", source)
        self.assertIn("T.simd.vld", source)
        self.assertIn("T.simd.vsts", source)
        self.assertIn("target='ascend'", source)
        self.assertNotIn("target='npuir'", source)
        rtol, atol = {
            torch.float32: (1e-4, 1e-5),
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
        }[args[0].dtype]
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        # Reuse compiled code with different inputs, exercising the cached launcher.
        second_args = tuple(value * 0.5 for value in args)
        torch.testing.assert_close(compiled(*second_args), fn(*second_args), rtol=rtol, atol=atol)


if __name__ == "__main__":
    import unittest

    unittest.main()
