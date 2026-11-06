import importlib.util
import os
import unittest

os.environ["INDUCTOR_ASCEND_TILELANG_AUTOTUNE"] = "0"

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code


TILELANG_AVAILABLE = (
    torch_npu.npu.is_available()
    and importlib.util.find_spec("tilelang") is not None
)
requires_tilelang = unittest.skipUnless(
    TILELANG_AVAILABLE,
    "TileLang and an available NPU are required",
)


class TileLangInductorTest(unittest.TestCase):
    rtol = 1e-2
    atol = 1e-2
    force_disable_caches = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._old_autotune = os.environ.get(
            "INDUCTOR_ASCEND_TILELANG_AUTOTUNE"
        )
        cls._old_force_disable_caches = (
            torch._inductor.config.force_disable_caches
        )
        os.environ["INDUCTOR_ASCEND_TILELANG_AUTOTUNE"] = "0"
        torch._inductor.config.force_disable_caches = cls.force_disable_caches

    @classmethod
    def tearDownClass(cls):
        torch._inductor.config.force_disable_caches = (
            cls._old_force_disable_caches
        )
        if cls._old_autotune is None:
            os.environ.pop("INDUCTOR_ASCEND_TILELANG_AUTOTUNE", None)
        else:
            os.environ["INDUCTOR_ASCEND_TILELANG_AUTOTUNE"] = cls._old_autotune
        torch._dynamo.reset()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def assert_all_kernels_are_tilelang(self, generated_codes):
        self.assertTrue(
            generated_codes,
            "Inductor did not return generated code for the compiled graph",
        )

        full_code = "\n".join(generated_codes)
        tilelang_kernel_count = (
            full_code.count("# TileLang kernel")
            + full_code.count("# TileLang T.gemm kernel")
        )
        tilelang_factory_count = full_code.count("def _prim_factory_tilelang")
        triton_kernel_count = full_code.count("async_compile.triton")
        direct_kernel_lines = [
            line.strip()
            for line in full_code.splitlines()
            if not line.lstrip().startswith("#")
            and (
                "extern_kernels." in line
                or "torch.ops.aten." in line
                or "torch.ops.npu." in line
            )
        ]
        relevant_lines = "\n".join(
            line
            for line in full_code.splitlines()
            if "# TileLang" in line
            or "def _prim_factory_tilelang" in line
            or "async_compile.triton" in line
            or "extern_kernels." in line
            or "torch.ops.aten." in line
            or "torch.ops.npu." in line
        )
        if not relevant_lines:
            excerpt_size = 2000
            if len(full_code) <= excerpt_size * 2:
                relevant_lines = full_code
            else:
                relevant_lines = (
                    full_code[:excerpt_size]
                    + "\n... generated code omitted ...\n"
                    + full_code[-excerpt_size:]
                )

        self.assertEqual(
            triton_kernel_count,
            0,
            "expected every generated kernel to use TileLang, but found "
            f"{triton_kernel_count} NPU Triton kernel(s):\n{relevant_lines}",
        )
        self.assertFalse(
            direct_kernel_lines,
            "expected every generated kernel to use TileLang, but found "
            "direct ATen/NPU/extern kernel call(s):\n"
            + "\n".join(direct_kernel_lines),
        )
        self.assertGreater(
            tilelang_kernel_count,
            0,
            "expected at least one generated TileLang kernel, but found none:\n"
            f"{relevant_lines}",
        )
        self.assertEqual(
            tilelang_factory_count,
            tilelang_kernel_count,
            "TileLang kernel comments and prim-func factories do not match:\n"
            f"{relevant_lines}",
        )
        self.assertIn(
            "target='ascend'",
            full_code,
            "generated TileLang kernel does not compile for the Ascend target:\n"
            f"{relevant_lines}",
        )

    def assert_compiled_close(
        self,
        fn,
        *inputs,
        dynamic=False,
        rtol=None,
        atol=None,
    ):
        actual, _ = self.assert_compiled_close_and_get_code(
            fn,
            *inputs,
            dynamic=dynamic,
            rtol=rtol,
            atol=atol,
        )
        return actual

    def assert_compiled_close_and_get_code(
        self,
        fn,
        *inputs,
        dynamic=False,
        rtol=None,
        atol=None,
    ):
        torch._dynamo.reset()
        expected = fn(*inputs)
        compiled = torch.compile(
            fn,
            backend="inductor",
            dynamic=dynamic,
            options={"npu_backend": "tilelang"},
        )
        actual, generated_codes = run_and_get_code(compiled, *inputs)
        torch.npu.synchronize()
        self.assert_all_kernels_are_tilelang(generated_codes)
        if expected.dtype != actual.dtype:
            expected = expected.to(actual.dtype)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=self.rtol if rtol is None else rtol,
            atol=self.atol if atol is None else atol,
        )
        return actual, generated_codes
