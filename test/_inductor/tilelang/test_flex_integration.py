"""Run in a fresh process with matching torch_npu and Triton-Ascend installed.

Initialization is checked on CPU with mocked device properties. The mixed-graph
test requires an allocated NPU and executes the example against a CPU reference.
"""

import os
from pathlib import Path
import runpy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TORCH_WARM_POOL"] = "0"

import torch
import torch_npu  # noqa: F401 - Register the NPU device interface.


class TestFlexInitialization(unittest.TestCase):
    def test_cold_tilelang_and_backend_switches(self):
        """Keep lowering registration and template dispatch correct across backend switches."""
        with (
            patch.dict(os.environ, {"TORCHINDUCTOR_NPU_BACKEND": "tilelang"}),
            patch("torch.utils._triton.has_triton", return_value=True),
            patch("torch_npu.npu._backends.get_soc_version", return_value=260),
            patch("torch_npu.npu.current_device", return_value=0),
            patch(
                "torch_npu.npu.get_device_properties",
                return_value=SimpleNamespace(cube_core_num=28, vector_core_num=56),
            ),
            patch(
                "torch_npu.npu._lazy_init",
                side_effect=AssertionError(
                    "Device initialization is forbidden in this test"
                ),
            ),
        ):
            import torch_npu._inductor as inductor
            from torch._inductor.lowering import lowerings
            from torch_npu._inductor import config

            original_choice = config.flex_attention.implementation
            for backend in ("tilelang", "default", "tilelang", "default"):
                os.environ["TORCHINDUCTOR_NPU_BACKEND"] = backend
                inductor._load_backend()
                handler = lowerings[torch.ops.higher_order.flex_attention]
                self.assertEqual(
                    handler.__module__, "torch_npu._inductor.kernel.flex_attention"
                )
                self.assertEqual(config.flex_attention.implementation, original_choice)
                if backend == "default":
                    with patch.object(
                        config.flex_attention, "implementation", "tilelang"
                    ):
                        with self.assertRaisesRegex(
                            NotImplementedError, "requires npu_backend"
                        ):
                            handler(
                                SimpleNamespace(
                                    get_device=lambda: torch.device("npu", 0)
                                ),
                                *([None] * 8),
                            )
            from torch_npu._inductor.tilelang.codegen import (
                tilelang_scheduling as scheduling,
            )
            from torch_npu._inductor.codegen.tilelang import TileLangTemplateBuffer
            from unittest.mock import Mock

            # Templates must reach their renderer and remain a fusion boundary.
            scheduler = object.__new__(scheduling.TileLangScheduling)
            scheduler._triton_scheduling = Mock()
            template = Mock()
            template.is_template.return_value = True
            template.get_template_node.return_value = object.__new__(
                TileLangTemplateBuffer
            )
            template.get_nodes.return_value = [template]
            pointwise = Mock()
            pointwise.is_template.return_value = False
            pointwise.get_nodes.return_value = [pointwise]
            for left, right in ((template, pointwise), (pointwise, template)):
                self.assertFalse(scheduler.can_fuse_vertical(left, right))
                self.assertFalse(scheduler.can_fuse_horizontal(left, right))
            with patch.object(scheduling, "codegen_tilelang_template") as render:
                scheduler.codegen_template(template, [], [])
                render.assert_called_once_with(scheduler, template, [], [])
                scheduler._triton_scheduling.codegen_template.assert_not_called()
            with self.assertRaises(NotImplementedError):
                scheduler.codegen_template(template, [], only_gen_src_code=True)
            with self.assertRaisesRegex(NotImplementedError, "template fusion"):
                scheduler.codegen_template(template, [], [pointwise])
            scheduler.codegen_template(pointwise, [])
            scheduler._triton_scheduling.codegen_template.assert_called_once_with(
                pointwise, [], only_gen_src_code=False
            )


class TestMixedGraph(unittest.TestCase):
    @unittest.skipUnless(
        torch.npu.is_available(), "requires an allocated NPU and TileLang"
    )
    def test_pointwise_attention_pointwise(self):
        """Run the example on NPU; catch rank-4 launcher errors and wrong mixed-graph outputs."""
        example = (
            Path(__file__).resolve().parents[3] / "examples/flex_attention/mixed.py"
        )
        attention_with_pointwise = runpy.run_path(str(example))[
            "attention_with_pointwise"
        ]
        from torch_npu._inductor import config

        with (
            patch.object(config.flex_attention, "implementation", "tilelang"),
            torch.no_grad(),
        ):
            compiled = torch.compile(
                attention_with_pointwise,
                fullgraph=True,
                dynamic=False,
                options={"npu_backend": "tilelang"},
            )
            # New inputs on the second call also check reuse of the compiled graph.
            for _ in range(2):
                q, k, v, residual = [
                    torch.randn(1, 4, 256, 128, device="npu", dtype=torch.bfloat16)
                    for _ in range(4)
                ]
                actual = compiled(q, k, v, residual)
                # Match BF16 rounding at the two kernel boundaries in the example.
                q_ref = ((q.cpu().float() + 0.125) * 0.5).to(q.dtype).float()
                scores = q_ref @ k.cpu().float().transpose(-2, -1) / 128**0.5
                attention = (scores.softmax(-1) @ v.cpu().float()).to(q.dtype)
                expected = (attention.float() + residual.cpu().float()).to(q.dtype)
                torch.testing.assert_close(actual.cpu(), expected, atol=0.03, rtol=0.03)

    @unittest.skipUnless(
        torch.npu.is_available(), "requires an allocated NPU and TileLang"
    )
    def test_dense_boundaries(self):
        """Check single/three KV tiles, unequal lengths, scale and grad-enabled inference."""
        import torch_npu._inductor  # noqa: F401 - Patch before binding FlexAttention.
        from torch.nn.attention.flex_attention import flex_attention, AuxRequest
        from torch_npu._inductor import config

        with (
            patch.object(config.flex_attention, "implementation", "tilelang"),
            torch.enable_grad(),
        ):
            for b, h, hk, m, n, d in (
                (2, 2, 2, 256, 128, 128),
                (1, 3, 3, 128, 384, 128),
                (2, 8, 2, 256, 128, 256),
                (2, 4, 1, 128, 384, 512),
            ):
                with self.subTest(shape=(b, h, hk, m, n, d)):
                    compiled = torch.compile(
                        lambda q, k, v: flex_attention(
                            q, k, v, scale=0.125, enable_gqa=True
                        ),
                        fullgraph=True,
                        dynamic=False,
                        options={"npu_backend": "tilelang"},
                    )
                    for _ in range(2):
                        q = torch.randn(b, h, m, d, device="npu", dtype=torch.bfloat16)
                        k = torch.randn(b, hk, n, d, device="npu", dtype=q.dtype)
                        v = torch.randn_like(k)
                        k_ref = k.cpu().float().repeat_interleave(h // hk, dim=1)
                        v_ref = v.cpu().float().repeat_interleave(h // hk, dim=1)
                        scores = (q.cpu().float() @ k_ref.transpose(-2, -1)) * 0.125
                        expected = (scores.softmax(-1) @ v_ref).to(q.dtype)
                        torch.testing.assert_close(
                            compiled(q, k, v).cpu(), expected, atol=0.03, rtol=0.03
                        )
            requested_lse = torch.compile(
                lambda q, k, v: flex_attention(
                    q, k, v, enable_gqa=True, return_aux=AuxRequest(lse=True)
                ),
                fullgraph=True,
                options={"npu_backend": "tilelang"},
            )
            with self.assertRaisesRegex(
                Exception, "auxiliary outputs are not supported"
            ):
                requested_lse(q, k, v)


if __name__ == "__main__":
    unittest.main()
