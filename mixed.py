"""Run TileLang pointwise -> FlexAttention template -> pointwise without fusion."""

import re

import torch
import torch_npu._inductor  # noqa: F401 - Register NPU patches before binding FlexAttention.
from torch.nn.attention.flex_attention import flex_attention
from torch_npu._inductor import config


def attention_with_pointwise(q, k, v, residual):
    q = (q + 0.125) * 0.5
    return flex_attention(q, k, v) + residual


def main():
    from torch._inductor.utils import run_and_get_code

    torch.npu.set_device(0)
    torch.manual_seed(2026)
    config.flex_attention.implementation = "tilelang"
    compiled = torch.compile(
        attention_with_pointwise,
        backend="inductor",
        options={"npu_backend": "tilelang"},
        fullgraph=True,
        dynamic=False,
    )
    with torch.no_grad():
        for iteration in range(2):
            q, k, v, residual = [
                torch.randn(1, 4, 256, 128, device="npu", dtype=torch.bfloat16)
                for _ in range(4)
            ]
            if iteration == 0:
                actual, codes = run_and_get_code(compiled, q, k, v, residual)
                source = "\n".join(codes)
                assert "compile_tilelang(" in source and ".launch(" in source
                assert source.count("def _prim_factory_tilelang") >= 2
                # Integer BlockMask metadata may use the existing Triton fallback.
                triton_names = re.findall(
                    r"async_compile\.triton\(['\"]([^'\"]+)", source
                )
                assert all(
                    re.fullmatch(r"triton_poi_fused_(ones|zeros)_\d+", name)
                    for name in triton_names
                ), triton_names
                assert "extern_kernels." not in source
            else:
                actual = compiled(q, k, v, residual)
            # Round the pointwise result to BF16 at the template input boundary.
            q_ref = ((q.cpu().float() + 0.125) * 0.5).to(q.dtype).float()
            k_ref, v_ref = k.cpu().float(), v.cpu().float()
            scores = (q_ref @ k_ref.transpose(-2, -1)) / q.shape[-1] ** 0.5
            attention = (scores.softmax(-1) @ v_ref).to(q.dtype)
            expected = (attention.float() + residual.cpu().float()).to(q.dtype)
            torch.testing.assert_close(actual.cpu(), expected, atol=0.03, rtol=0.03)
    print("PASS: pointwise and template kernels coexist; new inputs reuse the graph.")


if __name__ == "__main__":
    main()
