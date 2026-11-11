"""Run TileLang pointwise -> FlexAttention template -> pointwise."""

import torch
import torch_npu._inductor  # noqa: F401 - Register NPU patches before FlexAttention.
from torch.nn.attention.flex_attention import flex_attention
from torch_npu._inductor import config


def attention_with_pointwise(q, k, v, residual):
    q = (q + 0.125) * 0.5
    return flex_attention(q, k, v) + residual


def main():
    torch.npu.set_device(0)

    # Select the attention template and ordinary TileLang codegen separately.
    config.flex_attention.implementation = "tilelang"
    compiled = torch.compile(
        attention_with_pointwise,
        backend="inductor",
        options={"npu_backend": "tilelang"},
        fullgraph=True,
        dynamic=False,
    )
    q, k, v, residual = [
        torch.randn(1, 4, 256, 128, device="npu", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    with torch.no_grad():
        output = compiled(q, k, v, residual)
    print("Output:", output.shape, output.dtype)
    print(output[0, 0, 0, :8].cpu())


if __name__ == "__main__":
    main()
