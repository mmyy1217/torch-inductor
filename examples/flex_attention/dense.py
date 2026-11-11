"""Run default dense FlexAttention with the TileLang template."""

import torch
import torch_npu._inductor  # noqa: F401 - Patch FlexAttention before importing it.
from torch.nn.attention.flex_attention import flex_attention
from torch_npu._inductor import config


def dense_attention(q, k, v):
    return flex_attention(q, k, v)


def main():
    torch.npu.set_device(0)
    config.flex_attention.implementation = "tilelang"
    compiled = torch.compile(
        dense_attention,
        fullgraph=True,
        dynamic=False,
        options={"npu_backend": "tilelang"},
    )
    q, k, v = [
        torch.randn(1, 4, 256, 128, device="npu", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    with torch.no_grad():
        out = compiled(q, k, v)
    print("Output:", out.shape, out.dtype)
    print(out[0, 0, 0, :8].cpu())


if __name__ == "__main__":
    main()
