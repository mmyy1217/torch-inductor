"""8 个 Query head 共享 1 个 KV head。"""

CONFIG = {"hq": 8, "hkv": 1}


def build(c, device):
    import torch_npu._inductor  # noqa: F401 - Apply NPU patches before binding FlexAttention.
    from torch.nn.attention.flex_attention import flex_attention

    def attention(q, k, v):
        return flex_attention(q, k, v, enable_gqa=True)

    return attention, None, None


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run import main

    raise SystemExit(main(default_case=Path(__file__).stem))
