"""返回输出和掩码后、softmax 前的最大分数。"""

CONFIG = {"aux": "max_scores"}


def build(c, device):
    import torch_npu._inductor  # noqa: F401 - Apply NPU patches before binding FlexAttention.
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    from torch.nn.attention.flex_attention import AuxRequest

    def mask_mod(b, h, m, n):
        return m >= n

    score_mod = None

    # CPU build supplies callbacks for the independent dense reference only.
    block_mask = None
    if mask_mod is not None and device != "cpu":
        block_mask = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=c["sq"], KV_LEN=c["sk"], device=device
        )

    def attention(q, k, v):
        return flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            score_mod=score_mod,
            enable_gqa=c["hq"] != c["hkv"],
            return_aux=AuxRequest(max_scores=True),
        )

    return attention, mask_mod, score_mod


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run import main

    raise SystemExit(main(default_case=Path(__file__).stem))
