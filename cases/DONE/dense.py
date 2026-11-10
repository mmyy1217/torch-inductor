"""批量 runner 的 dense 用例定义；独立学习入口见根目录 first_dense.py。"""

CONFIG = {}


def build(c, device):
    """由 run.py 的 run_worker() 调用，返回 attention、mask_mod、score_mod。

    c 是 runner 合并后的配置；device 区分 NPU 执行与 CPU 参考回调构造。
    build 本身不接收 Q/K/V，不执行计算；返回的 attention 才接收张量。
    """
    import torch_npu._inductor  # noqa: F401 - Apply NPU patches before binding FlexAttention.
    from torch.nn.attention.flex_attention import flex_attention

    def attention(q, k, v):
        return flex_attention(q, k, v, enable_gqa=c["hq"] != c["hkv"])

    return attention, None, None


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run import main

    # 直接执行本文件也会进入批量 runner，只是默认 --case dense。
    raise SystemExit(main(default_case=Path(__file__).stem))
