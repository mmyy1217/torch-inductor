"""从头到尾的 dense 示例：先按 SETUP.md 准备环境和 overlay，再执行本文件。"""

import torch
import torch_npu._inductor  # noqa: F401 - 先注册 NPU lowering，再导入 flex_attention。
from torch.nn.attention.flex_attention import flex_attention
from torch_npu._inductor import config


# 1. 用户写的计算：默认 score、无 mask，就是普通 dense attention。
def attention(q, k, v):
    return flex_attention(q, k, v)


def main():
    # 2. 选择已分配的可见设备；这里的 0 是可见设备中的逻辑编号。
    torch.npu.set_device(0)
    torch.manual_seed(2026)
    q = torch.randn(1, 4, 256, 128, device="npu:0", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # 3. 选择 TileLang 模板；与下面的 npu_backend 一起决定模板和普通 codegen。
    config.flex_attention.implementation = "tilelang"

    # 4. 创建编译后的函数；backend 仍是 inductor，不是 tilelang。
    compiled_attention = torch.compile(
        attention,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"npu_backend": "tilelang"},
    )
    with torch.no_grad():
        # 5. 第一次调用触发 FX 捕获、lowering、模板代码生成、编译和 NPU 执行。
        actual = compiled_attention(q, k, v)

        # 6. 用 CPU FP32 公式核对结果：softmax(Q @ K.T / sqrt(D)) @ V。
        q_cpu, k_cpu, v_cpu = (x.cpu().float() for x in (q, k, v))
        scores = (q_cpu @ k_cpu.transpose(-2, -1)) / q.shape[-1] ** 0.5
        expected = torch.softmax(scores, dim=-1) @ v_cpu
        torch.testing.assert_close(actual.cpu().float(), expected, atol=0.03, rtol=0.03)
    print("PASS: dense attention matches the FP32 reference.")


if __name__ == "__main__":
    main()
