# 当前支持范围：dense 前向

实现版本见 [community-source.json](community-source.json)。这版按实用 shape 扩展，不表示实现了某个完整模型。

| 项目 | 本版支持 |
| --- | --- |
| Q | `[B, HQ, M, D]` |
| K/V | `[B, HKV, N, D]`，两者 shape 相同 |
| D | **128、256、512**；不是任意 128 的倍数 |
| M/N | 各自为 128 的正倍数；可以相等，也可以不同 |
| MHA | `HQ = HKV` |
| GQA | `HQ % HKV = 0`，多个 Q head 共用一个 KV head |
| MQA | `HKV = 1`，所有 Q head 共用一组 KV |
| B/H | 正整数，没有额外固定上限；受张量元素数和设备内存限制 |
| dtype / 布局 | BF16、连续 rank-4，同一 NPU |
| 大小 | 每个输入少于 `2**31` 个元素 |
| scale | 有限静态值，默认 `1/sqrt(D)` |
| 执行 | 静态 shape、前向推理、Python wrapper；没有梯度的输入可用默认 grad mode |

- GQA/MQA 调用需加 `enable_gqa=True`，见 [gqa.py](cases/DONE/gqa.py)。第 h 个 Q head 使用第 `h // (HQ/HKV)` 个 KV head。
- D 是**每个 head 的维度**，不是模型 hidden size。Q 投影宽度通常为 HQ×D，KV 投影宽度为 HKV×D。
- M 是 Q 长度，N 是 KV 长度。这版覆盖对齐的 prefill/分块 Q 场景；`M=1` 的逐 token decode 不支持。
- 默认无 mask 是全可见 dense；它不等于自回归 causal attention。
- shape 变化可能重新编译；用 `dynamic=False`。B/H 无固定上限不代表所有组合都上板测过。

## 暂不支持

| 类别 | 后续范围 |
| --- | --- |
| callback | score_mod、mask_mod、BlockMask、captured tensor |
| shape | D=64/96/384 等、Dqk≠Dv、batch 广播、M/N 尾块、M=1 decode、动态 shape |
| 类型/布局 | FP16/FP32 输入、非连续输入 |
| 训练/接口 | backward、LSE/max 辅助输出、C++ wrapper、AOTInductor |
| 优化 | 模板前后融合、多候选 autotuning |

不支持的输入会报错，不会自动改走 Triton。不要传入 Triton 专用 tile/warps 选项。
后续先在同一 shape 范围扩展 score/mask，并分别验证正确性和性能。

## 跑测与排错

- [DONE](cases/DONE/)：dense、gqa、mqa；[TODO](cases/TODO/)：其余场景是待办，不是本版承诺。
- 看 [PERFORMANCE.md](PERFORMANCE.md) 的典型矩阵和结果；契约范围大于已测试集合。
- dtype/layout 错误先检查 BF16 和 `is_contiguous()`；shape 错误检查上述各条件。
- 编译/导入错误先核对 [SETUP.md](SETUP.md) 的匹配环境，不能把工具链不匹配当作 shape 不支持。
- 重建编译器后重启进程并换新 Inductor/TileLang 缓存目录，不需要手工设置 compiler revision。
