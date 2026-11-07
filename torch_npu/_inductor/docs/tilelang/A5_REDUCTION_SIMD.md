# A5 reduction SIMD lowering

2026-11-07。开发分支 `tilelang-backend-a5`。官方 TileLang 固定为
`64b52b4b1a505ba93e6e49a35cc0c5dc2a4b020d`，与已有 pointwise 验证环境一致。

## 当前实现

| 层 | 输入 | 处理与输出 |
| --- | --- | --- |
| Inductor lowering | sum / amax / softmax FX 节点 | 保留 reduction dtype 和 `strict_sum`，生成 sum/max reduction 节点及逐元素表达式 |
| TileLang scheduling | reduction 节点、输出行数 X、归约长度 R | 静态 `1 <= R <= 4096`；一个 block 处理一行，R tile 向上对齐到 64，buffer versions 固定为 1 |
| Region planner | 连续末轴 load/store、reduction 与逐元素依赖 | 沿用 reduction region；softmax 分为 max、exp/sum、exp/div 三段，可重算表达式和重读输入 |
| SIMD emitter | UB tile、有效长度 `_remain_R`、sum/max 类型 | `T.SimdVF` 内按 64 个 FP32 lane 累积，再用 `vcadd` / `vcmax` 跨 lane 归约；单元素写回 UB |
| 官方编译器 | Ascend PrimFunc，显式 SIMD VF、copy、buffer versions | 自动调度搬运与同步，生成 native kernel；wrapper 传入 Inductor 分配的输出 |

FP16/BF16 输入先扩展到 FP32；完整行累加结束后再按结果 dtype 转换。
无效尾 lane 以 sum 的 0 或 max 的负无穷替换，不能使用纯 pointwise 的尾处理规则。
max 显式检测并传播任意有效 lane 的 NaN。scalar 输出采用 ONEPT store，避免完整向量写越界。
softmax 使用减最大值的稳定公式；全负无穷行等特殊值以 PyTorch eager 为参照。

首版不覆盖动态长度、R > 4096、非连续/转置/ranked 布局、argmax 返回索引、gather、分段归约和多 buffer 调优。
不支持的情况会拒绝 TileLang 路径；已有 Triton fallback 尚未通过该隔离环境验收，不能保证兜底运行。
本轮没有性能结论，也没有引入 SIMT 路由。

## 验证状态

2026-11-07 16:53–17:03 CST，在 A5 上完成 **54/54** 计算用例；3 个测试方法，0 failures / 0 errors / 0 skipped。实际运行源码为 `24e421a6a4081d11462844c4d3862cc220532f47`，后续仅合入文档与证据，Python overlay 的逐文件 SHA256 已与该提交核对。

| 用例 | dtype | Shape / 内容 | 数量 |
| --- | --- | --- | --- |
| sum / amax（仅最大值） | FP32 / FP16 / BF16 | `[3, R]`，R = 63、64、65、257、2051、4096 | 36 |
| softmax | FP32 / FP16 / BF16 | `[3, R]`，R = 63、64、65、257、2051 | 15 |
| sum / amax / softmax 特殊值 | FP32 | `[4, 65]`，有效尾元素 NaN、全 −∞、+∞、负数行 | 3 |

每例先与 PyTorch eager 对比，再更换输入复用已编译 callable。随机种子为 0。sum/max 的 FP32 atol/rtol 均为 `1e-4`，FP16/BF16 均为 `2e-2`；softmax 为 `atol=2e-3, rtol=2e-2`；特殊值使用 `equal_nan=True`。验收不等同于位级一致。

54 份 wrapper 均只有一个 TileLang kernel，禁止 Triton/ATen fallback；55 份 native source（含直接 pointwise smoke）均含 `__simd_vf__`，均不含 `__simt_vf__`。每份 native source 只有一个 global kernel。softmax 的 max/sum/逐元素阶段在同一次 kernel launch 内：FP32 为 4 段 VF，低精度为 5 段 VF（多一段输入转换），尚未融合成单个 VF。官方编译器负责阶段间搬运和同步，当前仍重读输入三次。

全过程受 `npu-guard` 保护，读回 32 AIC/64 AIV，退出码 0，postflight 空闲，无冲突或可见性丢失。本轮仅做正确性与生成代码验收，没有性能或 profiler 结论。

本地 15/15 通过：13 个 emitter 语义测试覆盖被 NaN 污染的物理尾部、归约边界和单元素写回；2 个 cast lowering 回归测试覆盖 CSE 提升 dtype 后的最终窄化与窄化/扩展链。这些本地测试独立于设备验收。

## 本轮修复

- 适配 PyTorch 2.14 `strict_sum` 参数，按轴列表计算 reduction 总长度。
- 修复 sum/max 输出的 FFI 维数不匹配：wrapper 为 caller-owned 输出传递 `[X, 1]` view，保留 Inductor 原输出 shape 和 storage。
- 修复低精度 softmax 的 buffer dtype 冲突：CSE 可报告 FP32 计算 dtype，cast lowering 必须保留显式请求的 FP16/BF16 dtype，不能因此消掉输出转换或窄化/扩展精度边界。

首轮全量记录 `evidence-reduction-01/` 为 44/54；修复后的低精度专项 `evidence-reduction-smoke-08/` 为 2/2；最终全量 `evidence-reduction-02/` 为 54/54。早期断线/占用检查与失败记录保留，不用于通过结论。

## 证据与复现

结构化报告：[a5_reduction_simd_validation.json](evidence/a5_reduction_simd_validation.json)，包含每例 shape/dtype/结果、wrapper 哈希、55 份 native source 哈希、源码/overlay 和 guard 完成记录。
选定源码：[FP32 softmax R=2051](evidence/a5_reduction_softmax_fp32_2051.cpp)、[BF16 softmax R=2051](evidence/a5_reduction_softmax_bf16_2051.cpp)、[对应 PrimFunc](evidence/a5_reduction_softmax_bf16_2051.tir)、[FP16 sum R=4096](evidence/a5_reduction_sum_fp16_4096.cpp)、[特殊值 max](evidence/a5_reduction_max_special_fp32.cpp)。

远端根目录 `/home/yja/experiments/tilelang-backend-a5-20261106`；完整记录由
本轮原样脚本 [run-reduction-validation.sh](evidence/a5_reduction_runner/run-reduction-validation.sh) / [.py](evidence/a5_reduction_runner/run-reduction-validation.py) 生成。使用前须同步源码并完成 Python overlay；这些脚本固定使用上述隔离环境。脚本实时检查状态，整个进程受 `npu-guard` 保护，初始化后设置并读回 32 AIC/64 AIV。用户于 2026-11-07 明确允许在忙碌时进入 guard 队列等待；设置 `A5_ALLOW_GUARD_QUEUE=1` 后，由 guard 再确认空闲并取得锁才启动程序。
设置 `A5_REDUCTION_SMOKE=1` 跑三个 FP32 smoke，`lowprecision` 跑两个低精度 softmax；取消该变量跑完整矩阵。最终调用脚本见 [full-reduction.sh](evidence/a5_reduction_runner/full-reduction.sh)。
用新的 `A5_REDUCTION_EVIDENCE` 目录区分重试，保留 `environment.json`、Python overlay 哈希、`results.json`、PrimFunc 与 native source。
