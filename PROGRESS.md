# 当前进展

精确社区实现提交见 [community-source.json](community-source.json)。本次在已有集成提交 `70ea2d0ca5` 上追加一个 shape 扩展提交，保留同事历史。只更新个人 GitCode/GitHub，小组分支未更新。

| 本轮交付 | 状态 |
| --- | --- |
| BF16 dense，D=128/256/512 | 已实现 |
| MHA/GQA/MQA，同 batch，M/N 各为 128 的正倍数 | 已实现；完整边界见 SUPPORT |
| 大 D 分块与缓冲区优化 | 已保留上板验证方案；不启用 autotuning |
| 三方性能矩阵 | 26 shapes × 3 backends × 2 rounds，156/156 正确性通过 |
| CPU 编译链回归 | 9 tests 通过，设备编译 mocked |
| 集成回归 | 3 tests 通过，其中 2 项实际执行 NPU：混合图及参数化 dense |
| 特殊 scale | D=512 MQA scale=0、D=256 GQA scale=-0.125 上板通过 |
| 冷/热缓存 | 冷编译与跨进程 FX cache replay 通过；热阶段不重新渲染模板 |
| 个人 DONE 用例 | dense/gqa/mqa，B=2、D=512，全部上板通过，已保存 IR/wrapper |
| 本仓工具测试 | 5 tests 通过：用例发现、路由开关、overlay、旧 graph CLI 拒绝行为 |

**性能结论：** TileLang 对 Triton 的汇总加速 1.12–4.56×，但所有已测 shape 仍慢于 torch_npu 原生 API。详见 [PERFORMANCE.md](PERFORMANCE.md)，包含不利结果和波动限制。

**后续而非本版能力：** score/mask、captured tensor、尾块、decode、其他 dtype/head dim、backward、辅助输出、动态 shape、模板融合。旧稀疏矩阵脚本已移除，原 mask/score 用例移到 TODO；不再保留旧支持承诺。
