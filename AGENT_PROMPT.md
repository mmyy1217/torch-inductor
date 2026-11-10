# 后续 agent

先读 README、community-source.json、SUPPORT 和 PERFORMANCE。

- 当前阶段：普通 TileLang codegen + dense 模板，D=128/256/512，MHA/GQA/MQA；完整契约见 SUPPORT。score/mask 留待后续，不把 TODO 误报为支持。
- 本仓仅放说明、用例、跑测工具。功能在独立源码 checkout，显式 `--source-repo` 创建 overlay；不修改安装包，不复制实现进本仓。
- 社区测试：`test/_inductor/test_tilelang_flex_attention.py` 检查 CPU 编译链；`test/_inductor/tilelang/test_flex_integration.py` 检查 NPU 数值及普通 codegen 集成。CPU/mock 通过不代表 NPU 正确。
- 三方性能统一用源码仓 bench.py；保留 shape、源码版本、工具链、正确性、计时样本；不能只报有利 shape。NPU event 图时间不是纯 kernel 时间。
- 上板遵守工作区 AGENTS.md 和设备授权；只用自己目录、低并发单内核 JIT，不做大型构建，不动他人进程。本机不运行 TileLang Huawei Ascend 编译链。
- 当前更新只允许个人 GitHub/GitCode；**不得推小组仓库**，用户验收后再决定。不得擅自创建/发布/重开 PR。
