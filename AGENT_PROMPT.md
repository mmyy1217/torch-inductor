# 后续 agent

先读 README、community-source.json、PROGRESS 和 SUPPORT。

- 本仓仅放中文说明、用例和跑测工具。功能在独立源码 checkout，显式 `--source-repo` 创建 overlay；不修改安装包，不复制实现进本仓。
- 当前目标是普通 TileLang codegen 与 attention 模板共存。不要擅自开启模板融合、扩大算子范围或做性能优化。
- 社区测试入口：`test/_inductor/test_tilelang_flex_attention.py`（模板回归）、`test/_inductor/tilelang/test_flex_integration.py`（初始化、分派、融合边界和参数 ABI）。CPU/mock 通过不代表 NPU 正确。
- 上板先跑 mixed.py，再按修改范围跑 DONE、validate_tilelang_matrix.py 和 cold/warm 验收。保留结果、源码 hash、实际 wheel/TileLang/CANN 版本；不放宽容差或静默切回 Triton。
- A5-BLUE 只在自己账号目录操作，只用物理 0/1/2，启动前检查空闲。只允许低并发单内核 JIT，不做大型构建、不动他人进程；本机不运行 TileLang Ascend 编译。
- 普通 codegen 与 attention 模板有两个配置开关，统一 runner 已同时设置；换后端必须记录实际选择。
- 当前代码分支只保留一个功能 commit，用户 review 前不 push。没有针对具体仓库和 PR 的直接授权，不创建、发布或重开 PR。
