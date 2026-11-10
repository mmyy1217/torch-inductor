# 当前进展

源码版本见 community-source.json。本地分支 `pytorch-tilelang-flex-integration`，基于同事的 `tilelang-backend-a5`，尚未推送。

- 旧模板的七个核心文件已完整迁入：lowering、FX 翻译、模板 IR、源码渲染、内核、runtime、缓存。
- 普通 TileLang codegen 与 attention 模板已接通，模板前后融合关闭。
- 修复普通线性 kernel 接收四维 tensor 时的 FFI 维度不匹配，使用共享存储的 view。
- 2026-11-10 A5-BLUE：BF16 `[1,4,256,128]` 的 add/mul → attention → residual 混合图通过，两组不同输入复用同一 compiled callable；对比 CPU FP32 attention，atol=rtol=0.03。
- 生成三个 TileLang 计算 kernel、两个 Triton 整数 BlockMask 元数据 kernel。
- 精简后保留 32 项模板回归、2 项后端集成检查；同事原有的 emitter/cast 本地 15 项此前通过。

未验证：当前集成分支的完整 DONE/稀疏矩阵、跨进程 cold/warm 全套、性能。旧模板的既有覆盖不能替代集成版本重新验收。DONE 表示实现支持，不能理解成这轮全部复测。
