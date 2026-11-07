# A5 pointwise 改为 SIMD VF

2026-11-06，官方 TileLang 固定为 `64b52b4b1a505ba93e6e49a35cc0c5dc2a4b020d`。

## 路线选择

首批已验收的 pointwise 原来使用 `SimtVF(threads=128)`。现在改为 `SimdVF()` 内显式向量指令；不是简单替换上下文名称。官方 `AscendSimdVFLowerParallel` 在这个固定版本中的表达式分解没有覆盖当前 typed plan 的全部变量绑定与 cast，因此暂不依赖高层 Parallel 自动向量化。

| 输入 | 本层处理 | 输出 |
| --- | --- | --- |
| Inductor typed pointwise operations、buffer dtype/shape | 保留 load、运算、execution/result dtype 和输出边界 | 已校验的 pointwise plan |
| GM tensor 与有效 tile 范围 | 原有 `T.copy` DMA，保持有效长度 | 连续 UB 输入 tile；scalar 输入只有一个元素 |
| UB tile、运算图 | `T.SimdVF()`，每轮 64 个逻辑元素；显式 vld/vdup/vadd/vmul/vexp/vcvt/vsts | UB 边界输出，中间结果是向量寄存器 |
| UB 输出与逻辑有效长度 | 原有 DMA，只写回有效元素 | caller-owned 输出 tensor |

没有启用 Triton gather 标注到 TileLang SIMT 的路由。旧 `INDEXED_LOAD` 是索引化广播搬运，也不等同于完整 gather 实现；现在遇到它会明确抛出 `TileLangUnsupported`。调度器原有 Triton fallback 仍独立存在，本次并未修复或验收该 fallback 环境。当前验收用例严格要求不回退。

后续若支持 gather/scatter 等非规则访问，应先建立明确的访问模式和能力选择，再独立增加 SIMT emitter；不能让任意索引表达式默默生成 SIMT VF。部分索引访问也可能用 SIMD 指令实现，具体路线应由操作、布局和硬件能力决定。

## 向量布局与精度

- 寄存器计算以 64 个 FP32 元素为一组。FP32 连续输入用 `NORM` load，FP16/BF16 用 `UNPK_B16` 后转换为 FP32。
- scalar 输入用 `BRC_B32` / `BRC_B16`，只读取一个 UB 元素。FP16/BF16 输出用非饱和、最近偶数转换及 `PK_B32` store。
- 保留 typed plan 的低精度舍入边界：operand 转换、运算结果舍入和 result dtype 转换分开处理。例如 half 乘法随后转 FP32，必须先得到 half 舍入结果，不能直接保留完整 FP32 乘积。
- 当前转换采用保守的 narrow/widen 方式；可能存在冗余 cast，未做性能或指令数优化。不能仅因切换为 SIMD 就声称加速。
- 物理 tile 行宽必须是 64 的正整数倍。固定默认宽度为 128；factory 检查运行时 tile 参数，拒绝 0、32、65 等不满足完整向量装载约束的值。
- SIMD 处理完整物理 tile，尾块只从 GM 搬运/写回逻辑有效元素；无效 lane 的结果不外传。这个约定仅适用于当前逐元素操作，不能直接用于跨 lane reduction。
- 多维 broadcast 的纯 emitter 语义有本地覆盖，完整 Inductor ranked broadcast、stride、动态 shape、CV/Attention 尚未验收。2026-11-07 的 sum/max/softmax reduction 已完成 A5 54/54 验收，见 [A5_REDUCTION_SIMD.md](A5_REDUCTION_SIMD.md)。

## 验证

最终实现源码 `c2f510f2542a6c9976c95b344aed6a5b2600672c` 在 A5 上通过：

- 4 个测试方法，0 failures / 0 errors / 0 skipped。
- FP32/FP16/BF16 × 长度 1/64/257/2051：12 例；FP32 多输出：1 例；三种 dtype 的向量加 scalar 广播：3 例。共 16 个端到端计算用例，每例更换输入复用缓存 callable。
- 直接 PrimFunc 与 caller-owned 输出、后端切换和 foreach 恢复通过。
- 17 份 native source（含直接 smoke）全部包含 `__simd_vf__`，全部不含 `__simt_vf__`；wrapper 检查没有 Triton/ATen fallback。
- 全过程由 `npu-guard` 保护，读回 32 AIC/64 AIV 限核，前台进程退出码为 0。

首轮源码 `36b2710017c0a3afb274ad484b543a0db7c2a866` 已通过上述全部用例；补入 factory tile 宽度保护后，于 2026-11-06 17:09–17:13 CST 对最终实现再次完成全量验收。本地 9/9 通过，选定 FP32/BF16 native source 和 PrimFunc 与首轮逐字节一致。源码 SHA 指验收时的提交，后续仅合入文档与证据。

结构化结果：[a5_simd_validation.json](evidence/a5_simd_validation.json)。选定源码：[FP32 size=2051](evidence/a5_simd_fp32_2051.cpp)、[BF16 scalar broadcast](evidence/a5_simd_scalar_broadcast_bf16_257.cpp)、[对应 PrimFunc](evidence/a5_simd_scalar_broadcast_bf16_257.tir)。FP32 native 中可见 `vlds_norm → vadd → vmul → vexp → vsts_norm`；BF16 可见 unpack、broadcast、cast 和 pack。MTE2→V→MTE3 同步仍由官方编译器处理。

服务器根目录 `/home/yja/experiments/tilelang-backend-a5-20261106`，最终完整成功记录位于 `evidence-simd-02/`（首轮为 `evidence-simd-01/`）。复测脚本 `run-simd-validation.sh` 会重新检查空闲并申请 guard；`evidence/` 用于下一轮。设备代码版本要同时核对源码 SHA 与 `python-overlay.json`，wheel 的原始版本字符串不随 Python-only overlay 改变。

## 官方源码依据

- `tilelang/ascend/language/simd.py`：vld distributions、vcvt round/sat/part、vsts packing。
- `testing/ascend/language/test_tilelang_ascend_simdvf_cast.py`：UNPK_B16 与 PK_B32。
- `testing/ascend/language/simdvf/test_simdvf_arithmetic.py`：SIMD 算术调用。
- `src/ascend/transform/ascend_simdvf_lower_parallel.cc`：高层 Parallel 的支持边界。
