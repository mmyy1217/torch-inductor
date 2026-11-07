# A5 TileLang：首批运行链迁移

日期：2026-11-06。开发分支：[tilelang-backend-a5](https://gitcode.com/wamdus/pytorch/tree/tilelang-backend-a5)。本文件区分已实现接口与实际验收；完整路线见 [迁移计划](A5_MIGRATION_PLAN.md)。

## 最新路线：SIMD VF（2026-11-06）

当前 pointwise 发射器改为 `T.SimdVF()` + 显式 `T.simd.*`。下文原 SIMT 验收与生成代码保留为历史记录；当前路线、精度/布局约束和 SIMD 验收见 [A5_SIMD_VF.md](A5_SIMD_VF.md)。现有 TileLang 发射器不生成 `SimtVF`，索引化搬运在 codegen 阶段明确拒绝；没有启用 gather 专用路由。

2026-11-07 首版 sum/max/softmax SIMD reduction 已完成 A5 **54/54** 验收，包含三种浮点 dtype、尾块和特殊值。支持范围、修复及原始证据见 [A5_REDUCTION_SIMD.md](A5_REDUCTION_SIMD.md)。

## 固定来源

- torch_npu 发布基线：`v26.2.0-pytorch2.14.0`，`54e365d3e3614299bccd17f06916544c3880c9fc`。
- A5 DSL/编译器只以 [tile-ai/tilelang](https://github.com/tile-ai/tilelang/tree/64b52b4b1a505ba93e6e49a35cc0c5dc2a4b020d) 为准：`64b52b4b1a505ba93e6e49a35cc0c5dc2a4b020d`；TVM gitlink：`0215a96cba91eabac903afa4ec1e42c99257e4c3`。
- 首批实现提交：`a864ba51a2aa150c2ebe9625eaaef164e7d797bd`。旧 A2/A3 分支与主工作树的未提交修改保留。
- 本地工作树：`D:\pytorch-ascend\.worktrees\tilelang-a5-migration`。
- A5 隔离根目录：`/home/yja/experiments/tilelang-backend-a5-20261106`，分别使用 `pytorch/`、`tilelang/`、`env/`。不覆盖服务器原有实验源码或基础环境。

## 各层输入、处理与输出

| 层 | 输入 | 当前处理 | 输出 |
| --- | --- | --- | --- |
| Dynamo / Inductor | `torch.compile(..., options={"npu_backend": "tilelang"})`，FX 图及 shape/stride/dtype guards | 使用发布版公共 patches，注册 TileLang scheduler、Python wrapper、decomposition | Inductor scheduler nodes、LoopBody 和参数/输出信息 |
| TileLang 捕获与计划 | 节点 schedule、range tree、load/store 索引、标量运算及 dtype | 复用旧 typed operation/region/buffer plan；保留共享 NPU 索引分析 | `TileLangKernelPlan`：签名、launch、region、transfer、pointwise operations、buffer 元数据 |
| A5 pointwise emitter | `PointwiseOpPlan`、输入/输出边界、tile shape、execution/result dtype | 生成 `T.SimdVF()`、64-lane 向量循环和 `T.simd.*`；中间值驻留向量寄存器；标量使用 BRC load | Ascend DSL VF body；只将边界输出写回 UB |
| PrimFunc emitter | region/transfer + VF body | `tilelang.ascend.language`，一维 `T.Kernel`；边界 UB `T.alloc_shared`，显式 `T.annotate_buffer_versions(...: 1)` | 参数化 `@T.prim_func` 工厂，shape/tile 实参确定具体 PrimFunc |
| 编译 wrapper | 具体 PrimFunc、shape 缓存键、单 buffer 配置 | `tilelang.compile(target="ascend", execution_backend="tvm_ffi", pass_configs={"tl.enable_auto_schedule": True})` | 缓存的 JIT callable；底层 native 编译由官方 TileLang 完成 |
| 运行 | 输入 tensor 与 Inductor 已分配的输出 tensor | 普通执行不传 `out_idx`；按签名把输入/输出都传给 callable | 写入 caller-owned 输出，供 Inductor 后续节点/返回值使用 |

这阶段没有新增完整的 VF/storage/pipeline cost planner。使用现有 plan 的 pointwise 边界决定 VF，逐 buffer 固定一个版本；后续再显式表达 lifetime、ownership、版本数及 Stage 候选。内部表达式最终是否保持寄存器、VF 如何 lowering，必须检查 native source/IR，不能仅凭 Python DSL 判断硬件行为。

## 已修改的兼容接口

- `NPUWrapperCodeGen` 改为 `NPUPythonWrapperCodeGen`；补入公共 patches。
- decomposition 入口适配发布版 `_register_triton_decompositions`；`TileGenerator` 使用 `runtime.tile_generator`。
- `define_kernel` 接收新版 `kernel_graph_hash` 参数；CSE 保留 PyTorch 2.14 新增的 shape 与 dtype；显式绑定 `TileLangOverrides`，缺失的运算捕获在生成前拒绝。
- foreach 禁用纳入中央 baseline 快照，以支持恢复；default → TileLang → default 切换及 foreach 恢复已通过设备验收进程内的检查。`728ac2a1bd` 另修复了实际测试发现的 `aten.erfc` 重复 decomposition 注册：将其加入原有的覆盖列表，和 `expm1` 一样先移除再注册。
- 编译 target 从 `npuir` 改为 `ascend`，执行后端固定 `tvm_ffi`，不再传旧 NPUIR pass key。
- autotune 默认关闭。旧布尔 multibuffer 不能映射为 A5 的逐 buffer versions，当前只保留 single-buffer 配置，显式拒绝旧 multibuffer=True。

## 支持边界

- 首批 emitter 实现浮点 `copy/cast`、`add/sub/mul/truediv`、`exp/log/sqrt/rsqrt/abs`；实际端到端验收首先覆盖 `add → mul → exp`。
- 计划中 operand 可按尾维对齐做 extent=1 广播；不支持的 tile/layout/dtype/运算明确拒绝。设备已验证线性 vector + scalar 输入广播；ranked broadcast 尚未验收。
- 连续末轴、静态 R <= 4096 的 sum/max 与 softmax 已接入 SIMD emitter；sum/max 验证到 R=4096，softmax 验证到 R=2051。其他 reduction 仍尝试现有 Triton fallback；该 fallback 在当前隔离环境未通过验收。模板仍使用原 fallback，尚未生成新的 TileLang CV/Attention 模板。
- 共享 `NPUIndexTritonKernel` / `NPUTritonScheduling` 的索引和调度设施，因此仍依赖 `triton-ascend`。不能将本实现称为不依赖 Triton 的独立 backend。
- legacy autotuner 代码尚未完成 A5 验收；当前禁止把它的结果用作 A5 调优结论。

## 验证与复现

本地纯逻辑测试：

```powershell
python test/_inductor/tilelang/test_ascend_emitter.py
```

SIMT 阶段本地复查 6/6 通过：融合表达式且无中间 tensor storage、物理临时名复用、广播与多输出、布局拒绝、未启用运算拒绝、整数 dtype 拒绝、VF 内标量中间值广播。其中部分性质合并在同一个测试中；这些测试不导入 TileLang，不证明 native compiler 或设备正确性。

设备验收入口：`test/_inductor/tilelang/test_ascend_pointwise.py`，现已要求显式 SIMD、禁止 SIMT。覆盖 float32/float16/bfloat16 × size 1/64/257/2051，以及多输出、标量输入广播；每例复用缓存 callable 再换输入运行。断言生成代码无 Triton/ATen/NPU fallback，且含 Ascend dialect 和显式 VF。

设备测试必须在 `npu-guard` 的整个前台进程内执行，运行前实时确认空闲。初始化后、编译或分配 tensor 前，通过 `set_device_limit(0, cube_num=32, vector_num=64)` 并读取核实限核。记录源码、wheel、native library、CANN、设备、cache、generated source/IR。

### SIMT 阶段验收历史

| 项目 | 结果与对应源码 |
| --- | --- |
| emitter 语义测试 | 当前 `a5eb1ec34d`：6/6 通过；AST 与 diff 检查通过 |
| 官方 TileLang native build | `64b52b4b1a505ba93e6e49a35cc0c5dc2a4b020d` 构建成功；`libtilelang.so` SHA256：`3ed5750e6afb4c9e3598435090d23c8b42dde5365f4457b54bd4fbe3cbd43ca2` |
| torch_npu 源码 wheel | `2.14.0+gita864ba5` 已构建安装；SHA256：`ca72704641a3b9a04d8158bbd40b31bfa6a7cf1460324dfe08b963d0f71b5944` |
| 软件环境 | PyTorch `2.14.0+cpu`、CANN `/home/yja/envs/tilelang-950/Ascend/cann-9.2.0`、triton-ascend `3.2.1` / triton `3.2.0`；`pip check` 无依赖冲突 |
| 直接 PrimFunc | `add → mul → exp`、caller-owned 输出 ABI、float32 数值检查通过 |
| 后端切换 | `default → tilelang → default` 的 foreach 恢复通过 |
| torch.compile，源码 `7b9ff29749` | float32/float16/bfloat16 × size 64/257/2051 共 9 例通过；float32 size=257 多输出 1 例通过；每例新输入复用 compiled callable 也通过。生成代码检查无 Triton/ATen fallback |
| size=1，源码 `7b9ff29749` | 三种 dtype 均进入旧物化广播 fallback；Triton 预编译头失败。该轮 3 个测试方法、3 个 error、0 skipped；11 次 TileLang compile 调用包含 direct smoke 和 10 个通过用例 |
| 最新 scalar 修复 `a5eb1ec34d` | 删除线性 scalar 输入的物化广播，改为 consumer VF 内索引 lane zero；允许标量中间值按 VF domain 重算。6 个 CPU 测试通过，新增 size=257 + scalar 输入设备用例；在仅追加文档后的 `15b290a828` 上已完成设备复测，见下表 |
| 最新完整验收 `15b290a828` | 4 个测试方法，0 failures / 0 errors / 0 skipped；12 个 dtype/size 用例、多输出、scalar broadcast 共 14 个端到端计算用例全部通过；15 次 TileLang compile 调用包含 direct smoke。每例新输入复用 callable 和无 fallback 检查通过 |
| 性能、profiling | 未进行；没有加速比或 MFU 结论 |

后续修复仅修改 Python，按仓库的 Python-only 更新约定覆盖到首次从本仓库构建的安装包；不是使用不匹配的官方 wheel 冒充源码验证。`python-overlay.json` 记录基础源码 wheel SHA、当前 Python 源码提交、动态查询的安装目录和逐文件 SHA256。原始 wheel 的版本字符串仍为 `gita864ba5`，应同时读取 overlay 清单判断实际 Python 代码版本。

已检查实际生成的 C++：size=2051 的融合计算只有一个 `__simt_vf__` 和一次 `asc_vf_call`；内部加法值是局部 `float _vf_v0`，没有为每个算子中间值分配 UB。两次 GM→UB 和一次 UB→GM 的尾块长度为 `min(128, 2051 - block_idx*128)`；MTE2→V→MTE3 同步由编译器生成。多输出也在同一个 VF 内写回两个边界输出。源码证据不代替寄存器、指令或性能测量。选定源码与精简验证元数据见 [evidence/](evidence/)。

### 标量修复的设备复测已完成

2026-11-06 08:31 UTC 启动前，物理 device 0 的 guard 状态为 idle、无活动计算进程、无 reservation 占用。整个测试进程在 guard 内运行，读回限核为 32 AIC/64 AIV。08:36:38 UTC guard 记录 `workload_completed=true`、`workload_exit_code=0`，无冲突、无可见性丢失；08:39 UTC 再查设备已空闲、锁已释放。

SSH 跳板在测试期间短暂断开；重连后读取完整结果与 guard 完成记录确认成功，没有重复启动任务。先前 14:17 CST 的忙碌仅是上一次复测未启动的历史原因，不再阻塞这批验收。

新增源码证据明确了标量广播的实现：257 元素输入每个 tile 从 GM 搬运有效元素，标量只搬运 4 字节到一个独立 UB 元素，VF 的所有 lane 读取相同的 UB 地址；没有生成物化广播 kernel。size=1 的两个输入也各搬运 4 字节，只有一个 VF，输出只写回一个元素。首批实现仍固定 128 个 SIMT lane，因此 size=1 的功能正确性不代表执行效率已优化。

本轮无需追加 backend 代码修复：`a5eb1ec34d` 的 scalar 修复通过，验收源码为只追加了文档的 `15b290a828`。完整结构化记录见 [a5_pointwise_validation_complete.json](evidence/a5_pointwise_validation_complete.json)；此前 `a5_pointwise_validation.json` 保留为旧失败轮次的历史证据。

复现入口仍为以下 LF 脚本（每次运行都会重新检查空闲并申请 guard）；历史完整记录已归档到 `evidence-attempt-05/`：

```powershell
ssh -F C:\Users\yejiaang\.ssh\a5-server-950dt\config a5-server 'bash /home/yja/experiments/tilelang-backend-a5-20261106/run-validation.sh'
```

下一步扩展 ranked broadcast、stride/dynamic shape、长 reduction 和 Norm，再推进 pipeline versions 与 CV/Attention。现有 fallback 环境问题和性能尚未验收。

### 证据位置

隔离根目录 `/home/yja/experiments/tilelang-backend-a5-20261106`：

- `evidence-attempt-01/`：direct smoke 通过，重复 erfc 注册失败。
- `evidence-attempt-02/`：后端切换通过，CSE shape 接口失败。
- `evidence-attempt-03/`：CSE 适配后发现 overrides 未绑定，生成未定义临时值。
- `evidence-attempt-04/`：10 个端到端用例通过、3 个 size=1 失败；含 `environment.json`、`results.json`、`kernel-*.tir`、`kernel-*.cpp`、完整日志与缓存。
- `evidence-preflight-20261106-0820/`：上一次忙碌检查的历史记录。
- `evidence-attempt-05/`：最新完整通过轮次；含版本/overlay、4 个测试方法的结果、15 份 PrimFunc/native source 和完整验证日志。
- `configure.log`、`build.log`、`pytorch-install.log`、`pip-check.log`、`python-overlay.json`：构建、依赖与安装来源。`pytorch-build.log` 的最后部分可能属于已中断的重复构建；首次成功 wheel 以 SHA256 和已安装版本为证。
