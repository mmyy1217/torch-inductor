# TileLang backend：A2/A3 → A5 迁移开发计划

更新时间：2026-11-06（Asia/Shanghai）。本文保留基线 rebase 审计与分阶段计划；当前已开始 P1/P2 实现。实施范围和运行验收状态见 [A5_IMPLEMENTATION_STATUS.md](A5_IMPLEMENTATION_STATUS.md)。

## 2026-11-06 实施更新

用户指定：以官方 `tile-ai/tilelang` 为唯一 A5 DSL 基准；开发分支改名为 `tilelang-backend-a5`，在 GitCode 开发，通过 A5 服务器验证。下文 rebase 审计为历史起点，当前代码已开始适配三项导入变化及显式 SIMT pointwise。运行验收状态以 [实施记录](A5_IMPLEMENTATION_STATUS.md) 为准。

## 1. 当前状态与基线

| 项目 | 固定引用 / 状态 |
| --- | --- |
| 开发分支 | `tilelang-backend-a5` |
| 独立工作树 | `D:\pytorch-ascend\.worktrees\tilelang-a5-migration` |
| A2/A3 来源分支 | `codex/tilelang-inductor-v271-260` |
| 来源 HEAD | `c935070140f6d3b9f712ee43a66891b23ef4f5e9` |
| 来源补丁的父提交 | `715b8b942f76f38f9bef5acf9d4b398b8d09db72` |
| PyTorch Ascend 发布基线 | `v26.2.0-pytorch2.14.0` → `54e365d3e3614299bccd17f06916544c3880c9fc` |
| 上游 master（观察值） | `63488450876e63d20c1a1a45fcfa9f725105d957` |
| rebase 后迁移起点 | `4e7846b32e1a28f37d54debb268584438d0698c3` |
| A5 TileLang 来源仓库 | `D:\tilelang-950`，`tile-ai=https://github.com/tile-ai/tilelang.git` |
| A5 TileLang 目标源码 | 本轮 fetch 后的官方 `tile-ai/main`：`64b52b4b1a505ba93e6e49a35cc0c5dc2a4b020d` |
| TileLang 本地开发 HEAD | `codex/harness-framework`：`cec5acfac377a8738435f527493caf62c8621379` |
| TileLang 本地编译器祖先 | 与最新 upstream 的 merge-base：`47976f210de3598444003e9b05de8204dfa7ff88` |
| TileLang fork main（观察值） | `fe8c34618ffb527329f745350934d517175f1880`，落后其 upstream 79 个提交 |

“最新发布版本”在本轮按 fetch 到的最高正式、非 beta/0day tag 选择，即 26.2.0 发布系列中最高的 PyTorch 2.14.0 线。GitHub Releases API 返回空列表，故依据远端 Git tag 固定版本；不把 master 或更新中的 `v2.14.0-26.2.0` 分支当作不可变发布版。该发布 tag 的 `version.txt` 列出多个可构建版本，不能仅凭它认定运行环境已安装 PyTorch 2.14。

迁移以 **PyTorch 2.14 / 对应 torch_npu 发布源码 + 最新 A5 TileLang 源码** 为目标。服务器现有环境和历史 wheel 不视为满足此组合；开发阶段必须独立核对、构建和验证。如果目标环境暂不提供兼容的 2.14 依赖，应记录阻塞并单独建立 2.10 对照实验，不能悄悄替换本分支的发布基线。

### 仓库更新结果

- `origin`（Ascend/pytorch）、`wamdus`（Wamdus3/pytorch）、`gitcode`（wamdus/pytorch）已 fetch。
- GitCode fork `master` 从 `01d2647b84` 快进到 `6348845087`；本地 `master` 同步到该上游提交。旧功能分支未重写。
- GitHub fork `master` 仍为 `c71a9f7325540a8c91d013dfeb77348d01848aa0`，同步未完成：GitHub 连接器返回 `Resource not accessible by integration`，本地 Git 无法取得用户名凭据。没有执行强推。
- `D:\tilelang-950` 的 upstream/origin 引用已 fetch；其开发工作树有 harness、attention 文档及 `src/ascend/transform/estimate_latency.cc` 等未提交修改，未被切换、合并或覆盖。迁移引用上表固定的官方 tile-ai SHA；这不代表该版本已在服务器构建安装。
- 主工作树 `D:\pytorch-ascend` 仍在旧分支，有 `__init__.py` / `utils.py` 两处 foreach 恢复相关未提交修改，未带入本分支。需要在 P1 对照其意图与新版 baseline 恢复机制重新实现并验证。

### rebase 方法与冲突处理

仅重放旧分支顶部的 5 个 TileLang 提交，避免把旧 v2.7.1 维护线的全部历史重放到新发布版：

```powershell
# 已执行，记录用；不要在当前分支重复执行。
git worktree add -b codex/tilelang-a5-migration D:/pytorch-ascend/.worktrees/tilelang-a5-migration codex/tilelang-inductor-v271-260
git -C D:/pytorch-ascend/.worktrees/tilelang-a5-migration rebase --onto v26.2.0-pytorch2.14.0 715b8b942f codex/tilelang-a5-migration
```

| 旧提交 | rebase 后提交 | 内容 |
| --- | --- | --- |
| `922a1ba407` | `7d48f43c23` | TileLang backend 初始实现 |
| `5afe64ed03` | `b102bc6b70` | 示例与测试 |
| `4ef5b18a7a` | `d726949436` | backend 拆分、region plan、文档及测试整理 |
| `cf696c5b31` | `55fab20c5d` | 文档迁移 |
| `c935070140` | `4e7846b32e` | 最终代码与文档整理 |

处理策略：保留发布版 `ascendc`、`triton_experimental` 注册和 `patch_device_supports_tma`；加入 TileLang 注册；public bindings 只增加两个 TileLang 模块；保留发布版六篇 compilation_backend 文档的原位置；`_dynamo.py` 沿用新版 `Config(value_type=str)`，不恢复已淘汰的 `Literal` 定义。

最终相对发布 tag 为 16 个文件、9284 行新增。`torch_npu/_inductor/tilelang/` 与 `test/_inductor/tilelang/` 相对旧 HEAD 无差异。rebase 是源码历史迁移，**不是运行兼容性验收**。

## 2. 已核实的差异与首要问题

### 2.1 Inductor / torch_npu 集成

静态审计见 [A5_REBASE_AUDIT.json](A5_REBASE_AUDIT.json)。以下三项会阻止旧后端按原样初始化，应先于 A5 kernel 性能工作处理：

| 旧依赖 | 发布版实际状态 | P1 工作 |
| --- | --- | --- |
| `codegen.wrapper.NPUWrapperCodeGen` | 类改为 `NPUPythonWrapperCodeGen` 等 | 适配 Python wrapper 注册、kernel call ABI；单独核对 C++ wrapper |
| `_register_npu_inductor_decompositions(backend="triton")` | 已拆为 `_register_triton_decompositions`、`_register_mlir_dvm_decompositions`、`_register_triton_experimental_decompositions` | 按 TileLang 支持集选择 decomposition；不能无条件启用 Triton 专用 lowering |
| `codegen.tile_generator.TileGenerator` | 位于 `runtime.tile_generator` | 更新导入，并核对配置结构、调用参数和 A5 资源假设 |

还需运行验证：PyTorch 2.14 的 `SIMDKernel`、range tree、SchedulerNode、CSE、wrapper 和 autotune 私有接口；`NPUTritonScheduling` / `NPUIndexTritonKernel` 的继承假设；新建/复用进程中的后端切换；动态 shape、输出别名、mutation、stream 与设备选择。静态名称审计未覆盖这些调用语义。

旧调度器把模板节点及不支持的算子交给 `NPUTritonScheduling`。A5 迁移需显式定义 fallback 能力，验证具体 Triton 路径与设备相容；不能把“发生 fallback”当成 TileLang kernel 成功，也不能假定默认 Triton 与 `triton_experimental` 可任意互换。

### 2.2 A5 TileLang DSL 与编译链

下表以固定目标 SHA 的实际源代码为准；本地开发分支较旧，不能混用其 API 快照。

| A2/A3 旧后端 | A5 目标与实施约束 |
| --- | --- |
| `import tilelang.language as T` | `import tilelang.ascend.language as T` |
| `compile(..., target="npuir")` | 第一阶段显式 `target="ascend", execution_backend="tvm_ffi"`，固定 AscendC 路径；PTO 后续单列能力，不纳入首个里程碑 |
| `npuir.enable_auto_multi_buffer` 布尔开关 | 逐 storage 的 `T.annotate_buffer_versions` + loop ownership + `num_stages` 搜索边界；禁止继续传旧 NPUIR pass key |
| `T.Kernel(..., is_npu=True)` / 双返回值 | `with T.Kernel(N) as bx`，1-D grid，kernel 上没有 `threads`；需要 subcore ID 时使用 `T.MixedKernel(N, sids=...)` |
| `T.vadd/vexp/...` 对 UB buffer 操作 | 显式 `T.SimdVF()` + `T.simd.*` 寄存器操作，或 `T.SimtVF(threads=...)` + scalar/Parallel 表达；操作能力逐项验证 |
| 每个中间 VecOp 提前分配 buffer | 先保留逻辑值和 VF 内部表达；只对边界值、外部输出或需跨任务存活的值物化 |
| 单一 multibuffer 配置 | eligibility、owner、count、index mode、task stage 分开建模 |
| 编译结果可直接复用的旧缓存 | 源码/计划、shape/stride/dtype、TileLang/PyTorch/torch_npu、架构、CANN/Bisheng 及参数共同参与迁移侧缓存身份 |

最新源码的重要约束：

- `T.annotate_buffer_versions({buf: 1})` 禁用该 storage 及 aliases 的 multibuffer eligibility；`{buf: (1, "auto")}` 保留 eligibility，只固定一个物理版本。两者不可等同。
- 模式包含 `auto`、`iteration`、`counter`、`write_counter`。`counter` 在同一 owner 内不能用于该 storage 的跨 stage 访问；不能把“遇到 if 就使用 counter”写成规则。`write_counter` 需明确版本数，并证明初始化与读写条件，首期不自动选择。
- `T.Stage(k)` 包裹完整 Task/VF；应写 `with T.Stage(1), T.SimtVF(...):`。非零 stage 需要 enclosing loop 的 `enable_offset=True`；`num_stages` 限制自动版本数搜索，不是 stage 编号上限。
- 普通自动调度和 `T.Stage` 路径均保留 AutoSchedule。不要同时手写 ping-pong 和让编译器自动添加物理版本。
- `T.annotate_buffer_addresses` 是精确物理放置能力，需要明确版本数、对齐及含 padding 的物理大小；首期保留编译器分配，不增加地址搜索。`annotate_unlimit_memory` 不能用于通过资源验收。
- 已有 `VFChecker`、`AutoSchedule`、`PrepareMultiBuffer`、`InsertSync`、`MaterializeMultiBuffer`、`LowerScheduledTIR` 等后端 pass。codegen 输出可检查的 VF/存储/阶段约束；TileLang 后端落实同步、版本索引和物理分配，Bisheng 继续完成目标代码编译。没有必要在 Inductor 再复制一套完整的底层调度器。

## 3. 分层接口：每层必须交付什么

下述计划类型是拟新增/拆分的内部接口，不代表目前已有同名实现。尽量复用现有 typed operation graph、region builder 和 emitter；避免在 5900 行 kernel 文件里继续堆放目标特例。

| 层 | 输入 | 输出 | 必须保留/验证的内容 |
| --- | --- | --- | --- |
| Capture / Inductor adapter | fused SchedulerNodes、loop/index 表达式、buffer reads/writes | `LogicalKernelGraph`：ValueId、OpId、dtype、symbolic shape、stride、mask、alias、外部输出 | 运算次序、类型提升、mutation、输出 ABI；unsupported reason |
| Tile/loop planner | 逻辑图、shape guards、候选 tile、`TargetCaps` | `LoopRegionPlan`：X/R 分块、tile 坐标、有效 extent、tail predicate、循环携带状态 | reduction identity、循环初始化/更新/退出，多输出可见范围 |
| VF planner | region 内 operation DAG、dtype/op 能力、候选 SIMD/SIMT 模式 | `VFPlan`：分区、模式、成员操作、边界 ValueId、mask/thread/lane 映射 | 不把一个 region 强行对应一个 VF；保留跨 VF 的语义依赖 |
| Storage planner | VF 边界、Value liveness、alias/access footprints、layout | `StoragePlan`：逻辑值→寄存器表达或 UB/L1/L0 storage、形状、dtype、padding、读写集合 | 源码 last-use 不等于异步消费者完成；首期不做跨任务激进复用 |
| Pipeline planner | storage、producer/consumer、loop owners、依赖距离 | `PipelinePlan`：逐 storage eligibility/count/mode、可选 Task stage、loop enable_offset、依赖边 | accumulator 单独建模；版本数和 stage 分开；自动调度作对照 |
| Validator | 完整候选、TargetCaps、形状与资源约束 | `ValidatedKernelPlan` 或结构化 reject/fallback 原因 | UB/L1/L0 分池预算、padding/versions/reservation、mode-stage 合法性 |
| A5 emitter/runtime adapter | ValidatedKernelPlan、签名、编译配置 | Ascend dialect PrimFunc、compile 参数、wrapper、缓存身份 | VF、load/store、copy、版本/stage 标注；显式 target；调用方输出/stream 约定 |
| Compile & measurement | PrimFunc、源码版本、toolchain、参考输入输出 | 编译产物、生成代码/IR、实际 VF/资源/版本报告、正确性、latency、最佳候选 | 编译失败与资源拒绝分开；禁止只凭 Python 源码宣称融合/双缓冲已生效 |

建议模块边界：`tilelang/target.py`（能力与版本）、`tilelang/plan.py`（序列化计划）、`tilelang/planning/{vf,storage,pipeline}.py`、`tilelang/codegen/ascend.py`、`tilelang/runtime.py`。这是按职责拆分的起点；以避免循环依赖和便于测试为准，不要求预先创建空模块。

`TargetCaps` 至少包含：arch、目标 DSL/编译器 SHA、execution backend、支持的 VF 模式与 op/dtype、各存储池/对齐/预留、核心预算、版本模式/stage 能力。不得把旧 A2/A3 的 UB 大小、并行度或 dtype 表直接当成 A5 常量。

## 4. 实施阶段与验收

| 阶段 | 工作 | 产物与退出条件 |
| --- | --- | --- |
| P0 基线准备（本轮） | fetch、fork 同步、新分支、限定范围 rebase、接口盘点 | 本文、固定 SHA、静态审计；本地 rebase 已完成，GitHub fork 认证为未完成项 |
| P1 初始化与运行环境 | 修复三项已知导入差异；适配 PyTorch 2.14 私有接口、decomposition/fallback、foreach 恢复；建立独立的 A5 构建环境 | 可导入 backend；默认/TileLang/其他后端切换不泄漏全局 patch；匹配源码构建的 torch_npu/TileLang；失败有明确原因 |
| P2 最小显式 VF | 抽出 target/emitter；先固定 tile、单 buffer；float32 连续 pointwise；SIMT 语义基线 + SIMD 寄存器链 | `add → mul → exp` 正确；内部值不强制 UB 落地；输出源码与 IR 能证明实际 VF 边界；再覆盖 fp16/bf16、tail、broadcast/stride |
| P3 Reduction / 多输出 | 移植 sum/max/min；reducer 生命周期、broadcast、跨 R tile 状态；RMSNorm/LayerNorm/Softmax；多输出和 alias | 短/长 reduction、非整 tile、极值/NaN 语义、输出 mutation 对照通过；unsupported dtype/布局明确回退 |
| P4 多缓冲与受限调优 | 使用 compiler-managed versions；逐 buffer 1/2 版本与 auto 对照；随后引入小范围 Stage 候选 | 长循环、多轮调用和条件路径无提前覆盖；实际 versions/内存/依赖可检查；shape/stride/dtype/版本完整缓存键；无效候选被淘汰 |
| P5 CV 模板接入 | GEMM+epilogue → Attention；模板承载算法与 core/dataflow 骨架，VF body 复用通用 codegen | 分别核对 Cube/Vector 分工、dual_copy 形状、同步、累计精度、端到端数值与延迟；记录 fallback 覆盖 |

首个可验收里程碑是 **P1 + P2：在固定发布版本上运行由 torch.compile 生成的 A5 显式 VF pointwise kernel**。第二个里程碑是 RMSNorm/Softmax。Attention、多输出复杂融合、自动 stage 搜索和 PTO 不阻塞首个里程碑。

每阶段分开提交：公共接口适配、计划数据结构、A5 emitter、运行/缓存、测试与证据。不要在一个提交里同时升级 compiler、改变 tile、改变算法和宣称性能收益。

### 模板与 Attention 的后续边界

模板需要实例化 shape/stride/dtype、mask、tile、layout、VF 与流水配置，并生成/编译 PrimFunc；并非只导入 shape 调用一个既有二进制算子。先用受控模板承载 GEMM/Attention 的循环与 Cube/VF 边界，再逐步泛化。

首个 Attention 范围：无 dropout 前向、先固定规则布局；Q/K/V、scale、causal/tail mask 为语义输入。每个 Q tile 保留 `m=-inf`、`l=0`、`O_acc=0`，迭代 KV tile：

1. Cube：`S = Q @ K.T * scale`；Vector：施加有效位置 mask。
2. Vector：`m_new=max(m,rowmax(S))`、`alpha=exp(m-m_new)`、`P=exp(S-m_new)`、`l_new=alpha*l+rowsum(P)`。
3. Cube/Vector：`O_acc=alpha*O_acc + P @ V`；更新 `m,l`；最后 `O=O_acc/l`。

上述是数学数据流。实现中必须处理全 mask 行、空/短序列和尾块，避免 `-inf - -inf` 产生错误；明确 FP32 累计、P 的转换精度和输出约定。Q/K/V→L1、L0C→UB、P→Cube 输入的布局、版本与 task ownership 按目标能力具体化；不能假定 Vector 区域必然是单个 VF，也不能保证 CV 融合仅靠编译器从任意通用图自动发现。

## 5. 验证方案与证据

### rebase 阶段已完成（历史记录）

- `git merge-base --is-ancestor v26.2.0-pytorch2.14.0 HEAD` 验证发布基线祖先关系。
- `git range-diff 715b8b942f..c935070140 v26.2.0-pytorch2.14.0..4e7846b32e` 核对 5 个提交及冲突处理。
- `git diff --check`；13 个变更 Python 文件 AST parse；TileLang 本地导入静态审计。
- 核对旧后端/测试文件与 rebase 后内容一致；上游非 TileLang 文档未被错误移动。
- **未运行** torch_npu import、torch.compile、TileLang native build、NPU correctness、性能或 profiling。此项记录 rebase 完成时的状态。三项初始化适配现在已实现，实际运行验收见实施记录。

### 后续门槛

1. **环境一致性**：记录实际 Python/torch/torch_npu/TileLang 路径、native 扩展、wheel SHA、源码 SHA、子模块 SHA、CANN/Bisheng、NPU arch、编译参数。新增独立环境，保持已有工作环境可复用。源码更新后核验安装包来源；涉及二进制/后端集成变动须构建并安装匹配产物。
2. **纯逻辑测试**：operation/VF/storage/pipeline plan 的语义不变量、拒绝原因、cache identity、输出 ABI。测真实边界，不用仅比较生成字符串的测试替代语义检查。
3. **编译验证**：最小 PrimFunc、SIMD/SIMT、版本模式、stage、resource rejection，保存 lowered IR 与 generated source。对照目标仓库 `testing/ascend/language/{simtvf,simdvf}/`、`testing/ascend/auto_schedule/` 相关测试。
4. **端到端正确性**：复用并适配 `test/_inductor/tilelang/` 的 ops/broadcast/mixed；增加同进程后端切换、动态 shape、stride、tails、dtype、mutation、重复运行、条件 epoch。分别统计 native TileLang 与 fallback 用例，输出 expected/completed/failed/skipped 数量。
5. **性能**：固定输入/版本/设备与测量边界；分别测 compile/cold start、steady-state kernel 与 torch.compile 调用；与 eager、经验证的 A5 Triton、自动调度基线比较。保存完整样本与变异，预先确定回归阈值；不把编译成功或 solver II 当设备加速证据。

A5 设备执行按 [a5-server skill](C:/Users/yejiaang/.codex/skills/a5-server/SKILL.md)：使用专用 SSH 配置，执行前实时确认物理卡空闲；整个设备进程放入 `npu-guard run --wait 3600 --timeout 3600`；36 AICore DT 任务预算限制为最多 32 AICore，并记录生效依据。纯 Vector 的 launch grid 与 AIC 配额不是同一概念，需记录 AIV 映射；不能把示例中的 block 数直接作为设备限核机制。

后续需先创建 A5 专用工作目录及 LF 脚本，按仓库远端执行约定同步、初始化 CANN 后执行。A2/A3 的 `arm_npu/wangchenyu` 环境和 CANN 8.5/9.0 路径只用于旧基线回归，不直接用于 A5；A5 的最新路径和软件状态需实时核对。rebase 阶段未登录服务器；当前已在 A5 上建立独立工作目录，见实施记录。

## 6. 下一步可执行清单

- [x] 建立独立迁移分支并保留旧分支/未提交修改。
- [x] fetch 上游及 fork，快进 GitCode fork 和本地 master。
- [x] 按最新要求改为 GitCode 开发；GitHub fork 同步不再阻塞本次工作。
- [x] 只 rebase 5 个 TileLang 提交到固定正式发布 tag。
- [x] 固定最新 A5 TileLang upstream SHA，核对 VF/versions/stages/compile API。
- [x] 记录静态审计和开发计划。
- [x] P1：修复 wrapper/decomposition/TileGenerator，验证后端切换与公共 hook 恢复。
- [x] P1：准备并验证 PyTorch 2.14 + 对应 torch_npu + 固定 TileLang SHA 的隔离环境。
- [x] P2：产出固定 tile、单 buffer、显式 VF 的首个 torch.compile pointwise kernel。

## 7. 源码依据

- [PyTorch Ascend 发布 tag](https://github.com/Ascend/pytorch/tree/v26.2.0-pytorch2.14.0)：`torch_npu/_inductor/__init__.py`、`decomposition.py`、`lowering_patch.py`、`codegen/wrapper.py`、`runtime/tile_generator.py`。
- [A5 TileLang 固定源码](https://github.com/tile-ai/tilelang/tree/64b52b4b1a505ba93e6e49a35cc0c5dc2a4b020d)：`tilelang/ascend/language/{kernel,frame,annotations,schedule_hint}.py`、`pipeline.py`、`README.md`；`.agents/skills/tilelang-ascend/SKILL.md` 及 `references/synchronization.md`。
- 旧后端：本分支 `tilelang/__init__.py`、`codegen/tilelang_kernel.py`、`codegen/tilelang_scheduling.py`。历史说明 [TILELANG_BACKEND.md](TILELANG_BACKEND.md) 仍描述 A2/A3 NPUIR 路径，不能作为 A5 使用说明。
- 之前的 [A5 架构图册](D:/Exdraw/output/tilelang-a5-codegen-v2/index.html) 用作分层讨论背景；本文的固定源码和具体 API 约束优先于旧图中的概念性名称。
