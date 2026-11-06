# TileLang Inductor Backend for Ascend NPU

## 概述

`torch_npu/_inductor/tilelang/codegen` 实现了面向 Ascend NPU 的 PyTorch
Inductor codegen 后端。它复用 NPU Triton 的 scheduling、range tree 与轴分析，
把 Inductor 的 pointwise/reduction 计算记录为结构化 operation graph，再生成
TileLang `@T.prim_func`，最终通过 `tilelang.compile(..., target='npuir')`
编译和执行。

本文是 TileLang 后端的总览和支持边界说明。更细的资料见：

- [TileLang Codegen 调用链](tilelang_codegen.md)：从
  `torch.compile` 到 wrapper 调用的逐阶段说明。
- [NPU Triton Codegen 与 Autotune Pipeline](TRITON_PIPELINE.md)：共享的
  scheduling、tiling、融合与 Triton fallback 路径。

---

## 激活方式

推荐通过 `torch.compile` 的 per-compile option 选择 TileLang：

```python
import torch
import torch_npu

compiled_model = torch.compile(
    model,
    backend="inductor",
    options={"npu_backend": "tilelang"},
)
```

该配置只作用于对应的 compiled callable，便于在同一进程中分别使用 TileLang、
Triton、MLIR 或 DVM。`TORCHINDUCTOR_NPU_BACKEND=tilelang` 仍作为兼容入口保留；
使用环境变量时必须在导入 `torch_npu` 前设置。

`torch_npu/_inductor/__init__.py` 在该次编译选择 TileLang 后端时调用
`tilelang.initialize_tilelang_backend()`。该入口把
`TileLangScheduling`、`NPUWrapperCodeGen` 和 `CppWrapperNpu` 注册为 NPU device
codegen，并安装 decomposition、fallback、MM lowering 及 Inductor 所需的 NPU
patch。

---

## 架构与路由

### 继承层次

```text
上游 PyTorch Inductor
├── SIMDKernel
│   └── TritonKernel
│       └── NPUIndexTritonKernel      (codegen/triton.py)
│           └── TileLangKernel        (tilelang_kernel.py)
│
└── SIMDScheduling
    └── TritonScheduling
        └── NPUTritonScheduling       (codegen/scheduling.py)
            └── TileLangScheduling    (tilelang_scheduling.py)
```

`TileLangKernel` 继承 `NPUIndexTritonKernel`，主要是复用 range tree、indexing
变换和 `decide_codegen_dims_in_kernel` 产生的轴信息。TileLang 的最终源码不读取
Triton `self.body`，而是从捕获的元数据生成结构化 plan 后独立 emit。

`TileLangScheduling` 继承 `NPUTritonScheduling`，并负责：

- 指定 `TileLangKernel` 为 SIMD kernel 类型；
- 收紧 TileLang 可接受的融合组合；
- 将 template 节点转发给独立的 `NPUTritonScheduling`；
- 在 TileLang 不支持当前 kernel 时按节点回退到 NPU Triton；
- 生成 shape-keyed TileLang compile/autotune wrapper。

### 路由结果

| 节点或场景 | 处理路径 |
|---|---|
| 可规划的 pointwise/reduction `ComputedBuffer` | TileLang |
| Template 节点 | 直接由 NPU Triton `codegen_template()` 处理 |
| TileLang 不支持的融合组合 | `can_fuse()` 拒绝融合；拆分后的节点再分别选择 TileLang 或 fallback |
| Split-scan、实际进入 codegen 的超过两个 reduction kernel | NPU Triton |
| 不支持的 op/dtype、tuple reduction、atomic add、无法映射的 layout/indexing | TileLang codegen 抛出 `NotImplementedError`/`TileLangUnsupported`，`codegen_node()` 捕获后回退 NPU Triton |
| TileLang autotune 失败 | 仍使用默认 tiling 进行 TileLang 编译，不切换到 Triton |

---

## 完整 Pipeline

```text
torch.compile(fn, backend="inductor")
        │
        ▼
Dynamo → FX graph → Inductor IR → Scheduler
        │
        ▼
TileLangScheduling.codegen_node()
  ├─ template/split-scan/reduction 数量预检查
  ├─ SIMDScheduling.codegen_node()
  │    └─ NPUTritonScheduling.codegen_node_schedule()
  │         ├─ select_tiling() → {"x": xnumel, "r": rnumel?}
  │         ├─ TileLangKernel(tiling, features=...)
  │         ├─ decide_codegen_dims_in_kernel()
  │         └─ 遍历 node body
  │              ├─ load()/store() 记录 buffer/value/output
  │              └─ reduction()/store_reduction() 记录 reduction
  │
  ├─ TileLangKernel.codegen_kernel()
  │    ├─ Pointwise → TileLangPointwiseProgramBuilder
  │    ├─ Reduction → TileLangReductionProgramBuilder
  │    ├─ build() → TileLangKernelPlan
  │    └─ KernelProgramEmitter.emit(plan) → @T.prim_func 源码
  │
  ├─ TileLangScheduling.define_kernel()
  │    ├─ 生成 prim_func factory
  │    ├─ 生成候选 tiling 与 shape-keyed cache
  │    ├─ 可选 TileLang autotune
  │    └─ tilelang.compile(..., target="npuir")
  │
  └─ TileLangKernel.call_kernel()
       └─ emit: kernel(tensors..., numels..., static_axis_args...)
```

### 共享的维度分析

`decide_codegen_dims_in_kernel` 是 TileLang 与 NPU Triton 的共同基础。它不生成
TileLang 源码，只把 schedule indexing 转换为 kernel axis，并给后续 planner 提供
布局信息。

| 阶段 | 作用 |
|---|---|
| `_mark_store_index_keys` | 标记 store indexing，避免 unified 写回轴在替换中丢失 |
| `_transform_schedule_indexing` | `split_and_set_ranges()` 对齐 node ranges 与 range tree；转换、拆分和化简 indexing；以子轴替换冗余父轴 |
| `_record_store_unified_indexing` | 记录转换后的最终 store 地址形式 |
| `_remove_substituted_dims_from_kernel` | 删除不再需要且不承担 store anchor 的父轴 |
| `_finalize_kernel_codegen_dims` | 运行 `SplitTiling` 与 `ReductionAnalysis`，得到 split、tiling、no-loop 轴和 `reduced_dim` |

最终数据流为：

```text
node.var_ranges
  → range tree (x0/x1/r0/...)
  → transformed body.indexing
  → 精简后的 axis 集合
  → split_axis       # AI Core grid
  → tiling_axis      # 片上 tile
  → no_loop_axis     # 可整块处理的 tile
  → reduced_dim      # T.reduce 的规约维
```

详细的 tiling 与融合约束见
[NPU Triton Codegen 与 Autotune Pipeline](TRITON_PIPELINE.md)。

---

## 结构化 Operation Graph 与 Kernel Plan

TileLang 不能直接复用 Triton 的标量表达式源码，因此 codegen 分成“捕获、规划、
发射”三层：

1. `TileLangOverrides` 在 Inductor CSE 计算时写入 `PendingOp`。
2. `TileLangKernel.load/store/reduction` 将 CSE 值、buffer、index、dtype 和输出关系
   记录到 `_values`、`_buffers`、`_outputs`、`_reduction_vars` 与
   `_reduction_outputs`。
3. Pointwise/reduction builder 把这些记录转换为显式的 operation graph 和 region
   plan；`KernelProgramEmitter` 只读取 plan，不在发射阶段查询 `V.graph`。

主要结构如下：

| 结构 | 职责 |
|---|---|
| `TileLangTensor` / `TileLangBlock` / `TileLangForLoop` | 描述逻辑维、tile shape、切片和循环 |
| `PointwiseOpPlan` / `ReductionOpPlan` / `BroadcastOpPlan` | 描述显式计算依赖 |
| `TileLangOperationGraph` | 校验 producer、consumer、顺序与环依赖 |
| `KernelRegionPlan` / `XTileRegionPlan` / `RTileRegionPlan` | 描述分配、搬运、计算与 X/R 循环区域 |
| `TileLangKernelPlan` | 汇总签名、launch、operation 与 region，并校验每个 operation 只调度一次 |
| `KernelProgramEmitter` | 将已验证 plan 发射为 `@T.prim_func` |

预期内的不支持场景在 emission 前使用 `TileLangUnsupported` 提供阶段、operation
和 origin 信息，并触发 NPU Triton fallback。`TileLangKernelPlan.validate()` 的
`RuntimeError` 表示 planner 内部不变量被破坏，应当作为实现错误处理，而不是静默
fallback。

---

## Pointwise 与 Reduction 生成

### Pointwise

Pointwise builder 优先构建保留逻辑轴和 broadcast 依赖的 ranked layout；若无法建立，
则使用展平的 linear layout。两种路径都使用：

- `T.alloc_shared` 分配输入、输出和必要的中间 tile；
- `T.copy` 完成 GM 与片上 buffer 的搬运；
- `T.v*` intrinsic 表达向量计算；
- block/remain 信息处理非整除尾块；
- `BroadcastInsertionPass` 显式插入 `T.vbrc` broadcast operation。

### Reduction

Reduction builder 根据轴分析选择 linear 或 ranked block：

1. linear reduction 使用 `_XBLOCK`、`_XBLOCK_SUB` 与 `_RBLOCK`；ranked reduction
   为每个保留的逻辑轴生成独立 block，例如 `[x,y,r]` 对应
   `X1BLOCK`、`X0BLOCK`、`R2BLOCK`。
2. persistent reduction 沿用 NPU Triton 的判定标签；完整 R 域能放入一个 tile 时
   不生成 R 循环，否则生成 `_RBLOCK` serial loop。
3. R 循环首块 `clear=True`，后续块 `clear=False`，由 `_remain_X`、
   `_remain_R` 处理 X/R 尾块。
4. 支持符合布局约束的 scalar/vector epilogue，以及两路受限 reduction 融合。

---

## 支持的 Primitive

以下表格描述 `_TILELANG_OPS` 中当前启用的原生向量 lowering，不等同于所有可从
PyTorch 前端写出的表达式；部分前端 op 会先被 Inductor decomposition 改写。

### Binary Ops

| Inductor op | TileLang intrinsic | 支持 dtype |
|---|---|---|
| `add` | `T.vadd` | fp16, fp32 |
| `sub` | `T.vsub` | fp16, fp32 |
| `mul` | `T.vmul` | fp16, fp32, int16, int32, int64 |
| `truediv` | `T.vdiv` | fp16, fp32, int64；非零标量除数可改写为乘倒数 |
| `maximum` | `T.vmax` | fp16, fp32 |
| `minimum` | `T.vmin` | fp16, fp32, bf16, int16, int32, int64 |
| `bitwise_and` | `T.vand` | int8, int64, fp16, fp32, bool |

`pow` 没有原生 `T.vpow` registry 项。测试中的 `torch.pow(x, 2)` 依赖 Inductor
将其分解为已支持表达式，不能据此推断任意 `pow` 都受支持。

### Unary Ops

| Inductor op | TileLang intrinsic/改写 | 支持 dtype |
|---|---|---|
| `neg` | `T.vmul(x, -1)` | fp16, fp32, int16, int32, int64 |
| `exp` | `T.vexp` | fp16, fp32 |
| `log` | `T.vln` | fp16, fp32 |
| `relu` | `T.vrelu` | fp16, fp32 |
| `sigmoid` | `T.vsigmoid` | fp16, fp32 |
| `sqrt` | `T.vsqrt` | fp16, fp32 |
| `rsqrt` | `T.vrsqrt` | fp16, fp32 |
| `abs` | `T.vabs` | fp16, fp32, uint8, int32, int64 |
| `cos` | `T.vcos` | fp16, fp32 |
| `sin` | `T.vsin` | fp16, fp32 |
| `erf` | `T.verf` | fp16, fp32 |
| `tanh` | `T.vtanh` | fp16, fp32 |

predicate/comparison、`where`、`bitwise_not` 当前因运行时稳定性或验证状态关闭；
`bitwise_or`/`bitwise_xor` 没有已验证 dtype。这些组合会回退 Triton。

### Reduction Ops

| Inductor reduction type | TileLang reduce mode |
|---|---|
| `sum` | `sum` |
| `max` | `max` |
| `min` | `min` |

`mean` 可由 Inductor 分解为 `sum` 与 pointwise 除法。`prod`、`xor_sum`、
argmax/argmin、Welford 等不是当前 TileLang reduction primitive。

---

## Autotune 与缓存

`define_kernel()` 按 shape、静态轴参数和候选 tiling 生成 wrapper。
`_tilelang_xblock_configs()` 使用轴元数据和 `TileGenerator` 生成候选；每个候选除
block 参数外还保留 TileGenerator 产生的 `multibuffer` 布尔值。相同 block 的
`multibuffer=False/True` 是两个不同候选，不会在去重阶段合并。

默认开启 TileLang autotune；测试矩阵通常将其关闭以缩短运行时间：

- 非 autotune 路径优先使用保守 config。generic reduction 默认限制为
  `X0BLOCK=8`、`R0BLOCK<=1024`、`multibuffer=False`，避免未经 profile 直接选中
  过大的 R block 导致 UB 溢出；
- autotune 将候选按 `multibuffer=False/True` 分成两个编译组。每组使用固定的
  `npuir.enable_auto_multi_buffer` pass config，分别运行调优后再比较组间延迟；
- 选中 config 后，最终 `tilelang.compile()` 使用该 config 的 `multibuffer` 值。
  单组失败不会丢弃另一组；所有组失败时才回到默认 TileLang config；
- `TILELANG_NPUIR_ENABLE_AUTO_MULTI_BUFFER` 是显式调试覆盖项，设置后优先于
  config 中的值。

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `INDUCTOR_ASCEND_TILELANG_AUTOTUNE` | `1` | 开关 TileLang autotune |
| `INDUCTOR_ASCEND_TILELANG_AUTOTUNE_CANDIDATES` | 未设置 | 以逗号/分号分隔的 `X0BLOCK` 过滤集合 |
| `INDUCTOR_ASCEND_TILELANG_AUTOTUNE_WARMUP` | `5` | profile warmup 次数 |
| `INDUCTOR_ASCEND_TILELANG_AUTOTUNE_REP` | `30` | profile 重复次数 |
| `INDUCTOR_ASCEND_TILELANG_AUTOTUNE_TIMEOUT` | `30` | 单次 autotune 超时（秒） |
| `INDUCTOR_ASCEND_TILELANG_CONFIG_GENERATORS` | `tile_generator` | 候选生成器，可选 `tile_generator`、`carver` 或二者 |
| `TILELANG_NPUIR_ENABLE_AUTO_MULTI_BUFFER` | 未设置（按 config） | 强制覆盖 NPUIR auto multi-buffer pass |

---

## 测试与 Benchmark

`test/_inductor/tilelang/tilelang_test_utils.py` 会同时检查数值结果和生成代码，
确保被测图：

- 至少生成一个 TileLang kernel/factory；
- 不包含 `async_compile.triton`；
- 不绕过到 ATen/NPU/extern kernel；
- 使用 `target='npuir'`。

最终测试布局：

| 文件 | 覆盖内容 |
|---|---|
| `test_tilelang_ops.py` | tensor/scalar binary、unary、融合链、不同 rank、sum/max/min/mean |
| `test_tilelang_broadcast.py` | 2D/3D、单轴/多轴、尾块、动态 shape，以及最小 autotune case |
| `test_tilelang_mixed.py` | softmax、layer/RMS norm、大小 R reduction、广播后 reduction、ranked 高维 reduction |
| `benchmark_tilelang_softmax.py` | softmax 正确性与性能 |
| `benchmark_tilelang_layernorm.py` | layer norm 正确性与性能 |

测试文件在模块导入前设置 TileLang backend，并默认关闭 autotune。需要在已安装当前
仓库 `torch_npu`、TileLang 可用且 NPU 可见的环境中执行。

---

## 文件结构

```text
torch_npu/_inductor/
├── __init__.py                         调用 TileLang 单一初始化入口
└── codegen/
    ├── scheduling.py                   NPUTritonScheduling 与共享轴分析
    ├── triton.py                       NPUIndexTritonKernel / Triton fallback
    ├── split_tiling.py                 split/tiling/no-loop 轴选择
    ├── kernel_analysis.py              ReductionAnalysis
    └── tilelang/
        ├── __init__.py                 TileLang backend 注册与 NPU patch
        ├── tilelang_kernel.py          kernel、planner、emitter
        └── tilelang_scheduling.py      scheduling、autotune wrapper

test/_inductor/tilelang/
├── tilelang_test_utils.py
├── test_tilelang_ops.py
├── test_tilelang_broadcast.py
├── test_tilelang_mixed.py
├── benchmark_tilelang_softmax.py
└── benchmark_tilelang_layernorm.py

torch_npu/_inductor/docs/tilelang/
├── TILELANG_BACKEND.md                 本文：总览、边界与配置
├── tilelang_codegen.md                 调用链与生成代码细节
└── TRITON_PIPELINE.md                  共享 Triton scheduling/tiling/fusion
```

---

## 已知限制与调试

主要限制：

- 原生 reduction 仅 `sum`、`max`、`min`；一个 kernel 最多两个 reduction，双
  reduction 仅 `max + sum` 和 `sum + sum`。
- tuple reduction、`atomic_add`、split-scan、未验证的 predicate/comparison、
  不可映射 layout/indexing 会回退 Triton。
- 展开的多规约轴要求能确定静态轴范围。
- 动态 shape 由 wrapper 按 shape key 懒编译和缓存，支持范围取决于 layout plan
  能否对当前符号 shape 建模。
- 当前测试明确跳过 fp16/int32 多级融合链，因为中间值仍可能被物化为 fp32。

`TileLangKernel.codegen_kernel()` 当前会打印：

```text
====== TileLang planned prim_func ======
<生成的 @T.prim_func>
```

`TileLangScheduling._fallback_to_triton()` 通过 Inductor scheduling debug logger
记录 fallback 原因。Autotune 还会打印选中的 config，失败时打印异常并使用默认
TileLang config。需要调试 Inductor wrapper 时，可配合 `TORCH_COMPILE_DEBUG=1`
和 `TORCH_LOGS="+inductor"` 查看生成代码与路由过程。
