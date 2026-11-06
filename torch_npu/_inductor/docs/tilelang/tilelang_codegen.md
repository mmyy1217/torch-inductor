# TileLang Codegen 调用链

本文说明当前 TileLang 后端从 Inductor scheduler 到 `@T.prim_func` 的代码生成
流程，重点描述 `torch_npu/_inductor/tilelang/codegen` 内部的职责边界。

后端能力、配置和测试矩阵见 [TileLang Inductor Backend](TILELANG_BACKEND.md)；
共享的 NPU Triton scheduling、tiling 与融合策略见
[NPU Triton Pipeline](TRITON_PIPELINE.md)。

---

## 激活与注册

推荐在每次编译时通过 `npu_backend` option 选择 TileLang：

```python
compiled = torch.compile(
    fn,
    backend="inductor",
    options={"npu_backend": "tilelang"},
)
```

环境变量 `TORCHINDUCTOR_NPU_BACKEND=tilelang` 仅作为兼容入口保留，并且必须在
导入 `torch_npu` 前设置。

`torch_npu/_inductor/__init__.py` 在该次编译选择 TileLang 后端时调用
`tilelang.initialize_tilelang_backend()`，由该入口完成以下注册：

```text
register_backend_for_device(
    "npu",
    TileLangScheduling,
    NPUWrapperCodeGen,
    CppWrapperNpu,
)
```

同时注册 NPU device overrides、decomposition、fallback 和 MM lowering，并安装
Inductor 调度 NPU kernel 所需的 patch。`patch_has_triton()` 在这里表示 NPU 可以
使用 SIMD scheduling 基础设施，不代表 TileLang 最终生成 Triton 源码。

---

## 当前架构

```text
上游 PyTorch Inductor
├── SIMDKernel
│   └── TritonKernel
│       └── NPUIndexTritonKernel
│           └── TileLangKernel
│
└── SIMDScheduling
    └── TritonScheduling
        └── NPUTritonScheduling
            └── TileLangScheduling
```

继承的目的不是复用 Triton 源码生成，而是复用 Inductor 的 SIMD 调度入口以及
NPU 的 range tree、indexing 和轴分析能力。

| 组件 | 复用的能力 | TileLang 自身职责 |
|---|---|---|
| `TileLangScheduling` | `NPUTritonScheduling` 的 node schedule、维度分析和基础融合判断 | TileLang 融合限制、节点级 fallback、template 转发、wrapper 生成 |
| `TileLangKernel` | `NPUIndexTritonKernel` 的 range tree、indexing 映射和 axis 元数据 | 捕获 value/buffer/reduction，构建 plan，生成 TileLang 源码与调用 |

`TileLangKernel` 没有用 Triton `self.body` 生成最终源码。调度过程中继承的
`codegen_body()` 仍负责维持父类的 scheduling/CSE 生命周期，但
`KernelProgramEmitter` 不读取 `self.body`。

---

## 端到端调用链

```text
torch.compile(..., backend="inductor")
  │
  ▼
Inductor Scheduler
  │
  ▼
TileLangScheduling.codegen_node(node)
  ├─ 检查 template、split-scan 和 reduction 数量
  ├─ 不支持 → _triton_scheduling.codegen_node(node)
  └─ 支持 → SIMDScheduling.codegen_node(self, node)
       │
       ▼
TileLangScheduling.codegen_node_schedule(features, nodes)
  └─ NPUTritonScheduling.codegen_node_schedule(...)
       ├─ TileLangScheduling.select_tiling()
       ├─ TileLangKernel(tiling, features=...)
       ├─ decide_codegen_dims_in_kernel(...)
       ├─ 遍历 node body，触发 load/store/reduction
       ├─ TileLangKernel.codegen_kernel()
       ├─ TileLangScheduling.define_kernel()
       └─ TileLangKernel.call_kernel()
```

### 1. 调度与轴分析

`TileLangScheduling.select_tiling()` 将 pointwise numel 归一为 `x`，存在 reduction
时再加入 `r`：

```text
pointwise → {"x": xnumel}
reduction → {"x": xnumel, "r": rnumel}
```

随后共享的 `decide_codegen_dims_in_kernel()` 完成：

- node ranges 与 kernel range tree 的对齐；
- load/store indexing 的转换与冗余轴替换；
- store unified indexing 的保护；
- split、tiling、no-loop axis 选择；
- reduction 维度分析。

这些结果以 `sorted_axis`、`split_axis`、`tiling_axis`、`low_dims` 等元数据保留在
`TileLangKernel` 上，供 layout planner 和 tiling config 生成使用。

### 2. 执行 node body 并捕获元数据

Inductor 遍历 schedule node 时仍通过标准 ops handler 调用 kernel 方法，但
TileLang 不在此阶段直接拼接最终 `T.*` 源码。

| 入口 | 捕获内容 |
|---|---|
| `TileLangOverrides.*` | 通过 `PendingOp` 记录即将产生的 primitive、operand 和 dtype |
| `load()` | 输入 buffer、index、numel、dtype，以及 CSE value 到 buffer 的绑定 |
| `store()` | 输出 buffer、输出 index、结果 value，并校验整条计算图的 op/dtype |
| `reduction()` | reduction 类型、源 value、结果 value 和 accumulator dtype |
| `store_reduction()` | reduction 结果与最终输出 buffer 的绑定 |
| `create_cse_var()` | 将 `PendingOp` 绑定到新产生的 CSE value |

主要状态集合：

| 字段 | 含义 |
|---|---|
| `_values` | CSE value 的 dtype、producer op、来源 buffer 或常量 |
| `_buffers` | 输入/输出 buffer 的指针名、local 名、角色、dtype、index 和 numel |
| `_outputs` | pointwise 输出 local buffer 与结果 value/index 的绑定 |
| `_reduction_vars` | reduction 结果到 reduction 类型、源 value 和 dtype 的映射 |
| `_reduction_outputs` | reduction 输出 local buffer 与结果 value/dtype 的映射 |

这一阶段得到的是 planner 输入，不是最终 kernel program。

### 3. 构建 Operation Graph

Builder 将捕获状态转换为显式 operation：

- `PointwiseOpPlan`：一段类型化向量计算；
- `ReductionOpPlan`：reduction source、accumulator、mode 和 reduction dims；
- `BroadcastOpPlan`：source/destination tensor 与 broadcast dims。

`BroadcastInsertionPass` 在 consumer 前插入必要的 broadcast operation。
`TileLangOperationGraph.validate()` 检查：

- operation id 唯一；
- 一个逻辑输出只有一个 producer；
- 每个输入来自已有 producer 或 external value；
- operation graph 无环；
- producer 位于 consumer 之前。

预期内的不支持场景使用 `TileLangUnsupported` 记录失败阶段、operation 和 origin，
由 `TileLangScheduling.codegen_node()` 捕获为 `NotImplementedError` 并回退 NPU
Triton。

### 4. 构建 Kernel Plan

`TileLangKernel.codegen_kernel()` 按是否存在 reduction 输出选择 builder：

```python
if self._reduction_outputs:
    plan = TileLangReductionProgramBuilder(self).build(name)
else:
    plan = TileLangPointwiseProgramBuilder(self).build(name)
```

Builder 的产物统一为 `TileLangKernelPlan`：

| Plan | 作用 |
|---|---|
| `KernelSignaturePlan` | `@T.prim_func` 名称和 tensor 参数 |
| `KernelLaunchPlan` | `T.Kernel` grid 与 binding |
| `BufferPlan` | shared buffer 的 shape、dtype 和 scope |
| `TransferPlan` | GM/shared 之间的 copy |
| `XTileRegionPlan` | pointwise/X 轴循环、分配、搬运和子区域 |
| `RTileRegionPlan` | R block 循环、reduction operation 和累积方式 |
| `TileLangKernelPlan` | 汇总 signature、launch、operation、region 和 preamble |

`TileLangKernelPlan.validate()` 验证每个 operation 恰好被一个 region 调度，并验证
region 顺序符合 producer/consumer 依赖。这里的 `RuntimeError` 表示 planner 内部
不变量被破坏，不作为普通能力缺失静默 fallback。

#### Pointwise Builder

`TileLangPointwiseProgramBuilder`：

1. 优先构建保留逻辑轴与 broadcast 关系的 ranked layout；无法建立时使用 linear
   layout。
2. 从输出 value 递归构建类型化 `VecOp`，并通过 `TileLangOpSpec` 做 dtype、改写和
   intrinsic 校验。
3. 规划输入/输出/中间 buffer、GM/shared transfer、broadcast 和 pointwise
   operation。
4. 使用 block/remain 元数据覆盖非整除尾块。

#### Reduction Builder

`TileLangReductionProgramBuilder`：

1. 根据 indexing 区分 matrix、scalar、row broadcast 等输入布局。
2. 为 X tile 与 R tile 分别建立 region。
3. 用 `_XBLOCK`/`_XBLOCK_SUB` 划分 pointwise 轴，用 `_RBLOCK` 划分 reduction
   轴。
4. 首个 R block 的 reduction 使用 `clear=True`，后续 block 使用
   `clear=False` 累积。
5. 将支持的 scalar/vector epilogue 安排到对应 region。

### 5. 发射 TileLang 源码

`KernelProgramEmitter.emit(plan)` 只读取已经验证的 plan，不再查询 `V.graph`：

```text
TileLangKernelPlan
  ├─ SignatureEmitter         → @T.prim_func signature
  ├─ KernelLaunchPlan         → with T.Kernel(...)
  └─ LoopRegionEmitter
       ├─ BufferPlan          → T.alloc_shared
       ├─ TransferPlan        → T.copy
       ├─ BroadcastOpPlan     → T.vbrc
       ├─ PointwiseOpPlan     → T.v*
       └─ ReductionOpPlan     → T.reduce
```

这种分层使 capture、合法性校验、调度计划和字符串 emission 相互独立，避免在
源码发射阶段重新推断 graph 或 layout。

---

## Wrapper、缓存与 Autotune

`TileLangScheduling.define_kernel()` 将生成源码写入 `wrapper.header`，并生成：

1. prim_func factory：接收 numel、静态轴参数和 tiling 参数；
2. shape-keyed cache：key 由当前 numel 与静态轴值组成；
3. runtime wrapper：cache miss 时生成 tiling candidates，并编译选中的 prim_func；
4. kernel call：只把 tensor 参数传给已经编译的 TileLang kernel。

Autotune 使用当前 axis 元数据和 `TileGenerator` 生成候选 config。候选同时携带
block 参数和 `multibuffer`：去重 key 包含 `multibuffer`，因此同一 block 的
`False/True` 版本都会保留。

关闭 autotune 时直接使用首个保守 config；generic reduction 的默认值限制为
`X0BLOCK=8`、`R0BLOCK<=1024`、`multibuffer=False`。开启 autotune 时，wrapper
按 `multibuffer` 将候选分组，为每组创建独立 `AutoTuner` 并传入固定的 NPUIR
pass config。这样并行候选编译不会通过共享可变状态串扰。两个组的结果再按 latency
比较，获胜 config 用于最终编译。Autotune 全部失败时仍回到默认 TileLang compile，
不会因为调优失败切换到 Triton。

编译入口为：

```python
tilelang.compile(
    prim_func,
    target="npuir",
    pass_configs=_tilelang_npuir_pass_configs(selected_multibuffer),
)
```

`_tilelang_npuir_pass_configs()` 将这个布尔值映射为
`npuir.enable_auto_multi_buffer`。环境变量
`TILELANG_NPUIR_ENABLE_AUTO_MULTI_BUFFER` 若已设置，会作为显式覆盖值优先生效。

`TileLangKernel.call_kernel()` 向 wrapper call section 写入 tensor、numel 和必要的
静态轴实参。动态 shape 的不同具体值会形成不同 cache key；是否能生成 TileLang
kernel 仍取决于当前 layout plan 能否建模该 shape。

---

## Fallback 边界

Fallback 分为两个阶段：

### 融合阶段

`TileLangScheduling.can_fuse()` 对 reduction 组合和 epilogue 关系施加比 NPU Triton
更严格的限制。返回 `False` 只表示拒绝把节点融合在一起；拆分后的节点仍会分别
尝试 TileLang codegen。

### 节点 Codegen 阶段

以下情况由 `_triton_scheduling.codegen_node(node)` 重新生成整个节点：

- template 或 split-scan 节点；
- 实际进入 codegen 的 reduction 数量超过限制；
- buffer dtype 或 primitive/dtype 组合不支持；
- tuple reduction、未支持的 reduction 类型或 `atomic_add`；
- operation graph、broadcast、layout/indexing 无法构造为 TileLang plan。

Template 路径通过 `codegen_template()` 直接委托 NPU Triton，不进入 TileLang
pointwise/reduction planner。

---

## 与 NPU Triton 的关键差异

| 阶段 | NPU Triton | TileLang |
|---|---|---|
| Node body | 生成 `tl.load/tl.store` 等表达式 buffer | 捕获 value、buffer、index、dtype 和 operation |
| 中间表示 | Triton body/code buffers | `TileLangOperationGraph` + `TileLangKernelPlan` |
| 最终 emission | `codegen_kernel()` 读取 `self.body` | `KernelProgramEmitter` 读取已验证 plan |
| 计算形式 | Triton 标量表达式、`tl.arange` 与 mask | `T.copy`、`T.v*`、`T.vbrc`、`T.reduce` |
| 编译 | `async_compile.triton(...)` | `tilelang.compile(..., target="npuir")` |
| 缓存 | Triton compile/autotune cache | wrapper 内 shape-keyed compiled-kernel cache |
| 调用 | `kernel.run(..., stream=...)` | wrapper function 调用已编译 TileLang kernel |

两者共享的轴分析和融合基础详见
[NPU Triton Pipeline](TRITON_PIPELINE.md)。

---

## 调试

`TileLangKernel.codegen_kernel()` 当前会打印规划完成后的源码：

```text
====== TileLang planned prim_func ======
<生成的 @T.prim_func>
```

节点 fallback 原因通过 Inductor scheduling debug logger 记录。可配合：

```bash
TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+inductor" python your_script.py
```

查看 wrapper、kernel source 和路由过程。
