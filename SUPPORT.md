# TileLang 的 PyTorch 支持契约

本页描述匹配社区提交的实现范围（见 community-source.json）。权威检查位于 `torch_npu/_inductor/kernel/tilelang_flex_attention.py`，callback 子集位于 `torch_npu/_inductor/kernel/tilelang_fx.py`。通过前端检查不等于所有规格均已在设备验证。

## 张量、shape 与执行方式

| 项目 | 支持条件 |
| --- | --- |
| 入口 | `torch.nn.attention.flex_attention.flex_attention`，经 `torch.compile(backend="inductor", fullgraph=True)`；设置 `torch_npu._inductor.config.flex_attention.implementation="tilelang"` 和 `options={"npu_backend":"tilelang"}` |
| 模式 | 前向推理；Python wrapper；不支持 backward、C++ wrapper、AOTInductor |
| 设备与 dtype | Q/K/V 在同一 NPU，均为 BF16 |
| Q | `[B,H,M,128]` |
| K / V | 均为 `[B,H,N,128]`，batch/head 必须与 Q 一致 |
| 布局 | rank 4、静态正整数尺寸、连续布局；size=1 维度不约束 stride |
| 序列长度 | M、N 分别为 128 的正倍数，可以不同；最小 128，不支持尾块或 decode 长度 1 |
| 大小上限 | `B*H*M*128 < 2**31` 且 `B*H*N*128 < 2**31`；B/H 没有额外固定上限，但受该条件与设备内存约束 |
| scale | 有限的静态 Python float/int，包括零和负数；默认值由 PyTorch 传入 |
| 输出 | 仅 Attention 张量，shape 与 Q 相同、BF16；不能消费 LSE/max 等辅助输出 |

不同 shape 可以分别静态编译。复用同一 callable 更换 shape 时，PyTorch 自动动态化可能产生被拒绝的符号尺寸；需要静态特化时可自行使用 `torch.compile(..., dynamic=False)`，不保证无需重编译。统一 runner 已设置 `dynamic=False`。

这里没有承诺某个最大可运行 M/N 或任意 B/H 都经过测试。整数元素数限制也不是内存容量保证。runner 的独立 CPU 前向参考逐 batch/head、按最多 128 个 Q 行分块构造 scores；反向会保留各块的 autograd 图。独立设备矩阵脚本则构造完整密集 scores，大 shape 仍可能耗尽参考计算内存；这不等同于 TileLang 内核拒绝该 shape。不支持 FP16/FP32 输入、GQA/MQA、Q/K/V batch 广播、不同 Dv、非连续张量、动态 shape。模板前后融合、多候选择优也未实现。

## 自己写一个 PyTorch 调用

先按 [SETUP.md](SETUP.md) 准备环境，并从显式指定的社区 checkout 创建 overlay。以下是最小前向调用；此片段只演示接口，数值验收使用 runner。

```python
import torch
import torch_npu
from torch_npu._inductor import config
from torch.nn.attention.flex_attention import flex_attention

config.flex_attention.implementation = "tilelang"
attention = torch.compile(flex_attention, backend="inductor", fullgraph=True, dynamic=False,
                          options={"npu_backend": "tilelang"})
q, k, v = [torch.randn(1, 4, 256, 128, device="npu", dtype=torch.bfloat16)
           for _ in range(3)]
with torch.no_grad():
    output = attention(q, k, v)
```

可以把 Q 长度改为 128、K/V 长度改为 384；K 与 V 的长度必须一致。改成 257 长度、D=64 或 FP16 当前会拒绝。shape、callback 或静态参数变化可能触发新特化和编译；第一次调用包含编译成本，不应当作稳定执行时延。

## score_mod 与 mask_mod

`score_mod(score, b, h, q_idx, kv_idx)` 和 `mask_mod(b, h, q_idx, kv_idx)` 必须能被追踪为支持的标量 FX 图。支持内联有限常数和索引表达式；捕获张量不支持，额外的捕获标量 placeholder 也不支持。Python 闭包中可被追踪内联的常量不同于运行时捕获输入。

| 类别 | 可翻译操作 |
| --- | --- |
| 算术 | 加、减、乘、真除、整数整除/取余、负号、abs |
| 比较 | `== != < <= > >=` |
| 组合 | bitwise/logical and、or、not；`torch.where` |
| 数学函数 | exp、tanh |
| 标量构造/转换 | bool、int32、int64、float32；零维 `aten.full` |
| 常量 | bool/int/有限 float；不接受 NaN/Inf 字面量 |

以最终 FX target、overload 和参数为准，不承诺名字类似的任意 Python/Torch 操作。例如 `aten.div.Tensor_mode` 仅接受 `rounding_mode="floor"`。索引算术保留整数语义，负数整除/取余有模板回归覆盖；当前集成版本的这部分尚未重新上板。子集支持不等于对除零、指数溢出等无效数值输入作保证。张量索引加载、任意 Python 控制流、未知函数/FX 节点会被拒绝。

典型已实现表达式见 [relative_bias](cases/DONE/relative_bias.py)、[softcap](cases/DONE/softcap.py)、[causal_with_bias](cases/DONE/causal_with_bias.py)。[document_mask](cases/DONE/document_mask.py) 使用索引算术划分文档，不表示支持读取捕获的 document-id 张量。

整除/取余要求两边均为整数；浮点 `s // 2`、`s % 2` 会在 lowering 拒绝。`tanh` 要求浮点输入，索引可先 `.to(torch.float32)`；整数 `exp` 仅接受 FX 输出 float32，布尔 `exp` 不支持。`where` 支持已列明的标量子图，但 BF16 `s` 与 Python 字面量分支可能产生未支持的 `scalar_tensor`/BF16 cast，当前明确拒绝；可用 `torch.where(m >= n, s, s * 0.0)` 表达零分支。不能将显式 BF16 cast 自动等价为 FP32。

## BlockMask

支持 dense，以及 partial/full/mixed 块；支持 batch/head 元数据广播、非连续 KV 块索引、零块行和块内全 mask。Q/KV block size 必须是静态正整数、128 的倍数且不超过 `2**30`；两者均为 `2**30` 时识别为 dense。

partial 的 counts/indices 必须存在；full 的 counts/indices 必须同时存在或同时缺省。每对 counts/indices 必须为同设备连续 int32，shape 分别是 `[Bm,Hm,ceil(M/Q_BLOCK)]` 与 `[Bm,Hm,ceil(M/Q_BLOCK),capacity]`，`Bm∈{1,B}`、`Hm∈{1,H}`。支持内部 13 字段 ABI，以及经规范化的 17 字段 ABI。

元数据内容应由合法的 `create_block_mask` 等路径构造；上述检查主要覆盖 dtype/layout/shape，不意味着逐元素验证手工元数据的计数、索引范围或重复块。完整块会跳过逐元素 mask 判断，不能错误地将 partial 块标为 full。

## 选项与错误

默认 `kernel_options` 即可。仅允许以下内部键：`BACKEND`（只能为 `"AUTO"`）、`PRESCALE_QK`（不可为真）、`ROWS_GUARANTEED_SAFE`、`BLOCKS_ARE_CONTIGUOUS`、`WRITE_DQ`、`OUTPUT_LOGSUMEXP`、`OUTPUT_MAX`、`_NPU_EXPLICIT_SCORE_MOD`。后两个 OUTPUT 选项不可为真；允许某个键不表示对应优化或 backward 已实现。不要将 Triton 的块大小/warps 调优参数直接传给此后端。

不支持的条件在 lowering/翻译阶段抛出带 `TileLang FlexAttention v2:` 前缀的 `NotImplementedError`，可能由 torch.compile 包装为编译错误；不会承诺自动退回 Triton。

| 典型报错片段 | 检查方向 |
| --- | --- |
| `contiguous static BF16 rank-4 Q/K/V required` | dtype、布局、rank；动态/非正尺寸也会单独拒绝 |
| `matching batch/heads ... dimension 128` | GQA/MQA、broadcast、Dk/Dv |
| `sequence lengths must be multiples of 128` | 尾块与 decode |
| `captured tensors are not supported` / `captured tensors/scalars are unsupported` | callback 是否读取外部输入 |
| `unsupported callback operation` | 对照上面的 FX 子集 |
| `auxiliary outputs are not supported` | 关闭 LSE/max 输出 |
| `unsupported kernel_options` | 删除不支持的选项 |
| `invalid BlockMask metadata shape/broadcast` | block size、counts/indices 维度与广播 |
| `Python wrapper required` / `inference-only` | wrapper、AOT、训练/反向模式 |

环境导入失败、缺失 DSL/SDK 接口或编译器错误不等于输入契约拒绝；先核对 [安装环境](SETUP.md)。选择 `--impl triton` 可作独立对照，但不保证其全部 21 个场景在当前环境通过。

## 用例与验证证据

[已实现的 7 个用例](cases/DONE/)：dense、causal、sliding_window、document_mask、relative_bias、softcap、causal_with_bias。

[尚不支持的 14 个用例](cases/TODO/)：gqa、mqa、batch_broadcast、fp16、different_value_dim、noncontiguous、captured_bias、captured_document、tail_length、decode、lse、max_scores、backward、backward_score。目录表示实现范围，不表示本次设备复测结果。

最新集成验证见 [PROGRESS.md](PROGRESS.md)：本轮仅验证了混合 dense 图的两组输入，其他 DONE 与稀疏场景尚未整套复测。
