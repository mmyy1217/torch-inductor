# 两条路线在 scheduler 汇合

```text
PyTorch → FX → lowering
                ├─ 普通算子 → Loop IR → TileLangScheduling → 普通 TileLang kernel
                └─ FlexAttention → TileLangTemplateBuffer → 渲染专用模板
                                      ↓
                     wrapper 分配内存，依赖顺序调用多个 kernel
```

模板前后融合关闭；普通运算之间仍可融合。混合例子生成前置 add/mul、attention、后置 add 三个 TileLang 计算 kernel；dense TileLang 模板不读取 BlockMask 元数据。

以下路径相对于 `--source-repo`。精确版本见 [当前实现提交](https://gitcode.com/qq_51358522/pytorch/commit/5ea224c1a41905c7704304091d51dcb134651be7)。

| 目的 | 源码路径（torch_npu/_inductor/ 下） |
| --- | --- |
| 同时注册两条路线 | `tilelang/__init__.py` |
| 调度、模板分派、普通 kernel 参数适配 | `tilelang/codegen/tilelang_scheduling.py` |
| 普通 IR 生成 TileLang | `tilelang/codegen/tilelang_kernel.py` |
| attention 参数校验与 lowering | `kernel/tilelang_flex_attention.py` |
| 模板 IR 和 wrapper | `codegen/tilelang.py` |
| 模板源码渲染 | `codegen/tilelang_template.py` |
| attention 分块与流水线 | `kernel/tilelang_flex_kernel.py` |
| attention 编译与 launch | `runtime/tilelang.py` |
| attention 缓存标识 | `codegen/tilelang.py` 的模板源码摘要 |

普通 kernel 在第一次调用时编译；attention 模板在 wrapper 加载时编译。两条编译生命周期暂未统一。当前模板仅支持 dense，score/mask 尚未接入。GQA/MQA 在模板里映射 KV head，不复制展开 KV 张量。
