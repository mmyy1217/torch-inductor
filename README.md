# FlexAttention 用例与学习仓

本仓只放用例、说明和跑测工具。实现位于个人 [GitCode 分支](https://gitcode.com/qq_51358522/pytorch/tree/pytorch-tilelang-flex-integration)，精确提交见 [community-source.json](community-source.json)。不复制 PyTorch/torch_npu，不自动修改安装环境。

| 从哪里看 | 内容 |
| --- | --- |
| **1. [first_dense.py](first_dense.py)** | 最简单的完整调用，从上往下读 |
| 2. [SUPPORT.md](SUPPORT.md) | 本版支持的 shape、MHA/GQA/MQA，以及不支持的功能 |
| 3. [SETUP.md](SETUP.md) | 新机器的环境与 overlay |
| 4. [CODE_MAP.md](CODE_MAP.md) | 普通 codegen 和模板的调用路线 |
| 5. [PERFORMANCE.md](PERFORMANCE.md) | TileLang / Inductor Triton / torch_npu 三方性能 |
| [mixed.py](mixed.py) | 普通算子 → 模板 → 普通算子的简单例子 |
| [PROGRESS.md](PROGRESS.md) / [AGENT_PROMPT.md](AGENT_PROMPT.md) | 进展与后续交接 |

**当前只做 dense 前向，支持 D=128/256/512、MHA/GQA/MQA。**
[DONE](cases/DONE/) 只有 dense、gqa、mqa；其他在 [TODO](cases/TODO/)，不能当成已支持。

## 第一次运行

按 SETUP.md 建好 overlay 后：

```bash
python -P first_dense.py
bash run.sh --case gqa --impl tilelang --output /your/work/gqa
python -P bench.py --source-repo /path/to/pytorch-npu --impl tilelang --shape 2 8 512 2048 --kv-heads 2 --head-dim 256
```

`bench.py` 直接运行指定源码仓的同名性能脚本，避免维护两套计时逻辑。三方都用它；`run.py` 留给批量用例和中间产物收集。

## build 是谁调用的？

```text
run.py: main → 子进程 run_worker
  → load_case("dense")           加载 cases/DONE/dense.py
  → module.build(config, "npu")  返回 attention 函数，还未计算
  → compile_attention(fn, ...)   配置后端，调用 torch.compile
  → compiled(q, k, v)            首次编译并执行
```

`--impl tilelang` 选择普通 TileLang codegen + attention 模板；`--impl triton` 选择 default codegen + Triton attention。两者的 compile backend 都叫 `inductor`。
runner 保存 FX、pre/post fusion IR、output_code.py、环境和结果。大产物放个人输出目录，不提交。
