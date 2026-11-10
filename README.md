# FlexAttention 用例与学习仓

当前对应 **普通 TileLang codegen + FlexAttention 模板** 的集成分支。这里只放用例和跑测工具，不复制 torch_npu 实现。

**源码仍在本地 review，尚未 push 到 GitCode。** 精确提交见 [community-source.json](community-source.json)；远端同事原分支不能直接运行当前模板。已有源码请显式传给 overlay，其他机器需先取得该提交。

| 阅读顺序 | 内容 |
| --- | --- |
| 1. [first_dense.py](first_dense.py) | 最简单的完整调用，按顺序写，不经过 runner |
| 2. [mixed.py](mixed.py) | add/mul → attention → residual；本轮已上板的集成例子 |
| 3. [CODE_MAP.md](CODE_MAP.md) | 普通 codegen 与模板在哪里分流 |
| 4. [SUPPORT.md](SUPPORT.md) | shape、dtype、callback 限制 |
| 5. [SETUP.md](SETUP.md) | 环境与运行命令 |
| [PROGRESS.md](PROGRESS.md) | 当前验证范围 |
| [AGENT_PROMPT.md](AGENT_PROMPT.md) | 后续工作边界 |

`--impl tilelang` 同时选择普通 TileLang codegen 和 attention 模板；`--impl triton` 选择 default codegen 和 Triton attention。两者都使用 `torch.compile(backend="inductor")`。

## build 是谁调用的？

```text
run.py: main() → 子进程 main(--worker) → run_worker()
  → load_case("dense")           加载 cases/DONE/dense.py
  → module.build(config, "npu")  返回 attention 函数，还未计算
  → compile_attention(fn, ...)   配置两个开关，调用 torch.compile
  → compiled(q, k, v)            首次编译并执行
  → module.build(config, "cpu")  仅取 mask/score 回调供 FP32 参考计算
```

[7 个 DONE](cases/DONE/)表示模板实现支持，[14 个 TODO](cases/TODO/)尚不支持。它们不是本次集成全部重新上板通过的清单。run.py 的 main A–D 负责批量调度，run_worker 1–8 负责单用例检查。

runner 保存 FX、pre/post fusion IR、output_code.py、环境和结果。大产物放在个人输出目录，不提交。性能工具保留，但当前集成分支没有性能结论；旧分支的测量记录已从当前目录移除。

## 跑测入口

准备好 SETUP.md 的 overlay 后：

```bash
python -P first_dense.py
python -P mixed.py
bash run.sh --case dense --impl tilelang --output /your/work/dense-check
bash run.sh --case dense --impl triton --output /your/work/triton-check
python -P validate_tilelang_matrix.py --output /your/work/matrix
python -P validate_tilelang_cache.py --phase cold --output /your/work/cache
python -P validate_tilelang_cache.py --phase warm --output /your/work/cache
```

`run.sh --mode bench` 做 event 计时，`graph_bench.py <output>/<impl>/dense` 做 dense Graph 计时；两种口径不要混用。目前保留工具，尚未对集成分支重测性能。
