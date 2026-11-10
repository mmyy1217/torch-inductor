# 环境与第一次运行

本仓库只是用例与辅助工具，不能 `pip install` 出一个新的 torch_npu 后端。需要另外取得社区源码、匹配 wheel、Huawei Ascend TileLang 与 CANN。历史验证依赖带既有修改的编译器和个人配套安装物；目前没有仅凭公开源码的一键从零复现方案。

## 准备两份源码

个人用例仓：`git clone https://github.com/mmyy1217/torch-inductor.git flex-attention-lab`。

实现目前在本地 `/home/zuomin/Projects/Compiler/wt/pytorch-tilelang-flex-integration`，分支和提交见 community-source.json。它尚未推送到 GitCode；新机器需要先取得这份提交，不能直接用同事原分支代替。拿到后使用自己的路径，不依赖上面的个人绝对路径。

个人仓没有 torch_npu、third_party 或子模块。overlay 只替换已安装匹配原生扩展之上的 `_inductor` Python 文件，不构建 PyTorch。

## 已验证的组件组合

| 组件 | 2026-11-10 A5-BLUE记录 |
| --- | --- |
| 设备 | Huawei Ascend 950PR，28 Cube / 56 Vector |
| Python | 3.11 |
| torch / torch_npu | 2.14.0+cpu / 2.14.0；NPU由torch_npu提供 |
| Triton-Ascend | 包元数据3.6.0.dev0+gitf3733306；triton模块报告3.5.0，保留该差异 |
| TileLang | 79eaaae37014abdd2ab7ee381732a1e906286e51，加服务器已有修改 |
| CANN运行时 | 9.2系列，历史安装记录为9.2.0-beta.2 |
| CANN设备编译器 | 9.3.0~weekly.20261009.01，对应20261009_000323778快照 |

TileLang修改的归档SHA256为 `aae5d77024255a0f95d6133d97af338c03390ad8dc40757bb3b3f7f96b029d63`，完整diff与安装物未随本仓分发。只checkout该TileLang commit不保证可运行。weekly `compiler/version.info` 的Version字段曾为9.2.0，不能只看目录名；应核对完整安装清单、timestamp和wrapper。

需要向环境提供者取得：匹配torch/torch_npu/Triton-Ascend wheel及SHA256、匹配TVM-FFI（历史0.1.12）、已构建的TileLang与完整补丁、CANN runtime/AICPU、weekly编译器/头文件/库、环境脚本与驱动配套说明。普通GPU Triton/TileLang不能替代Huawei Ascend后端。这里不提供未经验证的下载链接。

首次复现优先复用现成匹配环境，只做低并发单内核JIT；不要在共享设备服务器上构建PyTorch、TileLang或LLVM。不要混用其他历史实验的组件。

## 准备 overlay

先激活匹配Python环境，并加载环境提供者的运行时/编译器wrapper脚本；不要自行依次source两套CANN来猜测配套关系。

```bash
source /path/to/venv/bin/activate
source /path/to/matched-flex-environment.sh
export PYTHONSAFEPATH=1
export PYTHONPATH="$(python -P -c 'import os; print(":".join(p for p in os.environ.get("PYTHONPATH", "").split(":") if p))')"
python -P - <<'CHECK'
import torch, torch_npu, triton, tilelang
for module in (torch, torch_npu, triton, tilelang):
    print(module.__name__, getattr(module, '__version__', 'unknown'), module.__file__)
assert torch.__version__.startswith('2.14.')
assert torch_npu.__version__.startswith('2.14.')
CHECK
python -P prepare_python_overlay.py --source-repo /path/to/pytorch-npu --output /your/work/overlay
export PYTHONPATH="/your/work/overlay${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
export ASCEND_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCHNPU_PRECOMPILE_THREADS=1 OMP_NUM_THREADS=2
```

只选择获分配的卡。overlay不修改安装环境；若已经导入旧torch_npu，应重启进程。`overlay.json` 的源码摘要比只有Git HEAD更准确：它会覆盖当前checkout中的未提交Python修改。

## 第一个例子与后续验收

```bash
python -P first_dense.py
python -P mixed.py
bash run.sh --case dense --impl tilelang --output /your/work/dense-check
bash run.sh --case all --impl tilelang --output /your/work/tilelang-check
bash run.sh --case dense --impl triton --output /your/work/triton-check
python -P validate_tilelang_matrix.py --output /your/work/boundaries
python -P validate_tilelang_cache.py --phase cold --output /your/work/cache-check
python -P validate_tilelang_cache.py --phase warm --output /your/work/cache-check
```

先看 `summary.json`、各case的 `result.json` 和日志。默认check以固定seed对独立FP32参考比较所有输出，atol=rtol=0.03，不代表模型级精度验收。首次编译时间与稳态时延分开报告。

不同shape用 `--batch`、`--heads`、`--q-seq`、`--kv-seq` 配置；许可范围与明确拒绝条件见 [SUPPORT.md](SUPPORT.md)。当前矩阵通过不表示所有允许shape都已验证。Triton pointwise可以参与BlockMask元数据构造，判断attention后端要看模板和launch源码，不能只看是否出现Triton。
