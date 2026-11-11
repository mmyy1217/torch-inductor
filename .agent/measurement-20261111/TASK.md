# Yellow：全量测量任务交接（2026-11-11）

用户明确要求将全部测量任务同步给你。你在自己已获授权的服务器独立复测；协调方现有 Grok 持有本地实验服务器队列，129/130由它继续。不要操作、重启或重复提交协调方队列。不同服务器结果分别保存，不能拼成同场加速比。无需再等阶段确认，但不得安装/重置共享环境、创建/重开/发布PR。

## 当前事实

- Run116：26/26正确性、156/156计时完成；TL全部超过Triton，FAS V4/TL为0.536–0.989。附完整表。
- 单独PR shape [B,HQ,HKV,M,N,D]=[1,32,1,8192,8192,128]：TL3312.125us、FAS V4 3318.206us，1.0018x基本持平（轮次胜负翻转）。不能等同PR16对FIA V5的1.03x。
- Run128连同125：三候选全矩阵结束，没有整体提升，不采用crossQ/D512slice/D256DN替换；附结果。
- Run129（Grok仍负责）：原版PR16对仅DN改动，同物理28 AIC、逻辑32块，PR方法5轮，原版5745.97us、DN5355.50us，1.0729x；28块补测待完成。此为worker交付结论，非FIA比较，不乘到作者1.03x上。
- Run130（Grok后继）：27shape TL/FAS V4/FIA V5，162次计时，尚未交付。你独立服务器的结果作为复测，不替代或混合它的记录。

## 你的测量任务

1. 记录硬件、实测AIC/AIV、Torch/torch_npu、CANN运行时/编译器、TileLang commit与dirty diff。先检查环境兼容，不静默更换编译器。
2. matrix26.json的26个shape，加上述PR shape，共27。BF16连续BNSD、dense forward、scale=D**-0.5、无mask/dropout；数组顺序B,HQ,HKV,M,N,D。
3. 当前TL源码为随包current_dn_kernel.py（SHA256见清单），不要使用老persistent.py b800d9f5或退化候选。这是完整Python kernel builder，接口shape顺序为B,HQ,M,N,D,HKV，必须用现有Inductor runtime overlay调用，不能误当字符串模板。先复核prepare_overlay.py与生产runtime接口；只在新实验overlay替换kernel文件，不改安装包。源码D128固定28块策略须如实记录，不擅自改成32并仍称相同候选；若硬件不同另列适配结果。
4. 基线：现有.agent/pr16/bench_fia.py里的flash和fia调用可以复用；FAS是npu_fusion_attention，FIA是npu_fused_infer_attention_score，BNSD、明确HQ/HKV、scale、softmax_lse_flag=False。重新证实实际dispatch到aclnnFusedInferAttentionScoreV5，不能只靠Python函数名。缺接口/不支持shape按unsupported记录，不能替换版本。它是安装版V5，不声称复现作者自编译V5。
5. 每个shape做TL、FAS V4、FIA V5同设备两轮反序，fresh process，10warmup+30NPUevent，合计162个任务。完整分块FP32 oracle、显式finite、atol=rtol=.03、pre/post正确性；失败停对应实验留证，不降阈值。保存每个样本、退出码、输入与源码hash、占用和中断记录，重算中位数。共享卡采用10秒占用检查、仅终止自己的进程组、保留完成数据续跑；不得同时竞争协调方队列。
6. Triton建议同场补齐27shape两轮（另外54任务，共216），沿现有生产Triton调用，数值与计时协议相同；若无法同场，引用116历史列并明确不同服务器/会话，PR未测则NA。不得混合历史数据宣称同场胜负。
7. PR16原版vsDN是单独测量口径：10warmup、每轮50次、5轮交替，profiler任务时间求每轮均值再取中位数；不要与上述API Event时间混用。当前原版/DN源码实验由Grok持有，你暂不自行重新实现；收到可移植129包后再做独立复测。此项明确pending-package，不阻塞V5矩阵。

## 回传

独立JSON保存全部原始样本、环境/源码/输入身份、每轮中位数和正确性；一张27行表含shape、TL、Triton、FAS V4、FIA V5、V4/TL、V5/TL、Triton/TL（大于1为TL快），NA注明原因。报告进度分为完成/等待设备/失败/未开始，不把排队当运行。不把不同core数、环境、scope的结果合并。

只在个人codeagent分支单一交流commit中追加CODEAGENT回复和小型结果；fetch后amend并用精确旧SHA的force-with-lease，保留双方历史。禁止PR及小组仓库push。原始大日志/tensor/profiler留本机。请先回传接收、环境及实际可执行范围，然后自主完成。
