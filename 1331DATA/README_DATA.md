# DSAI4207 · SFT:DPO 样本比例实验

涵盖摘要（summary_v1.2）和指令遵循（instruction_following_v1.0）。比例按
SFT 示范条数 : DPO 偏好对数计算。每个任务、每档预算的两阶段合计使用原版
`pool_N` 的全部独立问题。对照同档 1:1、1:3 和 3:1 时，样本 ID 和原始偏好对一致。

| 总预算 N | 1:3 SFT | 1:3 DPO | 3:1 SFT | 3:1 DPO |
|---:|---:|---:|---:|---:|
| 500 | 125 | 375 | 375 | 125 |
| 2,000 | 500 | 1,500 | 1,500 | 500 |
| 10,000 | 2,500 | 7,500 | 7,500 | 2,500 |

## 训练组

在 `<任务目录>/ratio_1_3/arms/` 或 `<任务目录>/ratio_3_1/arms/` 选择预算 N。

| 实验臂 | Stage 1 | Stage 2 |
|---|---|---|
| G4（SFT → DPO） | `G4_stage1_sft_N.jsonl` | `G4_stage2_dpo_N.jsonl` |
| G5（DPO → SFT） | `G5_stage1_dpo_N.jsonl` | `G5_stage2_sft_N.jsonl` |

SFT 读取 `messages`，DPO 读取 `prompt`、`chosen`、`rejected`。
同任务、同比例、同预算下，G4 与 G5 的同训练方法文件逐字节相同。
沿用既有 M0 checkpoint、Qwen 模板、两阶段衔接和评测设置。若启用
`assistant_only_loss=True`，对 SFT 专用分词器调用随包
`qwen_training_template.py` 的 `enable_assistant_loss_mask`。
预热源文件位于 `summary_v1.2/warmup_5k.jsonl`。

## 分配与验证

沿用 1:1 的 A/B 配对。每档选中一半配对。1:3 将选中配对的 A 样本改派 DPO，
3:1 将相同配对的 B 样本改派 SFT。分层控制摘要的 chosen/rejected 联合来源与
指令遵循的约束数，并平衡文本长度和子版块或约束标签。小预算是同一比例、
同一训练方法大预算的有序前缀。每种比例的 SFT 和 DPO 无重复分配。

`<任务目录>/splits/assignments_N.jsonl` 记录原始 A/B 分配、样本和偏好对哈希，
以及两个新比例的训练方法。`manifests/manifest.json` 记录源文件、样本计数、
长度 SMD、分层偏差及文件 SHA-256。各任务的 `manifests/source_manifest.json`
保留原版的来源与数据清洗记录。解压后在包根目录运行 `shasum -a 256 -c CHECKSUMS.sha256`。

`evaluation_plan.json` 的路径相对包根目录，包含 TL;DR、IFEval 和两任务共用的
MMLU 及 ARC-Challenge 路径。包内评测与预热文件沿用原版字节。
数据按原交付约定在课程组内使用。
