# Agent handoff · LIU Shiyu 训练部分

读完这一份就可以接着做，不需要再问用户环境、模型或实验规则。人类可读的逐步清单在同目录 `训练全流程.md`。两份冲突时，以本文件里的**机器事实**和**实验合同**为准；本文件写于 2026-11-04 16:50（UTC+8）。

用户是 LIU Shiyu（），负责训练。数据工程是 PAN Haoren，评测是 PAN Yubing。用户只有大约一周（从 2026-11-04 到 2026-11-10）完成训练交付。现在环境已经就绪，训练脚本和全部 run 都还没开始。

---

## 0. 你接下来立刻做什么

按这个顺序，不要跳到 44 次正式训练：

1. 用下面的 venv Python 写训练脚本，放到 `LSYNEED/training/`。checkpoint 只写到 `03_runs/`。
2. 用真实 0.5B + LoRA，在 GPU 上对 `G2_sft_500.jsonl` 跑 **1 step SFT**，对 `G3_dpo_500.jsonl` 跑 **1 step DPO**。输出放 `LSYNEED/training/smoke/`，不要放进 `03_runs/`。
3. 用 `test_200_main.jsonl` 的 3 条做贪婪生成（`max_new_tokens=128`，`do_sample=False`），确认会在 `<|im_end|>` 停下。
4. 1 step 和生成都通过之后，训 **M0**（`warmup_5k.jsonl`，种子 42），合并 LoRA。
5. 再跑通 `G2_sft_500` 种子 42，以及 `G4` 500 种子 42 的 stage1→合并→stage2。
6. 然后按第 6 节的优先级跑完其余实验，写 `03_runs/RUN_INDEX.md`。

评测脚本、LCWR、MMLU、ARC、IFEval、盲评都不是这一周的训练任务。不要改 `01_data/v1.0/` 里的任何文件。

---

## 1. 已经完成的事

### 1.1 数据

2026-11-04 16:48 用 venv Python 对 `01_data/v1.0/manifests/manifest.json` 做了 SHA256 校验：

```text
manifest files 46
problems 0
```

训练需要的文件都在，并且和 manifest 一致，包括早先这台机器上缺过、现在已经补上的：

- `01_data/v1.0/warmup_5k.jsonl`
- `01_data/v1.0/canonical_pool.jsonl`
- `arms/G2_sft_{500,2000,10000}.jsonl`
- `arms/G3_dpo_{500,2000,10000}.jsonl`
- `arms/G3prime_dpo_500.jsonl`
- `arms/G4_stage1_sft_{500,2000,10000}.jsonl`
- `arms/G4_stage2_dpo_{500,2000,10000}.jsonl`
- `arms/G5_stage1_dpo_{500,2000,10000}.jsonl`
- `arms/G5_stage2_sft_{500,2000,10000}.jsonl`

数据版本 `v1.0`，协议 `v1.2-final`，`synthetic = false`。只读。

### 1.2 Python 环境

不要往 Anaconda base 里装包。Anaconda 是 Python 3.14，而且 `requirements.txt` 装不进去。

项目 venv：

```text
C:\Users\liush\Desktop\DSAI4207_shared\02_code\DSAI4207_code\.venv\Scripts\python.exe
```

- 解释器：Python 3.12.14（`C:\Users\liush\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe` 创建的 venv）
- 不用 Python 3.11。`numpy==2.5.4` 要求 `>=3.12`，3.11 会直接失败。
- 本机 PowerShell 禁止运行脚本（`profile.ps1` / `Activate.ps1` 会报 UnauthorizedAccess）。调用上面的 `python.exe` 绝对路径，不要依赖 `Activate.ps1`。

已验证的导入（2026-11-04 16:48）：

| 包 | 版本 | 说明 |
|---|---|---|
| torch | `2.11.0+cu128` | `torch.cuda.is_available() == True`，设备 `NVIDIA GeForce RTX 4060 Laptop GPU` |
| trl | `1.16.0` | 与 `requirements.txt` 一致 |
| transformers | `5.20.1` | 与 `requirements.txt` 一致 |
| datasets | `5.1.0` | 与 `requirements.txt` 一致 |
| peft | `0.21.2` | **不在** `requirements.txt` 里。TRL 把它放在 extra `peft>=0.13.0`。已经另装。不要卸掉 |
| accelerate | `1.15.0` | 与 `requirements.txt` 一致 |

`requirements.txt` 写的是 `torch==2.14.1`。PyPI 上这个版本的 Windows 轮子是 **CPU**（`2.14.1+cpu`，`cuda False`）。官方 CUDA 索引里没有 `2.14.1+cu128`。有 `2.14.1+cu130`，但这台驱动是 `566.26`，`nvidia-smi` 报告的最高 CUDA 是 **12.7**，cu130 不能用。因此只把 torch 换成了 `2.11.0+cu128`，其余包仍是 `requirements.txt` 的版本。

不要重新执行 `pip install -r requirements.txt`。那会把 CUDA 版 torch 盖回 CPU 版。

Windows 上 import torch 时可能打印：

```text
NOTE: Redirects are currently not supported in Windows or MacOs.
```

这是 torch elastic 的提示，不是失败。

### 1.3 Base 模型

已下载，不是 Instruct：

```text
C:\Users\liush\Desktop\DSAI4207_shared\LSYNEED\models\Qwen2.5-0.5B
```

- repo：`Qwen/Qwen2.5-0.5B`
- revision：`060db6499f32faf8b98477b0a26969ef7d8b9987`
- `config.json` 存在，`architectures = ["Qwen2ForCausalLM"]`，`torch_dtype = bfloat16`
- `model.safetensors` 存在，988,097,824 字节
- 词表 `vocab_size = 151936`（配置里的值）。协议里的 tokenizer 长度是 151,665（151,643 BPE + 22 个特殊 token）。训练脚本加载这个本地目录，不要再从 Hub 下 Instruct。

### 1.4 还没有的东西

- `LSYNEED/training/` 不存在。训练脚本还没写。
- `03_runs/` 里只有 `README_RUNS.md`。没有 M0，没有微调 run。
- 没有 1-step smoke，没有合并过的 checkpoint，没有 `RUN_INDEX.md`。

---

## 2. 机器

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8188 MiB（约 8GB） |
| 驱动 | 566.26，nvidia-smi CUDA Version 12.7 |
| 系统 | Windows，PowerShell |

8GB 上的固定做法：`per_device_train_batch_size = 1`，`gradient_accumulation_steps = 16`，等效 batch **16**。显存不够时只允许把这两个数朝相反方向调，乘积保持 16。所有 run 用同一套。4060 支持 bf16，用 bf16。

M0 序列最长 2048，DPO 还要再放一份冻住的参考模型。这两类最容易 OOM。OOM 时先降 micro-batch，不要改等效 batch，不要改学习率。

提案里的「一张 4090、10–14 GPU 小时」是速度估算。这张 4060 笔记本慢几倍，全部跑完大约要机器连续跑一两天。算力够用，不需要换 H100 / A100。

---

## 3. 实验合同

这些数字来自最终 proposal。主结果必须照做。为了 loss 更好看去改它们，实验作废。

| 项目 | 锁定值 |
|---|---|
| 起点模型 | 本地 `LSYNEED/models/Qwen2.5-0.5B`，Base，不是 Instruct |
| 方法 | LoRA，r = 32，alpha = 64 |
| LoRA dropout | 0.05（工程默认，44 次相同） |
| LoRA 目标层 | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| SFT 学习率 | 1e-4 |
| DPO 学习率 | 5e-6 |
| DPO beta | 0.1 |
| 每个阶段 epoch | 正好 1 |
| 优化器 | AdamW |
| 日程 | cosine，`warmup_ratio = 0.03` |
| 梯度裁剪 | 1.0 |
| `dataloader_drop_last` | false |
| 种子 | 42、1337、2024。10,000 档只用 42。M0 只用 42 |
| TL;DR `max_length` | 1024 |
| M0 `max_length` | 2048 |
| DPO `max_prompt_length` | 768（协议上限 `prompt_tokens ≤ 768`） |
| SFT loss | 只计算 assistant 回复和结尾的 `<\|im_end\|>`。TRL 1.16 的 `SFTConfig` 使用 `assistant_only_loss=True` |
| 精度 | bf16 |
| 日志 | `report_to = none`，`logging_steps = 1`，只在 epoch 结束保存 |
| 预期步数 | `ceil(样本数 / 16)` |

Tokenizer 每次加载本地模型后立刻执行：

```python
tokenizer.eos_token = "<|im_end|>"       # id 151645
tokenizer.pad_token = "<|endoftext|>"    # id 151643
tokenizer.padding_side = "right"         # 训练 right；生成评测才 left
```

`pad_token` 和 `eos_token` 必须是两个 id。设成同一个，模型学不会停止。

DPO 的参考模型是另一次 `from_pretrained` 得到的冻住权重，和正在训练的 LoRA 模型不共享参数。

DPO 出现 NaN 或 loss 飞掉：所有 DPO run 一起改成学习率 `1e-6`、beta `0.3`，已训完的 DPO 用新设定重跑，并在该 run 的 README 写明「偏离预注册，使用 proposal 的发散预案」。不要只改一条。

500 档每个阶段只有大约 16 或 32 步。这是提案接受的情况。主表保持 1 epoch。3 epoch 只能在 44 个目录都交完之后，另开 `_epoch3` 目录做敏感性，不能覆盖 1 epoch 的结果。

### 谁从哪份权重开始

| 臂 | 做什么 | 起点 | DPO 参考模型 |
|---|---|---|---|
| M0 | 5,000 条通用指令 SFT | 原始 Base | 无 |
| G2 | 纯 SFT，N 条 chosen | M0 merged | 无 |
| G3 | 纯 DPO，N 对 | M0 merged | M0 merged |
| G4 stage1 | SFT，part A（0.5N） | M0 merged | 无 |
| G4 stage2 | DPO，part B（0.5N） | **该种子**的 G4 stage1 merged | 同一份 stage1 merged |
| G5 stage1 | DPO，part B（0.5N） | M0 merged | M0 merged |
| G5 stage2 | SFT，part A（0.5N） | **该种子**的 G5 stage1 merged | 无 |
| G3′ | 冷启动 DPO，只有 500、种子 42 | 原始 Base | 原始 Base |

两阶段之间：把 LoRA `merge_and_unload` 成完整模型，stage2 是新进程、新优化器、新的 LoRA。不要把 stage1 的 optimizer 状态拷过去。不同种子的 stage2 不要共用别人的 stage1。

G4 和 G5 在同一预算下看到的是同一批句子，只是阶段顺序相反。这是课题要回答的问题。不要改数据、不要重采样、不要为了某一个臂调学习率。

不要用 `Qwen2.5-0.5B-Instruct` 或任何已经做过通用 SFT 的现成模型代替 M0。M0 的意义是：5,000 条 `no_robots` 已经去掉摘要，使 500 档成为这条后训练管线里第一次接触 TL;DR。

---

## 4. 数据怎么读

根目录：`C:\Users\liush\Desktop\DSAI4207_shared\01_data\v1.0`

SFT 文件（`G2_sft_*`、`G4_stage1_sft_*`、`G5_stage2_sft_*`、`warmup_5k.jsonl`）训练字段用 `messages`。

DPO 文件（`G3_dpo_*`、`G4_stage2_dpo_*`、`G5_stage1_dpo_*`、`G3prime_dpo_500.jsonl`）训练字段用对话列表 `prompt`、`chosen`、`rejected`。不要改成纯字符串。

样本数：

| 文件 | 行数 | 等效 batch 16 时的预期步数 |
|---|---:|---:|
| `warmup_5k.jsonl` | 5,000 | 313 |
| `G2` / `G3` / `G3prime` 的 500 | 500 | 32 |
| `G4` / `G5` 每个阶段的 500 | 250 | 16 |
| `G2` / `G3` 的 2000 | 2,000 | 125 |
| `G4` / `G5` 每个阶段的 2000 | 1,000 | 63 |
| `G2` / `G3` 的 10000 | 10,000 | 625 |
| `G4` / `G5` 每个阶段的 10000 | 5,000 | 313 |

实际 `global_step` 和上表差超过 1 时，在该 run 的 README 写原因。常见原因是 `drop_last=True` 或拿错了预算文件。

训练开始前，对 1 条样本把 `(token_id, token 文本, label)` 打进 `train.log` 开头。SFT 里 system 和 user 的 label 是 `-100`，assistant 正文和结尾 `<|im_end|>` 的 label 是真实 token id。

种子要设到 Python、NumPy、PyTorch 和 Trainer 的 `seed`。

---

## 5. 要写的代码

放在 `C:\Users\liush\Desktop\DSAI4207_shared\LSYNEED\training\`：

| 文件 | 作用 |
|---|---|
| `train_sft.py` | 读 `messages`，`assistant_only_loss=True`，1 epoch，保存 LoRA adapter |
| `train_dpo.py` | 读 `prompt` / `chosen` / `rejected`，加载冻住的 ref，1 epoch，保存 LoRA adapter |
| `merge_lora.py` | adapter 合并进当时的起点模型，写出 `checkpoint/merged/` |
| `make_configs.py` | 生成全部 yaml，避免手改 |
| `configs/*.yaml` | 每个 run 一份。开训时把这份 yaml 原样复制进该 run 目录，命名为 `config.yaml` |

每份 yaml 至少包含：臂名、阶段、预算、种子、数据文件绝对路径、起点模型绝对路径、DPO 参考模型绝对路径（没有则写 `null`）、第 3 节的全部超参、数据版本 `v1.0`、输出目录、实际 torch / trl / peft / transformers 版本。

调用方式用 venv 的 python.exe 绝对路径，例如：

```powershell
& "C:\Users\liush\Desktop\DSAI4207_shared\02_code\DSAI4207_code\.venv\Scripts\python.exe" C:\Users\liush\Desktop\DSAI4207_shared\LSYNEED\training\train_sft.py --config <yaml>
```

接口以本机已安装的 TRL 1.16 为准，不要照抄 2024 年的 TRL 教程。仓库里可参考的写法在 `02_code/DSAI4207_code/data_engineering/tests/test_data_invariants.py` 的 `test_trl_smoke`（那个测试用的是临时小模型，正式训练必须用本地 0.5B + LoRA）。

---

## 6. 正式 run 的顺序和目录

目录名：`03_runs/{arm}_{budget}_seed{S}_{YYYYMMDD}/`。两阶段把 stage 写进 arm 名。重跑不要覆盖，加 `_rerun1` 或新日期。

每个目录至少有 `config.yaml`、`train.log`、`checkpoint/`（含 `merged/`）、`README.md`。README 一句话写清：臂、预算、种子、数据版本 v1.0、耗时、最终 loss、实际步数、显卡 `RTX 4060 Laptop 8GB`。

建议日期用实际开训日。下面用后缀表示目录名的前半段。

### 6.1 先做这两条，作为模板

| 目录后缀 | 数据 | 起点 | 参考 | 步数 |
|---|---|---|---|---|
| `M0_warmup_seed42` | `warmup_5k.jsonl` | 本地 Base | 无 | 313 |
| `G2_sft_500_seed42` | `arms/G2_sft_500.jsonl` | M0 merged | 无 | 32 |
| `G4_stage1_sft_500_seed42` | `arms/G4_stage1_sft_500.jsonl` | M0 merged | 无 | 16 |
| `G4_stage2_dpo_500_seed42` | `arms/G4_stage2_dpo_500.jsonl` | 上一行的 merged | 同一份 merged | 16 |

M0 没合并完，不要开始 G2–G5。G3′ 可以在脚本跑通后随时做，它不依赖 M0。

### 6.2 优先级

1. 500 档 G4 与 G5，种子 42、1337、2024。这是课题要比较的顺序。跑完可以把 stage2 merged 路径先交给评测。
2. 500 档 G2、G3 的三个种子，加 `G3prime_dpo_500_seed42`。
3. 2,000 档 G2、G3、G4、G5，三个种子。
4. 10,000 档 G2、G3、G4、G5，只跑种子 42。时间不够时这一档可以最后做。提案允许把它降级成描述性结果。

来不及就保留：M0，以及 500 和 2,000 的 G4/G5 三个种子。其次保留 500 和 2,000 的 G2/G3 与 G3′。

目录总数：M0 1 + G2 7 + G3 7 + G4 14 + G5 14 + G3′ 1 = **44**。评测要加载的最终模型是 M0 + 29 个实验；两阶段只把 **stage2 merged** 登记为该实验的最终权重，stage1 仍留在磁盘上。

G5 stage1 数据是 `G5_stage1_dpo_{N}.jsonl`，stage2 数据是 `G5_stage2_sft_{N}.jsonl`。G4 与此对称。

### 6.3 交卷

写 `03_runs/RUN_INDEX.md`，每行一个最终模型：

```text
arm | budget | seed | train_dir | merged_path | started_from | dpo_ref | steps | final_loss | gpu | hours
```

自检：config 与实际运行一致；loss 有限；步数与第 4 节的表一致；`AutoModelForCausalLM.from_pretrained(merged)` 能加载。

---

## 7. 不要做的事

- 不要改 `01_data/v1.0`，不要重跑数据管线去「重新生成」v1.0。
- 不要下载或加载 `Qwen2.5-*-Instruct`。
- 不要 `pip install -r requirements.txt` 覆盖当前 venv。
- 不要全参数微调。提案规定 LoRA。
- 不要把主实验改成 3 个 epoch。
- 不要把 smoke 输出写进 `03_runs` 的正式命名。
- 不要在 stage2 从 M0 重新开始。stage2 的 `model_path` 必须是该种子自己的 stage1 merged。
- 不要只上传 `train.log` 而不带 `config.yaml`。

更长的人类清单、故障表和七天日历在 `LSYNEED/训练全流程.md`。协议原文在 `00_docs/DATA_PROTOCOL.md`，提案在 `00_docs/proposal/DSAI4207_Final_Proposal_main.tex`，run 命名在 `03_runs/README_RUNS.md`。
