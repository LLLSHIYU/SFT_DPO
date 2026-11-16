# DSAI4207 Part C 评测代码

本目录补齐端到端评测所需脚本：固定测试集生成、GPT-4o-mini 双位置裁判、MMLU/ARC likelihood、官方 IFEval、LCWR/OA 统计与图表、100 对人工盲评及 Cohen's κ。脚本只读 `01_data/v1.0`；生成文本和逐题分数写入各 run 的 `03_runs/{run}/eval/`，汇总表和图写入 `04_results/`。

> 当前共享工作区的 `03_runs/` 没有模型 checkpoint。代码已写好，但正式评测要等训练组上传 M0 与各最终 checkpoint，并在 manifest 中填入实际路径。先用 `--dry-run` 检查命令，不会产生模型结果。

## 1. 环境与模型清单

在项目根目录执行，安装到能读取训练 checkpoint 的同一 Python 环境：

```bash
python -m pip install -r 02_code/evaluation/requirements-eval.txt
```

复制并编辑 `models.example.json` 为自己的运行清单：**每个最终 checkpoint 一行**，必须包含唯一 M0 baseline 和所有待评测最终模型（G2/G3/G4/G5/G3prime）；`adapter` 填 LoRA adapter 路径，若 `model` 本身是完整合并 checkpoint 则省略 `adapter`。M0 也应填共享起点 checkpoint。每项 `arm/budget/seed` 必须与训练 run 对应。

默认基础模型是 `Qwen/Qwen2.5-0.5B`。若项目训练使用不同 Hub revision / 本地 tokenizer，应将 `--base-model` 指向**训练完全相同的 tokenizer/base revision**；脚本会校验 `<|im_end|>`=151645、`<|endoftext|>`=151643，不一致时停止，不能带错配置继续跑。

## 2. 全矩阵一键评测

先预览命令：

```bash
python 02_code/evaluation/run_matrix.py --manifest 02_code/evaluation/models.json --dry-run
```

正式运行（需要 GPU/足够内存；IFEval 官方 checker 依赖随代码快照 ZIP 分发）：

```bash
python 02_code/evaluation/run_matrix.py --manifest 02_code/evaluation/models.json
```

脚本逐模型执行 TL;DR/IFEval 生成、IFEval 评分、MMLU/ARC 两种 likelihood 评分，再将每个非 M0 模型与同一 M0 的 200 主集输出进行双位置裁判。可用 `--skip-judge`、`--skip-capability` 或 `--skip-ifeval` 分阶段执行。脚本支持断点续跑生成/裁判；不要把不同 checkpoint 的结果写到同一个 run 目录。

若需手动调试单个步骤，可直接运行相应脚本：

- `generate.py --task tldr|ifeval`：贪心解码；TL;DR 上限 128，IFEval 上限 1280。
- `judge_lcwr.py`：使用 DeepSeek 的 OpenAI 兼容 API，默认模型为 `deepseek-chat`，从 `DEEPSEEK_API_KEY` 环境变量读取密钥，默认 API 地址为 `https://api.deepseek.com`。不要将密钥写入 manifest/脚本。System prompt 和 user prompt 按 Balancing the Budget 附录 A.2.1：system 是 TL;DR 摘要指令，user 是带例子的名次比较，返回 `model_1`、`model_2` 的名次，名次 1 更好。同一题把两份摘要对调后再判一次。结果用 `json.loads` 读取，不执行模型返回的文本。完整 29 个最终模型 × 200 题约 11,600 次 API 调用（不含重试）。先对一个模型用 `--limit 2` 验证。只在已确认符合项目数据处理/外部 API 政策后运行；请求会包含 Reddit 测试文本和模型摘要。
- `score_capability.py --benchmark mmlu|arc`：固定 500/300 题，不重新抽样；MMLU 使用单 token 选项标签条件 log-prob，ARC 同时记录长度归一化全文选项 log-prob 与单字母 log-prob。
- `score_ifeval.py`：调用 ZIP 中的官方 `instruction_following_eval`，缺少官方 checker 时失败，不提供弱化 fallback。

## 3. LCWR、OA、bootstrap、McNemar 与图表

所有 run 的 `eval/lcwr_*.jsonl` 就绪后执行：

```bash
python 02_code/evaluation/analyze.py --input-dir 03_runs --output-dir 04_results --bootstrap 1000
```

`lcwr_v1.0_YYYYMMDD.csv` 保存逐 run 调整后胜率和 95% prompt bootstrap CI；长度控制使用 fractional logistic regression：`judge_win ~ log_length(candidate) - log_length(M0)`，报告长度差为 0 时的预测胜率。多 seed 的 mean CI 使用跨 seed 的共同 prompt ID 聚类重采样。`oa_v1.0_YYYYMMDD.csv` 保存 $OA(N)=LCWR(G4)-LCWR(G5)$、配对 prompt bootstrap CI，以及**每个 seed 单独**的 McNemar 检验（不把同一题的不同 seed 当成独立样本合并）。500/2k 多 seed 会保留 seed 级结果及均值/标准差；10k 单 seed 标记为描述性。另生成 OA 曲线和 gain–tax PDF；图中 MMLU 单 token 标签 log-prob 与 ARC 长度归一化全文选项 log-prob 分开画，不将两种不同量纲合并（提供能力 summary 时）。

若 capability summary 不在 `04_results/` 下，额外传入文件路径：

```bash
python 02_code/evaluation/analyze.py --judgments 03_runs/**/eval/lcwr_*.jsonl --capability-summaries path/to/*.summary.json
```

## 4. 100 对人机一致性盲评

从一批 judge 输出抽样、随机交换人工评审 A/B，并把映射 key 分开保存（key 文件不要发给盲评员）：

```bash
python 02_code/evaluation/human_audit.py sample \
  --judgments 03_runs/G4_run/eval/lcwr_G4_run.jsonl \
  --n 100 --seed 42 \
  --sheet 04_results/human_judge_blind.csv \
  --key /secure/location/human_judge_key.json
```

三位评审在 CSV 的 `annotator_1/2/3` 列中填写 `A`、`B` 或 `TIE`。完成后计算每位评审对 GPT 裁判及评审之间的 Cohen's κ：

```bash
python 02_code/evaluation/human_audit.py agreement \
  --sheet 04_results/human_judge_blind.csv \
  --key /secure/location/human_judge_key.json \
  --output 04_results/human_judge_agreement_YYYYMMDD.json
```

测试文本含 Reddit 用户内容，盲评表和原始 generation/judge 输出只用于课程项目内部，按团队权限保存；不要公开分享原始文本或 key。

## 5. 输出与复现信息

| 产物 | 位置 |
|---|---|
| 原始摘要/IFEval generation、逐题裁判、能力分数 | `03_runs/{run}/eval/` |
| LCWR、OA、IFEval/能力汇总、统计图 | `04_results/` |
| 评测脚本和依赖 | 本目录 |

结果表需连同模型/checkpoint 路径、训练配置、数据版本、评测日期保存。脚本不会改动冻结数据，也不会自动上传或覆盖训练 run。模型结果为空/缺题、tokenizer ID 不匹配、官方 IFEval checker 缺失时均会报错停止。
