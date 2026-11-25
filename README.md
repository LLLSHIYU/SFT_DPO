# Order vs. Scale

**项目：** Order vs. Scale: A Controlled Study of SFT/DPO Sequencing Across Data Budgets in Small LM Alignment  
**课程：** DSAI 4207 Introduction to Large Language Models, PolyU  
**团队：** PAN Haoren（数据）· LIU Shiyu（训练）· PAN Yubing（评测）  
**仓库：** https://github.com/LLLSHIYU/SFT_DPO  
**最后更新：** 2026-11-25

基座是 Qwen2.5-0.5B。训练用 LoRA，比较 SFT 和 DPO 的顺序、样本比例和数据量。摘要和指令遵循各跑一遍。种子是 42、1337、2024。

## 仓库里有什么

| 路径 | 内容 |
|---|---|
| `LSYNEED/training/` | 训练脚本和每次运行的 yaml |
| `PAN_Evaluation/` | 生成、LCWR 裁判、MMLU/ARC、IFEval、结果导出 |
| `SUMMARY_DATA/` | 摘要任务的 G2–G5，以及 MMLU、ARC、测试集 |
| `IF_DATA/` | 指令遵循任务的 G2–G5，以及 IFEval |
| `1331DATA/` | G6–G9 的 1:3 和 3:1。摘要在 `summary_v1.2`，指令遵循在 `instruction_following_v1.0` |
| `04_results/` | 汇总表和图 |

`01_data/v1.0/` 是早期交接目录，不在这个仓库里。G2–G5 用上面的 `SUMMARY_DATA` 和 `IF_DATA`。评测代码在 `PAN_Evaluation/`，不在 `02_code/evaluation/`。

权重和逐题记录不在仓库里。`03_runs/` 大约 115 GB，里面有 checkpoint、合并后的模型、生成文本和 200 题的裁判记录。这些文件超过 GitHub 单个文件 100 MB 的限制，所以 `.gitignore` 排除了整个目录。克隆这个仓库可以看代码和汇总表，不能下载模型，也不能从原始裁判记录重算汇总表。

## 八个实验臂

比例是 SFT 示范条数 : DPO 偏好对数。N 是题目数，取 500、2,000、10,000。

| 臂 | 顺序 | 比例 | 数据目录 |
|---|---|---|---|
| G2 | 只做 SFT | 全部 N | `SUMMARY_DATA` 或 `IF_DATA` 的 `arms/` |
| G3 | 只做 DPO | 全部 N | 同上 |
| G4 | SFT 然后 DPO | 1:1 | 同上 |
| G5 | DPO 然后 SFT | 1:1 | 同上 |
| G6 | SFT 然后 DPO | 1:3 | `1331DATA/.../ratio_1_3/arms/` |
| G7 | SFT 然后 DPO | 3:1 | `1331DATA/.../ratio_3_1/arms/` |
| G8 | DPO 然后 SFT | 3:1 | `1331DATA/.../ratio_3_1/arms/` |
| G9 | DPO 然后 SFT | 1:3 | `1331DATA/.../ratio_1_3/arms/` |

G8 的 3:1 指 SFT 占三份、DPO 占一份，所以第一阶段是较小的那组 DPO。同一比例下，两种顺序用的是同一批样本。

## 在自己的电脑上重跑

训练脚本和 yaml 写的是训练那台电脑的绝对路径。克隆之后不能直接开训。只看 `04_results/` 里的表，不用改路径。

要重跑，先改这三个文件开头的 `SHARED` 和 `PYTHON`：

- `LSYNEED/training/run_seed42.py`
- `LSYNEED/training/run_seeds_1337_2024.py`
- `LSYNEED/training/run_if.py`

`SHARED` 改成你自己的仓库目录。`PYTHON` 改成你自己的虚拟环境解释器。现在文件里写的是：

```text
C:\Users\liush\Desktop\DSAI4207_shared
C:\Users\liush\Desktop\DSAI4207_shared\02_code\DSAI4207_code\.venv\Scripts\python.exe
```

`LSYNEED/training/configs/` 里的 yaml 也写死了 `model_path`、`data_path` 和 `output_dir`。这些 yaml 是上面的脚本生成的。改完 `SHARED` 之后让脚本重新生成，或者把 yaml 里的 `C:/Users/liush/Desktop/DSAI4207_shared` 换成你的目录。

基座权重不在仓库里，需要自己放到 `LSYNEED/models/Qwen2.5-0.5B`。两阶段训练从已经合并好的 M0 开始，M0 在 `03_runs/` 里，仓库里没有。

## 结果

主表是 `04_results/eval_results_v1.0_20261125.xlsx`。同一天的文件：

- `lcwr_v1.0_20261125.csv`：G2–G9 的摘要胜率，96 行
- `oa_v1.0_20261125.csv`：只有 G4 减 G5 的顺序差和区间
- `capability_tax_v1.0_20261125.csv`
- `fig_oa_curve_v1.0_20261125.pdf`
- `fig_gain_tax_v1.0_20261125.pdf`

`20261116` 那一组只有 G2–G5。G6 对 G8、G7 对 G9 的顺序差在 Excel 的「顺序差」表里，是三个种子各自的差值。
