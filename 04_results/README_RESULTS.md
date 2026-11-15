# 04_results · 评测结果

当前已有**数据工程结果**；训练/评测结果等待 PAN Yubing 补充。

## 已有文件

| 文件 | 内容 |
|---|---|
| `data_funnel_summary.json` | 数据漏斗：raw rows、unique prompts、最终 pool、warmup、test |
| `split_smd_and_policy_balance.json` | 嵌套切分：A/B SMD、策略家族 χ²、子版块漂移、test_200 分层 |

## 命名约定

```text
04_results/
  data_funnel_summary.json
  split_smd_and_policy_balance.json
  lcwr_{data_version}_{date}.csv
  capability_tax_{data_version}_{date}.csv
  ifeval_{data_version}_{date}.csv
  fig_gain_tax_{data_version}.pdf
  fig_oa_curve_{data_version}.pdf
  human_judge_agreement_{date}.json
```

- 表格优先 `.csv`（易读、易合并）；图优先 `.pdf`（矢量）或 `.png`（300dpi）。
- 每个结果文件在 `README_RESULTS.md` 里写：生成脚本、checkpoint、数据版本、日期。
- 原始生成文本/日志放 `03_runs/{run}/eval/`，不要塞进 `04_results/`。
- 不要手工改结果表；改脚本重跑。

## 主结果清单（待补）

- [ ] LCWR：G2/G3/G4/G5/G3′ vs M0，200 主集 + 500 扩展集
- [ ] 能力税：MMLU-500 / ARC-300（likelihood）
- [ ] IFEval：Strict / Loose
- [ ] gain–tax 散点
- [ ] OA(n) 曲线：500/2k/10k
- [ ] 三种子 mean ± std、bootstrap 95% CI、McNemar
- [ ] 100 对人机裁判一致率（Cohen's κ）
- [ ] 失败案例分析样例
