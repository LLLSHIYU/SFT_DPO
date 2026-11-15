# 00_docs · 最新文档索引

本目录**只保留最新版本**；旧版本由 OneDrive 版本历史管理，不要在这里堆 `v1/v2/v3` 文件。

| 文件 | 版本/日期 | 说明 |
|---|---|---|
| `proposal/DSAI4207_Final_Proposal_main.pdf` | 2026-10-13 | 最终 proposal（**权威版本**） |
| `proposal/DSAI4207_Final_Proposal_main.tex` | 2026-10-13 | LaTeX 源文件 |
| `DATA_PROTOCOL.md` | v1.2-final | 数据协议：数据源、canonicalization、去污染、切分、镜像绑定、warmup、模板、验收 |
| `DATA_ISSUES.md` | v1.2 | 风险清单：Stiennon Table 8 纠偏、CNN/DM、PII、签字 |
| `DATA_FIX_PLAN.md` | v1.4 | 修复清单：P0–P3、R1–R6 已清零；P3-5 已由 README 缓解 |
| `DATA_CARD.md` | 2026-10-21 | 数据卡：漏斗、SMD、策略分布、Limitations、验收状态 |
| `PROJECT_README.md` | 2026-10-21 | 项目主 README：环境、数据获取、复现命令、训练/评测文件契约 |
| `SOURCES_LOCK.json` | 2026-10-21 | 官方数据/模型 commit + parquet SHA256 |
| `meeting_notes/MEETING_NOTES_TEMPLATE.md` | — | 会议记录模板 |

## 阅读顺序

1. `proposal/DSAI4207_Final_Proposal_main.pdf`：研究问题与实验设计。
2. `DATA_PROTOCOL.md`：数据怎么造、为什么这样造。
3. `DATA_CARD.md`：数据实际长什么样、有哪些限制。
4. `PROJECT_README.md`：怎么安装、下载、复现。
5. `DATA_FIX_PLAN.md`：修过哪些坑、当前状态。

## 注意

- `main.pdf` 与 `main.tex` 以 PDF 为准；若两者不一致，先问 PAN Haoren。
- 旧 proposal（v1/v2/v3/v4）不在共享目录；需要溯源时从 OneDrive 版本历史或本地 `proposal v*` 文件夹找。
- 修改文档后请在群内说明；不要同时编辑同一个文件，避免 OneDrive conflicted copy。
