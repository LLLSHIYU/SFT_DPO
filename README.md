# DSAI 4207 小组共享目录

**项目：** Order vs. Scale: A Controlled Study of SFT/DPO Sequencing Across Data Budgets in Small LM Alignment  
**课程：** DSAI 4207 Introduction to Large Language Models, PolyU  
**团队：** PAN Haoren（数据工程）· LIU Shiyu（训练）· PAN Yubing（评测）  
**数据版本：** `data/v1.0`（protocol `v1.2-final`，`synthetic = false`）  
**最后更新：** 2026-10-21

---

## 1. 这个目录是做什么的

这是小组内部共享目录，按“文档 → 数据 → 代码 → 训练 → 结果 → 交接”分区，避免聊天窗口里反复传大文件。

**规则：**

1. `01_data/v1.0/` 是**冻结数据**，只读；不要原地修改。需要变更时新建 `v1.1`。
2. `00_docs/` 只放**最新版本**文档；旧版本交给 OneDrive 版本历史，不在目录里堆 v1/v2/v3。
3. `03_runs/` 每次训练一个独立子目录，禁止覆盖别人的 run。
4. `04_results/` 只放最终表格/图；原始日志放 `03_runs/`。
5. 大文件不要发微信/邮件；用 OneDrive 共享链接，权限设为“特定人可查看”。

---

## 2. 目录结构

```text
DSAI4207_shared/
├── README.md                     # 本文件
├── SHARING_GUIDE.md              # OneDrive 共享教程
├── UPDATE_GUIDE.md               # 后续修改/新增文件的更新流程
├── 00_docs/                      # 最新 proposal、协议、DATA_CARD、会议记录
│   ├── proposal/
│   │   ├── DSAI4207_Final_Proposal_main.pdf   # 最终 proposal（权威）
│   │   └── DSAI4207_Final_Proposal_main.tex   # LaTeX 源文件
│   ├── DATA_PROTOCOL.md          # v1.2-final
│   ├── DATA_ISSUES.md            # 风险清单（Table 8 / PII / 签字）
│   ├── DATA_FIX_PLAN.md          # v1.4 修复清单（R1–R6 已清零）
│   ├── DATA_CARD.md              # 数据卡（漏斗、SMD、Limitations）
│   ├── PROJECT_README.md         # 项目主 README（安装/复现）
│   ├── SOURCES_LOCK.json         # 官方数据/模型 commit + SHA256
│   └── meeting_notes/
│       └── MEETING_NOTES_TEMPLATE.md
├── 01_data/
│   └── v1.0/                     # 冻结数据，只读（约 285MB，52 个数据文件 + README）
│       ├── canonical_pool.jsonl
│       ├── test_200_main.jsonl
│       ├── test_500_full.jsonl
│       ├── warmup_5k.jsonl
│       ├── eval_benchmarks/      # MMLU-500 / ARC-300 / IFEval-541
│       ├── splits/               # pool_500/2000/10000 + partA/B + reserve
│       ├── arms/                 # 19 个训练臂 JSONL
│       ├── manifests/            # manifest、split、decontamination、run logs
│       ├── DATA_CARD.md
│       ├── sources_lock.json
│       └── README_DATA.md
├── 02_code/
│   ├── DSAI4207_code_20261021.zip   # 代码快照（约 128KB）
│   └── README_CODE.md
├── 03_runs/                      # 训练 checkpoint / log（空，等待训练组）
│   └── README_RUNS.md
├── 04_results/                   # 评测表格、图
│   ├── data_funnel_summary.json
│   ├── split_smd_and_policy_balance.json
│   └── README_RESULTS.md
└── 05_handoff/
    ├── HANDOFF.md                # 交接说明（先读这个）
    ├── manifest.json
    ├── split_report.json
    ├── decontamination_report.json
    ├── warmup_stats.json
    ├── run_1_sha256.txt
    ├── run_2_sha256.txt
    ├── run_diff.txt
    ├── sources_lock.json
    └── CHECKSUMS.sha256          # 数据/代码/结果校验
```

---

## 3. 最快上手

1. 先读 **`05_handoff/HANDOFF.md`**。
2. 校验数据：
   ```bash
   cd DSAI4207_shared
   shasum -a 256 -c 05_handoff/CHECKSUMS.sha256
   ```
3. 训练用：
   - `01_data/v1.0/warmup_5k.jsonl`
   - `01_data/v1.0/arms/G2_sft_*.jsonl`
   - `01_data/v1.0/arms/G3_dpo_*.jsonl`
   - `01_data/v1.0/arms/G4_stage1_sft_*.jsonl` / `G4_stage2_dpo_*.jsonl`
   - `01_data/v1.0/arms/G5_stage1_dpo_*.jsonl` / `G5_stage2_sft_*.jsonl`
   - `01_data/v1.0/arms/G3prime_dpo_500.jsonl`
4. 评测用：
   - `01_data/v1.0/test_200_main.jsonl`（主分析）
   - `01_data/v1.0/test_500_full.jsonl`（扩展敏感性）
   - `01_data/v1.0/eval_benchmarks/mmlu_500.jsonl`
   - `01_data/v1.0/eval_benchmarks/arc_challenge_300.jsonl`
   - `01_data/v1.0/eval_benchmarks/ifeval_541.jsonl`
5. 代码：
   - 解压 `02_code/DSAI4207_code_20261021.zip`；
   - `pip install -r requirements.txt`；
   - 按 `00_docs/PROJECT_README.md` 运行数据管线/验证。

---

## 4. 当前状态

| 项 | 状态 |
|---|---|
| 数据 `v1.0` | ✅ 官方 TL;DR + no_robots + 官方 Qwen2.5 tokenizer；14,733 pool；6,210 test 候选；8,949 warmup |
| 官方 benchmark | ✅ MMLU 14,042/57 科、ARC 1,172、IFEval 541 |
| 验证 | ✅ `07_validate.py` 通过；8/8 unittest（含 TRL SFT/DPO 单步） |
| 双跑 | ✅ `run_1_sha256.txt` == `run_2_sha256.txt`；`run_diff.txt` 0 字节 |
| 修复清单 | ✅ P0–P3 与 R1–R6 清零；P3-5 已由 README 说明缓解 |
| 训练/评测 | ⏳ 未开始；`03_runs/`、`04_results/` 等待训练/评测组 |

---

## 5. 共享与权限

- **OneDrive 副本已就位（2026-10-28）：**
  ```text
  ~/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared
  ```
  已校验 83 个文件、285MB、`CHECKSUMS.sha256` 0 失败；下一步在 Finder 右键 Share。
- 详细 OneDrive 教程见 **`SHARING_GUIDE.md`**。
- **后续修改/新增文件怎么更新：见 `UPDATE_GUIDE.md`。**
- 建议权限：
  - `01_data/`、`04_results/`、`05_handoff/`：**Can view**（只读）。
  - `00_docs/`、`03_runs/`：**Can edit**（组员需要写会议记录/上传 run）。
- 链接类型选 **Specific people**（组员 PolyU 邮箱），不要用 “Anyone with the link”。
- Reddit 数据仅限课程非商业研究；不要把公开链接发到群外。

---

## 6. 联系人

| 角色 | 姓名 | 负责 |
|---|---|---|
| 数据工程 | PAN Haoren (24104216d) | 数据 v1.0、协议、去污染、IFEval harness |
| 训练 | LIU Shiyu (24106028d) | M0、SFT/DPO、两阶段 merge、29 runs |
| 评测 | PAN Yubing (24103431d) | LCWR、MMLU/ARC、统计、图表 |

---

## 7. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-21 | 建立共享目录；放入 `data/v1.0`、最终 proposal、协议 v1.2-final、代码快照、manifest、双跑日志与校验文件 |
