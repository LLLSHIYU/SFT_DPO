# HANDOFF · DSAI 4207 数据工程交接

| 项 | 内容 |
|---|---|
| 项目 | Order vs. Scale: A Controlled Study of SFT/DPO Sequencing Across Data Budgets in Small LM Alignment |
| 团队 | PAN Haoren（数据工程）· LIU Shiyu（训练）· PAN Yubing（评测） |
| 数据版本 | `data/v1.0`（冻结，`synthetic = false`） |
| 协议版本 | `DATA_PROTOCOL.md` v1.2-final |
| 代码快照 | `02_code/DSAI4207_code_20261021.zip`；本地 Git commit `e3ac629` |
| 交接日期 | 2026-10-21 |
| 状态 | 数据已冻结；`07_validate.py` + 8/8 unittest + TRL smoke 通过；训练/评测未开始 |

---

## 1. 先读什么

1. `00_docs/proposal/DSAI4207_Final_Proposal_main.pdf` — 研究问题与实验设计。
2. `00_docs/DATA_PROTOCOL.md` — 数据规则（为什么这样切、这样去污染）。
3. `00_docs/DATA_CARD.md` — 数据实际长什么样、有哪些限制。
4. `00_docs/PROJECT_README.md` — 安装、数据获取、复现命令、训练/评测文件契约。
5. `00_docs/DATA_FIX_PLAN.md` — 修复记录与当前状态。

---

## 2. 数据在哪里

所有训练/评测文件在 `01_data/v1.0/`。**只读，不要原地改。**

| 用途 | 文件 |
|---|---|
| M0 预热 | `warmup_5k.jsonl`（`messages`，`assistant_only_loss=True`，`max_length=2048`） |
| G2 纯 SFT | `arms/G2_sft_{500,2000,10000}.jsonl` |
| G3 纯 DPO | `arms/G3_dpo_{500,2000,10000}.jsonl` |
| G4 SFT→DPO | `arms/G4_stage1_sft_*.jsonl` → `arms/G4_stage2_dpo_*.jsonl` |
| G5 DPO→SFT | `arms/G5_stage1_dpo_*.jsonl` → `arms/G5_stage2_sft_*.jsonl` |
| G3′ 冷启动探针 | `arms/G3prime_dpo_500.jsonl` |
| 主评测集 | `test_200_main.jsonl`（200 条） |
| 扩展敏感性集 | `test_500_full.jsonl`（500 条，前 200 == 主集） |
| 能力税 | `eval_benchmarks/mmlu_500.jsonl`、`arc_challenge_300.jsonl` |
| IFEval | `eval_benchmarks/ifeval_541.jsonl` |
| manifest | `manifests/manifest.json` |
| 切分/去污染报告 | `manifests/split_report.json`、`decontamination_report.json` |
| 双跑 | `manifests/run_1_sha256.txt`、`run_2_sha256.txt`、`run_diff.txt` |

### 文件 schema

**SFT：**

```json
{"pid": "t3_…", "messages": [{"role":"system","content":"…"},{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}
```

**DPO：**

```json
{"pid": "t3_…", "prompt": [{"role":"system","content":"…"},{"role":"user","content":"…"}], "chosen": [{"role":"assistant","content":"…"}], "rejected": [{"role":"assistant","content":"…"}]}
```

**Token / 模板：**

- `Qwen/Qwen2.5-0.5B-Base` 官方 tokenizer，`vocab_size=151,643`，`len=151,665`
- `eos_token="<|im_end|>"`（151645），`pad_token="<|endoftext|>"`（151643）
- ChatML 与官方 `apply_chat_template` 逐字节一致
- TL;DR `max_length=1024`；warmup `max_length=2048`
- 评测 greedy，`max_new_tokens=128`；IFEval `max_new_tokens=1280`（不要套 TL;DR 的 128）

---

## 3. 校验（训练前必做）

在共享根目录：

```bash
cd DSAI4207_shared
shasum -a 256 -c 05_handoff/CHECKSUMS.sha256
```

只校验数据 manifest：

```bash
python - <<'PY'
import json, hashlib
from pathlib import Path
root = Path("01_data/v1.0")
m = json.load(open(root / "manifests/manifest.json"))
for f in m["files"]:
    p = root / f["path"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"], p
print("manifest OK:", len(m["files"]), "files")
PY
```

如果校验失败：先检查 OneDrive 是否同步完成、是否有 conflicted copy；再联系 PAN Haoren。

---

## 4. 数据关键数字

| 项 | 值 |
|---|---:|
| `canonical_pool` 唯一 prompt | 14,733 |
| 10k 档余量 | +4,733 |
| validation 候选 Reddit prompt | 6,210 |
| warmup 干净池 | 8,949 |
| warmup 实际抽样 | 5,000 |
| 测试集 | 200 主 + 500 全 |
| 双跑 | 48 文件 SHA256 完全一致 |
| 子版块漂移（500/2k） | 0.66% / 0.64% |
| test_200 vs 500 层偏差 | 0.30% |
| A/B SMD | 全部 <0.10 |
| 官方 benchmark | MMLU 14,042/57 科；ARC 1,172；IFEval 541 |

---

## 5. 当前状态与遗留项

### 已完成

- P0：官方 tokenizer、官方 MMLU/ARC/IFEval、离线 fallback 修复、ChatML 对齐、课程 repo 文件。
- P1：rejected 策略平衡、官方 IFEval checker、真实性/集成测试、双跑、盲评。
- P2/P3：子版块漂移、数据卡分布、Table 8、PII/预训练限制、manifest env/revision、别名清理、签字。
- R1–R6：无效 commit SHA、χ² 措辞、manifest env、tokenizer 标签、git commit、audit key 均已清零。

### 低优先/可选

- P3-5：`07_validate.py` 在数据-only 环境会因读取 `trl` 版本失败；README §4.3 已要求先装完整 `requirements.txt`。`safe_version()` 作为可选代码改进。
- 训练/评测尚未开始；`03_runs/` 与 `04_results/` 等待对应组员。
- Reddit 数据含潜在 PII/情绪化内容；仅限课程非商业研究，不要公开链接。

---

## 6. 训练组（LIU Shiyu）交接要点

1. 用 `02_code/DSAI4207_code_20261021.zip` 或本地 Git 仓库；安装完整 `requirements.txt`。
2. M0：`warmup_5k.jsonl`，`assistant_only_loss=True`，`max_length=2048`，1 epoch。
3. 主实验：按 `00_docs/PROJECT_README.md` 的 29-run 矩阵；两阶段臂 merge LoRA + reset optimizer。
4. 每个 run 按 `03_runs/README_RUNS.md` 命名，保存 `config.yaml` + `train.log` + checkpoint。
5. 训练前跑一次 `07_validate.py`；如果换 tokenizer/模板，必须重新跑 token-level 断言。

## 7. 评测组（PAN Yubing）交接要点

1. 主集 `test_200_main.jsonl`；扩展集 `test_500_full.jsonl`（先跑主集，边界 p 值再补 300）。
2. 官方 benchmark 文件已冻结；不要重新在线抽样。
3. LCWR：A/B 与 B/A 双向换序 + 长度协变量；McNemar；bootstrap 1,000 次。
4. IFEval：官方 `instruction_following_eval`，`max_new_tokens=1280`。
5. 结果按 `04_results/README_RESULTS.md` 命名；原始生成放 `03_runs/{run}/eval/`。

---

## 8. 权限与合规

- 共享链接用 **Specific people**（三位组员 PolyU 邮箱），不要 “Anyone with the link”。
- `01_data/`、`04_results/`、`05_handoff/`：Can view；`00_docs/`、`03_runs/`：Can edit。
- Reddit 数据仅限课程非商业研究；不要公开、不要重识别个人。
- 数据文件已设只读；需要更新时新建 `data/v1.1`，不要改 `v1.0`。

---

## 9. 联系人

| 角色 | 姓名 | 负责 |
|---|---|---|
| 数据工程 | PAN Haoren (24104216d) | 数据、协议、去污染、IFEval harness |
| 训练 | LIU Shiyu (24106028d) | M0、SFT/DPO、两阶段 merge、29 runs |
| 评测 | PAN Yubing (24103431d) | LCWR、MMLU/ARC、统计、图 |

---

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-21 | 建立共享目录；数据 v1.0、最终 proposal、协议 v1.2-final、代码快照、manifest、双跑日志、校验文件交接 |
