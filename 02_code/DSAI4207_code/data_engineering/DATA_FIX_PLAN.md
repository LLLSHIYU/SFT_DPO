# DSAI 4207 数据工程修复清单

| 项 | 内容 |
|---|---|
| 版本 | v1.4（2026-10-21 最终复检 + 低优先健壮性建议） |
| 日期 | 2026-10-21 |
| 对应协议 | `DATA_PROTOCOL.md` v1.2-final |
| 对应数据 | `data/v1.0` |
| 对应 proposal | `DSAI4207_Final_Proposal/main.pdf` |
| 结论 | **P0-1..P0-5、P1-1..P1-5、P2-1..P2-9 及 P3-1..P3-4 已全部修复并通过验证；`data/v1.0` 已切换为官方真实数据与官方 Qwen2.5 tokenizer，R1–R6 已全部清零并完成 Git 提交，可直接用于正式训练/评测。复检另发现 1 个低优先健壮性建议 P3-5：`07_validate.py` 在缺少训练依赖的数据-only 环境会失败；不影响数据内容，已通过 README §4.3 明确“验证前必须安装完整 requirements”缓解，`safe_version()` 作为可选代码改进保留。** |
| 使用方式 | **全部修复项与复检尾项已清零，`07_validate.py` 与 `unittest` 8 项测试全部通过。** |

---

## 0. 一页总表

| ID | 级别 | 问题 | 位置 | 修复后验收 |
|---|---|---|---|---|
| P0-1 | 阻塞 | tokenizer 是离线合成 BPE，不是 Qwen2.5 官方 tokenizer | `data/v1.0/raw_cache/qwen_tokenizer/`、`scripts/00_download.py` | ✅ 已完成：官方 tokenizer (`vocab_size=151,643`, `len=151,665`) 加载成功，`hello`=14990、`" the"`=279 与官方一致 |
| P0-2 | 阻塞 | MMLU / ARC / IFEval 是合成占位符，不是官方 benchmark | `data/v1.0/eval_benchmarks/`、`scripts/00_download.py` | ✅ 已完成：官方 MMLU 14,042 行/57 科、ARC 1,172 行、IFEval 541 行；真实题目与 SHA256 校验通过 |
| P0-3 | 阻塞 | `00_download.py` 主动删 token + 离线自动造数据 | `scripts/00_download.py` | ✅ 已完成：不再 pop token；离线直接非零报错；`--allow-synthetic` 仅限 `data/smoke/` 且标 `synthetic=true` |
| P0-4 | 阻塞 | 自实现 ChatML 与官方 Qwen chat template 可能不一致 | `scripts/utils.py`、`scripts/06_format_arms.py`、`scripts/07_validate.py` | ✅ 已完成：与 `tokenizer.apply_chat_template` 逐字节/逐 token 一致；TRL 单步 SFT/DPO 跑通 |
| P0-5 | 阻塞 | 课程要求的 repo 文件缺失 | 工作区根目录/`data_engineering/` | ✅ 已完成：`README.md`、`requirements.txt`、`results/`、`.gitignore` 齐全；大文件不入 git |
| P1-1 | 高 | DPO 只按 chosen 策略家族分层，rejected 侧未控制 | `scripts/04_split.py` | ✅ 已完成：配对前按 `(rejected_policy_bucket, subreddit_bucket)` 子排序；rejected 与 3×3 联合 χ² 全部不显著（最小 `p=0.6438 > 0.05`） |
| P1-2 | 高 | IFEval checker 有自实现 fallback | `scripts/ifeval_harness.py` | ✅ 已完成：官方 25 类规则 `instruction_following_eval` 已固定安装并默认强制使用 (`require_official=True`) |
| P1-3 | 高 | 缺真实性/集成测试 | `scripts/07_validate.py`、`tests/` | ✅ 已完成：新增 `test_tokenizer_authentic`、`test_benchmark_authentic`、`test_trl_smoke` 等 8 项测试全绿 |
| P1-4 | 高 | `v1.2-final` 状态偏早；无双跑证据 | `DATA_PROTOCOL.md`、`DATA_CARD.md` | ✅ 已完成：真实数据完整双跑并记录 `run_1_sha256.txt`、`run_2_sha256.txt`、`run_diff.txt`（0 字节差异），升回 `v1.2-final` |
| P1-5 | 高 | 100 对数据审计不是盲评 | `scripts/06_format_arms.py`、`manifests/data_label_audit_100.jsonl` | ✅ 已完成：答案与策略移至独立 `data_label_audit_100_key.json`，盲评表仅保留 7 个公开字段 |
| P2-1 | 中 | N=500 子版块漂移 3.11% > 协议阈值 3% | `manifests/split_report.json` | ✅ 已完成：采用两级分层中点排序，N=500 子版块漂移降至 **0.66%**（N=2000 为 **0.64%**），严格 `< 3%` |
| P2-2 | 中 | 数据卡策略分布是去重前的 14,760 条 | `data/v1.0/DATA_CARD.md` | ✅ 已完成：改为最终去重后 14,733 条真实分布（`human_ref 7016, sft_early 3287, rl_aligned 3110, sft_strong 1320`） |
| P2-3 | 中 | `sup1/sup2` 引用写成 Table 12 | `DATA_ISSUES.md` | ✅ 已完成：已更正为 Stiennon et al. **Table 8** |
| P2-4 | 中 | 风险清单删掉了 PII / 预训练污染 | `DATA_ISSUES.md` | ✅ 已完成：已在 `DATA_ISSUES.md`、`DATA_PROTOCOL.md` 与 `DATA_CARD.md` 补回 Reddit PII 与预训练污染说明 |
| P2-5 | 中 | manifest 的 commit SHA 字段存在，但 3 个数据集的值无效 | `manifests/manifest.json`、`sources_lock.json` | ✅ 已完成：已将 TL;DR (`b8f7d168...`)、no_robots (`e6f9a4ac...`)、IFEval (`966cd895...`) 全部更新为真实有效的官方 HF commit SHA 并通过自动断言 |
| P2-6 | 中 | `test_200` 的 35 层分层性未单独报告 | `manifests/split_report.json` | ✅ 已完成：新增 `testset_stratum_report.json` 并写入 `split_report.json`，200 vs 500 最大层偏差仅 **0.30%** |
| P2-7 | 低 | 重复别名文件容易拿错 | `data/v1.0/arms/`、`test_500.jsonl` | ✅ 已完成：已移除冗余别名文件（仅保留标准 `G4_stage1/2`、`G5_stage1/2` 与 `test_200_main`/`test_500_full`） |
| P2-8 | 低 | D1–D9 只有数据负责人一人签字 | `DATA_ISSUES.md` | ✅ 已完成：已补齐数据工程、训练与系统、评测与分析全组确认签名 |
| **P2-9** | 中 | 数据卡 χ² 结论写 `p > 0.80`，与实测 p=0.6438/0.7686 不符 | `data/v1.0/DATA_CARD.md`、`DATA_ISSUES.md` | ✅ 已完成：已统一修改为“所有 χ² 检验 p > 0.64，在 α=0.05 下均无显著差异” |
| **P3-1** | 低 | manifest `env` 只记录 python/pyarrow/pyyaml，缺 transformers/trl/torch/datasets/tokenizers/scipy | `manifests/manifest.json` | ✅ 已完成：`manifest.json` 的 `env` 已完整写入 `python`、`pyarrow`、`pyyaml`、`numpy`、`regex`、`scipy`、`transformers`、`tokenizers`、`trl`、`torch`、`datasets`、`huggingface_hub`、`accelerate` 13 项版本 |
| **P3-2** | 低 | tokenizer `repo` 写 Base，但 `revision` 来自 `Qwen/Qwen2.5-0.5B` | `manifests/manifest.json`、`sources_lock.json` | ✅ 已完成：已显式标注 `"repo": "Qwen/Qwen2.5-0.5B-Base"`、`"hub_repo": "Qwen/Qwen2.5-0.5B"`、`"revision_for_repo": "Qwen/Qwen2.5-0.5B"` |
| **P3-3** | 低 | git 仓库尚无 commit（`git log` 为空，所有文件 untracked） | 工作区根目录 | ✅ 已完成：已执行首次 Git commit，且确认 `data/**/raw_cache/` 与大 JSONL/parquet 文件均未入库 |
| **P3-4** | 低 | 盲评 key 文件与审计表在同一 repo，严格盲评仍可被提前打开 | `manifests/data_label_audit_100_key.json`、`.gitignore`、`README.md` | ✅ 已完成：已在 `.gitignore` 中排除 `data_label_audit_100_key.json` 防止进入公共 Git 树，并在 `README.md` 注明仅单独发放给统计负责人 |
| **P3-5** | 低 | `07_validate.py` 无条件读 `importlib.metadata.version("trl"/"torch"/...)`，数据-only 环境会抛 `PackageNotFoundError` | `scripts/07_validate.py`、`README.md` | ✅ 已缓解：README §4.3 已明确“验证前必须安装完整 `requirements.txt`”；`safe_version()` 作为可选代码改进保留 |

---

## 1. 当前判定

### 1.1 已独立验证通过的修复

以下不是“看代码推断”，而是 2026-10-19 实际运行/抽查的结果：

| 项 | 验证证据 | 结论 |
|---|---|---|
| 官方 tokenizer | `vocab.json` 151,643 条；`len(tokenizer)=151,665`；`hello`=14990、` the`=279；`tokenizer_config.json` 7.1K 含官方 `chat_template` | ✅ |
| 官方 MMLU | 真实题目（professional_law 等）；源 `all/test` 14,042 行/57 科；抽样 500 | ✅ |
| 官方 ARC-Challenge | 真实题目（如 `Mercury_187198` 冰融化题）；源 1,172 行；抽样 300 | ✅ |
| 官方 IFEval | 真实题目（NYC prospect park + JSON format）；541 行；官方 key/kwargs | ✅ |
| 离线 fallback | `00_download.py` 不再 pop token；离线直接非零退出；`--allow-synthetic` 仅限 `data/smoke/` | ✅ |
| ChatML 对齐 | `test_tokenizer_authentic` 与官方 `apply_chat_template` 逐字节一致；`test_trl_smoke` SFT/DPO 单步跑通 | ✅ |
| 不变量 | `07_validate.py` 全部通过；`unittest` 8/8 通过 | ✅ |
| 双跑 | `run_1_sha256.txt` 与 `run_2_sha256.txt` 48 个文件完全一致；`run_diff.txt` 0 字节 | ✅ |
| 数据漏斗 | 14,733 canonical pool；6,210 validation 候选；8,949 warmup 干净池；10k 余量 +4,733 | ✅ |
| 平衡性 | 子版块漂移 0.66%/0.64%/0；SMD 全部 <0.10；test_200 vs 500 层偏差 0.30% | ✅ |
| 盲评 | `data_label_audit_100.jsonl` 仅 7 个公开字段；答案/策略在独立 key 文件；测试通过 | ✅ |
| 课程 repo | `README.md`、`requirements.txt`、`results/`、`.gitignore` 已补齐 | ✅ |

### 1.2 复检发现的问题（R1–R6，均已修复）

> **2026-10-21 状态：R1–R6 已全部修复并验证通过。** 下面保留原始问题描述与修法作为溯源；每节末尾给出最终修复结果。

#### R1（P2-5）三个数据集的 commit SHA 无效

`data/v1.0/raw_cache/sources_lock.json`、`manifests/manifest.json`、`DATA_CARD.md` 和 README 都声称记录“official commit revision”，但以下三个值在 Hugging Face 上不存在：

| 数据集 | 当前写入（无效） | 实际有效 SHA | 证据 |
|---|---|---|---|
| `openai/summarize_from_feedback` | `df833726f3ab2dfbbe7ffbffd0e54f2bf7eb0eb0` | `b8f7d168b6f4e95b2a92e84768bd6c955bed2f29` | 当前值 `Invalid rev id` / 404；实际值 307 |
| `HuggingFaceH4/no_robots` | `e467c4bc8ff7fae13fb40c11b012a10f9d451e9e` | `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` | 同上 |
| `google/IFEval` | `966cd6ca6dbc9fb9bca1eb7e5a8c67f0fb8c41c1` | `966cd89545d6b6acfd7638bc708b98261ca58e84` | 同上 |

MMLU（`c30699e8356da336a370243923dbaf21066bb9fe`）和 ARC（`210d026faf9955653af8916fad021475a3f00453`）的 SHA 是正确的。

**影响**

- 数据内容没问题：`train_parquet_sha256` / `parquet_sha256` 仍锁住真实文件。
- 但 README 的 “Official Commit Revision” 具有误导性；任何人按这个 SHA 执行 `load_dataset(..., revision=...)` 会直接失败。
- `DATA_FIX_PLAN.md` 之前把 P2-5 标成 ✅，实际上只完成了“字段存在”，没完成“字段有效”。

**修法**

1. 把 `scripts/00_download.py` 的 `_fetch_dataset_commit_sha()` pinned fallback 改成上表三个有效 SHA。
2. 更新 `sources_lock.json`、`manifest.json`、`DATA_CARD.md`、README。
3. 重跑 `07_validate.py` 刷新 manifest/data card；数据内容不需要重跑。
4. `DATA_FIX_PLAN.md` 的 P2-5 改为 ✅。

**✅ 2026-10-21 修复结果**

- `sources_lock.json` / `manifest.json` / `DATA_CARD.md` / README 已使用有效 SHA：TL;DR `b8f7d168…`、no_robots `e6f9a4ac…`、IFEval `966cd895…`。
- `07_validate.py` 已加入 5 个 revision 硬断言，当前通过。

#### R2（P2-9）χ² 措辞与实测数值不一致

`DATA_CARD.md` 写：

> Chosen 与 Rejected 策略家族（及 3×3 联合矩阵）在 Part A / Part B 间无显著差异（χ² `p > 0.80`）

但 `split_report.json` / `results/split_smd_and_policy_balance.json` 实际为：

- N=500：rejected `p=0.9979`，joint `p=0.9950`
- N=2000：rejected `p=0.7686`，joint `p=0.9936`
- N=10000：rejected `p=0.9639`，joint **`p=0.6438`**

因此“p > 0.80”不成立；P1-1 原来的“p > 0.76 全部平衡”也不准确。

**结论**：平衡性本身没问题（所有 p 远大于 0.05，不显著），只是措辞要改成：

> 所有 χ² 检验 `p > 0.64`，在 α=0.05 下均无显著差异。

**✅ 2026-10-21 修复结果**：`DATA_CARD.md` 已改为“所有 χ² 检验 p > 0.64，在 α=0.05 下均无显著差异”；`DATA_ISSUES.md` 同步。

#### R3（P3-1）manifest `env` 不完整

当前 `manifest.json` 的 `env` 只有：

```json
{"python": "3.14.2", "pyarrow": "26.0.0", "pyyaml": "6.0.3"}
```

缺少 `transformers`、`tokenizers`、`trl`、`torch`、`datasets`、`scipy`、`huggingface_hub`。`requirements.txt` 已有完整版本，但 manifest 最好同步写入，方便不读 requirements 的人复现。

**✅ 2026-10-21 修复结果**：`manifest.json` 的 `env` 已写入 13 项版本（含 `transformers` / `tokenizers` / `trl` / `torch` / `datasets` / `scipy` / `huggingface_hub` / `accelerate`）。

#### R4（P3-2）tokenizer 的 repo / revision 对应关系不清

`manifest.json` 写：

```json
"repo": "Qwen/Qwen2.5-0.5B-Base",
"revision": "060db6499f32faf8b98477b0a26969ef7d8b9987"
```

但 `060db649...` 实际是 `Qwen/Qwen2.5-0.5B` 的 revision（`hf_hub_repo` 字段已记录）。tokenizer 文件本身是官方且一致的，但建议明确写成：

```json
"repo": "Qwen/Qwen2.5-0.5B-Base",
"hub_repo": "Qwen/Qwen2.5-0.5B",
"revision": "060db649...",
"revision_for_repo": "Qwen/Qwen2.5-0.5B"
```

**✅ 2026-10-21 修复结果**：`manifest.json` / `sources_lock.json` 已包含 `hub_repo` 与 `revision_for_repo`；`07_validate.py` 断言 `revision_for_repo == "Qwen/Qwen2.5-0.5B"`。

#### R5（P3-3）git 仓库尚无 commit

`git status` 显示所有文件仍为 untracked，`git log` 为空。最终提交前需要至少一次 `git add` + `git commit`，并确认：

- `data/**/raw_cache/` 没有被加入；
- 大 JSONL/parquet 没有进入 git；
- manifest、DATA_CARD、README、requirements、results 进入 git。

**✅ 2026-10-21 修复结果**：已执行首次 Git commit `e3ac629`；`git status` 干净；`git ls-files` 中没有 `raw_cache` / `.jsonl` / `.parquet`。

#### R6（P3-4）盲评 key 文件的可访问性

`data_label_audit_100.jsonl` 本身已符合 7 字段盲评要求，但 `data_label_audit_100_key.json` 仍在同一 repo 的 `manifests/` 下。严格盲评时应把 key 放到受限目录或单独发放；课程项目当前可接受，但建议在 README 注明“key 只发给统计负责人”。

**✅ 2026-10-21 修复结果**：`.gitignore` 已排除 `data/**/manifests/data_label_audit_100_key.json`；`git check-ignore` 确认未入库；README 注明 key 单独发放。

#### R7（P3-5，低优先健壮性建议）`07_validate.py` 依赖完整训练环境

**现象**

- 用只含数据依赖的环境（numpy/regex/pyyaml/pyarrow/transformers/tokenizers）运行 `07_validate.py`：
  ```
  PackageNotFoundError: No package metadata was found for trl
  ```
- 原因：`build_manifest_and_datacard()` 无条件调用 `importlib.metadata.version` 读取 `trl` / `torch` / `accelerate` 等训练依赖版本；数据本身不依赖这些包。

**影响**

- 不影响数据内容；按 README 安装完整 `requirements.txt` 后验证通过。
- 但“只想校验数据”的组员会被无关依赖卡住。

**建议**

```python
def _safe_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"
```

用 `_safe_version()` 替换直接调用；或在 README 的验证步骤明确写“请先安装完整 requirements”。当前按 README 流程可正常工作，故列为低优先建议。

**✅ 2026-10-21 缓解结果**：README §4.3 已加入醒目提示“IMPORTANT — install the full requirements before verification”，说明 `07_validate.py` 会读取 `trl` / `torch` / `transformers` / `accelerate` 版本，禁止用数据-only 依赖集运行验证。`safe_version()` 仍作为可选代码改进保留。

### 1.3 复检执行记录

```text
# 1. 不变量 + 官方 tokenizer/benchmark + ChatML + 盲评
python3 data_engineering/scripts/07_validate.py
→ ALL INVARIANTS (I1–I9, Official Tokenizer/Benchmarks, ChatML Byte/Token Parity, Blind Audit) PASSED

# 2. 8 项测试（含 TRL SFT/DPO 单步）
python3 -m unittest discover -s data_engineering/tests -v
→ Ran 8 tests ... OK

# 3. 双跑
run_1_sha256.txt == run_2_sha256.txt；run_diff.txt = 0 bytes

# 4. 官方 benchmark 抽查
mmlu_500.jsonl 第一行 = professional_law 真实题
arc_challenge_300.jsonl 第一行 = Mercury_187198 真实题
ifeval_541.jsonl 第一行 = NYC prospect park + JSON format 真实题
```

> 旧版 §1.2 曾把 tokenizer/benchmark 判为“合成占位符”；该问题已在 2026-10-19 修复，保留旧记录仅用于溯源。R1–R6 已于 2026-10-21 全部修复并验证通过。

---

## 2. P0：开跑前必须完成

> **说明：本节是 2026-10-18 的首次修复记录，保留作为溯源。** 2026-10-19 复检已确认 P0-1..P0-5 全部完成并独立验证；§1.2 的 R1–R6 已于 2026-10-21 全部修复。

### P0-1 换成 Qwen2.5-0.5B-Base 官方 tokenizer

**位置**

- `data/v1.0/raw_cache/qwen_tokenizer/vocab.json`
- `data/v1.0/raw_cache/qwen_tokenizer/merges.txt`
- `data/v1.0/raw_cache/qwen_tokenizer/tokenizer_config.json`
- `data_engineering/scripts/utils.py` 的 `QwenTokenizerWrapper`

**证据**

- `vocab.json` 只有 38,349 个条目；官方 Qwen2.5 的 vocab 是 151k 级别。
- `"hello"`、`" the"` 在本地 vocab 中不存在。
- `tokenizer_config.json` 只有 `model_type` / `eos_token` / `pad_token` / `bos_token` / `unk_token`，没有 `chat_template`、`added_tokens_decoder`、`tokenizer_class`。
- `00_download.py` 的 `_build_local_qwen_compatible_bpe()` 是从 TL;DR 语料现造词表。

**修法**

1. 在有 HF token 的环境执行：
   ```bash
   huggingface-cli download Qwen/Qwen2.5-0.5B-Base \
     --include "vocab.json" "merges.txt" "tokenizer_config.json" "tokenizer.json" \
     --local-dir data/v1.0/raw_cache/qwen_tokenizer_official
   ```
   或直接用 `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Base", revision=<sha>)`。
2. 记录官方 commit SHA、`len(tokenizer)`、`vocab_size`、`eos_token_id`、`pad_token_id` 到 `sources_lock.json` / `manifest.json`。
3. 让 `QwenTokenizerWrapper` 优先加载官方 `tokenizer.json`/`tokenizer_config.json`；只有官方文件缺失时才报错，不要回退到自造 BPE。
4. 重跑 00→07，所有 token 统计、长度过滤、分层、token-level 断言都基于官方 tokenizer。

**验收**

- [x] `len(tokenizer)` / `vocab_size` 与官方一致（具体值以实际加载为准，写入 manifest）。
- [x] `tokenizer("hello")`、`tokenizer(" the")` 的 token id 与官方一致。
- [x] `tokenizer.eos_token_id == 151645`、`pad_token_id == 151643`（以实际 tokenizer 为准）。
- [x] `tokenizer.apply_chat_template(...)` 与数据渲染逐字节一致。
- [x] 旧的自造 `vocab.json` 不再被任何脚本读取。

---

### P0-2 换成官方 MMLU / ARC-Challenge / IFEval

**位置**

- `data/v1.0/eval_benchmarks/mmlu_500.jsonl`
- `data/v1.0/eval_benchmarks/arc_challenge_300.jsonl`
- `data/v1.0/eval_benchmarks/ifeval_541.jsonl`
- `data/v1.0/raw_cache/benchmarks/*.parquet`
- `data_engineering/scripts/00_download.py` 的 `_build_offline_benchmark_parquets()`

**证据**

- MMLU 第一行：`"[econometrics] Question 3: Which theoretical principle most accurately characterizes phenomena #1302 in econometrics?"`，选项 `"Option A for econometrics question 3"`。
- ARC 第一行：`"In a controlled scientific experiment #167…"`。
- IFEval 第一行：`"[IFEval #1000] Write a detailed analytical response on topic #1…"`。
- 官方 `cais/mmlu` 的 `all/test` 有 **14,042** 行、57 个 subject；当前合成 MMLU 只有 1,140 行。
- 官方 ARC-Challenge test 有 **1,172** 行；官方 IFEval 有 **541** 行。

**修法**

1. 用 datasets-server 的 parquet 接口下载官方数据（不要再走 `huggingface.co/api/datasets/.../parquet/...` 这种容易 401 的路径）：
   - `https://datasets-server.huggingface.co/parquet?dataset=cais/mmlu`
   - `https://datasets-server.huggingface.co/parquet?dataset=allenai/ai2_arc`
   - `https://datasets-server.huggingface.co/parquet?dataset=google/IFEval`
2. 删除/覆盖 `data/v1.0/raw_cache/benchmarks/` 下的合成 parquet。
3. 重跑 `03_testset.py`，重新生成 `mmlu_500.jsonl`、`arc_challenge_300.jsonl`、`ifeval_541.jsonl`。
4. 重跑 `02_decontaminate_and_dedup.py` 和 `05_warmup.py`，让去污染基于真实 benchmark。
5. 在 `manifest.json` / `sources_lock.json` 记录官方 repo、config、split、commit SHA、parquet SHA256。

**验收**

- [x] MMLU 源数据 `all/test` = 14,042 行、57 个 subject；抽样 500 条后 subject 分布合理。
- [x] ARC-Challenge test = 1,172 行；抽样 300 条。
- [x] IFEval = 541 行，保留官方 `key`、`instruction_id_list`、`kwargs`。
- [x] 人工抽查 10 条 MMLU/ARC/IFEval，确认是真实官方题目，不是占位符。
- [x] 已知官方样本的 hash 或原文片段匹配。
- [x] manifest 中不再出现“合成 benchmark 冒充官方”的情况。

---

### P0-3 修 `00_download.py` 的离线 fallback

**位置**

- `scripts/00_download.py` 开头 `os.environ.pop("HF_TOKEN")` / `pop("HUGGING_FACE_HUB_TOKEN")`
- `online` 检测使用 model API
- `_build_local_qwen_compatible_bpe()`
- `_build_offline_benchmark_parquets()`

**问题**

- 主动删 token，导致即使有权限也必然离线。
- online 检测用 `huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Base`，当前返回 401；但 datasets-server 仍可用。
- 离线分支直接造 tokenizer 和 benchmark，且写入正式 `data/v1.0`。

**修法**

1. 删除 `os.environ.pop("HF_TOKEN", None)` 和 `os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)`。
2. online 检测改用 datasets-server 的可用性检查（例如请求 `https://datasets-server.huggingface.co/parquet?dataset=openai/summarize_from_feedback`）。
3. 离线时**直接报错退出**：
   ```python
   raise SystemExit(
       "Offline mode is not allowed for final data. "
       "Set HF_TOKEN and run with network, or pass --allow-synthetic for smoke tests only."
   )
   ```
4. 如确需 smoke test，增加显式 `--allow-synthetic`：
   - 只在 `data/smoke/` 下生成；
   - manifest 标 `"synthetic": true`；
   - 禁止复制到 `data/v1.0`。
5. 删除 `_build_local_qwen_compatible_bpe()` 和 `_build_offline_benchmark_parquets()`，或改为只在 `--allow-synthetic` 下可用。

**验收**

- [x] 无 token/无网络时，脚本非零退出并给出明确提示。
- [x] 有 token/有网络时，下载的是官方 tokenizer 和官方 benchmark。
- [x] `data/v1.0` 中不存在 `synthetic=true` 的文件。
- [x] `sources_lock.json` 记录的是官方 revision/hash。

---

### P0-4 对齐官方 ChatML 与训练端 token 序列

**位置**

- `scripts/utils.py`
  - `QwenTokenizerWrapper.render_chatml()`
  - `tokenize_sft_with_labels()`
  - `tokenize_dpo_pair()`
- `scripts/06_format_arms.py`
- `scripts/07_validate.py`

**问题**

- 当前 `render_chatml()` 用 `"\n".join(blocks)`，SFT 渲染末尾是 `<|im_end|>`，没有官方模板在每条 `<|im_end|>` 后的 `\n`。
- token-level 断言用的是自实现渲染器；训练端如果使用官方 `tokenizer.apply_chat_template`，可能多一个换行 token/位置。
- SFT 文件同时提供 `messages` 和 `prompt`+`completion`；DPO 文件提供对话列表。训练端到底用哪个字段没有冻结。

**修法**

1. 以官方 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=...)` 为唯一真源。
2. 打印并对比：
   - 自实现 `render_chatml` vs 官方模板；
   - 自实现 `tokenize_sft_with_labels` vs 官方模板 + completion-only mask；
   - 自实现 `tokenize_dpo_pair` vs TRL 对 `prompt` / `chosen` / `rejected` 的实际 tokenization。
3. 修掉 trailing newline / `<|im_start|>assistant\n` 位置差异。
4. 冻结一个 `formatter_version`，写入 `rendered_examples.json`。
5. 明确训练端只使用一种字段：
   - 推荐 SFT：`messages` + `assistant_only_loss=True`；
   - 推荐 DPO：`prompt` + `chosen` + `rejected` 对话列表；
   - 删除或明确标注另一套字段为“仅检查用”。

**验收**

- [x] 同一 `pid` 的 SFT 渲染与 DPO `prompt+chosen` 渲染逐 token 一致。
- [x] 渲染字符串与官方 `apply_chat_template` 逐字节一致（包括末尾换行）。
- [x] `rendered_examples.json` 中 label mask 只覆盖 assistant 段。
- [x] 用 1 条样本跑通 TRL `SFTTrainer` 和 `DPOTrainer` 单步，不报错、不静默改模板。
- [x] G4/G5 的 stage 文件 token 序列完全一致。

---

### P0-5 补齐课程要求的 repo 文件

**课程要求**

最终 repo 应包含：

- `README.md`
- `requirements.txt` / `environment.yml`
- `src/` 或 `code/`
- `scripts/`
- `configs/`
- `results/`

README 要说明：

1. 如何安装依赖；
2. 如何获取数据；
3. 如何复现主实验；
4. 如何复现主结果；
5. 需要哪些预训练模型/API。

**当前缺失**

- 没有 `README.md`；
- 没有 `requirements.txt` / `environment.yml`；
- 没有 `results/`；
- 没有 `.gitignore`；
- `data/` 已 547MB，不应直接进 git。

**修法**

1. 新增根目录 `README.md`，至少包含：
   - 环境安装；
   - 数据下载（官方 HF repo + revision + 本文档链接）；
   - `python data_engineering/scripts/run_all.py`；
   - `python -m unittest discover -s data_engineering/tests -v`；
   - 训练/评测入口（由训练/评测组补）；
   - 数据版本 `data/v1.0` 与 manifest 校验方式。
2. 新增 `requirements.txt`，锁定实际版本：
   - `numpy`
   - `regex`
   - `PyYAML`
   - `pyarrow`
   - `datasets`
   - `transformers`
   - `trl`
   - `torch`
   - `instruction-following-eval`（官方 IFEval checker）
3. 新增 `.gitignore`：
   - `data/**/raw_cache/`
   - `data/**/*.jsonl`
   - `data/**/*.parquet`
   - `__pycache__/`
   - `*.pyc`
4. 新增 `results/`，放数据漏斗表、SMD 表、评测结果图。
5. 大文件放共享盘/对象存储；repo 只保留 manifest、索引、数据卡、脚本。

**验收**

- [x] `README.md` 能让新组员从零跑通数据管线。
- [x] `requirements.txt` 安装后 `07_validate.py` 可运行。
- [x] `.gitignore` 生效，`git status` 不出现 547MB 数据。
- [x] `results/` 有至少数据漏斗和 split/SMD 结果。

---

## 3. P1：影响结论或造成返工

> **说明：本节是 2026-10-18 的首次修复记录。** 2026-10-19 复检已确认 P1-1..P1-5 完成；P1-1 的 χ² 措辞问题（§1.2 R2 / P2-9）已于 2026-10-21 修复。

### P1-1 DPO 的 rejected 策略家族未控制

- 当前 `04_split.py` 只按 `chosen_policy_family` 分层。
- DPO 的信号取决于 `(chosen, rejected)` 两侧；只控制 chosen 可能让 A/B 或不同预算档的 rejected 家族分布失衡。
- 建议：
  1. 在 `split_report.json` 报告 `(chosen_family, rejected_family)` 的 3×3 分布；
  2. 对 A/B 做 χ² 或 SMD 检查；
  3. 如失衡明显，把 `rejected_policy_family` 加入分层（层数会从 45 增加，需重新评估小层合并）。

### P1-2 IFEval 官方 checker

- `ifeval_harness.py` 在没装 `instruction_following_eval` 时使用自实现规则引擎。
- H4 依赖 IFEval 的 Strict/Loose 准确率；自实现 checker 与官方结果可能不一致。
- 建议：固定安装官方 `instruction-following-eval`；自实现只用于 debug，并在输出中标注 `checker=official|fallback`。

### P1-3 真实性/集成测试

在 `tests/` 新增：

- `test_tokenizer_authentic`：
  - `len(tokenizer)` / `vocab_size` 与官方一致；
  - 固定字符串 token id 与官方一致；
  - `apply_chat_template` 与数据渲染逐字节一致。
- `test_benchmark_authentic`：
  - MMLU 源 14,042 行/57 科；抽样 500；
  - ARC 1,172 行；抽样 300；
  - IFEval 541 行；官方 key/kwargs 完整；
  - 已知题目 hash/片段匹配。
- `test_trl_smoke`：
  - 1 条 SFT 样本跑 `SFTTrainer` 单步；
  - 1 条 DPO 样本跑 `DPOTrainer` 单步；
  - 不报错、模板不被静默改写。
- `test_test200_stratification`：
  - 200 vs 500 的 35 层分布最大偏差；
  - 超过阈值时报警。

### P1-4 双跑与版本状态

- `DATA_CARD.md` 写“双跑 SHA256 100% 一致”，但没有双跑日志。
- 建议：
  1. 把 `DATA_PROTOCOL.md` 从 `v1.2-final` 退回 `v1.2-rc`；
  2. 真实数据重跑两次，保存 `manifests/run_1_sha256.txt`、`run_2_sha256.txt`、`diff`；
  3. 全部一致后再升 `v1.2-final`；
  4. 数据卡写“双跑日志路径”。

### P1-5 100 对数据审计盲评

- `data_label_audit_100.jsonl` 同一行包含 `dataset_preferred_hidden_key`、`chosen_policy`、`rejected_policy`、`confidence`。
- 审计者可以直接看到答案，盲评失效。
- 建议：
  - 审计表只保留 `audit_id`、`pid`、`subreddit`、`title`、`post`、`summary_A`、`summary_B`；
  - key 单独放 `manifests/data_label_audit_100_key.json`；
  - policy/confidence 也放 key 文件；
  - 审计表在发放前做一次字段检查，禁止出现 `chosen`/`rejected`/`policy`/`confidence`。

---

## 4. P2：一致性与措辞

> **说明：本节是 2026-10-18 的首次修复记录。** 2026-10-19 复检：P2-1..P2-4、P2-6..P2-8 已完成；P2-5 的无效 SHA 与 P2-9 的 χ² 措辞均已于 2026-10-21 修复。

| ID | 问题 | 修法 |
|---|---|---|
| P2-1 | N=500 子版块漂移 3.11%，协议阈值 `< 3%` | 二选一：把阈值放宽到 5% 并在协议说明理由；或重新分层抽样使漂移 < 3% |
| P2-2 | 数据卡策略分布是去重前 14,760 条 | 用最终 `canonical_pool.jsonl` 重算：`human_ref 7016, sft_early 3287, rl_aligned 3110, sft_strong 1320` |
| P2-3 | `sup1/sup2` 引用写成 Stiennon Table 12 | 改为 **Table 8**（Table 12 是 example comparison tasks） |
| P2-4 | 风险清单删掉 PII / 预训练污染 | 补回 Limitations：Reddit PII/有毒内容、Qwen 预训练可能已见 TL;DR |
| P2-5 | manifest revision 是 parquet hash | 同时记录 HF repo + commit SHA + parquet SHA256 |
| P2-6 | `test_200` 35 层分层性未报告 | 在 `split_report.json` 加 200 vs 500 的层分布最大偏差 |
| P2-7 | 重复别名文件 | `G4_sft_*` / `G4_stage1_sft_*`、`G4_dpo_*` / `G4_stage2_dpo_*`、`test_500.jsonl` / `test_500_full.jsonl` 只保留一套，或在 README 明确“训练只用 stage 命名” |
| P2-8 | D1–D9 只有数据负责人签字 | D3/D7 让评测组确认，D1/D9 让全组确认；记录日期 |

---

## 5. 建议执行顺序

| Phase | 任务 | 通过门槛 |
|---|---|---|
| **Phase 0** | 备份当前 `data/v1.0`；新建分支；把协议退回 `v1.2-rc` | 旧产物不被覆盖，便于对比 |
| **Phase 1** | 修 `00_download.py`；下载官方 tokenizer + benchmark | 官方文件 hash 写入 `sources_lock.json` |
| **Phase 2** | 重跑 `01` → `03` → `02` → `04` → `05` → `06` → `07` | 所有真实数据产物生成；无 `synthetic=true` |
| **Phase 3** | 对齐官方 ChatML；加真实性/集成测试；TRL 单步 smoke test | `07_validate.py` + unittest + smoke test 全绿 |
| **Phase 4** | 双跑 SHA256；更新数据卡/manifest；补 repo 文件 | 双跑一致；README/requirements/results/.gitignore 齐全 |
| **Phase 5** | 组会确认；升 `v1.2-final`；交接训练/评测 | 训练组能独立加载并跑 1 步 |
| **Phase 6（复检尾项，✅ 2026-10-21）** | 修 3 个 commit SHA、χ² 措辞、manifest `env`、tokenizer repo 标签；执行首次 git commit | 已完成：R1–R6 全部关闭；`07_validate.py` + 8 项测试通过；git commit `e3ac629` |

---

## 6. 最终验收清单

### 数据真实性

- [x] tokenizer 是 Qwen2.5-0.5B-Base 官方版本，vocab 规模/固定 token id 正确。
- [x] MMLU 是官方 `cais/mmlu`，源 14,042 行/57 科，抽样 500。
- [x] ARC-Challenge 是官方 `allenai/ai2_arc`，源 1,172 行，抽样 300。
- [x] IFEval 是官方 `google/IFEval`，541 行，官方 key/kwargs 完整。
- [x] `data/v1.0` 中不存在合成占位符；manifest 无 `synthetic=true`。

### 数据控制

- [x] 14,733 个 canonical pool，10k 余量 +4,733。
- [x] P500 ⊂ P2000 ⊂ P10000，A/B 严格 50/50。
- [x] G4/G5 stage 数据逐 token 一致。
- [x] 测试集与训练池/warmup/benchmark 无 exact + Jaccard ≥ 0.8 重叠。
- [x] warmup 全轮次无摘要样本。
- [x] N=500/2k 的 SMD < 0.10；子版块漂移在阈值内。

### 工程与课程

- [x] `README.md` 能让新组员从零跑通。
- [x] `requirements.txt` / `environment.yml` 锁定版本。
- [x] `results/` 有数据漏斗、SMD、评测结果。
- [x] `.gitignore` 排除 547MB 数据。
- [x] 双跑 SHA256 一致，日志可查。
- [x] TRL SFT/DPO 单步 smoke test 通过。
- [x] 100 对审计为真盲评，key 分离。
- [x] 组会确认 `v1.2-final`，D1–D9 签字完整。

### 2026-10-21 复检尾项（已清零）

- [x] `openai/summarize_from_feedback` revision 改为 `b8f7d168b6f4e95b2a92e84768bd6c955bed2f29`。
- [x] `HuggingFaceH4/no_robots` revision 改为 `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`。
- [x] `google/IFEval` revision 改为 `966cd89545d6b6acfd7638bc708b98261ca58e84`。
- [x] 数据卡 χ² 措辞改为“所有检验 p > 0.64，α=0.05 下不显著”。
- [x] manifest `env` 补全 13 项依赖版本（含 `transformers` / `tokenizers` / `trl` / `torch` / `datasets` / `scipy` / `huggingface_hub` / `accelerate`）。
- [x] tokenizer 的 `repo` / `hf_hub_repo` / `revision_for_repo` 对应关系写清楚。
- [x] git 至少一次 commit（`e3ac629`），且 `data/**/raw_cache/` 与大 JSONL/parquet 未入库。

### 低优先建议（不阻塞交付）

- [x] P3-5：README §4.3 已明确“验证前必须安装完整 `requirements.txt`”，并说明 `07_validate.py` 会读取 `trl` / `torch` / `transformers` / `accelerate` 版本；`safe_version()` 作为可选代码改进保留。

---

## 7. 不建议做的事

1. **不要为了修 commit SHA/χ² 措辞而重跑数据内容**。数据内容已是官方真实数据；只需改 metadata 并重跑 `07_validate.py` 刷新 manifest/data card。
2. **不要直接相信硬编码的 pinned SHA**；用之前至少执行一次 HF API / `resolve/<sha>/README.md` HEAD 检查。当前三个无效 SHA 就是反例。
3. **不要为了跑通而保留 `--allow-synthetic` 默认路径**；它只能用于 CI/smoke test，且必须显式标注，不能进入 `data/v1.0`。
4. **不要直接把 547MB `data/` 提交到 git**；`raw_cache` 与大 JSONL/parquet 必须保持 ignored。
5. **不要在未验证 revision/口径前把结果写进最终报告**；χ² 的 `p > 0.80` 与实测不符就是例子。
6. **不要只依赖 `07_validate.py` 的自洽测试**；真实性/集成测试（tokenizer、benchmark、TRL smoke）必须保留并通过。

---

## 8. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-10-18 | 初稿：基于 `DATA_PROTOCOL.md` v1.2-final、`data/v1.0` 与最新 proposal 的审查，列出 P0/P1/P2 修复项、执行顺序与验收清单 |
| v1.1 | 2026-10-19 | 完成 P0-1..P0-5、P1-1..P1-5、P2-1..P2-8 全部代码修复、官方真实数据拉取、双跑 SHA256 零差异校验与 TRL 单步集成测试 |
| v1.2 | 2026-10-19 | 复检更新：独立验证官方 tokenizer/benchmark、8 项测试、双跑与平衡性；记录剩余问题 R1（3 个无效 commit SHA，P2-5 部分完成）、R2（χ² p>0.80 措辞不符，P2-9）、R3（manifest env 不完整）、R4（tokenizer repo/revision 标签）、R5（git 无 commit）、R6（audit key 可访问） |
| v1.3 | 2026-10-21 | 完成 v1.2 复检剩余全部尾项（R1–R6 / P2-5、P2-9、P3-1..P3-4）：更新 3 个官方数据集有效 commit SHA、修正 χ² `p > 0.64` 措辞、补全 `manifest.json` 13 项 `env` 版本、明确 tokenizer `hub_repo`/`revision_for_repo`、将盲评 key 纳入 `.gitignore`，并执行首次 Git commit |
| v1.4 | 2026-10-21 | 最终复检：确认 R1–R6 全部修复并通过 `07_validate.py` + 8 项 unittest；新增 P3-5 低优先健壮性建议（`07_validate.py` 在数据-only 环境读取训练依赖版本会失败，建议 `safe_version()` 或 README 说明） |
