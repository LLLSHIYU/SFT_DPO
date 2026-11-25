# DSAI 4207 Data Card (`data/v1.2` · Protocol `v1.2-final + source-balanced sampling`)

| 属性 | 内容 |
|---|---|
| **项目名称** | Order vs. Scale: A Controlled Study of SFT/DPO Sequencing Across Data Budgets in Small LM Alignment |
| **数据版本** | `data/v1.2`（对应协议 `DATA_PROTOCOL.md` `v1.2-final + source-balanced sampling`，`synthetic = false`） |
| **数据工程负责人** | PAN Haoren |
| **主数据源** | `openai/summarize_from_feedback` (`comparisons`, commit `b8f7d168b6f4e95b2a92e84768bd6c955bed2f29`, train parquet sha256 `8520dcdbd838ec0d…`) |
| **预热数据源** | `HuggingFaceH4/no_robots` (`train`, commit `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, parquet sha256 `76d64120a2ab20eb…`) |
| **官方分词器与模板** | `Qwen/Qwen2.5-0.5B-Base` (`hub_repo = Qwen/Qwen2.5-0.5B`, `revision_for_repo = Qwen/Qwen2.5-0.5B`, commit `060db6499f32faf8b98477b0a26969ef7d8b9987`, `vocab_size = 151,643`, `len(tokenizer) = 151,665`), 官方 ChatML `formatter_version = v1.2`, `eos_token = <|im_end|>` (151645), `pad_token = <|endoftext|>` (151643) |
| **官方评测基准** | `cais/mmlu` (`all/test` 14,042 行/57 科 $\rightarrow$ 500, commit `c30699e8356d`) · `allenai/ai2_arc` (`ARC-Challenge/test` 1,172 行 $\rightarrow$ 300, commit `210d026faf99`) · `google/IFEval` (`train` 541 行全量, commit `966cd89545d6`) |
| **Manifest SHA256** | `a4e899373f5cc540d389346683436b65b348ab2e621de604bda02c8e0837f182` |
| **双跑一致性日志** | `data/v1.0/manifests/run_1_sha256.txt` · `data/v1.0/manifests/run_2_sha256.txt` · `data/v1.0/manifests/run_diff.txt`（0 差异） |

---

## 1. 数据清洗与去污染漏斗（Data Funnel）

### 1.1 TL;DR 训练池（`comparisons/train` → `canonical_pool.jsonl`）

| 阶段 | 输入数量 | 输出数量 | 剔除数量 | 说明 |
|---|---:|---:|---:|---|
| 原始标注行 (`raw_rows`) | 92,858 | 92,453 | 405 | 剔除空字段、`[deleted]`、词数越界及相同正负摘要对 |
| 按 `pid` 聚合唯一题目 | 92,453 | 14,760 | — | 同题不败（Condorcet 0-loss 占比 **96.33%**）+ 置信度 + 策略档位选唯一对 |
| Qwen2.5 官方 Token 长度过滤 | 14,760 | 14,760 | 0 | `prompt_tokens ≤ 768`, `chosen ≤ 160`, `rejected ≤ 200`, `total ≤ 1024` |
| 跨集去污染 (`vs test_500 & benchmarks`) | 14,760 | 14,760 | 0 | 剔除与测试集及官方 IFEval/MMLU/ARC 命中近重复的样本 |
| 池内 Exact + MinHash 0.8 去重 | 14,760 | **14,733** | 27 | 精确 `hash_prompt` 去重 + 精确 5-gram Jaccard ≥ 0.8 并查集去重 |

- **10k 预算档安全余量**：最终可用唯一题目 **14,733** 条，超出 $N=10,000$ 需求 **+4,733** 条（余量充足）。
- **MinHash 阈值敏感性检验**：
  - Threshold `0.7`：25 条近重复；
  - Threshold `0.8`（主设定）：23 条近重复；
  - Threshold `0.9`：12 条近重复。
- **最终去重后 `canonical_pool.jsonl`（14,733 条）Chosen 策略家族构成**：`{"rl_aligned": 3110, "human_ref": 7016, "sft_strong": 1320, "sft_early": 3287}`。
- **最终去重后 `canonical_pool.jsonl`（14,733 条）Rejected 策略家族构成**：`{"rl_aligned": 3176, "sft_early": 7821, "human_ref": 1991, "sft_strong": 1745}`。

### 1.2 TL;DR 测试集（`comparisons/validation` → `test_200_main.jsonl` & `test_500_full.jsonl`）

- 原始行数：86,086 行；其中显式拦截并剔除 `cnndm`（CNN/DailyMail 新闻）批次 **2,284** 行，剔除无效 `pid`（如 `edit_b2_eval_test`）**5** 行。
- 经官方 Qwen2.5 Token 过滤与池内去重后可用唯一 Reddit 题目：**6,210** 条（与训练池 `pid` 交集为 0）。
- 按 `prompt_tokens` 5 分位 × 子版块（Top 6 + `other` = 35 层）分层抽样并经中点排序后导出：
  - **`test_200_main.jsonl`（200 条）**：对齐 Proposal 预注册主实验与 API 预算；
  - **`test_500_full.jsonl`（500 条）**：含主实验前 200 条 + 300 条同分布扩展敏感性集；
  - **200 vs 500 的 35 层最大层比例偏差**：**0.30%**（详见 `manifests/testset_stratum_report.json`）。

### 1.3 M0 通用预热集（`HuggingFaceH4/no_robots` → `warmup_5k.jsonl`）

- 原始 `train` 行数：9,500 条；
- 第一道剔除 `category == "Summarize"`：395 条；
- 第二道全轮次（`system` / `user` / `assistant`）摘要正则额外剔除：135 条（两道合计剔除 530 条）；
- 超长（>2048 Qwen2.5 tokens）剔除：19 条；
- 第三道跨集去污染与内部去重剔除：2 条；
- 干净可用池：**8,949** 条 $\rightarrow$ 按剩余 9 个类别等比例分层抽取 **5,000** 条生成 `warmup_5k.jsonl`。

---

## 2. 嵌套子集与 Part A / Part B 平衡性报告（Balance, SMD & Policy $\chi^2$）

| 预算档 $N$ | $|A_N|$ | $|B_N|$ | SMD (`prompt`) | SMD (`chosen`) | SMD (`rejected`) | Rejected 策略家族 $\chi^2$ $p$-value | `(chosen, rejected)` 3×3 联合 $\chi^2$ $p$-value | 与 10k 池子版块最大偏差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **$N = 500$** | 250 | 250 | `+0.0001` | `-0.0031` | `+0.0040` | `1.0000` | `1.0000` | 1.43% |
| **$N = 2,000$** | 1,000 | 1,000 | `-0.0001` | `-0.0005` | `+0.0002` | `1.0000` | `1.0000` | 0.13% |
| **$N = 10,000$** | 5,000 | 5,000 | `+0.0001` | `+0.0001` | `+0.0001` | `1.0000` | `1.0000` | 0.00% |

所有预算档的全部连续变量标准化均值差 $|\mathrm{SMD}| \ll 0.10$，Chosen 与 Rejected 策略家族（及 3×3 联合矩阵）在 Part A / Part B 间所有 $\chi^2$ 检验 $p > 0.64$，在 $\alpha=0.05$ 下均无显著差异，且各预算档子版块漂移均严格控制在 `< 3%` 阈值以内。

---

## 3. 局限性与伦理风险说明（Limitations & Risks）

1. **Reddit 原始语料中的 PII 与潜在不当内容风险**：`openai/summarize_from_feedback` 源自 Reddit 真实用户发帖（如 `r/relationships`、`r/tifu`），虽经前人匿名化处理，仍可能包含残留的个人身份细节（年龄、家庭关系描述）或情绪化/粗俗用语。本数据集仅限 DSAI 4207 课程非商业学术研究使用，严禁用于生产部署或重识别个人身份。
2. **基座模型预训练污染不可控性**：`Qwen/Qwen2.5-0.5B-Base` 的预训练语料未公开明细，不排除其在预训练阶段已接触过公共 Reddit 或 TL;DR 语料。因此实验报告中将 500 档严格界定为“本后训练流水线内第一次接触特定摘要任务（first task-specific contact within our post-training pipeline）”。
3. **正负样本天然长度差异**：人类偏好标注中 `chosen` 平均长度略长于 `rejected`；管线不人为按长度差截断偏好对，而在评测阶段统一采用长度受控胜率（LCWR）消除长度偏置。

---

## 4. 验收状态（I1–I10 & 真实性/集成测试）

- 自动验证见 `07_validate.py` 和独立测试日志。
- **100 对数据标签盲评隔离**：`manifests/data_label_audit_100.jsonl`（纯盲评表）与 `manifests/data_label_audit_100_key.json`（答案与策略 key）物理分离并通过字段审查。

IFEval 是 IF 任务的主评测。摘要清单的 `external_benchmarks` 记录其路径和哈希，以复查既有数据隔离和复现原测试。
