# DSAI 4207 数据协议 v1.2-final

> 本文档定义 G2 / G3 / G4 / G5 / G3′ 与 M0 预热集所依赖的全部数据构造规则。
> **状态：已确认（v1.2-final，D1–D9 全部按 v1.2 推荐默认值冻结生效）。**
> 数据产物标记为 `data/v1.0`；开跑后任何改动必须升数据版本并重跑受影响实验臂。

| 项 | 内容 |
|---|---|
| 项目 | Order vs. Scale: A Controlled Study of SFT/DPO Sequencing Across Data Budgets in Small LM Alignment |
| 课程 | DSAI 4207 Introduction to Large Language Models |
| 文档版本 | v1.2-final（对齐最新 proposal + D1–D9 全部按默认值确认冻结） |
| 日期 | 2026-10-18 |
| 对应 proposal | `DSAI4207_Final_Proposal/main.pdf`（2026-10-13，4 页，最新版） |
| 版本口径 | 以 `main.pdf` 为准；`main.tex` 与 `main.pdf` 内容一致 |
| 团队 | PAN Haoren (24104216d)、LIU Shiyu (24106028d)、PAN Yubing (24103431d) |
| 数据负责人 | PAN Haoren |
| 关联文档 | `DSAI4207_Final_Proposal/main.pdf`、`balancing the budget.pdf`、`DATA_ISSUES.md` |

---

## 1. 目的与适用范围

本协议只解决一件事：**让 G4 与 G5 的差异可以被唯一归因于“顺序”。**

为此，数据层必须同时满足：

1. 每个 prompt 只有一对经“同题不败（Condorcet）”筛选的 canonical chosen/rejected；
2. 500 ⊂ 2,000 ⊂ 10,000 是真子集，且前缀采样无小层抢跑偏差；
3. 每个预算档的 Part A / Part B 数量严格相等（0.5N / 0.5N）；
4. G4 与 G5 看到的 SFT 文本、DPO 对逐字且逐 token 相同；
5. 测试集（100% 纯 Reddit TL;DR，剔除 CNN/DM 批次）、warmup、评测基准与训练池互不污染；
6. 所有产物可复现、可审计、带版本号。

本协议不规定训练超参、评测脚本和统计方法；这些由其他组员负责，但数据接口必须与本文档一致。

### 1.1 与最新 proposal 的对应关系

| 最新 proposal 的说法 | 本协议位置 | 处理 |
|---|---|---|
| H1/H2/H3：OA(500) > OA(2000) > OA(10k) ≈ 0 | §8 嵌套与配对切分；附录 D | 已覆盖，§8.6 修正中点排序消除小样本偏差 |
| H4：风格收益 vs 能力税 | §12 MMLU/ARC/IFEval 接口 | 已覆盖，固定索引见 D8 |
| “200 held-out prompts” | §7 | **同时导出 `test_200_main.jsonl`（对齐 Proposal 主分析）与 `test_500_full.jsonl`（扩展敏感性），训练池对全部 500 条去污染，见 D3** |
| “first contact with the task” | §10.2 可执行筛查 | 扫 `messages` 全轮次；报告措辞改为“本流水线内第一次”，见 `DATA_ISSUES.md` P1-5 |
| “chosen side” 镜像绑定 | §5.2–5.3, §9 | 已覆盖；**纠正 v1.1 将 `sup1/sup2` 误作人类摘要的错误**，引入同题不败规则，见 D2 |
| “Part A/B stratified by length” | §8.4 | 按 prompt 长度 × chosen 长度 × 策略家族分层，见 D9 |
| “decontamination ... benchmarks” | §6.3 | 先跨集去污染、再内部去重，LSH 命中后精确核验 Jaccard ≥ 0.8 |
| “IFEval harness” 归 Name A | §12.4 | **需确认 Name A 是否=你，见 D7** |
| “29 runs” | 附录 D | 一致 |

---

## 2. 术语与口径

| 术语 | 定义 |
|---|---|
| **prompt / 题目** | 一条 Reddit 原始帖子，以 `info.id`（如 `t3_34xale`）唯一标识；预算单位是“唯一 prompt 数”，不是标注行数 |
| **pair / 偏好对** | 同一 prompt 下经人类标注员判别好坏的一条 `(chosen, rejected)` 摘要对比 |
| **chosen / rejected** | 人类标注者更偏好 / 较不偏好的摘要；由 `choice` 字段（0 或 1）决定 |
| **canonical pair** | 一个 prompt 有多个标注行时，按预注册规则（同题不败 → 置信度 → 策略质量 → 确定性哈希）选出的唯一一对 |
| **policy / 采样策略** | 生成候选摘要的来源标识（详见 Stiennon et al., 2020 Table 8）：**仅 `ref` 为 Reddit 原帖作者编写的人类参考摘要**；`sup1`（750M）、`sup2`（1.3B）、`sup3`（1.3B）、`sup3_6b`（6.7B）、`sup4`（1.3B）、`sup4_6b`（6.7B）均为 OpenAI 的 SFT 模型；含 `ppo` 或 `bo` 的策略为 RL / Best-of-N 对齐模型 |
| **policy_family / 策略家族** | 按 `chosen` 与 `rejected` 的策略质量分为四档：`rl_aligned`（含 `ppo`/`bo`）、`sft_strong`（`sup4*`/`sup3_6b`）、`human_ref`（`ref`）、`sft_early`（`sup1`/`sup2`/`sup3` 等早期小模型） |
| **budget N** | 该实验臂消耗的唯一 prompt 总数，N ∈ {500, 2000, 10000} |
| **Part A / Part B** | 两阶段臂把 N 个 prompt 互斥切成两个等长子集，各 0.5N |
| **mirror binding / 镜像绑定** | SFT 文本 100% 取自同一 canonical pair 的 chosen；G4 与 G5 看到同一批 A/B 数据，只交换阶段顺序 |
| **nested subsets / 嵌套子集** | P500 ⊂ P2000 ⊂ P10000；换预算不换题，只加题 |
| **held-out** | 从训练池中完全排除、任何训练阶段都不使用的测试集 prompt |
| **decontamination / 去污染** | 训练池与测试集、warmup、评测基准之间做 exact + 精确 5-gram Jaccard ≥ 0.8 查重 |
| **MinHash** | 用哈希草图快速召回候选近重复对的方法；召回后必须计算精确 5-gram Jaccard 过滤假阳性 |
| **M0** | Qwen2.5-0.5B-Base 在 5k `no_robots` 无摘要通用指令上预热 1 epoch 后的共享起点 |

---

## 3. 不可违反的 10 条不变量

| ID | 不变量 |
|---|---|
| I1 | canonical pool 中每个 `pid` 恰好一条记录、恰好一对 chosen/rejected，且 100% 来自 Reddit（无 `cnndm` 批次） |
| I2 | `chosen != rejected`，二者均非空，且通过长度/有效性过滤 |
| I3 | P500 ⊂ P2000 ⊂ P10000，且各档大小精确为 500 / 2,000 / 10,000 |
| I4 | 每个预算档 `len(A) == len(B) == N/2`，且 `A ∩ B = ∅`、`A ∪ B = P_N` |
| I5 | G4 Stage-1 的 SFT 文本及 token 序列多重集 == G5 Stage-2 的 SFT 文本及 token 序列多重集 |
| I6 | G4 Stage-2 的 DPO 对及 token 序列多重集 == G5 Stage-1 的 DPO 对及 token 序列多重集 |
| I7 | G2 的 SFT 文本 == P_N 全部 chosen；G3 的 DPO 对 == P_N 全部 pair；同题 SFT 与 DPO chosen 的 token 序列严格一致 |
| I8 | 测试集（500 全集及 200 主集）与 P10000、warmup、任何训练臂均无交集，且无 5-gram Jaccard ≥ 0.8 的近重复 |
| I9 | warmup 全轮次 `messages` 中不含摘要类样本，且与 TL;DR 池/测试集/评测基准无污染 |
| I10 | 同一份协议连续运行两次，所有产物 SHA256 完全一致 |

任何一条不满足，管线必须**报错退出**，不允许静默降级或自动改抽样。

---

## 4. 数据源与版本固定

### 4.1 主数据源

| 项 | 值 |
|---|---|
| 数据集 | `openai/summarize_from_feedback` |
| config | `comparisons` |
| 训练池 split | `train` |
| 测试集候选 split | `validation`（**必须剔除 `cnndm` 新闻批次**） |
| 许可 | MIT（Reddit 派生内容仅限非商业研究） |
| 固定方式 | `datasets.load_dataset(..., revision=<commit_sha>)`；revision 写入 manifest |
| 下载日期 | 记录在 manifest |

**为什么只用 `comparisons`：** 它是唯一同时提供 `(post, chosen, rejected)` 的 split，能保证 SFT 与 DPO 从同一份记录派生，镜像绑定才严格成立。

**为什么不用 `UCL-DARK/openai-tldr-filtered` 做 SFT：** 那是另一份独立过滤的 SFT 数据集（仅含 Reddit 原始 `ref` 摘要），其文本与 `comparisons` 中的 `chosen` 不一致，混用会直接破坏镜像绑定（G2/G4/G5 的 SFT 文本将不再等于 DPO 的 `chosen`）。

### 4.2 为什么测试集来自 `validation`（及 CNN/DM 过滤）

- `train` 与 `validation` 的 Reddit prompt 零重叠。
- **关键过滤**：`comparisons/validation` 中混有用于奖励模型跨领域评测的 CNN/DailyMail 新闻摘要批次（`batch` 为 `batch0_cnndm`、`cnndm0`、`cnndm2`）。管线第一步必须剔除所有含 `cnndm` 的行，并要求 `info.subreddit` 和 `info.title` 非空，保证测试集 100% 为 Reddit TL;DR。
- 过滤后 `validation` 仍有超过 5,000 个唯一 Reddit prompt，足够分层抽取 500 条（含 200 条主分析集 + 300 条敏感性集），且不消耗 `train` 的训练池余量。

### 4.3 备选数据源（仅在组会确认切换时启用）

若组里决定与种子论文数据源对齐，可改用 `UCL-DARK/openai-tldr-summarisation-preferences` 的 `train` + `validation`（必须从同一份 preference 数据派生 SFT 与 DPO 以维持镜像绑定）。切换后必须重跑第 5 节全部漏斗统计并在 manifest 中记录新 revision，不得与原版混用。

### 4.4 预调研计数与 v1.2 口径修正

> **v1.2 重要纠偏**：v1.1 误把 `sup1`（750M SFT 模型）和 `sup2`（1.3B SFT 模型）当成“人类编写摘要”，并限制 `chosen_policy ∈ {sup1, sup2, ref}`，导致误删了数据集中质量最高的 `sup4_6b` / `sup4_ppo_*` 优选摘要，且人为制造了 10k 档余量紧张的假象。v1.2 恢复标准定义：凡经人类标注员作出偏好判断（`choice ∈ {0, 1}`）的有效 Reddit 样本均纳入候选池。

| 指标 | 数值（预估，以固定 revision 实跑为准） |
|---|---:|
| `comparisons/train` 原始行数 | 92,858 |
| `comparisons/train` 唯一 Reddit prompt | 14,767 |
| 经有效性、token 长度与同题不败选对后唯一 prompt | ≈ 14,550 |
| 经跨集去污染 + MinHash 0.8 内部去重后唯一 prompt | ≈ 14,450（10k 档余量 > 4,400 条） |
| `comparisons/validation` 原始行数 | 86,086 |
| `comparisons/validation` 剔除 `cnndm` 后唯一 Reddit prompt | ≈ 6,000+ |

---

## 5. Canonicalization（规范化与一题一对）

### 5.1 输入字段

从 `comparisons` 的每一行读取：

```text
info.id            # prompt 唯一 id（如 t3_34xale）
info.post          # Reddit 帖子正文
info.title         # Reddit 帖子标题（进入 user prompt，防摘要幻觉）
info.subreddit     # 所属子版块（进入 user prompt，并用于分层与报告）
batch              # 批次名（用于过滤 cnndm 新闻数据）
summaries[0/1].text
summaries[0/1].policy   # ref 为人类参考摘要；sup1~sup4* 为 SFT 模型；*ppo*/*bo* 为 RL/BoN 模型
choice             # 0 或 1；chosen = summaries[choice], rejected = summaries[1 - choice]
extra.confidence   # 人类标注员置信度（1~9 或 null；数值越大越自信）
```

### 5.2 候选行有效性过滤

在选 canonical pair 之前，逐行过滤：

| 过滤项 | 规则 | 理由 |
|---|---|---|
| 领域过滤（非新闻） | `"cnndm" not in (batch or "").lower()` 且 `info.site is None` | 剔除 validation 中的 CNN/DailyMail 新闻样本 |
| 元数据完整 | `info.id`、`info.post`、`info.title`、`info.subreddit` 均非空 | 保证完整 Reddit 贴文结构 |
| 正文/标题非删除 | `post` 与 `title` 不匹配 `[removed]` / `[deleted]`（不区分大小写） | 无效帖无法摘要 |
| choice 有效 | `choice in (0, 1)` | 必须有明确的人类偏好标签 |
| post 长度 | 30–2,000 词（预筛）；最终完整 prompt ≤ 768 Qwen tokens | 太短无摘要价值，太长超出训练窗口 |
| chosen 长度 | 5–100 词；最终 ≤ 160 Qwen tokens | 去掉空/超短/异常截断或失控长文本 |
| rejected 长度 | 5–120 词；最终 ≤ 200 Qwen tokens | 同上 |
| 非相同 | `normalize_for_hash(chosen) != normalize_for_hash(rejected)` | 去掉无区分度的对 |

**策略分类标注（不做盲目删减，用于优先排序与分层）：**

定义辅助函数 `get_policy_tier(policy: str) -> int`（数值越大质量越高）：
- **Tier 3 (`rl_aligned`)**：含 `ppo` 或 `bo`（OpenAI RLHF / Best-of-N 策略，人类胜率最高）；
- **Tier 2 (`sft_strong`)**：`sup4`、`sup4_6b`、`sup3_6b` 等后期大容量/清洗版 SFT 策略；
- **Tier 2 (`human_ref`)**：`policy == "ref"`（Reddit 作者原版 TL;DR）；
- **Tier 1 (`sft_early`)**：`sup1`（750M）、`sup2`（1.3B）、`sup3`（1.3B）及其他早期/高温采样策略。

**不做**的过滤（v1.2 默认）：
- 不做子版块硬性配额（会导致分布失真；只做分层与报告）。
- 不清洗正文脏话（会改变文本分布，破坏与种子论文的可比性）。
- 不按 chosen/rejected 长度差强行删 pair（长度偏好交给 LCWR 协变量处理；数据卡报告长度差分布）。

### 5.3 一题一对的选择规则（预注册：同题不败 + 置信度 + 策略质量）

同一 `pid` 通常有 4–8 个有效标注行。为避免选出“在另一行对比中落败的劣质摘要”作为 SFT 范文，先在同一 `pid` 内统计每个候选摘要文本（按 `normalize_for_hash` 归并）的胜负记录：
- `wins[s]`：摘要 `s` 在该 `pid` 下作为 `chosen` 出现的次数；
- `losses[s]`：摘要 `s` 在该 `pid` 下作为 `rejected` 出现的次数。

对该 `pid` 下的所有有效候选行，按以下优先级元组**降序**排序，取第一条作为唯一 `canonical pair`：

1. **同题不败优先（Condorcet 约束）**：`int(losses[chosen] == 0)`（优先选择从未在该题其他对比中输过的 `chosen` 摘要）；
2. **高置信度门槛**：`int((extra.confidence or 5) >= 3)`（排斥 `confidence ∈ {1, 2}` 的极低把握标注，除非该题无其他候选行）；
3. **同题净胜场次**：`wins[chosen] - losses[chosen]`（越受一致偏好的 `chosen` 越优先）；
4. **标注员置信度**：`extra.confidence if extra.confidence is not None else -1`（越高越优先）；
5. **Chosen 策略质量档位**：`get_policy_tier(chosen_policy)`（确保 SFT 优先学到高质量摘要而非早期 750M 弱模型输出）；
6. **确定性 Hash 兜底**：按 `sha256(pid + "\x00" + chosen + "\x00" + rejected)` 字典序**升序**（取最小者）。

若同一 `pid` 的不同行出现轻微不同的 `post` / `title` 文本，取被选中行对应的 `post` 与 `title`。

**记录字段：** `n_candidate_rows`、`chosen_wins_in_pid`、`chosen_losses_in_pid`、`chosen_policy_family`、`rejected_policy_family`、`selected_confidence`、`selected_source_row_hash`。

### 5.4 文本规范化

**训练文本**（`normalize_for_training`，应用于 `title`、`post`、`chosen`、`rejected`）：

1. `html.unescape`（还原 `&amp;`、`&gt;` 等 HTML 实体）。
2. `\r\n` / `\r` → `\n`。
3. 去掉零宽字符（U+200B–U+200D、U+FEFF）。
4. NBSP（`\u00a0`）→ 普通空格。
5. 移除 Reddit 遗留的首尾多余空白（`strip()`），以及摘要开头自带的冗余前缀（若 `chosen`/`rejected` 以 `"TL;DR:"` 或 `"TLDR:"` 开头，仅剥离该前缀词并 `strip()`，防止部分样本带前缀、部分不带导致格式分裂；剥离情况计入报告）。
6. 连续 3 个以上换行压成 2 个（`\n\n`）。

**不**做：小写化、去标点、去重音、截断、改写。

**哈希文本**（`normalize_for_hash`，仅用于查重与去污染）：

1. 小写（`lower()`）。
2. 删除 URL（`https?://\S+|www\.\S+`）。
3. 非字母数字字符（`[^a-z0-9]+`）→ 单个空格。
4. 压缩连续空白并 `strip()`。

> **铁律**：训练文本与哈希文本分开存储。**永远不要用哈希文本训练**，否则会破坏大小写与标点分布。

### 5.5 最终 Token 过滤

使用**固定 revision 的 `Qwen/Qwen2.5-0.5B-Base` tokenizer**，按 §11.1 的完整模板渲染后计算：

- `prompt_tokens ≤ 768`（含 `Subreddit`、`Title` 与 `post` 的完整 user message tokens）
- `chosen_tokens ≤ 160`
- `rejected_tokens ≤ 200`
- `total_tokens ≤ 1024`（完整 ChatML 序列：`system + user + max(chosen, rejected)`）

超过的样本**直接丢弃，不截断**。理由：TL;DR 正常样本远低于此上限；硬截断会造成帖子结尾缺失或摘要半句中断的人为伪影。

### 5.6 输出 schema

`canonical_pool.jsonl`，每行：

```json
{
  "pid": "t3_34xale",
  "subreddit": "relationships",
  "title": "My [24F] boyfriend [26M] ...",
  "post": "…",
  "user_content": "Subreddit: r/relationships\nTitle: My [24F] boyfriend [26M] ...\n\n…",
  "chosen": "…",
  "rejected": "…",
  "chosen_policy": "sup4_6b",
  "rejected_policy": "sup2",
  "chosen_policy_family": "sft_strong",
  "rejected_policy_family": "sft_early",
  "n_candidate_rows": 6,
  "chosen_wins_in_pid": 3,
  "chosen_losses_in_pid": 0,
  "selected_confidence": 8,
  "prompt_tokens": 428,
  "chosen_tokens": 38,
  "rejected_tokens": 35,
  "hash_prompt": "…",
  "hash_pair": "…",
  "source_split": "train"
}
```

其中 `hash_prompt = sha256(normalize_for_hash(title + " " + post))`，`hash_pair = sha256(chosen + "\x00" + rejected)`（基于 `normalize_for_training` 后的精确训练文本，用于镜像绑定断言）。

### 5.7 漏斗报告

必须输出并写进数据卡：

```text
raw rows
  → 剔除 cnndm 批次 / 空字段 / [deleted] / 无效词数 / 相同答案
  → 按 pid 聚合成唯一 prompt（同题不败 + 置信度 + 策略档位选唯一 canonical pair）
  → Qwen tokenizer 精确 token 长度过滤
  → 跨集去污染（vs test_500 / vs IFEval / vs MMLU / vs ARC）
  → 池内 exact 去重（hash_prompt）
  → 池内 MinHash + 精确 5-gram Jaccard ≥ 0.8 去重
  → final canonical pool
```

每一步记录：输入数、输出数、删除数、删除原因 Top-5 示例。

---

## 6. 去污染与去重（先外后内）

### 6.1 执行顺序铁律：先跨集去污染，后池内去重

为避免池内去重保留的代表元恰好与测试集近重复（或池内去重掩盖二跳测试泄漏），管线必须严格按以下顺序执行：

1. **Step A（测试集内部净化）**：对 `validation` 候选池做内部 exact + MinHash 去重，抽取固定 `test_500_full.jsonl`（其中前 200 条为 `test_200_main.jsonl`）。
2. **Step B（训练池与预热集对外部基准/测试集去污染）**：将 `train` 候选池与 `warmup` 候选池分别对 `test_500_full`、`IFEval`、`MMLU`、`ARC-Challenge` 做 exact + MinHash（精确 5-gram Jaccard ≥ 0.8）查重；**凡命中者一律从 `train` 或 `warmup` 中直接删除**。 Also 将 `warmup` 对 `train` 候选池查重，命中者从 `warmup` 中删除。
3. **Step C（池内去重）**：在完成外部去污染后的干净 `train` 池与 `warmup` 池内部，各自执行 exact 去重与 MinHash 近重复聚类去重。

### 6.2 Exact 去重

对 `hash_prompt` 完全相同的记录，只保留一条，优先级与 §5.3 一致：
1. `chosen_losses_in_pid == 0` 优先；
2. `selected_confidence` 高优先；
3. `get_policy_tier(chosen_policy)` 高优先；
4. `hash_pair` 字典序最小优先。

### 6.3 MinHash 近重复与精确 Jaccard 核验（防单链假阳性）

| 参数 | 值 |
|---|---|
| 索引工具 | `datasketch.MinHashLSH` |
| 文本口径 | 对 `normalize_for_hash` 后的文本提取 **5-word shingles（词级 5-gram 集合 $S$）** |
| `num_perm` | 128（固定随机种子 `seed=42`） |
| `threshold` | **0.8**（精确 5-gram Jaccard：$\frac{|S_1 \cap S_2|}{|S_1 \cup S_2|} \ge 0.8$） |
| 短文本保护 | 词数 < 5 时只做 exact 匹配，不生成 5-gram |
| **假阳性过滤（关键）** | `lsh.query()` 仅作为候选召回（LSH 阈值可设为 0.75 以提高召回）；**召回的候选对必须计算精确 5-gram 集合 Jaccard，仅当精确 Jaccard ≥ 0.8 时才判定为近重复 / 连入并查集（Union-Find）**，严禁把未核验的 LSH 碰撞直接塞入并查集 |
| 簇内代表元选择 | 每个连通分量保留质量最高的一条（规则同 §6.2） |

> 主分析用 0.8；同时统计精确 Jaccard ≥ 0.7 与 ≥ 0.9 下的命中数作为敏感性检查写入数据卡。

### 6.4 查重范围与各集文本口径

| 对比 | 目的 | 文本口径 | 处理 |
|---|---|---|---|
| `pool vs test_500` | 防测试泄漏 | `title + " " + post` | 删 `pool` 样本，绝不动 `test` |
| `pool vs IFEval/MMLU/ARC` | 防评测基准污染 | `title + " " + post + " " + chosen` vs 基准文本 | 删 `pool` 样本 |
| `warmup vs pool/test` | 防 M0 预热接触摘要数据 | `warmup` 全轮次 `user + assistant` 文本 vs `title + post + chosen` | 删 `warmup` 样本 |
| `warmup vs IFEval/MMLU/ARC` | 防 M0 预热污染评测基准 | `warmup` 全轮次 `user + assistant` 文本 vs 基准文本 | 删 `warmup` 样本 |
| `pool internal` | 去 Reddit 一贴多发/转载 | `title + " " + post` | 每簇保留最优一条 |
| `warmup internal` | 去通用指令重复 | `warmup` 全轮次 `user + assistant` 文本 | 每簇保留 `prompt_id` 字典序最小一条 |

**外部基准文本拼接口径：**
- **MMLU**：`question + " " + " ".join(choices)`
- **ARC-Challenge**：`question + " " + " ".join(choices)`
- **IFEval**：官方 `prompt` 原文

### 6.5 报告 schema

`decontamination_report.json`：

```json
{
  "minhash": {"shingle_words": 5, "num_perm": 128, "lsh_recall_threshold": 0.75, "exact_jaccard_threshold": 0.8},
  "scopes": [
    {"name": "pool_vs_test500", "left_n": 14550, "right_n": 500, "matches": 2, "dropped_from_left": 2, "examples": []},
    {"name": "pool_internal", "input_n": 14548, "clusters": 28, "dropped": 30, "examples": []}
  ]
}
```

---

## 7. 测试集（200 主分析集 + 500 扩展全集）

### 7.1 来源、规模与双文件机制

| 项 | 设定 |
|---|---|
| 来源 | `openai/summarize_from_feedback` · `comparisons/validation`（已执行 §5.2–5.5 全部过滤，**100% 剔除 `cnndm` 新闻批次**） |
| 候选池规模 | 约 6,000 个唯一 Reddit prompt（内部先完成 exact + MinHash 0.8 去重） |
| 抽样总规模 | 500 条（分层抽样，固定排序后导出 `test_500_full.jsonl`） |
| **主分析集（对齐 Proposal）** | **取 `test_500_full.jsonl` 的前 200 条（`order ∈ [0, 199]`），单独导出为 `test_200_main.jsonl`** |
| 扩展敏感性集 | 剩余 300 条（`order ∈ [200, 499]`），仅在需要更高统计功效时作为补充分析，不覆盖 Proposal 预注册的 200 条主结论 |
| 抽样 seed / 盐值 | `SALT_TEST = "dsai4207_test_v1.2_20261018"`，`SALT_TEST_ORDER = "dsai4207_test_order_v1.2"` |

**为什么要同时导出 `test_200_main.jsonl` 和 `test_500_full.jsonl`：**
- Proposal 正文预注册的是 `200 held-out prompts`（对应 3,200 次裁判 API 调用，约 $3–5）；直接交付 `test_200_main.jsonl` 可防止评测组员误跑 500 条导致与 Proposal 预注册口径不一致。
- 同时，先按 500 条做分层抽样并用哈希排序取前 200 条，能保证前 200 条与后 300 条同分布，且训练池对全部 500 条做去污染，未来若需扩充样本量降低置信区间宽度（从 ±7% 降至 ±4.4%），无需重训任何模型即可直接无缝评测后 300 条。

### 7.2 分层抽样与确定性前缀

- **分层变量**：`prompt_tokens` 五分位箱（5 档）× 子版块桶（Top 6 子版块 + `other`，共 7 档）= 35 个层。
- 若某层候选数 < 5，并入同一长度分位下的 `other` 子版块桶。
- 为保证前 200 条（`test_200_main`）和全部 500 条（`test_500_full`）都严格保持各层比例，对先按比例抽出的 500 条测试题应用与 §8.6 相同的**层内中点分数 `(i + 0.5) / n_stratum` 全局排序**，再赋 `order = 0, 1, ..., 499`。这样前 200 条自动是全部 35 个层的等比例代表性子集！

### 7.3 输出 schema

`test_200_main.jsonl` 与 `test_500_full.jsonl` 每行：

```json
{
  "pid": "t3_…",
  "subreddit": "relationships",
  "title": "…",
  "post": "…",
  "user_content": "Subreddit: r/relationships\nTitle: …\n\n…",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant who is an expert at summarizing content into TL;DR sentences. Summarize the following text"},
    {"role": "user", "content": "Subreddit: r/relationships\nTitle: …\n\n…"}
  ],
  "reference": "…",
  "reference_policy": "sup4_6b",
  "order": 0,
  "is_main_200": true,
  "hash_prompt": "…"
}
```

- `reference` 是该测试题按 §5.3 选出的 canonical `chosen` 摘要，仅用于可读性对照或可选 ROUGE；**LCWR 主指标对比的是各实验臂 checkpoint 与 M0 的输出，不使用 `reference`**。
- 测试集绝不进入任何训练臂，也不得用于调参。

---

## 8. 预算、嵌套与 Part A/B 切分

### 8.1 预算

$N \in \{500, 2000, 10000\}$。预算单位是**唯一 prompt 数**，不是原始标注行数，也不是优化步数。

### 8.2 真子集嵌套

先构造 10k 池，再取前缀真子集：
- $P_{500} \subset P_{2000} \subset P_{10000}$
- 同时满足 $A_{500} \subset A_{2000} \subset A_{10000}$ 且 $B_{500} \subset B_{2000} \subset B_{10000}$。

### 8.3 配对设计（消除 A/B 数量与分布偏差）

在每个分层内部，将样本两两配成 `(A, B)` 对。每个预算档 $N$ 直接截取前 $k = N/2$ 对，从而在数学上 100% 保证：
- 每个预算档 $|A_N| = |B_N| = N/2$；
- $A_N$ 与 $B_N$ 在每一层上的样本数完全相等（差值 = 0）。

### 8.4 分层变量

| 变量 | 分箱 |
|---|---|
| `prompt_tokens` | 5 个等频分位箱（按 canonical pool 计算） |
| `chosen_tokens` | 3 个等频分位箱（short / medium / long） |
| `chosen_policy_family` | 3 类桶：`rl_or_strong`（Tier 3 + Tier 2 `sft_strong`）、`human_ref`（`ref`）、`sft_early`（Tier 1） |

共 $5 \times 3 \times 3 = 45$ 个初始层。

**小层合并规则：**
- 若某层样本数 < 6 条，先与同一 `chosen_policy_family`、同一 `prompt_tokens` 箱下相邻的 `chosen_tokens` 箱合并；
- 仍 < 6 条，再与相邻 `prompt_tokens` 箱合并；
- 所有合并记录写入 `split_report.json`。

### 8.5 层内配对算法

```python
# 输入: pool（已完成跨集去污染与池内去重的 canonical_pool）
# 输出: pairs（每个元素含 A/B 两条记录及所属 stratum）, reserve_odd

SALT_PAIR = "dsai4207_pair_v1.2"
SALT_AB   = "dsai4207_ab_v1.2"

pairs = []
reserve_odd = []

for stratum_id, stratum_items in sorted(strata.items()):
    items = sorted(stratum_items, key=lambda r: sha256(r["pid"] + SALT_PAIR))
    if len(items) % 2 == 1:
        reserve_odd.append(items.pop())  # 奇数余数进入 reserve，不参与配对
    for i in range(0, len(items), 2):
        a, b = items[i], items[i + 1]
        # 确定性伪随机决定谁进 A、谁进 B
        if int(sha256(a["pid"] + b["pid"] + SALT_AB)[:8], 16) % 2 == 1:
            a, b = b, a
        pairs.append({"A": a, "B": b, "stratum": stratum_id})
```

### 8.6 嵌套排序（中点公式修正）

> **v1.2 算法修正**：v1.1 使用左端点 `i / n`，导致全部层的第 0 对都取得 `0.0` 并挤在最前排，使 500 档（前 250 对）系统性高估小层比例。v1.2 改用**中点公式 `(i + 0.5) / n`**，使任意前缀 $k$ 对的层分布无偏逼近总体比例。

```python
SALT_ORDER = "dsai4207_order_v1.2"

by_stratum = group_by(pairs, key="stratum")
for stratum_id, ps in by_stratum.items():
    ps.sort(key=lambda p: sha256(p["A"]["pid"] + p["B"]["pid"] + SALT_ORDER))
    n = len(ps)
    for i, p in enumerate(ps):
        # 使用 (i + 0.5) / n 中点位置，消除 i=0 左端点扎堆偏差
        p["order_key"] = ((i + 0.5) / n, sha256(p["A"]["pid"] + p["B"]["pid"] + SALT_ORDER))

ordered_pairs = sorted(pairs, key=lambda p: p["order_key"])
```

### 8.7 预算抽取与交错写入

```python
for N in (500, 2000, 10000):
    k = N // 2
    if len(ordered_pairs) < k:
        raise RuntimeError(f"Not enough pairs for N={N}: got {len(ordered_pairs)}, need {k}")
    chosen_pairs = ordered_pairs[:k]
    A = [p["A"] for p in chosen_pairs]
    B = [p["B"] for p in chosen_pairs]
    # pool_N 按 (A_0, B_0, A_1, B_1, ...) 交错排列，禁止写成 A + B 拼接
    pool_N = [item for p in chosen_pairs for item in (p["A"], p["B"])]
    assert len(A) == len(B) == k and len(pool_N) == N
```

- $N=500 \rightarrow$ 前 250 对；
- $N=2000 \rightarrow$ 前 1,000 对；
- $N=10000 \rightarrow$ 前 5,000 对。
- 5,000 对之后的剩余配对与 `reserve_odd` 存入 `splits/reserve.jsonl`，不参与任何主实验。

### 8.8 Part A / Part B 输出

每个预算档输出至 `data/v1.0/splits/`：
- `pool_{N}.jsonl`：按配对交错排列的 $A_N \cup B_N$ 全部记录（供 G2/G3/G3′ 派生）。
- `partA_{N}.jsonl`：$A_N$ 记录（供 G4 Stage-1 / G5 Stage-2 派生）。
- `partB_{N}.jsonl`：$B_N$ 记录（供 G4 Stage-2 / G5 Stage-1 派生）。

### 8.9 平衡报告

每档必须输出至 `split_report.json` 并写入数据卡：
- $A_N$ 与 $B_N$ 的 `prompt_tokens`、`chosen_tokens`、`rejected_tokens` 均值/中位数/标准差；
- 标准化均值差（$\mathrm{SMD} = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{(s_A^2 + s_B^2)/2}}$）；要求所有连续变量 $|\mathrm{SMD}| < 0.10$（若 > 0.10 必须在报告中预警解释）；
- $A_N$ 与 $B_N$ 的策略家族比例、Top-6 子版块分布对比，以及 $P_{500}$、$P_{2000}$ 与 $P_{10000}$ 之间的分布最大绝对偏差（要求 < 3 个百分点）。

---

## 9. 镜像绑定与实验臂数据

### 9.1 镜像绑定定义

- SFT 训练目标文本 = 同一 canonical pair 的 `chosen` 字符串。
- DPO 训练偏好对 = 同一 canonical pair 的 `(chosen, rejected)` 字符串。
- 严禁针对不同实验臂做任何二次改写、重采样或重规范化。
- 所有实验臂共用同一个 `formatter_version = "v1.2"`。

### 9.2 实验臂矩阵

| 臂 | Stage 1 | Stage 2 | 数据来源 |
|---|---|---|---|
| **G2** | SFT: `pool_N` 全部 chosen | — | `pool_N`（$N$ 条） |
| **G3** | DPO: `pool_N` 全部 pair（ref = M0） | — | `pool_N`（$N$ 对） |
| **G4** | SFT: `partA_N` chosen | DPO: `partB_N` pair（ref = Stage-1 merged 输出） | `partA_N` ($0.5N$) $\rightarrow$ `partB_N` ($0.5N$) |
| **G5** | DPO: `partB_N` pair（ref = M0） | SFT: `partA_N` chosen | `partB_N` ($0.5N$) $\rightarrow$ `partA_N` ($0.5N$) |
| **G3′** | DPO: `pool_500` 全部 pair（ref = 裸 Base） | — | `pool_500`（500 对，仅 1 seed） |

### 9.3 臂级文件 Schema（兼容 TRL 对话格式 + 预渲染文本双保险）

> **v1.2 工程修复**：在 Hugging Face `trl` 中，若 `DPOTrainer` 的 `prompt` 为对话列表（`list[dict]`），则 `chosen` 和 `rejected` 也必须是 `[{"role": "assistant", "content": "..."}]` 对话列表格式，混用列表与裸字符串会导致报错或漏加 ChatML 模板。为此，v1.2 统一采用 **TRL 标准 Conversational 格式**，并在 `rendered` 字段中同时附上预渲染好的完整 ChatML 字符串，供训练脚本直接校验或加载。

**SFT 臂文件**（`G2_sft_{N}.jsonl`、`G4_stage1_sft_{N}.jsonl`、`G5_stage2_sft_{N}.jsonl`）：

```json
{
  "pid": "t3_34xale",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant who is an expert at summarizing content into TL;DR sentences. Summarize the following text"},
    {"role": "user", "content": "Subreddit: r/relationships\nTitle: …\n\n…"},
    {"role": "assistant", "content": "…"}
  ],
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant who is an expert at summarizing content into TL;DR sentences. Summarize the following text"},
    {"role": "user", "content": "Subreddit: r/relationships\nTitle: …\n\n…"}
  ],
  "completion": [
    {"role": "assistant", "content": "…"}
  ],
  "meta": {
    "chosen_policy": "sup4_6b",
    "chosen_policy_family": "sft_strong",
    "prompt_tokens": 428,
    "chosen_tokens": 38,
    "hash_pair": "…"
  }
}
```
*(注：同时提供 `messages` 与 `prompt` + `completion` 字段，使 `TRL SFTTrainer` 无论通过 `messages` 还是 `prompt-completion` 原生 completion-only loss 模式均可零转换直接读取。)*

**DPO 臂文件**（`G3_dpo_{N}.jsonl`、`G4_stage2_dpo_{N}.jsonl`、`G5_stage1_dpo_{N}.jsonl`、`G3prime_dpo_500.jsonl`）：

```json
{
  "pid": "t3_34xale",
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant who is an expert at summarizing content into TL;DR sentences. Summarize the following text"},
    {"role": "user", "content": "Subreddit: r/relationships\nTitle: …\n\n…"}
  ],
  "chosen": [
    {"role": "assistant", "content": "…"}
  ],
  "rejected": [
    {"role": "assistant", "content": "…"}
  ],
  "meta": {
    "chosen_policy": "sup4_6b",
    "rejected_policy": "sup2",
    "chosen_policy_family": "sft_strong",
    "prompt_tokens": 428,
    "chosen_tokens": 38,
    "rejected_tokens": 35,
    "hash_pair": "…"
  }
}
```

### 9.4 镜像绑定与跨方法 Token 级断言

在 `07_validate.py` 中强制执行以下两类断言：

1. **集合级镜像断言**：
   ```python
   assert [r["pid"] for r in G4_sft_N] == [r["pid"] for r in G5_sft_N]
   assert [r["meta"]["hash_pair"] for r in G4_sft_N] == [r["meta"]["hash_pair"] for r in G5_sft_N]
   assert [r["pid"] for r in G4_dpo_N] == [r["pid"] for r in G5_dpo_N]
   assert [r["meta"]["hash_pair"] for r in G4_dpo_N] == [r["meta"]["hash_pair"] for r in G5_dpo_N]
   ```
2. **跨阶段 & 跨训练器（SFT vs DPO）Token 序列一致性断言**：
   对任意 `pid`，将 SFT 样本的 `messages` 经 ChatML 渲染分词后的 `input_ids`，与同一 `pid` 在 DPO 样本中 `prompt + chosen` 渲染分词后的 `chosen_input_ids` 逐位比对，要求 **100% 逐 token 完全相同**。

---

## 10. 预热集 M0（`no_robots` 无摘要筛查）

### 10.1 来源

| 项 | 值 |
|---|---|
| 数据集 | `HuggingFaceH4/no_robots` |
| split | `train`（9,500 条；严禁混入 `test` split） |
| 许可 | Apache 2.0 |
| 字段 | `prompt`、`prompt_id`、`messages`、`category` |

### 10.2 三道全轮次筛查（v1.2 升级）

> **v1.2 修正**：v1.1 仅对 `prompt` 字段跑正则，会漏掉 `messages` 中的 `system` 提示词、多轮 `user` 追问以及 `assistant` 回复里的 `"TL;DR"`。v1.2 将筛查扩展至 `messages` 全轮次，并收窄易在代码/问答中误触发的裸词。

**第一道：Category 过滤**
- 删除 `category == "Summarize"`（约 395 条）。
- 删除 `messages` 中不存在非空 `assistant` 回复的异常样本。

**第二道：全轮次关键词正则过滤**
- 对 `messages` 中所有 `role in ("system", "user")` 的 `content`（以及顶层 `prompt`），使用以下不区分大小写正则匹配，命中即删：
  ```regex
  \b(summar(y|ies|ize|ise|ized|ised|izing|ising|ization|isation)|tl[;/]?dr|condense|recap|shorten|gist|synops[ie]s|key points|main points|in a nutshell|brief overview|abstract of)\b
  ```
- 对 `messages` 中所有 `role == "assistant"` 的 `content`，匹配以下摘要标志词，命中即删：
  ```regex
  \b(tl[;/]?dr|in summary|to summarize|here is a summary)\b
  ```
- 人工抽查 50 条被关键词剔除的样本记录于 `warmup_stats.json`，核验无系统性类别坍塌。

**第三道：跨集去污染与内部去重（MinHash + 精确 5-gram Jaccard ≥ 0.8）**
- 拼接每条 `no_robots` 样本中全部 `user` 与 `assistant` 轮次文本，经 `normalize_for_hash` 生成 5-word shingles。
- 按 §6.4 依次对 `canonical_pool`、`test_500_full`、`IFEval`、`MMLU`、`ARC-Challenge` 做去污染，命中即删 `warmup` 样本。
- 最后在剩余 `warmup` 样本内部做 exact + MinHash 0.8 去重。

### 10.3 分层抽样与格式

- 在通过三道筛查且总长 `total_tokens <= 2048`（超长直接丢弃，不截断）的干净池（预计约 8,500–8,800 条）中，按剩余 9 个 `category` 严格等比例分层抽样 **5,000 条**（固定排序键 `sha256(prompt_id + "dsai4207_warmup_v1.2")`）。
- 保留 `no_robots` 原生 `messages` 多轮对话结构，不注入 TL;DR 专用 system prompt。
- 输出：`warmup_5k.jsonl`、`warmup_indices.json`、`warmup_stats.json`。

---

## 11. Prompt 模板、Tokenization 与 EOS/PAD 规范

### 11.1 统一 TL;DR 模板（`formatter_version = "v1.2"`）

| 角色 | 内容 |
|---|---|
| `system` | `You are a helpful assistant who is an expert at summarizing content into TL;DR sentences. Summarize the following text` |
| `user` | `Subreddit: r/{subreddit}\nTitle: {title}\n\n{post}` |
| `assistant` | `{chosen}`（SFT）或 `{chosen}` / `{rejected}`（DPO） |

> **为什么默认包含 `Subreddit` 与 `Title`**：Reddit 贴文（尤其是占比过半的 `r/relationships` 和 `r/tifu`）大量核心背景写在标题中，正文常以 *"As the title says..."* 开头；若不给标题，模型在学习包含标题信息的 `chosen` 摘要时会被迫学习幻觉生成。同时，这也与 Stiennon et al. (2020) 及 UCL-DARK 数据集的标准构造保持一致。

### 11.2 ChatML 渲染与 EOS / PAD 显式绑定（关键防坑）

`Qwen/Qwen2.5-0.5B-Base` 的默认 `tokenizer_config.json` 中，`eos_token` 为 `<|endoftext|>`（ID `151643`），而 ChatML 对话结束符为 `<|im_end|>`（ID `151645`）。**所有数据脚本、训练脚本与评测脚本在加载 tokenizer 时必须执行以下统一初始化：**

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Base", revision=TOKENIZER_REVISION)
tokenizer.eos_token = "<|im_end|>"          # ID = 151645，生成与训练的真实停止符
tokenizer.pad_token = "<|endoftext|>"       # ID = 151643，严禁 pad_token == eos_token！
tokenizer.padding_side = "right"            # 训练阶段 right padding；评测生成阶段设为 "left"
```

> **为什么严禁 `pad_token == eos_token`**：若将 `pad_token` 设为与 `eos_token` 相同，DataCollator 在将 padding 位置的 label 设为 `-100` 时，会把回复末尾真正的 `eos_token` 也一起 mask 掉，导致模型永远学不会停止生成（详见 Huang et al., 2024 §3 Detail 3）。

统一使用如下 ChatML 模板渲染（若 `tokenizer.chat_template` 与标准 ChatML 存在尾部换行差异，统一冻结写入下述 Jinja 模板）：

```text
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Subreddit: r/{subreddit}
Title: {title}

{post}<|im_end|>
<|im_start|>assistant
{summary}<|im_end|>
```

评测生成时（`add_generation_prompt=True`），输入严格停在 `<|im_start|>assistant\n`，停止词为 `eos_token_id = 151645`（`<|im_end|>`）与 `151643`（`<|endoftext|>`）。

### 11.3 Loss Mask 规则

- **SFT（TL;DR 各臂与 M0 warmup）**：采用 **Completion-Only Loss**，仅对 `assistant` 响应正文及其结尾的 `<|im_end|>` 计算交叉熵 loss；`system` 段、`user` 段及 `<|im_start|>assistant\n` 提示头的 `labels` 全部设为 `-100`。
- **DPO**：采用 TRL 标准行为，`prompt` 段（含 `<|im_start|>assistant\n`）不计入序列对数概率求和，仅对 `chosen` / `rejected` 正文及结尾 `<|im_end|>` 计算 log-prob。
- 训练启动前，管线必须自动打印 1 条完整的 `(token_id, decoded_token, label)` 逐位对齐表并存入 `manifests/rendered_examples.json`。

### 11.4 最大长度汇总

| 数据集 | `max_length` | 细项上限与处理 |
|---|---:|---|
| TL;DR SFT | 1024 | `prompt_tokens ≤ 768`，`chosen_tokens ≤ 160`；超长在 §5.5 已丢弃 |
| TL;DR DPO | 1024 | `prompt_tokens ≤ 768`，`chosen ≤ 160`，`rejected ≤ 200`；超长在 §5.5 已丢弃 |
| Warmup SFT (`no_robots`) | 2048 | 完整多轮 tokens ≤ 2048；超长在 §10.3 直接丢弃 |

---

## 12. 评测数据接口

### 12.1 TL;DR 测试集文件

- **主评测集（必跑，对齐 Proposal）**：`data/v1.0/test_200_main.jsonl`（200 条，`order` 0–199）。
- **扩展全集（可选敏感性分析）**：`data/v1.0/test_500_full.jsonl`（500 条，`order` 0–499，前 200 条与 `test_200_main.jsonl` 逐字节一致）。
- 评测生成解码参数（Proposal 预注册）：Greedy decoding（`do_sample=False, temperature=0.0`），`max_new_tokens=128`，`eos_token_id=[151645, 151643]`。

### 12.2 MMLU / ARC-Challenge / IFEval 固定接口

| 基准 | 规模 | 抽样方式 | 评测方式 | 生成长度 / 打分口径 |
|---|---:|---|---|---|
| **MMLU** | 500 | 按 57 个 subject 等比例分层（每科 8–9 题），`seed=42` | Log-Likelihood | 4 选项（`A`/`B`/`C`/`D`）单 token 条件对数概率比较（同时记录生成式得分备查） |
| **ARC-Challenge** | 300 | 从 `test` split 随机无放回抽样，`seed=42` | Log-Likelihood | 选项完整文本长度归一化对数似然（`sum_logprob / num_choice_tokens`）与单字母 log-prob 双口径记录 |
| **IFEval** | 541（全量） | 官方 `google/IFEval` 全集，不抽样 | 官方 Rule Checker | Greedy decoding，**`max_new_tokens=1280`**（严禁误套用 TL;DR 的 128！） |

- 数据侧在 `data/v1.0/eval_benchmarks/` 下交付冻结后的 **`mmlu_500.jsonl`**、**`arc_challenge_300.jsonl`** 和 **`ifeval_541.jsonl`**（保留官方 `key`、`instruction_id_list`、`prompt`、`kwargs`）。
- 评测侧直接读取这三个固定 JSONL 文件，**严禁在评测脚本中重新在线随机抽样**，保证全部 29 个训练 checkpoint + M0 + 裸 Base 面对完全相同的考题与顺序。

### 12.3 裁判顺序与盲评样本

- 对 `test_200_main.jsonl` 的每条样本，模型输出与 M0 输出固定做 `(Model, M0)` 与 `(M0, Model)` 两次换序请求。
- Proposal 要求的“3 名组员盲评 100 对输出計算 Cohen's $\kappa$”由评测侧从主评测生成结果中按固定 seed 抽取，与本协议 §14.3 的“训练前 100 对数据集标签质量抽检”相互独立。

### 12.4 IFEval Harness 归属

- Proposal §5 Roles 将 `IFEval harness` 列在 Name A（PAN Haoren，数据工程）名下。
- 默认方案：数据工程侧一并提供 `ifeval_541.jsonl` 的 ChatML 渲染脚本与官方 `instruction_following_eval` 规则评测封装函数，评测侧只需传入模型生成的响应文件即可一键产出 Strict / Loose 准确率。

---

## 13. 产物、Manifest 与版本管理

### 13.1 标准目录结构

```text
data_engineering/
  DATA_PROTOCOL.md
  DATA_ISSUES.md
  scripts/
    00_download.py
    01_canonicalize.py
    02_decontaminate_and_dedup.py
    03_testset.py
    04_split.py
    05_warmup.py
    06_format_arms.py
    07_validate.py
  configs/
    data_v1.yaml
  tests/
    test_data_invariants.py
data/
  v1.0/
    canonical_pool.jsonl
    test_200_main.jsonl
    test_500_full.jsonl
    warmup_5k.jsonl
    eval_benchmarks/
      mmlu_500.jsonl
      arc_challenge_300.jsonl
      ifeval_541.jsonl
    splits/
      pool_500.jsonl
      pool_2000.jsonl
      pool_10000.jsonl
      partA_500.jsonl
      partB_500.jsonl
      partA_2000.jsonl
      partB_2000.jsonl
      partA_10000.jsonl
      partB_10000.jsonl
      reserve.jsonl
    arms/
      G2_sft_500.jsonl
      G3_dpo_500.jsonl
      G4_stage1_sft_500.jsonl
      G4_stage2_dpo_500.jsonl
      G5_stage1_dpo_500.jsonl
      G5_stage2_sft_500.jsonl
      G3prime_dpo_500.jsonl
      …（2000 / 10000 同理）
    manifests/
      manifest.json
      split_report.json
      decontamination_report.json
      warmup_stats.json
      rendered_examples.json
    DATA_CARD.md
```

### 13.2 Manifest Schema

```json
{
  "data_version": "v1.0",
  "protocol_version": "v1.2",
  "formatter_version": "v1.2",
  "created_at": "2026-10-18T…",
  "owner": "PAN Haoren",
  "sources": {
    "tldr": {
      "repo": "openai/summarize_from_feedback",
      "config": "comparisons",
      "splits": ["train", "validation"],
      "revision": "<commit_sha>"
    },
    "warmup": {
      "repo": "HuggingFaceH4/no_robots",
      "split": "train",
      "revision": "<commit_sha>"
    },
    "tokenizer": {
      "repo": "Qwen/Qwen2.5-0.5B-Base",
      "revision": "<commit_sha>",
      "eos_token": "<|im_end|>",
      "eos_token_id": 151645,
      "pad_token": "<|endoftext|>",
      "pad_token_id": 151643
    }
  },
  "salts": {
    "test": "dsai4207_test_v1.2_20261018",
    "test_order": "dsai4207_test_order_v1.2",
    "pair": "dsai4207_pair_v1.2",
    "ab": "dsai4207_ab_v1.2",
    "order": "dsai4207_order_v1.2",
    "warmup": "dsai4207_warmup_v1.2"
  },
  "training_seeds": [42, 1337, 2024],
  "filters": {
    "exclude_cnndm": true,
    "prompt_words": [30, 2000],
    "chosen_words": [5, 100],
    "rejected_words": [5, 120],
    "max_prompt_tokens": 768,
    "max_chosen_tokens": 160,
    "max_rejected_tokens": 200,
    "max_total_tokens": 1024,
    "max_warmup_tokens": 2048
  },
  "minhash": {
    "shingle_words": 5,
    "num_perm": 128,
    "lsh_recall_threshold": 0.75,
    "exact_jaccard_threshold": 0.8
  },
  "counts": {
    "raw_train_rows": 92858,
    "unique_train_prompts": 14767,
    "final_canonical_pool": 0,
    "test_main": 200,
    "test_full": 500,
    "warmup": 5000
  },
  "files": [
    {"path": "canonical_pool.jsonl", "sha256": "…", "n": 0}
  ],
  "env": {
    "python": "3.11",
    "datasets": "…",
    "transformers": "…",
    "trl": "…",
    "datasketch": "…",
    "torch": "…"
  }
}
```

---

## 14. 验收测试与交接门槛

### 14.1 自动不变量断言（`07_validate.py`）

```python
# I1 & 领域纯净性
assert len({r["pid"] for r in pool}) == len(pool)
assert all(r["subreddit"] and r["title"] and r["post"] for r in pool)
assert all(r["chosen"] and r["rejected"] and r["chosen"] != r["rejected"] for r in pool)

# I3 嵌套子集
assert len(P500) == 500 and len(P2000) == 2000 and len(P10000) == 10000
assert set(P500) < set(P2000) < set(P10000)
assert set(A[500]) < set(A[2000]) < set(A[10000])
assert set(B[500]) < set(B[2000]) < set(B[10000])

# I4 A/B 等长互斥
for N in (500, 2000, 10000):
    assert len(A[N]) == len(B[N]) == N // 2
    assert set(A[N]).isdisjoint(B[N])
    assert set(A[N]) | set(B[N]) == set(P[N])

# I5 / I6 / I7 镜像绑定与集合相等
for N in (500, 2000, 10000):
    assert [r["meta"]["hash_pair"] for r in G4_sft[N]] == [r["meta"]["hash_pair"] for r in G5_sft[N]]
    assert [r["meta"]["hash_pair"] for r in G4_dpo[N]] == [r["meta"]["hash_pair"] for r in G5_dpo[N]]
    assert multiset(r["meta"]["hash_pair"] for r in G2[N]) == multiset(r["meta"]["hash_pair"] for r in G3[N])

# I8 测试集隔离与双文件一致性
assert test_500[:200] == test_200
assert set(pids(test_500)).isdisjoint(pids(P10000))
assert verify_no_exact_5gram_jaccard_ge_08(test_500, P10000)

# I9 Warmup 全轮次无摘要
assert verify_warmup_all_turns_clean(warmup_5k)

# Token-level 跨阶段 & 跨训练器（SFT vs DPO）完全一致断言
for pid in P500:
    assert render_and_tokenize_sft(G4_sft[500][pid]) == render_and_tokenize_sft(G5_sft[500][pid])
    assert render_and_tokenize_dpo(G4_dpo[500][pid]) == render_and_tokenize_dpo(G5_dpo[500][pid])
    assert render_and_tokenize_sft(G2[500][pid]) == render_and_tokenize_dpo_chosen(G3[500][pid])
```

### 14.2 交接门槛 Checklist

训练负责人只有在以下条件全部满足后才能开跑：

- [ ] `07_validate.py` 全部断言通过（Exit Code 0）
- [ ] 连续两次完整运行管线，`manifest.json` 中所有数据文件 `sha256` 100% 一致
- [ ] `split_report.json` 中 500 / 2k / 10k 三档的 A/B 连续变量 $|\mathrm{SMD}| < 0.10$
- [ ] `manifests/rendered_examples.json` 中的 SFT / DPO / Eval 渲染文本与 `-100` label mask 经人工目检无误
- [ ] `eval_benchmarks/` 下的 MMLU-500、ARC-300、IFEval-541 与 `test_200_main.jsonl` 已生成并写入 manifest
- [ ] 组会确认协议升为 `v1.2-final`，数据产物冻结为 `data/v1.0`

---

## 15. 已确认决策表（D1–D9 全部按 v1.2 推荐默认值确认冻结）

| ID | 决策项 | 确认结论（v1.2 默认值） | 备选方案（未采用） | 状态 |
|---|---|---|---|---|
| **D1** | 数据源 | `openai/summarize_from_feedback` (`comparisons` train + 剔除 cnndm 的 validation) | `UCL-DARK/openai-tldr-summarisation-preferences` | **已确认** |
| **D2** | 一题一对与策略过滤 | **全策略纳入 + 同题不败（Condorcet）+ 置信度 + 策略档位优选**（纠正 v1.1 的 `sup1/sup2` 误判，训练池余量 > 4,400 条） | 仅保留 `chosen_policy == "ref"`（唯一 prompt 不足 10k） | **已确认** |
| **D3** | 测试集规模与来源 | `validation`（去 cnndm）抽 500 条，导出 **`test_200_main.jsonl`（主分析）** + `test_500_full.jsonl`（扩展敏感性） | 仅导出 200 条 | **已确认** |
| **D4** | Prompt 模板 | **包含 `Subreddit: r/{subreddit}\nTitle: {title}\n\n{post}`**（防无标题摘要幻觉） | 仅 `post` 正文 | **已确认** |
| **D5** | 最大长度 | TL;DR 1024 / Warmup 2048，超长直接丢弃 | 截断 | **已确认** |
| **D6** | Warmup 规模与筛查 | 扫 `messages` 全轮次去摘要后，按 category 分层抽 5,000 条 | 使用筛后全部约 8,600 条 | **已确认** |
| **D7** | IFEval Harness 归属 | 数据工程侧（PAN Haoren，对齐 Proposal Roles）提供固定题集与官方 checker 封装 | 评测侧独立实现 | **已确认** |
| **D8** | MMLU / ARC 子集 | 数据侧导出固定 `mmlu_500.jsonl`（分层）与 `arc_challenge_300.jsonl` | 跑全量 MMLU / ARC | **已确认** |
| **D9** | Part A/B 分层变量 | `prompt_tokens` (5) × `chosen_tokens` (3) × `chosen_policy_family` (3) + 中点排序 `(i+0.5)/n` | 仅按 `prompt_tokens` 分层 | **已确认** |

---

## 16. 变更记录

| 版本 | 日期 | 变更摘要 |
|---|---|---|
| v1.0-draft | 2026-10-18 | 初稿：数据源、canonicalization、去污染、测试集、配对嵌套、镜像绑定、warmup、模板、manifest、验收 |
| v1.1-draft | 2026-10-18 | 对齐最新 `main.pdf`：更新标题/团队/H2–H4；补充 200 模式、IFEval 归属、两类 100 对审计的区分、能力税固定索引 |
| v1.2-draft | 2026-10-18 | 全面审查修订：纠正 `sup1/sup2` 误判、引入同题不败 Condorcet 选对、剔除 `cnndm` 批次、纳入 `Subreddit/Title` 防幻觉、全轮次筛 `no_robots`、统一 TRL DPO 格式与 Qwen2.5 EOS/PAD、修正 `(i+0.5)/n` 中点排序、确立先去污染后去重及精确 Jaccard 核验 |
| **v1.2-final** | **2026-10-18** | **正式冻结：D1–D9 全部按 v1.2 推荐默认值确认生效，完成全套脚本实现与 `data/v1.0` 构建** |

---

## 附录 A：预调研数据漏斗（v1.2 修正口径）

```text
1. openai/summarize_from_feedback · comparisons/train
   92,858 rows
   → 14,767 unique Reddit prompts
   → 逐行有效性过滤（非空、非 deleted、choice∈{0,1}、词数与 Qwen token 长度合格）
   → 按 pid 聚合选唯一 canonical pair（同题零败绩 → confidence → 策略档位 → sha256）：≈ 14,550 unique prompts
   → 跨集去污染（vs test_500 / IFEval / MMLU / ARC）+ 池内 Exact & MinHash 0.8（精确 5-gram Jaccard）去重：
   → ≈ 14,450 final canonical pool（抽取 P10000 ⊂ P2000 ⊂ P500，余量 > 4,400 条）

2. openai/summarize_from_feedback · comparisons/validation
   86,086 rows
   → 剔除 batch 含 cnndm 的新闻行 + 有效性/token 长度过滤 + 同题选唯一 pair + 内部去重
   → ≈ 6,000+ unique Reddit prompts
   → 分层抽样 500 条（导出 test_500_full.jsonl，前 200 条导出为 test_200_main.jsonl）

3. HuggingFaceH4/no_robots · train
   9,500 rows
   → -395 (category == "Summarize")
   → -全轮次 messages 关键词正则命中 & >2048 tokens 超长样本
   → -跨集去污染（vs TL;DR pool / test_500 / IFEval / MMLU / ARC）& 内部去重
   → ≈ 8,600 干净通用指令池
   → 按剩余 9 个 category 等比例分层抽取 5,000 条（导出 warmup_5k.jsonl）
```

## 附录 B：文件 Schema 速查

| 文件 | 关键字段 |
|---|---|
| `canonical_pool.jsonl` | `pid, subreddit, title, post, user_content, chosen, rejected, chosen_policy, rejected_policy, chosen_policy_family, prompt_tokens, chosen_tokens, rejected_tokens, hash_prompt, hash_pair` |
| `test_200_main.jsonl` / `test_500_full.jsonl` | `pid, subreddit, title, post, user_content, messages, reference, order, is_main_200, hash_prompt` |
| `pool_N.jsonl` / `partA_N.jsonl` / `partB_N.jsonl` | 与 `canonical_pool.jsonl` 相同（`pool_N` 按 `A_i, B_i` 配对交错排列） |
| SFT 臂文件 (`G2/G4_s1/G5_s2`) | `pid, messages, prompt, completion, meta` |
| DPO 臂文件 (`G3/G4_s2/G5_s1/G3prime`) | `pid, prompt, chosen, rejected, meta`（均为 TRL 对话列表格式） |
| `warmup_5k.jsonl` | `prompt_id, messages, category` |

## 附录 C：实验臂 × 预算总表

| 臂 | 500 | 2,000 | 10,000 | 训练种子 |
|---|---|---|---|---|
| **G2** 纯 SFT | ✓ | ✓ | ✓ | 42 / 1337 / 2024（10k 仅 42） |
| **G3** 纯 DPO（ref = M0） | ✓ | ✓ | ✓ | 同上 |
| **G4** SFT → DPO（ref = Stage-1 输出） | ✓ | ✓ | ✓ | 同上 |
| **G5** DPO → SFT（DPO ref = M0） | ✓ | ✓ | ✓ | 同上 |
| **G3′** 裸 Base DPO（冷启动探针） | ✓ | — | — | 42 |
