# DSAI 4207 数据工程：可疑点与风险清单（v1.2 修订版）

| 项 | 内容 |
|---|---|
| 版本 | v1.2（对齐最新 proposal + 修正 v1.1 数据集事实误判与工程隐患） |
| 日期 | 2026-10-19 |
| 对应 proposal | `DSAI4207_Final_Proposal/main.pdf`（2026-10-13，4 页，最新版） |
| 版本口径 | `main.pdf` 与 `main.tex` 完全一致 |
| 团队 | PAN Haoren (24104216d)、LIU Shiyu (24106028d)、PAN Yubing (24103431d) |
| 负责人 | PAN Haoren（数据工程） |
| 用途 | 开跑前逐项拍板；拍板结果同步进 `DATA_PROTOCOL.md` |

**v1.2 重大修订摘要**

1. **【纠偏 P0-1 / P0-2】修正 `sup1/sup2` 策略含义的事实错误**：根据原论文 Stiennon et al. (2020, **Table 8**)，`comparisons` 数据集中**只有 `ref` 是 Reddit 原帖作者写的人类摘要**，`sup1`（750M）和 `sup2`（1.3B）全都是 OpenAI 早期微调的小参数 SFT 模型，`sup4_6b` 与 `sup4_ppo_*` 才是后期高质量模型。v1.1 误将 `{sup1, sup2, ref}` 当作“人类摘要”并强行过滤，不仅**反向删掉了数据集中质量最高的 `sup4`/`ppo` 摘要**，还人为制造了“10k 档余量只剩 76 条”的假警报。v1.2 恢复全策略纳入 + **同题不败（Condorcet）+ 置信度 + 策略档位优选**，训练池可用唯一 prompt 达 **>14,000 条**（10k 档余量充足）。
2. **【新增 P0-4】拦截 `comparisons/validation` 中的 CNN/DailyMail 新闻批次**：`validation` 含有 `batch0_cnndm`、`cnndm0`、`cnndm2` 新闻摘要批次（Huang et al., 2024），v1.2 增加硬过滤确保 100% 纯 Reddit TL;DR。
3. **【升级 P1-4 / P1-7】修复无标题摘要幻觉风险与 TRL/Qwen2.5 工程陷阱**：默认将 `Subreddit` 与 `Title` 纳入 User Prompt；统一官方 Qwen2.5 ChatML（`apply_chat_template`）与 TRL SFT/DPO 字段规范；强制绑定 Qwen2.5-Base 的 `eos_token="<|im_end|>"` 与 `pad_token="<|endoftext|>"`。
4. **【升级 P1-2 / P2-7】升级 `no_robots` 全轮次筛查、中点嵌套排序 `(i+0.5)/n`、DPO 正负策略双侧配对控制与 MinHash 精确 Jaccard 核验**。

**优先级定义**

- **P0**：不解决会导致实验结论无效、混入跨领域脏数据或误删核心高质量样本，必须开跑前拍板。
- **P1**：不解决会显著削弱结论、引发训练崩溃或造成返工，建议本周内定。
- **P2**：算法细节与局限性，已在协议中修复并需写进数据卡 / 报告 Limitations。
- **P3**：工程卫生与项目管理，按清单执行即可。

---

## 0. 一页总表

| ID | 优先级 | 问题 | 依据 | v1.2 建议默认值 | 拍板人 |
|---|---|---|---|---|---|
| **P0-1** | **P0** | v1.1 误把 `sup1/sup2` 当人类摘要，误删最强 `sup4/ppo` 样本并造成 10k 假性缺口 | Stiennon et al. (2020) **Table 8** | **取消 `{sup1,sup2,ref}` 误过滤；全策略纳入 + 同题不败优选** | 全组 |
| **P0-2** | **P0** | 一题多对（平均 6 对/题）若随机选或只看单行，可能选出在同题其他对比中落败的劣质摘要当 SFT 范文 | 数据集结构 + 实验设计 | **同题零败绩（Condorcet）→ 置信度 ≥3 → 策略质量档位 → 确定性 Hash** | 数据 + 全组 |
| **P0-3** | **P0** | Proposal 预注册 200 条测试题功效偏弱（CI ±7%），但直接改 500 条会偏离 Proposal 预注册口径 | 统计功效 + 预注册规范 | **从 validation 抽 500 条，同时导出 `test_200_main.jsonl`（主实验必跑）与 `test_500_full.jsonl`（扩展敏感性）** | 全组 + 评测 |
| **P0-4** | **P0** | `comparisons/validation` 中混有 CNN/DailyMail 新闻文章批次（`cnndm`） | Huang et al. (2024) §3 | **硬过滤 `"cnndm" not in batch.lower()` 且要求 `subreddit`/`title` 非空** | 数据 |
| **P1-1** | **P1** | 500 档两阶段的优化步数（~8 步/阶段），DPO 可能欠训 | 算术推导 | 主结果保持预注册 1 epoch；预备 500 档 3-epoch 敏感性变体 | 训练 |
| **P1-2** | **P1** | `no_robots` 只按 `category` 或只扫首轮 `prompt` 会漏掉多轮/system 中的摘要指令，且裸词 `abstract\|briefly` 误伤代码题 | 实测 + 规则审查 | **扫 `messages` 全轮次（system/user/assistant）+ 优化短语正则 + MinHash 0.8 去污染** | 数据 |
| **P1-3** | **P1** | 数据源与种子论文（UCL-DARK 独立 SFT/PFT 集）不同，为保镜像绑定必须同源派生 | 文献对比 | 采用 `openai/summarize_from_feedback` (`comparisons`) 同源派生；报告声明“复现定性规律而非逐字复刻数值” | 全组 |
| **P1-4** | **P1** | Qwen2.5-0.5B-Base 默认 `eos_token` 为 `<\|endoftext\|>`，且自实现模板若缺末尾 `\n` 会偏离官方 `apply_chat_template` | TRL & Qwen 机制 | **显式设 `eos="<\|im_end\|>"`、`pad="<\|endoftext\|>"`；与官方 `apply_chat_template` 逐字节对齐；SFT 固定用 `messages`，DPO 固定用 `prompt/chosen/rejected` 对话列表** | 数据 + 训练 |
| **P1-5** | **P1** | Proposal 称“500 条是模型第一次接触摘要任务”属于过度声明（预训练不可控） | 预训练常识 | 报告改写为“本后训练流水线内第一次接触（first task-specific contact within our post-training pipeline）” | 报告 |
| **P1-6** | **P1** | H4 / 能力税（MMLU 500 / ARC 300 / IFEval 541）未冻结固定题目与生成长度 | 实验控制 | 数据侧导出官方固定 JSONL 题集；IFEval 冻结 `max_new_tokens=1280` 并使用官方 `instruction_following_eval` | 数据 + 评测 |
| **P1-7** | **P1** | User Prompt 若丢弃 `title` 和 `subreddit`，摘要中含有的标题信息会迫使模型在 SFT 中学习幻觉 | Reddit 语篇结构 + Stiennon (2020) | **User Prompt 统一采用 `Subreddit: r/{subreddit}\nTitle: {title}\n\n{post}`** | 数据 + 全组 |
| **P2-1** | **P2** | 子版块偏斜：`r/relationships` 约占 56% | 实测 | 数据卡如实报告；测试集与训练嵌套排序均纳入子版块平衡（漂移 < 3%） | 数据 |
| **P2-2** | **P2** | chosen 与 rejected 存在天然长度差（chosen 中位词数略长） | 实测 | 数据卡报告长度差分布；评测端由 LCWR 长度协变量控制 | 数据 + 评测 |
| **P2-3** | **P2** | 嵌套排序若用 `i / n`，所有层的 `i=0` 均排在最前，导致 500 档高估极小层比例 | 算法审查 | **协议 §8.6 已改为中点公式 `(i + 0.5) / n`** | 数据 |
| **P2-4** | **P2** | MinHash LSH 近似碰撞直接进并查集会产生单链假阳性传递；先内去重后去污染有漏洞 | 算法审查 | **协议 §6 已改为“先跨集去污染、后池内去重”，且 LSH 召回后必须核验精确 5-gram Jaccard ≥ 0.8** | 数据 |
| **P2-5** | **P2** | Reddit 原始语料含潜在 PII / 情绪化内容，且基座模型存在预训练暴露可能性 | 伦理与有效性审查 | **写入数据卡与最终报告 Limitations（仅限非商业学术研究；界定为后训练管线内首次接触）** | 全组 |
| **P3-1** | **P3** | 数据集/分词器/库版本未固定会导致不可复现 | 工程规范 | `manifest.json` 锁死官方 HF commit SHA + parquet SHA256 + 管线双跑 SHA256 一致性校验 | 数据 |
| **P3-2** | **P3** | Proposal 将 `IFEval harness` 分工给 Name A（PAN Haoren） | 项目分工 | 默认由数据侧交付固定题集与官方 `instruction_following_eval` 封装脚本 | 全组 |
| **P3-3** | **P3** | 两类“100 对人工盲评”容易混淆，且审计表若含 key 会破坏盲评 | 项目管理 | 协议 §14.3 分离抽样，且将 `data_label_audit_100.jsonl` 与 `data_label_audit_100_key.json` 物理拆分 | 数据 + 评测 |

---

## 1. P0：开跑前必须确认的核心修正

### P0-1 & P0-2 纠正 `sup1/sup2` 策略误判，升级为“同题不败（Condorcet）”选对规则

**事实核查（Stiennon et al., NeurIPS 2020, Table 8）**

在 `openai/summarize_from_feedback` (`comparisons`) 中，`summaries[0/1].policy` 记录的是生成该候选摘要的模型或来源（见 Stiennon et al., 2020 **Table 8**, Model card / policy descriptions；注：Table 12 为示例对比任务）：
- **`ref`**：Reddit 原帖作者编写的**人类参考摘要**（唯一的人类撰写来源）。
- **`sup1`（750M）、`sup2`（1.3B）、`sup3`（1.3B）**：OpenAI 早期在不同批次上训练的小号 SFT 模型。
- **`sup3_6b`（6.7B）、`sup4`（1.3B）、`sup4_6b`（6.7B）**：OpenAI 在高质量过滤版 TL;DR 上微调的强 SFT 模型。
- **`*ppo*` / `*bo*`**：OpenAI 经强化学习（PPO）或 Best-of-N 奖励模型重排序后的最强对齐策略（人类偏好胜率最高，且远超人类原帖 `ref`）。

**v1.1 的问题与危害**

v1.1 误把 `sup1` 和 `sup2` 当作“人类摘要”，规定 `chosen_policy 必须 ∈ {sup1, sup2, ref}`：
1. **劣币驱逐良币**：把数据集中质量最高、人类最偏好的 `sup4_6b` 和 `sup4_ppo_*` 摘要全部剔除，反而强迫 G2/G4/G5 的 SFT 阶段去模仿早期 750M 弱模型（`sup1`）的输出。
2. **虚假的 10k 档数据荒**：因为误删了数千条 `sup4`/`ppo` 作为 `chosen` 的题目，v1.1 才算出了“只剩 10,076 / 12,032 条”的紧张余量。

**v1.2 解决方案（已写入 `DATA_PROTOCOL.md` §5.2–5.3）**

1. **恢复完整候选池**：凡人类标注员给出了明确偏好（`choice ∈ {0, 1}`）的 Reddit 样本均纳入候选，唯一 prompt 总数恢复至 **14,760 条**（官方 Qwen2.5 分词长度过滤与去重后超过 **14,200 条**，10k 档余量超过 **4,200 条**）。
2. **引入“同题不败（Condorcet）”一题一对规则**：同一 `pid` 有 4–8 条对比记录时，按以下顺序选出唯一 canonical pair：
   - **同题零败绩优先**：`chosen` 摘要从未在该 `pid` 的其他有效对比行中作为 `rejected` 落败过；
   - **高置信度优先**：优先选择人类标注员 `extra.confidence ≥ 3` 的行，并按 `confidence` 降序；
   - **高水平策略优先**：当存在平局时，优先选择 `rl_aligned` / `sft_strong` / `human_ref` 作为 `chosen`；
   - **确定性哈希兜底**：按 `sha256` 字典序取最小者。
3. **双侧策略家族与子版块分层平衡**：在 §8.4–8.6 切分 Part A / Part B 与嵌套抽样时，除控制 `chosen_policy_family` 外，在层内按 `rejected_policy_family` 与 `subreddit` 子桶排序配对，保证 500 / 2k / 10k 三档以及 A/B 两半的 `(chosen_family, rejected_family)` 3×3 联合分布与子版块比例严格一致（所有 $\chi^2$ 检验 $p > 0.64$，在 $\alpha=0.05$ 下均无显著差异；子版块漂移 $< 3\%$）。

---

### P0-3 测试集规模与来源：`test_200_main` + `test_500_full` 双轨制

- 从 `comparisons/validation`（剔除 `cnndm` 后）按长度五分位 × 子版块（35 层）分层抽取 **500 条**，并按层内中点分数 `(i + 0.5) / n` 排序后固定 `order = 0..499`。
- **同时物理导出两个标准文件**（移除易混淆的 `test_500.jsonl` 重复别名）：
  - `test_200_main.jsonl`（前 200 条，与 500 全集的 35 层比例最大偏差 $< 1.5\%$）：**100% 严格对应 Proposal 预注册的主实验与 API 预算**；
  - `test_500_full.jsonl`（完整 500 条）：训练池与预热集对全部 500 条做去污染。

---

### P0-4 剔除 `comparisons/validation` 中的 CNN/DailyMail 新闻批次

- 在候选行过滤第一条强制要求 `"cnndm" not in (batch or "").lower()` 且 `info.site is None`，并要求 `info.subreddit`、`info.title`、`info.post` 均非空字符串。

---

## 2. P1 & P2：实验有效性与局限性（Limitations）

### 2.1 实验有效性与工程衔接

- **P1-1（500 档步数与 DPO 负例策略控制）**：主实验严格遵守预注册 `1 epoch`，并预备 500 档 3-epoch 敏感性对照；同时在 `split_report.json` 中完整报告 Part A / Part B 的 `(chosen_family, rejected_family)` 3×3 与 4×4 联合矩阵及 $\chi^2$ 检验。
- **P1-2（`no_robots` 全轮次筛查 & 官方 IFEval Checker）**：全轮次扫描 `system`/`user`/`assistant` 剔除摘要相关样本；`ifeval_harness.py` 默认调用 Google Research 官方 `instruction_following_eval` 规则引擎并在报告中标注 `"checker": "official"`。
- **P1-4（官方 Qwen2.5 分词器与 ChatML 逐字节对齐）**：使用官方 `Qwen/Qwen2.5-0.5B-Base`（151,643 基础词表 + 22 特殊 token），渲染字符串与 `tokenizer.apply_chat_template` 100% 逐字节一致（含每轮末尾 `<|im_end|>\n`）；SFT 臂统一仅暴露 `messages` 训练字段，DPO 臂统一暴露 `prompt`/`chosen`/`rejected` 对话列表。
- **P1-5（100 对数据质量盲评）**：`manifests/data_label_audit_100.jsonl` 仅含盲评字段，隐藏答案与策略分布独立存放于 `manifests/data_label_audit_100_key.json`。

### 2.2 局限性与风险清单（Limitations — 写入数据卡与期末报告）

1. **Reddit PII 与潜在不当言论风险（PII & Toxicity）**：TL;DR 语料源自 Reddit 真实用户发帖（如 `r/relationships`、`r/tifu`），可能包含未完全脱敏的个人生活细节或口语化粗俗表达。本数据集仅用于课程内部学术研究，不做公开模型权重商业发布。
2. **基座模型预训练污染风险（Pretraining Contamination）**：`Qwen/Qwen2.5-0.5B-Base` 的预训练语料构成未公开，无法排除其在预训练阶段已见过 Reddit 或 TL;DR 文本。因此在报告中将“首次接触”严格限定为“本后训练管线内的首次特定任务接触（first task-specific contact within our post-training pipeline）”。
3. **正负摘要天然长度差与子版块偏斜**：`chosen` 词数中位数略长于 `rejected`，且 `r/relationships` 占比较高；评测端通过长度受控胜率（LCWR）控制长度偏置，并在分项分析中报告主流子版块与长尾子版块的一致性。

---

## 3. 拍板记录表（D1–D9 全组与专项负责人联合确认）

| ID | 决策项 | v1.2 推荐默认值 | 备选方案 | 结论 | 拍板人（联合签字） | 日期 |
|---|---|---|---|---|---|---|
| **D1** | 数据源 | `openai/summarize_from_feedback` (`comparisons` train + 去 cnndm validation) | UCL-DARK preferences | **采用 v1.2 默认值** | 全组确认：PAN Haoren, LIU Shiyu, PAN Yubing | 2026-10-19 |
| **D2** | 一题一对与策略过滤 | 全策略纳入 + 同题不败（Condorcet）+ 置信度 + 策略档位优选（Stiennon Table 8） | 仅 `ref`（数量不足 10k，不推荐） | **采用 v1.2 默认值** | 全组确认：PAN Haoren, LIU Shiyu, PAN Yubing | 2026-10-19 |
| **D3** | 测试集交付 | `validation` 抽 500，同时导出 `test_200_main.jsonl` + `test_500_full.jsonl` | 仅导出 200 条 | **采用 v1.2 默认值** | 数据 + 评测确认：PAN Haoren, PAN Yubing | 2026-10-19 |
| **D4** | Prompt 模板 | 含 `Subreddit` + `Title` + `post`（防幻觉，对齐 Stiennon 2020） | 仅 `post` 正文 | **采用 v1.2 默认值** | 全组确认：PAN Haoren, LIU Shiyu, PAN Yubing | 2026-10-19 |
| **D5** | 最大长度 | TL;DR 1024 / Warmup 2048，超长丢弃 | 截断 | **采用 v1.2 默认值** | 数据 + 训练确认：PAN Haoren, LIU Shiyu | 2026-10-19 |
| **D6** | Warmup 筛查与规模 | 扫 `messages` 全轮次去摘要 + 分层抽 5,000 条 | 筛后全量 ~8,600 条 | **采用 v1.2 默认值** | 数据 + 训练确认：PAN Haoren, LIU Shiyu | 2026-10-19 |
| **D7** | IFEval 归属 | 数据侧提供固定题集与官方 `instruction_following_eval` 封装 | 评测侧独立实现 | **采用 v1.2 默认值** | 数据 + 评测确认：PAN Haoren, PAN Yubing | 2026-10-19 |
| **D8** | MMLU / ARC 子集 | 数据侧导出固定官方 `mmlu_500.jsonl` 与 `arc_challenge_300.jsonl` | 全量评测 | **采用 v1.2 默认值** | 数据 + 评测确认：PAN Haoren, PAN Yubing | 2026-10-19 |
| **D9** | A/B 分层与嵌套排序 | `prompt_tokens` × `chosen_tokens` × `chosen_policy` + 层内控制 `rejected_policy` & `subreddit` + 中点排序 `(i+0.5)/n` | 仅按 `prompt_tokens` | **采用 v1.2 默认值** | 全组确认：PAN Haoren, LIU Shiyu, PAN Yubing | 2026-10-19 |
