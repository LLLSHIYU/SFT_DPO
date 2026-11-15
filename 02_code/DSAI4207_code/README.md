# Order vs. Scale: A Controlled Study of SFT/DPO Sequencing Across Data Budgets in Small LM Alignment

**Course:** DSAI 4207 Introduction to Large Language Models (The Hong Kong Polytechnic University)  
**Team Members:**  
- **PAN Haoren (`24104216d`)** — Data Engineering, Decontamination/Splitting Pipeline & IFEval Harness  
- **LIU Shiyu (`24106028d`)** — Model Training (SFT / DPO / Two-Stage Sequencing Arms G2–G5 & G3′)  
- **PAN Yubing (`24103431d`)** — Evaluation Pipeline (LCWR Judge Harness, Capability Tax MMLU/ARC & Statistical Analysis)  

---

## 1. Project Overview & Experimental Design

This repository contains the data engineering protocol, deterministic data construction pipeline (`data/v1.0`), evaluation harnesses, and reproducibility artifacts for our study investigating **how training stage ordering (SFT $\rightarrow$ DPO vs. DPO $\rightarrow$ SFT) interacts with total unique-prompt budget ($N \in \{500, 2000, 10000\}$)** when aligning a small language model (`Qwen/Qwen2.5-0.5B-Base`).

### Core Experimental Controls Enforced by `data/v1.0`
1. **Condorcet One-Pair-Per-Prompt Canonicalization:** Every Reddit TL;DR post (`pid`) is mapped to a single canonical `(chosen, rejected)` preference pair prioritizing zero-loss Condorcet winners, rater confidence $\ge 3$, and policy quality tier (Stiennon et al., 2020, Table 8).
2. **Strict Nested Subsets & Midpoint Ordering:** $P_{500} \subset P_{2000} \subset P_{10000}$ and $A_{500} \subset A_{2000} \subset A_{10000}$, $B_{500} \subset B_{2000} \subset B_{10000}$ using midpoint spacing $(i + 0.5)/n$ across strata (`prompt_tokens` $\times$ `chosen_tokens` $\times$ `chosen_policy_family` $\times$ `rejected_policy_family` $\times$ `subreddit`).
3. **Token-Level Mirror Binding:** Across every budget $N$, `G4_stage1_sft_{N}` and `G5_stage2_sft_{N}` share the exact same `partA_{N}` `chosen` targets, while `G4_stage2_dpo_{N}` and `G5_stage1_dpo_{N}` share the exact same `partB_{N}` `(chosen, rejected)` pairs—verified byte-for-byte and token-for-token under the official `Qwen/Qwen2.5-0.5B-Base` ChatML template (`apply_chat_template`).
4. **Outer-Before-Inner Decontamination:** Both the TL;DR training pool and the 5,000-sample `no_robots` general warmup set (`M0`) are decontaminated against `test_500_full.jsonl` and the three official capability benchmarks (`MMLU-500`, `ARC-Challenge-300`, `IFEval-541`) using 128-permutation 5-word shingle MinHash LSH followed by exact Jaccard $\ge 0.8$ verification.

---

## 2. Repository Structure

```text
.
├── README.md                          # Setup, data pipeline, training/eval entrypoints & verification guide
├── requirements.txt                   # Locked Python package versions
├── .gitignore                         # Excludes large raw parquets / JSONL datasets from git tracking
├── DSAI4207_Final_Proposal/           # Course project proposal (main.tex & main.pdf)
├── data_engineering/
│   ├── DATA_PROTOCOL.md               # Frozen data engineering specification (v1.2-final)
│   ├── DATA_ISSUES.md                 # Risk register, Stiennon Table 8 verification & D1–D9 team sign-offs
│   ├── DATA_FIX_PLAN.md               # Audit & remediation checklist (all P0/P1/P2 items resolved)
│   ├── configs/
│   │   └── data_v1.yaml               # Frozen salts, length thresholds, and source configurations
│   ├── scripts/
│   │   ├── 00_download.py             # Downloads & verifies official Qwen2.5 tokenizer & HF benchmarks
│   │   ├── 01_canonicalize.py         # Condorcet pair selection & official Qwen2.5 token filtering
│   │   ├── 02_decontaminate_and_dedup.py # Cross-set decontamination & Union-Find MinHash 0.8 dedup
│   │   ├── 03_testset.py              # Builds test_200_main, test_500_full & MMLU/ARC/IFEval subsets
│   │   ├── 04_split.py                # Paired stratified A/B splitting, SMD & chi-square balance checks
│   │   ├── 05_warmup.py               # 3-stage full-turn summarization removal & 5k M0 warmup sampling
│   │   ├── 06_format_arms.py          # Formats 19 canonical arm files & blind 100-pair audit sheet
│   │   ├── 07_validate.py             # Checks invariants I1–I10, builds manifest.json & DATA_CARD.md
│   │   ├── ifeval_harness.py          # Official Google Research instruction_following_eval wrapper
│   │   ├── instruction_following_eval/# Official Google Research IFEval rule-checker modules
│   │   ├── run_all.py                 # End-to-end deterministic pipeline runner (00 -> 07)
│   │   └── utils.py                   # Official QwenTokenizerWrapper, ChatML renderer & MinHash index
│   └── tests/
│       └── test_data_invariants.py    # Unit, authenticity & TRL SFT/DPO single-step smoke tests
├── data/
│   └── v1.0/                          # Generated dataset deliverables, manifests & DATA_CARD.md
└── results/                           # Exported data funnel tables, split SMD/chi2 reports & evaluation outputs
```

---

## 3. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 4. Data Acquisition & Reproducing `data/v1.0`

### 4.1 Official Upstream Sources Locked in `data/v1.0`
| Role | Hugging Face Repository | Config / Split | Official Commit Revision |
|---|---|---|---|
| **Primary Task (TL;DR)** | `openai/summarize_from_feedback` | `comparisons` (`train`, `validation`) | `b8f7d168b6f4e95b2a92e84768bd6c955bed2f29` |
| **M0 Warmup SFT** | `HuggingFaceH4/no_robots` | `default` (`train`, 9,500 rows) | `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` |
| **Base Tokenizer & Model** | `Qwen/Qwen2.5-0.5B-Base` (`hub_repo`: `Qwen/Qwen2.5-0.5B`) | Official 151,643 BPE + 22 special tokens (`len=151,665`) | `060db6499f32faf8b98477b0a26969ef7d8b9987` (`revision_for_repo = Qwen/Qwen2.5-0.5B`) |
| **Capability Tax — MMLU** | `cais/mmlu` | `all` / `test` (14,042 rows, 57 subjects $\rightarrow$ 500) | `c30699e8356da336a370243923dbaf21066bb9fe` |
| **Capability Tax — ARC** | `allenai/ai2_arc` | `ARC-Challenge` / `test` (1,172 rows $\rightarrow$ 300) | `210d026faf9955653af8916fad021475a3f00453` |
| **Instruction Following (H4)** | `google/IFEval` | `default` / `train` (541 rows, full set) | `966cd89545d6b6acfd7638bc708b98261ca58e84` |

### 4.2 Running the Full Data Pipeline (Steps `00` $\rightarrow$ `07`)
```bash
python data_engineering/scripts/run_all.py
```
> **Note on Offline Mode:** `00_download.py` strictly forbids offline synthetic fallback for `data/v1.0`. If executed without network access and without pre-cached official files, it exits with a non-zero error. The `--allow-synthetic` flag is reserved exclusively for isolated CI tests under `data/smoke/`.

### 4.3 Running Verification & Authenticity / TRL Smoke Tests

> **IMPORTANT — install the full requirements before verification.** `07_validate.py` reads `trl` / `torch` / `transformers` / `accelerate` package versions when regenerating `manifest.json`, and the test suite runs 1-step TRL `SFTTrainer` / `DPOTrainer`. A data-only environment may fail with `PackageNotFoundError: No package metadata was found for trl`. Run `pip install -r requirements.txt` first; do not use a trimmed data-only dependency set for this step.

```bash
# 1. Verify all 10 protocol invariants (I1–I10) and regenerate manifest.json / DATA_CARD.md
python data_engineering/scripts/07_validate.py

# 2. Run the full unit & integration test suite (including official tokenizer, benchmarks, and 1-step TRL SFT/DPO smoke tests)
python -m unittest discover -s data_engineering/tests -v
```

---

## 5. Canonical File Contract for Training & Evaluation

To prevent ambiguity, duplicate short-alias files have been removed. Training and evaluation scripts **must** use the canonical stage-explicit file names below:

### 5.1 Training Files (`data/v1.0/arms/`)
- **M0 Shared Warmup:** `data/v1.0/warmup_5k.jsonl` (field: `messages`, train with `assistant_only_loss=True`, `max_length=2048`)
- **Single-Stage SFT (`G2`):** `data/v1.0/arms/G2_sft_{500,2000,10000}.jsonl` (field: `messages`, train with `assistant_only_loss=True`, `max_length=1024`)
- **Single-Stage DPO (`G3`):** `data/v1.0/arms/G3_dpo_{500,2000,10000}.jsonl` (fields: `prompt`, `chosen`, `rejected` conversational lists, `max_length=1024`)
- **Two-Stage SFT $\rightarrow$ DPO (`G4`):**
  - Stage 1 (SFT on `partA_N`): `data/v1.0/arms/G4_stage1_sft_{500,2000,10000}.jsonl`
  - Stage 2 (DPO on `partB_N`): `data/v1.0/arms/G4_stage2_dpo_{500,2000,10000}.jsonl`
- **Two-Stage DPO $\rightarrow$ SFT (`G5`):**
  - Stage 1 (DPO on `partB_N`): `data/v1.0/arms/G5_stage1_dpo_{500,2000,10000}.jsonl`
  - Stage 2 (SFT on `partA_N`): `data/v1.0/arms/G5_stage2_sft_{500,2000,10000}.jsonl`
- **Cold-Start DPO Probe (`G3′` on raw Base at $N=500$):** `data/v1.0/arms/G3prime_dpo_500.jsonl`

**Mandatory Tokenizer Setup in Training & Evaluation Scripts:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=True)
tokenizer.eos_token = "<|im_end|>"      # ID = 151645
tokenizer.pad_token = "<|endoftext|>"   # ID = 151643 (NEVER set pad_token == eos_token)
tokenizer.padding_side = "right"        # Use "left" during batched generation
```

### 5.2 Evaluation Files (`data/v1.0/`)
- **Primary TL;DR Held-Out Test Set (200 prompts, pre-registered):** `data/v1.0/test_200_main.jsonl` (`max_new_tokens=128`, greedy decoding)
- **Extended TL;DR Sensitivity Test Set (500 prompts):** `data/v1.0/test_500_full.jsonl`
- **Fixed Benchmark Subsets:**
  - `data/v1.0/eval_benchmarks/mmlu_500.jsonl` (500 items stratified across 57 subjects)
  - `data/v1.0/eval_benchmarks/arc_challenge_300.jsonl` (300 items)
  - `data/v1.0/eval_benchmarks/ifeval_541.jsonl` & `ifeval_541_rendered.jsonl` (541 items, `max_new_tokens=1280`)
- **Blind 100-Pair Data Audit (`data/v1.0/manifests/`):**
  - `data_label_audit_100.jsonl`: Strictly blind 7-field annotation sheet (`audit_id`, `pid`, `subreddit`, `title`, `post`, `summary_A`, `summary_B`) distributed to annotators.
  - `data_label_audit_100_key.json`: Hidden answer & policy key (excluded from Git via `.gitignore` and distributed solely to the statistical lead after blind annotation is complete).

### 5.3 Running the Official IFEval Harness
```bash
# Render ChatML prompts for IFEval-541
python data_engineering/scripts/ifeval_harness.py --mode render

# Score model generations using the official Google Research instruction_following_eval checker
python data_engineering/scripts/ifeval_harness.py \
  --mode score \
  --responses results/model_ifeval_responses.jsonl \
  --report results/ifeval_scores.json
```
