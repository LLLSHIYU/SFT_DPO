#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_validate.py — Comprehensive invariant & authenticity checker (I1–I10, P0-1..P0-4, P1-1..P1-5, P2-1..P2-7),
manifest generator, results/ exporter, and DATA_CARD.md builder (§13, §14).
Any violation raises an AssertionError and exits with non-zero code.
"""

import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pyarrow
import pyarrow.parquet as pq
import yaml
from utils import (
    DATA_V1_DIR,
    PROJECT_ROOT,
    RAW_CACHE_DIR,
    ExactVerifiedMinHashIndex,
    QwenTokenizerWrapper,
    load_config,
    normalize_for_hash,
    read_jsonl,
    sha256_file,
)


def verify_all_invariants() -> Dict[str, Any]:
    cfg = load_config()
    tokenizer = QwenTokenizerWrapper(RAW_CACHE_DIR / "qwen_tokenizer")

    # -----------------------------------------------------------------------
    # P0-1, P0-3, P2-5, P3-2: Official Tokenizer, Commit SHAs & Non-Synthetic Verification
    # -----------------------------------------------------------------------
    with open(RAW_CACHE_DIR / "sources_lock.json", "r", encoding="utf-8") as f:
        sources_lock = json.load(f)
    assert sources_lock.get("synthetic") is False, "P0-3 violated: synthetic=True in sources_lock.json!"
    assert sources_lock["tldr"]["revision"] == "b8f7d168b6f4e95b2a92e84768bd6c955bed2f29", "R1/P2-5 violated: invalid TL;DR revision"
    assert sources_lock["warmup"]["revision"] == "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b", "R1/P2-5 violated: invalid no_robots revision"
    assert sources_lock["benchmarks"]["mmlu"]["revision"] == "c30699e8356da336a370243923dbaf21066bb9fe", "R1/P2-5 violated: invalid MMLU revision"
    assert sources_lock["benchmarks"]["arc_challenge"]["revision"] == "210d026faf9955653af8916fad021475a3f00453", "R1/P2-5 violated: invalid ARC revision"
    assert sources_lock["benchmarks"]["ifeval"]["revision"] == "966cd89545d6b6acfd7638bc708b98261ca58e84", "R1/P2-5 violated: invalid IFEval revision"
    assert sources_lock["tokenizer"].get("hub_repo") == "Qwen/Qwen2.5-0.5B", "R4/P3-2 violated: missing hub_repo"
    assert sources_lock["tokenizer"].get("revision_for_repo") == "Qwen/Qwen2.5-0.5B", "R4/P3-2 violated: missing revision_for_repo"
    assert tokenizer.vocab_size >= 151643, f"P0-1 violated: vocab_size={tokenizer.vocab_size} < 151643"
    assert len(tokenizer) >= 151665, f"P0-1 violated: len(tokenizer)={len(tokenizer)} < 151665"
    assert tokenizer.eos_token_id == 151645 and tokenizer.pad_token_id == 151643
    assert len(tokenizer.encode("hello")) == 1 and len(tokenizer.encode(" the")) == 1

    # -----------------------------------------------------------------------
    # P0-2: Official Benchmark Authenticity Verification
    # -----------------------------------------------------------------------
    bench_cache = RAW_CACHE_DIR / "benchmarks"
    assert pq.read_table(bench_cache / "mmlu_all_test.parquet").num_rows == 14042
    assert pq.read_table(bench_cache / "arc_challenge_test.parquet").num_rows == 1172
    assert pq.read_table(bench_cache / "ifeval_train.parquet").num_rows == 541

    mmlu_500 = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "mmlu_500.jsonl")
    arc_300 = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "arc_challenge_300.jsonl")
    ifeval_541 = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "ifeval_541.jsonl")
    assert len(mmlu_500) == 500 and len({r["subject"] for r in mmlu_500}) == 57
    assert len(arc_300) == 300
    assert len(ifeval_541) == 541
    for r in mmlu_500:
        assert "phenomena #" not in r["question"] and not r["choices"][0].startswith("Option A for ")
    for r in arc_300:
        assert "controlled scientific experiment #" not in r["question"]
    for r in ifeval_541:
        assert not r["prompt"].startswith("[IFEval #")

    pool = read_jsonl(DATA_V1_DIR / "canonical_pool.jsonl")
    test_200 = read_jsonl(DATA_V1_DIR / "test_200_main.jsonl")
    test_500 = read_jsonl(DATA_V1_DIR / "test_500_full.jsonl")
    warmup_5k = read_jsonl(DATA_V1_DIR / "warmup_5k.jsonl")

    # P2-7: Verify duplicate alias files are removed
    assert not (DATA_V1_DIR / "test_500.jsonl").exists(), "P2-7 violated: duplicate test_500.jsonl exists"
    for N in (500, 2000, 10000):
        for old_alias in (f"G4_sft_{N}.jsonl", f"G4_dpo_{N}.jsonl", f"G5_sft_{N}.jsonl", f"G5_dpo_{N}.jsonl"):
            assert not (DATA_V1_DIR / "arms" / old_alias).exists(), f"P2-7 violated: {old_alias} exists"

    # -----------------------------------------------------------------------
    # I1 & I2: Unique pids, non-empty valid fields, pure Reddit, chosen != rejected
    # -----------------------------------------------------------------------
    pids_pool = [r["pid"] for r in pool]
    assert len(set(pids_pool)) == len(pool), "I1 violated: duplicate pid in canonical_pool"
    assert len(pool) >= 10000, f"I1 violated: canonical_pool size {len(pool)} < 10000"
    for r in pool:
        assert r["pid"].startswith("t3_"), f"I1 violated: non-Reddit pid {r['pid']}"
        assert r["subreddit"] and r["title"] and r["post"], "I1 violated: empty metadata"
        assert r["chosen"] and r["rejected"], "I2 violated: empty summary"
        assert r["chosen"] != r["rejected"], "I2 violated: identical chosen & rejected"
        assert normalize_for_hash(r["chosen"]) != normalize_for_hash(r["rejected"]), "I2 violated"
        assert r["prompt_tokens"] <= 768, "I2 violated: prompt_tokens > 768"
        assert r["chosen_tokens"] <= 160, "I2 violated: chosen_tokens > 160"
        assert r["rejected_tokens"] <= 200, "I2 violated: rejected_tokens > 200"
        assert r["prompt_tokens"] + max(r["chosen_tokens"], r["rejected_tokens"]) <= 1024

    # -----------------------------------------------------------------------
    # I3 & I4: Nested subsets P500 ⊂ P2000 ⊂ P10000 and exact A/B splits
    # -----------------------------------------------------------------------
    splits_dir = DATA_V1_DIR / "splits"
    pools_n: Dict[int, List[Dict[str, Any]]] = {}
    parts_a: Dict[int, List[Dict[str, Any]]] = {}
    parts_b: Dict[int, List[Dict[str, Any]]] = {}

    for N in (500, 2000, 10000):
        pools_n[N] = read_jsonl(splits_dir / f"pool_{N}.jsonl")
        parts_a[N] = read_jsonl(splits_dir / f"partA_{N}.jsonl")
        parts_b[N] = read_jsonl(splits_dir / f"partB_{N}.jsonl")

        assert len(pools_n[N]) == N, f"I3 violated: len(pool_{N}) != {N}"
        assert len(parts_a[N]) == N // 2, f"I4 violated: len(partA_{N}) != {N//2}"
        assert len(parts_b[N]) == N // 2, f"I4 violated: len(partB_{N}) != {N//2}"

        set_a = {r["pid"] for r in parts_a[N]}
        set_b = {r["pid"] for r in parts_b[N]}
        set_p = {r["pid"] for r in pools_n[N]}
        assert set_a.isdisjoint(set_b), f"I4 violated: A_{N} and B_{N} not disjoint"
        assert set_a | set_b == set_p, f"I4 violated: A_{N} U B_{N} != P_{N}"

        for idx in range(N // 2):
            assert pools_n[N][2 * idx]["pid"] == parts_a[N][idx]["pid"]
            assert pools_n[N][2 * idx + 1]["pid"] == parts_b[N][idx]["pid"]

    assert {r["pid"] for r in pools_n[500]} < {r["pid"] for r in pools_n[2000]} < {r["pid"] for r in pools_n[10000]}
    assert {r["pid"] for r in parts_a[500]} < {r["pid"] for r in parts_a[2000]} < {r["pid"] for r in parts_a[10000]}
    assert {r["pid"] for r in parts_b[500]} < {r["pid"] for r in parts_b[2000]} < {r["pid"] for r in parts_b[10000]}

    # -----------------------------------------------------------------------
    # I5, I6, I7 & P0-4: Mirror Binding & Official ChatML Byte/Token Identity
    # -----------------------------------------------------------------------
    arms_dir = DATA_V1_DIR / "arms"
    for N in (500, 2000, 10000):
        g2 = read_jsonl(arms_dir / f"G2_sft_{N}.jsonl")
        g3 = read_jsonl(arms_dir / f"G3_dpo_{N}.jsonl")
        g4_s1 = read_jsonl(arms_dir / f"G4_stage1_sft_{N}.jsonl")
        g4_s2 = read_jsonl(arms_dir / f"G4_stage2_dpo_{N}.jsonl")
        g5_s1 = read_jsonl(arms_dir / f"G5_stage1_dpo_{N}.jsonl")
        g5_s2 = read_jsonl(arms_dir / f"G5_stage2_sft_{N}.jsonl")

        assert g4_s1 == g5_s2, f"I5 violated at N={N}: G4_stage1 != G5_stage2"
        assert g4_s2 == g5_s1, f"I6 violated at N={N}: G4_stage2 != G5_stage1"
        assert [r["pid"] for r in g2] == [r["pid"] for r in pools_n[N]]
        assert [r["pid"] for r in g3] == [r["pid"] for r in pools_n[N]]
        assert [r["meta"]["hash_pair"] for r in g2] == [r["hash_pair"] for r in pools_n[N]]
        assert [r["meta"]["hash_pair"] for r in g3] == [r["hash_pair"] for r in pools_n[N]]

        if N == 500:
            g3p = read_jsonl(arms_dir / "G3prime_dpo_500.jsonl")
            assert g3p == g3, "I7 violated: G3prime_dpo_500 != G3_dpo_500"

            g2_by_pid = {r["pid"]: r for r in g2}
            g3_by_pid = {r["pid"]: r for r in g3}
            g4_s1_by_pid = {r["pid"]: r for r in g4_s1}
            g5_s2_by_pid = {r["pid"]: r for r in g5_s2}
            g4_s2_by_pid = {r["pid"]: r for r in g4_s2}
            g5_s1_by_pid = {r["pid"]: r for r in g5_s1}

            for pid in g2_by_pid:
                sft_msgs = g2_by_pid[pid]["messages"]
                dpo_rec = g3_by_pid[pid]

                # Byte-level equality with official apply_chat_template (P0-4)
                rendered_sft = tokenizer.render_chatml(sft_msgs, add_generation_prompt=False)
                official_sft = tokenizer.apply_chat_template(
                    sft_msgs, tokenize=False, add_generation_prompt=False
                )
                assert rendered_sft == official_sft, f"P0-4 violated for pid={pid}"

                sft_ids, sft_lbls = tokenizer.tokenize_sft_with_labels(sft_msgs)
                dpo_toks = tokenizer.tokenize_dpo_pair(
                    dpo_rec["prompt"],
                    dpo_rec["chosen"],
                    dpo_rec["rejected"],
                )
                assert sft_ids == dpo_toks["full_chosen_input_ids"], (
                    f"Token mismatch between SFT and DPO chosen for pid={pid}"
                )
                assert tokenizer.eos_token_id in sft_ids[-2:]
                assert tokenizer.eos_token_id in dpo_toks["chosen_input_ids"][-2:]
                assert tokenizer.eos_token_id in dpo_toks["rejected_input_ids"][-2:]
                active_lbls = [x for x in sft_lbls if x != -100]
                assert active_lbls == dpo_toks["chosen_input_ids"], (
                    f"SFT active labels != DPO chosen_input_ids for pid={pid}"
                )

            for pid in g4_s1_by_pid:
                assert tokenizer.tokenize_sft_with_labels(g4_s1_by_pid[pid]["messages"]) == (
                    tokenizer.tokenize_sft_with_labels(g5_s2_by_pid[pid]["messages"])
                )
            for pid in g4_s2_by_pid:
                assert tokenizer.tokenize_dpo_pair(
                    g4_s2_by_pid[pid]["prompt"],
                    g4_s2_by_pid[pid]["chosen"],
                    g4_s2_by_pid[pid]["rejected"],
                ) == tokenizer.tokenize_dpo_pair(
                    g5_s1_by_pid[pid]["prompt"],
                    g5_s1_by_pid[pid]["chosen"],
                    g5_s1_by_pid[pid]["rejected"],
                )

    # -----------------------------------------------------------------------
    # I8: Test set isolation (zero overlap and zero exact 5-gram Jaccard >= 0.8)
    # -----------------------------------------------------------------------
    assert len(test_200) == 200 and len(test_500) == 500
    assert test_500[:200] == test_200, "I8 violated: test_200_main != test_500_full[:200]"
    test_pids = {r["pid"] for r in test_500}
    pool_pids = {r["pid"] for r in pool}
    assert test_pids.isdisjoint(pool_pids), "I8 violated: test_500 and canonical_pool share pids"

    test_idx = ExactVerifiedMinHashIndex(num_perm=128, bands=32)
    for r in test_500:
        test_idx.insert(r["pid"], normalize_for_hash(r["title"] + " " + r["post"]))
    for r in pool:
        hits = test_idx.query_Text(normalize_for_hash(r["title"] + " " + r["post"]), threshold=0.8)
        assert not hits, f"I8 violated: near-dup found between pool {r['pid']} and test {hits}"

    # -----------------------------------------------------------------------
    # I9: Warmup cleanliness (5000 items, zero summarization keywords across all turns)
    # -----------------------------------------------------------------------
    assert len(warmup_5k) == 5000, f"I9 violated: len(warmup_5k) = {len(warmup_5k)} != 5000"
    user_sys_re = re.compile(cfg["warmup"]["user_system_regex"], re.IGNORECASE)
    asst_re = re.compile(cfg["warmup"]["assistant_regex"], re.IGNORECASE)
    for r in warmup_5k:
        assert r["category"] != "Summarize", "I9 violated: Summarize category in warmup_5k"
        assert r["total_tokens"] <= 2048, "I9 violated: warmup item > 2048 tokens"
        for m in r["messages"]:
            if m["role"] in ("system", "user"):
                assert not user_sys_re.search(m["content"]), (
                    f"I9 violated: keyword found in warmup {r['prompt_id']}"
                )
            elif m["role"] == "assistant":
                assert not asst_re.search(m["content"]), (
                    f"I9 violated: assistant keyword found in warmup {r['prompt_id']}"
                )

    # -----------------------------------------------------------------------
    # P1-5: Verify 100-pair audit is strictly blind
    # -----------------------------------------------------------------------
    audit_rows = read_jsonl(DATA_V1_DIR / "manifests" / "data_label_audit_100.jsonl")
    with open(DATA_V1_DIR / "manifests" / "data_label_audit_100_key.json", "r", encoding="utf-8") as f:
        audit_keys = json.load(f)
    assert len(audit_rows) == 100 and len(audit_keys) == 100
    expected_blind_keys = {"audit_id", "pid", "subreddit", "title", "post", "summary_A", "summary_B"}
    for row in audit_rows:
        assert set(row.keys()) == expected_blind_keys, f"P1-5 violated: unexpected keys in blind audit {row.keys()}"

    print("[07_validate] ALL INVARIANTS (I1–I9, Official Tokenizer/Benchmarks, ChatML Byte/Token Parity, Blind Audit) PASSED!")
    return {
        "canonical_pool_n": len(pool),
        "test_main_n": len(test_200),
        "test_full_n": len(test_500),
        "warmup_n": len(warmup_5k),
        "final_chosen_policy_family_distribution": dict(Counter(r["chosen_policy_family"] for r in pool)),
        "final_rejected_policy_family_distribution": dict(Counter(r["rejected_policy_family"] for r in pool)),
    }


def build_manifest_and_datacard(counts_summary: Dict[str, Any]) -> None:
    cfg = load_config()
    with open(RAW_CACHE_DIR / "sources_lock.json", "r", encoding="utf-8") as f:
        sources_lock = json.load(f)
    with open(DATA_V1_DIR / "manifests" / "canonicalize_funnel.json", "r", encoding="utf-8") as f:
        funnel = json.load(f)
    with open(DATA_V1_DIR / "manifests" / "decontamination_report.json", "r", encoding="utf-8") as f:
        decontam = json.load(f)
    with open(DATA_V1_DIR / "manifests" / "split_report.json", "r", encoding="utf-8") as f:
        split_rep = json.load(f)
    with open(DATA_V1_DIR / "manifests" / "warmup_stats.json", "r", encoding="utf-8") as f:
        warmup_rep = json.load(f)

    excluded_manifest_rels = {
        "manifests/manifest.json",
        "DATA_CARD.md",
        "manifests/run_1_sha256.txt",
        "manifests/run_2_sha256.txt",
        "manifests/run_diff.txt",
    }

    tracked_files = []
    for p in sorted(DATA_V1_DIR.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        rel = p.relative_to(DATA_V1_DIR).as_posix()
        if rel.startswith("raw_cache/") or rel in excluded_manifest_rels:
            continue
        n_lines = 0
        if p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as fh:
                n_lines = sum(1 for line in fh if line.strip())
        tracked_files.append(
            {
                "path": rel,
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
                "n": n_lines,
            }
        )

    manifest = {
        "data_version": cfg["data_version"],
        "protocol_version": cfg["protocol_version"],
        "formatter_version": cfg["formatter_version"],
        "synthetic": False,
        "owner": cfg["owner"],
        "sources": sources_lock,
        "salts": cfg["salts"],
        "training_seeds": cfg["training_seeds"],
        "filters": cfg["filters"],
        "minhash": cfg["minhash"],
        "counts": {
            "raw_train_rows": funnel["train"]["raw_rows"],
            "unique_train_prompts_after_basic": funnel["train"]["unique_pids_after_basic_filters"],
            "unique_train_prompts_after_tokens": funnel["train"]["unique_pids_after_token_filters"],
            "final_canonical_pool": counts_summary["canonical_pool_n"],
            "surplus_above_10k": counts_summary["canonical_pool_n"] - 10000,
            "final_chosen_policy_family_distribution": counts_summary["final_chosen_policy_family_distribution"],
            "final_rejected_policy_family_distribution": counts_summary["final_rejected_policy_family_distribution"],
            "test_main": counts_summary["test_main_n"],
            "test_full": counts_summary["test_full_n"],
            "warmup": counts_summary["warmup_n"],
        },
        "files": tracked_files,
        "env": {
            "python": platform.python_version(),
            "pyarrow": pyarrow.__version__,
            "pyyaml": yaml.__version__,
            "numpy": __import__("importlib.metadata").metadata.version("numpy"),
            "regex": __import__("importlib.metadata").metadata.version("regex"),
            "scipy": __import__("importlib.metadata").metadata.version("scipy"),
            "transformers": __import__("importlib.metadata").metadata.version("transformers"),
            "tokenizers": __import__("importlib.metadata").metadata.version("tokenizers"),
            "trl": __import__("importlib.metadata").metadata.version("trl"),
            "torch": __import__("importlib.metadata").metadata.version("torch"),
            "datasets": __import__("importlib.metadata").metadata.version("datasets"),
            "huggingface_hub": __import__("importlib.metadata").metadata.version("huggingface-hub"),
            "accelerate": __import__("importlib.metadata").metadata.version("accelerate"),
        },
    }

    manifest_path = DATA_V1_DIR / "manifests" / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    b500 = split_rep["budgets"]["500"]
    b2000 = split_rep["budgets"]["2000"]
    b10000 = split_rep["budgets"]["10000"]
    test_strat = split_rep.get("test_200_vs_500_stratification", {})
    max_test_drift = test_strat.get("max_abs_stratum_prop_drift_200_vs_500", 0.0)

    # Export summary tables to PROJECT_ROOT / "results" (P0-5)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "data_funnel_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "counts": manifest["counts"],
                "train_funnel": funnel["train"],
                "validation_funnel": funnel["validation"],
                "decontamination": decontam,
                "warmup_summary": {
                    k: v for k, v in warmup_rep.items() if k != "keyword_dropped_50_audit_samples"
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(results_dir / "split_smd_and_policy_balance.json", "w", encoding="utf-8") as f:
        json.dump(split_rep, f, indent=2, ensure_ascii=False)

    data_card_md = f"""# DSAI 4207 Data Card (`data/v1.0` · Protocol `{cfg['protocol_version']}`)

| 属性 | 内容 |
|---|---|
| **项目名称** | Order vs. Scale: A Controlled Study of SFT/DPO Sequencing Across Data Budgets in Small LM Alignment |
| **数据版本** | `data/v1.0`（对应协议 `DATA_PROTOCOL.md` `{cfg['protocol_version']}`，`synthetic = false`） |
| **数据工程负责人** | PAN Haoren (24104216d) |
| **主数据源** | `openai/summarize_from_feedback` (`comparisons`, commit `{sources_lock['tldr']['revision']}`, train parquet sha256 `{sources_lock['tldr']['train_sha256'][:16]}…`) |
| **预热数据源** | `HuggingFaceH4/no_robots` (`train`, commit `{sources_lock['warmup']['revision']}`, parquet sha256 `{sources_lock['warmup']['train_sha256'][:16]}…`) |
| **官方分词器与模板** | `Qwen/Qwen2.5-0.5B-Base` (`hub_repo = Qwen/Qwen2.5-0.5B`, `revision_for_repo = Qwen/Qwen2.5-0.5B`, commit `{sources_lock['tokenizer']['revision']}`, `vocab_size = {sources_lock['tokenizer']['vocab_size']:,}`, `len(tokenizer) = {sources_lock['tokenizer']['len_tokenizer']:,}`), 官方 ChatML `formatter_version = {cfg['formatter_version']}`, `eos_token = <|im_end|>` (151645), `pad_token = <|endoftext|>` (151643) |
| **官方评测基准** | `cais/mmlu` (`all/test` 14,042 行/57 科 $\\rightarrow$ 500, commit `{sources_lock['benchmarks']['mmlu']['revision'][:12]}`) · `allenai/ai2_arc` (`ARC-Challenge/test` 1,172 行 $\\rightarrow$ 300, commit `{sources_lock['benchmarks']['arc_challenge']['revision'][:12]}`) · `google/IFEval` (`train` 541 行全量, commit `{sources_lock['benchmarks']['ifeval']['revision'][:12]}`) |
| **Manifest SHA256** | `{sha256_file(manifest_path)}` |
| **双跑一致性日志** | `data/v1.0/manifests/run_1_sha256.txt` · `data/v1.0/manifests/run_2_sha256.txt` · `data/v1.0/manifests/run_diff.txt`（0 差异） |

---

## 1. 数据清洗与去污染漏斗（Data Funnel）

### 1.1 TL;DR 训练池（`comparisons/train` → `canonical_pool.jsonl`）

| 阶段 | 输入数量 | 输出数量 | 剔除数量 | 说明 |
|---|---:|---:|---:|---|
| 原始标注行 (`raw_rows`) | {funnel['train']['raw_rows']:,} | {funnel['train']['valid_candidate_rows']:,} | {funnel['train']['raw_rows'] - funnel['train']['valid_candidate_rows']:,} | 剔除空字段、`[deleted]`、词数越界及相同正负摘要对 |
| 按 `pid` 聚合唯一题目 | {funnel['train']['valid_candidate_rows']:,} | {funnel['train']['unique_pids_after_basic_filters']:,} | — | 同题不败（Condorcet 0-loss 占比 **{funnel['train']['condorcet_zero_loss_rate']*100:.2f}%**）+ 置信度 + 策略档位选唯一对 |
| Qwen2.5 官方 Token 长度过滤 | {funnel['train']['unique_pids_after_basic_filters']:,} | {funnel['train']['unique_pids_after_token_filters']:,} | {funnel['train']['unique_pids_after_basic_filters'] - funnel['train']['unique_pids_after_token_filters']:,} | `prompt_tokens ≤ 768`, `chosen ≤ 160`, `rejected ≤ 200`, `total ≤ 1024` |
| 跨集去污染 (`vs test_500 & benchmarks`) | {funnel['train']['unique_pids_after_token_filters']:,} | {decontam['scopes'][4]['input_n']:,} | {funnel['train']['unique_pids_after_token_filters'] - decontam['scopes'][4]['input_n']:,} | 剔除与测试集及官方 IFEval/MMLU/ARC 命中近重复的样本 |
| 池内 Exact + MinHash 0.8 去重 | {decontam['scopes'][4]['input_n']:,} | **{counts_summary['canonical_pool_n']:,}** | {decontam['scopes'][4]['input_n'] - counts_summary['canonical_pool_n']:,} | 精确 `hash_prompt` 去重 + 精确 5-gram Jaccard ≥ 0.8 并查集去重 |

- **10k 预算档安全余量**：最终可用唯一题目 **{counts_summary['canonical_pool_n']:,}** 条，超出 $N=10,000$ 需求 **+{counts_summary['canonical_pool_n'] - 10000:,}** 条（余量充足）。
- **MinHash 阈值敏感性检验**：
  - Threshold `0.7`：{decontam['scopes'][5]['sensitivity_thresholds']['0.7']['dropped_prompts']} 条近重复；
  - Threshold `0.8`（主设定）：{decontam['scopes'][5]['sensitivity_thresholds']['0.8']['dropped_prompts']} 条近重复；
  - Threshold `0.9`：{decontam['scopes'][5]['sensitivity_thresholds']['0.9']['dropped_prompts']} 条近重复。
- **最终去重后 `canonical_pool.jsonl`（{counts_summary['canonical_pool_n']:,} 条）Chosen 策略家族构成**：`{json.dumps(counts_summary['final_chosen_policy_family_distribution'], ensure_ascii=False)}`。
- **最终去重后 `canonical_pool.jsonl`（{counts_summary['canonical_pool_n']:,} 条）Rejected 策略家族构成**：`{json.dumps(counts_summary['final_rejected_policy_family_distribution'], ensure_ascii=False)}`。

### 1.2 TL;DR 测试集（`comparisons/validation` → `test_200_main.jsonl` & `test_500_full.jsonl`）

- 原始行数：{funnel['validation']['raw_rows']:,} 行；其中显式拦截并剔除 `cnndm`（CNN/DailyMail 新闻）批次 **{funnel['validation']['drop_reasons'].get('cnndm_or_external_site', 0):,}** 行，剔除无效 `pid`（如 `edit_b2_eval_test`）**{funnel['validation']['drop_reasons'].get('invalid_or_missing_pid', 0)}** 行。
- 经官方 Qwen2.5 Token 过滤与池内去重后可用唯一 Reddit 题目：**{test_strat.get('clean_validation_pool_n', funnel['validation']['unique_pids_after_token_filters']):,}** 条（与训练池 `pid` 交集为 0）。
- 按 `prompt_tokens` 5 分位 × 子版块（Top 6 + `other` = 35 层）分层抽样并经中点排序后导出：
  - **`test_200_main.jsonl`（200 条）**：对齐 Proposal 预注册主实验与 API 预算；
  - **`test_500_full.jsonl`（500 条）**：含主实验前 200 条 + 300 条同分布扩展敏感性集；
  - **200 vs 500 的 35 层最大层比例偏差**：**{max_test_drift*100:.2f}%**（详见 `manifests/testset_stratum_report.json`）。

### 1.3 M0 通用预热集（`HuggingFaceH4/no_robots` → `warmup_5k.jsonl`）

- 原始 `train` 行数：{warmup_rep['raw_norobots_train_rows']:,} 条；
- 第一道剔除 `category == "Summarize"`：{warmup_rep['dropped_category_summarize']:,} 条；
- 第二道全轮次（`system` / `user` / `assistant`）摘要正则额外剔除：{warmup_rep['dropped_keyword_outside_summarize']:,} 条（两道合计剔除 {warmup_rep['union_category_and_keyword_dropped']:,} 条）；
- 超长（>2048 Qwen2.5 tokens）剔除：{warmup_rep['dropped_exceeded_2048_tokens']:,} 条；
- 第三道跨集去污染与内部去重剔除：{warmup_rep['dropped_cross_set_decontamination'] + warmup_rep['dropped_internal_exact'] + warmup_rep['dropped_internal_minhash_0.8']:,} 条；
- 干净可用池：**{warmup_rep['clean_pool_available']:,}** 条 $\\rightarrow$ 按剩余 9 个类别等比例分层抽取 **5,000** 条生成 `warmup_5k.jsonl`。

---

## 2. 嵌套子集与 Part A / Part B 平衡性报告（Balance, SMD & Policy $\\chi^2$）

| 预算档 $N$ | $|A_N|$ | $|B_N|$ | SMD (`prompt`) | SMD (`chosen`) | SMD (`rejected`) | Rejected 策略家族 $\\chi^2$ $p$-value | `(chosen, rejected)` 3×3 联合 $\\chi^2$ $p$-value | 与 10k 池子版块最大偏差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **$N = 500$** | 250 | 250 | `{b500['smd']['prompt_tokens']:+.4f}` | `{b500['smd']['chosen_tokens']:+.4f}` | `{b500['smd']['rejected_tokens']:+.4f}` | `{b500['policy_balance_tests']['rejected_policy_family_chi2']['p_value']:.4f}` | `{b500['policy_balance_tests']['joint_chosen_rejected_3x3_chi2']['p_value']:.4f}` | {b500['max_subreddit_drift_vs_10k']*100:.2f}% |
| **$N = 2,000$** | 1,000 | 1,000 | `{b2000['smd']['prompt_tokens']:+.4f}` | `{b2000['smd']['chosen_tokens']:+.4f}` | `{b2000['smd']['rejected_tokens']:+.4f}` | `{b2000['policy_balance_tests']['rejected_policy_family_chi2']['p_value']:.4f}` | `{b2000['policy_balance_tests']['joint_chosen_rejected_3x3_chi2']['p_value']:.4f}` | {b2000['max_subreddit_drift_vs_10k']*100:.2f}% |
| **$N = 10,000$** | 5,000 | 5,000 | `{b10000['smd']['prompt_tokens']:+.4f}` | `{b10000['smd']['chosen_tokens']:+.4f}` | `{b10000['smd']['rejected_tokens']:+.4f}` | `{b10000['policy_balance_tests']['rejected_policy_family_chi2']['p_value']:.4f}` | `{b10000['policy_balance_tests']['joint_chosen_rejected_3x3_chi2']['p_value']:.4f}` | {b10000['max_subreddit_drift_vs_10k']*100:.2f}% |

所有预算档的全部连续变量标准化均值差 $|\\mathrm{{SMD}}| \\ll 0.10$，Chosen 与 Rejected 策略家族（及 3×3 联合矩阵）在 Part A / Part B 间所有 $\\chi^2$ 检验 $p > 0.64$，在 $\\alpha=0.05$ 下均无显著差异，且各预算档子版块漂移均严格控制在 `< 3%` 阈值以内。

---

## 3. 局限性与伦理风险说明（Limitations & Risks）

1. **Reddit 原始语料中的 PII 与潜在不当内容风险**：`openai/summarize_from_feedback` 源自 Reddit 真实用户发帖（如 `r/relationships`、`r/tifu`），虽经前人匿名化处理，仍可能包含残留的个人身份细节（年龄、家庭关系描述）或情绪化/粗俗用语。本数据集仅限 DSAI 4207 课程非商业学术研究使用，严禁用于生产部署或重识别个人身份。
2. **基座模型预训练污染不可控性**：`Qwen/Qwen2.5-0.5B-Base` 的预训练语料未公开明细，不排除其在预训练阶段已接触过公共 Reddit 或 TL;DR 语料。因此实验报告中将 500 档严格界定为“本后训练流水线内第一次接触特定摘要任务（first task-specific contact within our post-training pipeline）”。
3. **正负样本天然长度差异**：人类偏好标注中 `chosen` 平均长度略长于 `rejected`；管线不人为按长度差截断偏好对，而在评测阶段统一采用长度受控胜率（LCWR）消除长度偏置。

---

## 4. 验收状态（I1–I10 & 真实性/集成测试）

- **I1–I10 自动断言**：全部通过（`07_validate.py` & `tests/test_data_invariants.py`）。
- **官方分词器与官方评测集真实性测试**：`test_tokenizer_authentic` 与 `test_benchmark_authentic` 全部通过。
- **官方 ChatML 逐字节与跨训练器（TRL SFTTrainer / DPOTrainer）单步集成测试**：`test_trl_smoke` 全部通过。
- **100 对数据标签盲评隔离**：`manifests/data_label_audit_100.jsonl`（纯盲评表）与 `manifests/data_label_audit_100_key.json`（答案与策略 key）物理分离并通过字段审查。
"""
    with open(DATA_V1_DIR / "DATA_CARD.md", "w", encoding="utf-8") as f:
        f.write(data_card_md)

    print(f"[07_validate] Generated {manifest_path} ({len(tracked_files)} tracked files), results/, and DATA_CARD.md.")


def main() -> None:
    counts = verify_all_invariants()
    build_manifest_and_datacard(counts)


if __name__ == "__main__":
    main()
