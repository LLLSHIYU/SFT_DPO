#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_warmup.py — Build the 5,000-sample M0 general instruction warmup dataset from HuggingFaceH4/no_robots (§10):
1. Filter out category == 'Summarize' and rows missing non-empty assistant turns.
2. Full-turn regex screening across all system/user turns and assistant turns.
3. Filter by Qwen2.5-0.5B-Base total_tokens <= 2048.
4. Decontaminate against canonical_pool, test_500_full, IFEval, MMLU, and ARC-Challenge (exact + 5-gram Jaccard >= 0.8),
   followed by internal exact + MinHash 0.8 deduplication.
5. Stratified-sample 5,000 instructions across the remaining 9 categories and output warmup_5k.jsonl,
   warmup_indices.json, and warmup_stats.json.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from utils import (
    DATA_V1_DIR,
    RAW_CACHE_DIR,
    ExactVerifiedMinHashIndex,
    QwenTokenizerWrapper,
    cluster_pairs_union_find,
    load_config,
    normalize_for_hash,
    normalize_for_training,
    read_jsonl,
    sha256_str,
    write_jsonl,
)


def main() -> None:
    cfg = load_config()
    tokenizer = QwenTokenizerWrapper(RAW_CACHE_DIR / "qwen_tokenizer")
    nr_rows = pq.read_table(RAW_CACHE_DIR / "norobots_train.parquet").to_pylist()

    user_sys_re = re.compile(cfg["warmup"]["user_system_regex"], re.IGNORECASE)
    asst_re = re.compile(cfg["warmup"]["assistant_regex"], re.IGNORECASE)
    max_warmup_tokens = cfg["filters"]["max_warmup_tokens"]

    raw_count = len(nr_rows)
    dropped_category_summarize = 0
    dropped_empty_assistant = 0
    dropped_keyword = 0
    dropped_keyword_Also_in_summarize = 0
    dropped_token_length = 0

    keyword_hit_samples: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    for row_idx, r in enumerate(nr_rows):
        cat = str(r.get("category") or "unknown").strip()
        pid = str(r.get("prompt_id") or f"nr_{row_idx}")
        raw_prompt = str(r.get("prompt") or "")
        msgs = r.get("messages") or []

        # Check keyword hit first for accurate union/overlap statistics
        kw_match_reason = None
        m_top = user_sys_re.search(raw_prompt)
        if m_top:
            kw_match_reason = f"prompt:{m_top.group(0)}"

        cleaned_msgs: List[Dict[str, str]] = []
        has_valid_assistant = False
        for m in msgs:
            role = str(m.get("role") or "").strip().lower()
            content, _ = normalize_for_training(m.get("content"), strip_tldr_prefix=False)
            if not role or not content:
                continue
            if role == "assistant":
                has_valid_assistant = True
                if not kw_match_reason:
                    ma = asst_re.search(content)
                    if ma:
                        kw_match_reason = f"assistant:{ma.group(0)}"
            elif role in ("system", "user"):
                if not kw_match_reason:
                    mu = user_sys_re.search(content)
                    if mu:
                        kw_match_reason = f"{role}:{mu.group(0)}"
            cleaned_msgs.append({"role": role, "content": content})

        if cat == cfg["warmup"]["excluded_category"]:
            dropped_category_summarize += 1
            if kw_match_reason:
                dropped_keyword_Also_in_summarize += 1
            continue

        if not has_valid_assistant or len(cleaned_msgs) < 2:
            dropped_empty_assistant += 1
            continue

        if kw_match_reason:
            dropped_keyword += 1
            if len(keyword_hit_samples) < 50:
                keyword_hit_samples.append(
                    {
                        "prompt_id": pid,
                        "category": cat,
                        "match": kw_match_reason,
                        "prompt_snippet": raw_prompt[:180],
                    }
                )
            continue

        # Token length check (§10.4: max_length <= 2048, drop if exceeded)
        input_ids, _ = tokenizer.tokenize_sft_with_labels(cleaned_msgs)
        total_toks = len(input_ids)
        if total_toks > max_warmup_tokens:
            dropped_token_length += 1
            continue

        full_text = " ".join(m["content"] for m in cleaned_msgs)
        norm_text = normalize_for_hash(full_text)
        if not norm_text:
            dropped_empty_assistant += 1
            continue

        candidates.append(
            {
                "prompt_id": pid,
                "source_row_idx": row_idx,
                "category": cat,
                "prompt": normalize_for_training(raw_prompt)[0],
                "messages": cleaned_msgs,
                "total_tokens": total_toks,
                "norm_text": norm_text,
                "hash_text": sha256_str(norm_text),
            }
        )

    after_keyword_and_len_n = len(candidates)

    # 3. Third screen: Cross-set decontamination against TL;DR pool, test_500, and benchmarks (§10.2)
    ext_index = ExactVerifiedMinHashIndex(num_perm=128, bands=32)
    ext_hashes = set()

    canonical_pool = read_jsonl(DATA_V1_DIR / "canonical_pool.jsonl")
    test_500 = read_jsonl(DATA_V1_DIR / "test_500_full.jsonl")
    ifeval = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "ifeval_541.jsonl")
    mmlu = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "mmlu_500.jsonl")
    arc = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "arc_challenge_300.jsonl")

    for r in canonical_pool:
        nt = normalize_for_hash(r["title"] + " " + r["post"] + " " + r["chosen"])
        ext_index.insert(f"pool:{r['pid']}", nt)
        ext_hashes.add(r["hash_prompt"])
    for r in test_500:
        nt = normalize_for_hash(r["title"] + " " + r["post"] + " " + r["reference"])
        ext_index.insert(f"test:{r['pid']}", nt)
        ext_hashes.add(r["hash_prompt"])
    for b_name, b_list in [("ifeval", ifeval), ("mmlu", mmlu), ("arc", arc)]:
        for item in b_list:
            ext_index.insert(f"{b_name}:{item['id']}", item["norm_text"])
            ext_hashes.add(sha256_str(item["norm_text"]))

    decontam_candidates = []
    decontam_hits = []
    for c in candidates:
        if c["hash_text"] in ext_hashes:
            decontam_hits.append({"prompt_id": c["prompt_id"], "match": "exact_hash", "jaccard": 1.0})
            continue
        hits = ext_index.query_Text(c["norm_text"], threshold=0.8)
        if hits:
            decontam_hits.append(
                {"prompt_id": c["prompt_id"], "match": hits[0][0], "jaccard": hits[0][1]}
            )
            continue
        decontam_candidates.append(c)

    # 4. Internal exact + MinHash 0.8 deduplication within warmup
    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in decontam_candidates:
        by_hash[c["hash_text"]].append(c)
    exact_unique = [
        sorted(g, key=lambda x: x["prompt_id"])[0] for _, g in sorted(by_hash.items())
    ]
    internal_exact_dropped = len(decontam_candidates) - len(exact_unique)

    w_index = ExactVerifiedMinHashIndex(num_perm=128, bands=32)
    by_pid = {c["prompt_id"]: c for c in exact_unique}
    for c in exact_unique:
        w_index.insert(c["prompt_id"], c["norm_text"])

    w_pairs = w_index.find_all_internal_pairs(threshold=0.8)
    w_clusters = cluster_pairs_union_find(by_pid.keys(), w_pairs)
    w_minhash_dropped = set()
    for _, members in w_clusters.items():
        sorted_m = sorted(members)
        for loser_id in sorted_m[1:]:
            w_minhash_dropped.add(loser_id)

    clean_warmup_pool = [
        c for c in sorted(exact_unique, key=lambda x: x["prompt_id"]) if c["prompt_id"] not in w_minhash_dropped
    ]

    # 5. Stratified sampling of 5,000 items by category (§10.3)
    target_n = cfg["warmup"]["target_size"]
    salt_warmup = cfg["salts"]["warmup"]
    if len(clean_warmup_pool) < target_n:
        raise RuntimeError(
            f"Clean warmup pool has {len(clean_warmup_pool)} < {target_n} items!"
        )

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in clean_warmup_pool:
        by_cat[c["category"]].append(c)

    cats = sorted(by_cat.keys())
    total_clean = len(clean_warmup_pool)
    quotas: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for cat in cats:
        exact_q = len(by_cat[cat]) * target_n / total_clean
        floor_q = int(exact_q)
        quotas[cat] = min(len(by_cat[cat]), floor_q)
        remainders.append((exact_q - floor_q, cat))

    rem_needed = target_n - sum(quotas.values())
    remainders.sort(key=lambda x: (-x[0], x[1]))
    for _, cat in remainders[:rem_needed]:
        quotas[cat] += 1

    sampled_items = []
    for cat in cats:
        cat_items = sorted(
            by_cat[cat], key=lambda c: sha256_str(f"{c['prompt_id']}_{salt_warmup}")
        )
        chosen_cat = cat_items[: quotas[cat]]
        n_c = len(chosen_cat)
        for i, item in enumerate(chosen_cat):
            order_key = (
                (i + 0.5) / max(1, n_c),
                sha256_str(f"{item['prompt_id']}_{salt_warmup}"),
            )
            sampled_items.append((order_key, item))

    sampled_items.sort(key=lambda x: x[0])
    assert len(sampled_items) == target_n

    warmup_5k_records = []
    warmup_indices = []
    for order_idx, (_, item) in enumerate(sampled_items):
        warmup_5k_records.append(
            {
                "prompt_id": item["prompt_id"],
                "category": item["category"],
                "messages": item["messages"],
                "total_tokens": item["total_tokens"],
                "order": order_idx,
            }
        )
        warmup_indices.append(
            {
                "order": order_idx,
                "prompt_id": item["prompt_id"],
                "source_row_idx": item["source_row_idx"],
                "category": item["category"],
                "total_tokens": item["total_tokens"],
                "hash_text": item["hash_text"],
            }
        )

    write_jsonl(DATA_V1_DIR / "warmup_5k.jsonl", warmup_5k_records)

    manifests_dir = DATA_V1_DIR / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(manifests_dir / "warmup_indices.json", "w", encoding="utf-8") as f:
        json.dump(warmup_indices, f, indent=2, ensure_ascii=False)

    # Also update decontamination_report.json with warmup scopes
    decontam_path = manifests_dir / "decontamination_report.json"
    if decontam_path.exists():
        with open(decontam_path, "r", encoding="utf-8") as f:
            dec_rep = json.load(f)
        dec_rep["scopes"].extend(
            [
                {
                    "name": "warmup_vs_pool_test_benchmarks",
                    "left_n": after_keyword_and_len_n,
                    "matches": len(decontam_hits),
                    "dropped_from_left": len(decontam_hits),
                    "examples": decontam_hits[:10],
                },
                {
                    "name": "warmup_internal_dedup",
                    "input_n": len(decontam_candidates),
                    "output_n": len(clean_warmup_pool),
                    "exact_dropped": internal_exact_dropped,
                    "minhash_0.8_dropped": len(w_minhash_dropped),
                },
            ]
        )
        with open(decontam_path, "w", encoding="utf-8") as f:
            json.dump(dec_rep, f, indent=2, ensure_ascii=False)

    tok_arr = np.array([r["total_tokens"] for r in warmup_5k_records])
    warmup_stats = {
        "raw_norobots_train_rows": raw_count,
        "dropped_category_summarize": dropped_category_summarize,
        "dropped_keyword_outside_summarize": dropped_keyword,
        "union_category_and_keyword_dropped": dropped_category_summarize + dropped_keyword,
        "overlap_category_and_keyword": dropped_keyword_Also_in_summarize,
        "dropped_empty_assistant": dropped_empty_assistant,
        "dropped_exceeded_2048_tokens": dropped_token_length,
        "dropped_cross_set_decontamination": len(decontam_hits),
        "dropped_internal_exact": internal_exact_dropped,
        "dropped_internal_minhash_0.8": len(w_minhash_dropped),
        "clean_pool_available": len(clean_warmup_pool),
        "sampled_warmup_n": len(warmup_5k_records),
        "category_counts_in_clean_pool": dict(Counter(c["category"] for c in clean_warmup_pool)),
        "category_counts_in_warmup_5k": dict(Counter(r["category"] for r in warmup_5k_records)),
        "token_stats": {
            "mean": round(float(np.mean(tok_arr)), 2),
            "median": round(float(np.median(tok_arr)), 2),
            "p90": round(float(np.quantile(tok_arr, 0.90)), 2),
            "max": int(np.max(tok_arr)),
        },
        "keyword_dropped_50_audit_samples": keyword_hit_samples,
    }
    with open(manifests_dir / "warmup_stats.json", "w", encoding="utf-8") as f:
        json.dump(warmup_stats, f, indent=2, ensure_ascii=False)

    print(
        f"[05_warmup] 9500 raw -> -{dropped_category_summarize} (Summarize cat) "
        f"-{dropped_keyword} (full-turn regex) -{dropped_token_length} (>2048 tok) "
        f"-{len(decontam_hits)} (decontam) -{internal_exact_dropped + len(w_minhash_dropped)} (dedup) "
        f"= {len(clean_warmup_pool)} clean -> sampled warmup_5k.jsonl (5000)."
    )


if __name__ == "__main__":
    main()
