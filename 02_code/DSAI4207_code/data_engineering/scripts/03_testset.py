#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_testset.py — Build fixed evaluation benchmarks (MMLU-500, ARC-Challenge-300, IFEval-541)
from official Hugging Face benchmark parquets and construct the decontaminated, stratified
held-out TL;DR test sets:
- data/v1.0/test_200_main.jsonl (pre-registered 200-prompt main evaluation set)
- data/v1.0/test_500_full.jsonl (500-prompt extended sensitivity set, where first 200 == test_200_main)
Also outputs 35-stratum representation report for test_200_main vs test_500_full (P2-6)
and removes duplicate test_500.jsonl alias (P2-7).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from utils import (
    DATA_V1_DIR,
    RAW_CACHE_DIR,
    SYSTEM_PROMPT,
    ExactVerifiedMinHashIndex,
    cluster_pairs_union_find,
    get_policy_tier,
    load_config,
    normalize_for_hash,
    read_jsonl,
    sha256_str,
    write_jsonl,
)


def build_eval_benchmarks(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    bench_cache = RAW_CACHE_DIR / "benchmarks"
    out_dir = DATA_V1_DIR / "eval_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. IFEval (full 541 official items)
    ifeval_rows = pq.read_table(bench_cache / "ifeval_train.parquet").to_pylist()
    assert len(ifeval_rows) == 541, f"Expected 541 official IFEval rows, got {len(ifeval_rows)}"
    ifeval_rows.sort(key=lambda r: int(r["key"]))
    ifeval_records = []
    for idx, r in enumerate(ifeval_rows):
        prompt_text = str(r["prompt"])
        assert not prompt_text.startswith("[IFEval #"), "Synthetic IFEval placeholder detected!"
        ifeval_records.append(
            {
                "id": f"ifeval_{r['key']}",
                "key": int(r["key"]),
                "prompt": prompt_text,
                "instruction_id_list": list(r["instruction_id_list"]),
                "kwargs": [
                    {k: v for k, v in kw.items() if v is not None}
                    for kw in (r.get("kwargs") or [])
                ],
                "messages": [{"role": "user", "content": prompt_text}],
                "max_new_tokens": cfg["sources"]["benchmarks"]["ifeval"]["max_new_tokens"],
                "order": idx,
                "norm_text": normalize_for_hash(prompt_text),
            }
        )
    write_jsonl(out_dir / "ifeval_541.jsonl", ifeval_records)

    # 2. ARC-Challenge (300 items deterministically sampled from 1,172 official test items, seed=42)
    arc_rows = pq.read_table(bench_cache / "arc_challenge_test.parquet").to_pylist()
    assert len(arc_rows) == 1172, f"Expected 1,172 official ARC-Challenge test rows, got {len(arc_rows)}"
    arc_rows.sort(key=lambda r: sha256_str(f"arc_seed_42_{r['id']}"))
    arc_selected = arc_rows[: cfg["sources"]["benchmarks"]["arc_challenge"]["sample_size"]]
    arc_records = []
    for idx, r in enumerate(arc_selected):
        q_str = str(r["question"])
        assert "controlled scientific experiment #" not in q_str, "Synthetic ARC placeholder detected!"
        choices_obj = r["choices"]
        labels = list(choices_obj["label"])
        texts = list(choices_obj["text"])
        concat_text = f"{q_str} " + " ".join(texts)
        arc_records.append(
            {
                "id": str(r["id"]),
                "question": q_str,
                "choices": texts,
                "choice_labels": labels,
                "answerKey": str(r["answerKey"]),
                "order": idx,
                "norm_text": normalize_for_hash(concat_text),
            }
        )
    write_jsonl(out_dir / "arc_challenge_300.jsonl", arc_records)

    # 3. MMLU (500 items stratified across all 57 subjects from 14,042 official test items, seed=42)
    mmlu_rows = pq.read_table(bench_cache / "mmlu_all_test.parquet").to_pylist()
    assert len(mmlu_rows) == 14042, f"Expected 14,042 official MMLU test rows, got {len(mmlu_rows)}"
    by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in mmlu_rows:
        subj = str(r.get("subject") or "unknown")
        by_subject[subj].append(r)

    assert len(by_subject) == 57, f"Expected 57 official MMLU subjects, got {len(by_subject)}"
    total_mmlu = len(mmlu_rows)
    target_mmlu = cfg["sources"]["benchmarks"]["mmlu"]["sample_size"]
    subjects = sorted(by_subject.keys())

    # Proportional allocation via Hamilton / largest-remainder method
    quotas: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for subj in subjects:
        exact_q = len(by_subject[subj]) * target_mmlu / total_mmlu
        floor_q = max(1, int(exact_q))
        quotas[subj] = min(len(by_subject[subj]), floor_q)
        remainders.append((exact_q - floor_q, subj))

    current_sum = sum(quotas.values())
    if current_sum < target_mmlu:
        remainders.sort(key=lambda x: (-x[0], x[1]))
        for _, subj in remainders[: target_mmlu - current_sum]:
            quotas[subj] += 1
    elif current_sum > target_mmlu:
        remainders.sort(key=lambda x: (x[0], x[1]))
        for _, subj in remainders[: current_sum - target_mmlu]:
            if quotas[subj] > 1:
                quotas[subj] -= 1

    mmlu_sampled = []
    for subj in subjects:
        items = by_subject[subj]
        items.sort(
            key=lambda r: sha256_str(
                f"mmlu_seed_42_{subj}_{r['question']}_{''.join(r['choices'])}"
            )
        )
        for i, r in enumerate(items[: quotas[subj]]):
            mmlu_sampled.append((subj, i, len(items[: quotas[subj]]), r))

    mmlu_sampled.sort(
        key=lambda t: (
            (t[1] + 0.5) / max(1, t[2]),
            sha256_str(f"mmlu_order_42_{t[0]}_{t[3]['question']}"),
        )
    )
    mmlu_records = []
    choice_letters = ["A", "B", "C", "D"]
    for idx, (subj, _, _, r) in enumerate(mmlu_sampled):
        q_str = str(r["question"])
        assert "phenomena #" not in q_str, "Synthetic MMLU placeholder detected!"
        choices_list = [str(c) for c in r["choices"]]
        ans_idx = int(r["answer"])
        concat_text = f"{q_str} " + " ".join(choices_list)
        mmlu_records.append(
            {
                "id": f"mmlu_{subj}_{idx:04d}",
                "subject": subj,
                "question": q_str,
                "choices": choices_list,
                "answer_index": ans_idx,
                "answer_letter": choice_letters[ans_idx],
                "order": idx,
                "norm_text": normalize_for_hash(concat_text),
            }
        )
    assert len(mmlu_records) == target_mmlu
    write_jsonl(out_dir / "mmlu_500.jsonl", mmlu_records)

    return {
        "ifeval": ifeval_records,
        "arc_challenge": arc_records,
        "mmlu": mmlu_records,
    }


def record_priority_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
    zero_loss = 1 if r.get("chosen_losses_in_pid", 0) == 0 else 0
    conf = r.get("selected_confidence")
    conf_score = conf if conf is not None else -1
    tier = get_policy_tier(r.get("chosen_policy"))
    return (-zero_loss, -conf_score, -tier, r["hash_pair"], r["pid"])


def build_testset() -> None:
    cfg = load_config()
    benchmarks = build_eval_benchmarks(cfg)

    val_path = RAW_CACHE_DIR / "canonical_val_pre_dedup.jsonl"
    val_records = read_jsonl(val_path)

    # 1. Exact dedup within validation by hash_prompt
    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in val_records:
        by_hash[r["hash_prompt"]].append(r)
    exact_deduped = [
        sorted(group, key=record_priority_key)[0]
        for _, group in sorted(by_hash.items())
    ]

    # 2. Decontaminate validation against benchmarks + internal MinHash 0.8 dedup
    bench_index = ExactVerifiedMinHashIndex(num_perm=128, bands=32)
    for b_name, b_list in benchmarks.items():
        for b_item in b_list:
            bench_index.insert(f"{b_name}:{b_item['id']}", b_item["norm_text"])

    val_decontam = []
    for r in exact_deduped:
        norm_txt = normalize_for_hash(r["title"] + " " + r["post"] + " " + r["chosen"])
        hits = bench_index.query_Text(norm_txt, threshold=0.8)
        if not hits:
            val_decontam.append(r)

    val_index = ExactVerifiedMinHashIndex(num_perm=128, bands=32)
    by_pid = {r["pid"]: r for r in val_decontam}
    for r in val_decontam:
        val_index.insert(r["pid"], normalize_for_hash(r["title"] + " " + r["post"]))

    internal_pairs = val_index.find_all_internal_pairs(threshold=0.8)
    clusters = cluster_pairs_union_find(by_pid.keys(), internal_pairs)
    dropped_pids = set()
    for _, members in clusters.items():
        sorted_members = sorted((by_pid[m] for m in members), key=record_priority_key)
        for loser in sorted_members[1:]:
            dropped_pids.add(loser["pid"])

    clean_val = [r for r in val_decontam if r["pid"] not in dropped_pids]

    # 3. Stratified sampling of 500 prompts (§7.2: 5 length quantiles x (Top 6 subreddits + other) = 35 strata)
    sub_counts = Counter(r["subreddit"] for r in clean_val)
    top_subs = {
        s for s, _ in sorted(sub_counts.items(), key=lambda x: (-x[1], x[0]))[:6]
    }

    lengths = np.array([r["prompt_tokens"] for r in clean_val])
    quantiles = np.quantile(lengths, [0.2, 0.4, 0.6, 0.8])

    def get_len_bin(tok_len: int) -> int:
        return int(np.searchsorted(quantiles, tok_len, side="right"))

    strata: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in clean_val:
        l_bin = get_len_bin(r["prompt_tokens"])
        s_bucket = r["subreddit"] if r["subreddit"] in top_subs else "other"
        strata[(l_bin, s_bucket)].append(r)

    # Merge any stratum with < 5 candidates into (l_bin, "other")
    merged_strata_log = []
    for (l_bin, s_bucket) in list(strata.keys()):
        if s_bucket != "other" and len(strata[(l_bin, s_bucket)]) < 5:
            cnt = len(strata[(l_bin, s_bucket)])
            strata[(l_bin, "other")].extend(strata.pop((l_bin, s_bucket)))
            merged_strata_log.append(
                {"from": f"q{l_bin}_{s_bucket}", "to": f"q{l_bin}_other", "count": cnt}
            )

    salt_test = cfg["salts"]["test"]
    salt_order = cfg["salts"]["test_order"]
    full_size = cfg["testset"]["full_size"]
    main_size = cfg["testset"]["main_size"]
    total_clean = len(clean_val)

    stratum_keys = sorted(strata.keys())
    quotas: Dict[Tuple[int, str], int] = {}
    remainders: List[Tuple[float, Tuple[int, str]]] = []
    for sk in stratum_keys:
        exact_q = len(strata[sk]) * full_size / total_clean
        floor_q = int(exact_q)
        quotas[sk] = min(len(strata[sk]), floor_q)
        remainders.append((exact_q - floor_q, sk))

    rem_needed = full_size - sum(quotas.values())
    remainders.sort(key=lambda x: (-x[0], x[1]))
    for _, sk in remainders[:rem_needed]:
        quotas[sk] += 1

    sampled_with_keys = []
    for sk in stratum_keys:
        items = sorted(
            strata[sk], key=lambda r: sha256_str(f"{r['pid']}_{salt_test}")
        )
        chosen_k = items[: quotas[sk]]
        n_k = len(chosen_k)
        chosen_k.sort(key=lambda r: sha256_str(f"{r['pid']}_{salt_order}"))
        stratum_label = f"q{sk[0]}_{sk[1]}"
        for i, r in enumerate(chosen_k):
            order_key = (
                (i + 0.5) / max(1, n_k),
                sha256_str(f"{r['pid']}_{salt_order}"),
            )
            sampled_with_keys.append((order_key, stratum_label, r))

    sampled_with_keys.sort(key=lambda x: x[0])
    assert len(sampled_with_keys) == full_size

    test_500_records = []
    stratum_counts_500 = Counter()
    stratum_counts_200 = Counter()

    for idx, (_, stratum_label, r) in enumerate(sampled_with_keys):
        stratum_counts_500[stratum_label] += 1
        if idx < main_size:
            stratum_counts_200[stratum_label] += 1
        test_500_records.append(
            {
                "pid": r["pid"],
                "subreddit": r["subreddit"],
                "title": r["title"],
                "post": r["post"],
                "user_content": r["user_content"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["user_content"]},
                ],
                "reference": r["chosen"],
                "reference_policy": r["chosen_policy"],
                "prompt_tokens": r["prompt_tokens"],
                "reference_tokens": r["chosen_tokens"],
                "stratum": stratum_label,
                "order": idx,
                "is_main_200": idx < main_size,
                "hash_prompt": r["hash_prompt"],
            }
        )

    test_200_records = test_500_records[:main_size]
    write_jsonl(DATA_V1_DIR / "test_500_full.jsonl", test_500_records)
    write_jsonl(DATA_V1_DIR / "test_200_main.jsonl", test_200_records)

    # P2-7: Remove duplicate alias test_500.jsonl if it exists
    dup_test_500 = DATA_V1_DIR / "test_500.jsonl"
    if dup_test_500.exists():
        dup_test_500.unlink()

    # P2-6: Compute 35-stratum representation comparison between test_200_main and test_500_full
    all_stratum_labels = sorted(stratum_counts_500.keys())
    per_stratum_comparison = {}
    max_abs_drift = 0.0
    max_count_diff_from_ideal = 0.0
    for sl in all_stratum_labels:
        c500 = stratum_counts_500[sl]
        c200 = stratum_counts_200[sl]
        p500 = c500 / full_size
        p200 = c200 / main_size
        drift = abs(p200 - p500)
        ideal_200 = c500 * (main_size / full_size)
        count_err = abs(c200 - ideal_200)
        if drift > max_abs_drift:
            max_abs_drift = drift
        if count_err > max_count_diff_from_ideal:
            max_count_diff_from_ideal = count_err
        per_stratum_comparison[sl] = {
            "count_500": c500,
            "count_200": c200,
            "ideal_200": round(ideal_200, 2),
            "prop_500": round(p500, 4),
            "prop_200": round(p200, 4),
            "abs_prop_drift": round(drift, 4),
        }

    testset_stratum_report = {
        "clean_validation_pool_n": total_clean,
        "num_strata_configured": 35,
        "num_strata_active": len(all_stratum_labels),
        "merged_strata": merged_strata_log,
        "prompt_token_quantiles_20_40_60_80": [float(x) for x in quantiles],
        "top_6_subreddits": sorted(top_subs),
        "max_abs_stratum_prop_drift_200_vs_500": round(max_abs_drift, 4),
        "max_stratum_count_error_vs_exact_quota": round(max_count_diff_from_ideal, 4),
        "strata": per_stratum_comparison,
    }
    manifests_dir = DATA_V1_DIR / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(manifests_dir / "testset_stratum_report.json", "w", encoding="utf-8") as f:
        json.dump(testset_stratum_report, f, indent=2, ensure_ascii=False)

    print(
        f"[03_testset] Built clean validation pool ({total_clean} prompts) -> "
        f"test_200_main.jsonl (200) & test_500_full.jsonl (500), "
        f"35-stratum max drift (200 vs 500) = {max_abs_drift*100:.2f}%; "
        f"official benchmarks saved in eval_benchmarks/."
    )


if __name__ == "__main__":
    build_testset()
