#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_decontaminate_and_dedup.py — Execute strict outer-before-inner decontamination and deduplication (§6):
1. Ensure test_500_full.jsonl and eval_benchmarks/ (MMLU-500, ARC-300, IFEval-541) are built.
2. Cross-set decontamination: remove any training candidate matching test_500_full or benchmarks
   (exact match or exact 5-word shingle Jaccard >= 0.8).
3. Internal exact deduplication on hash_prompt.
4. Internal MinHash LSH + exact 5-word shingle Jaccard >= 0.8 Union-Find clustering deduplication
   (with 0.7 / 0.8 / 0.9 sensitivity reporting).
Outputs data/v1.0/canonical_pool.jsonl and data/v1.0/manifests/decontamination_report.json.
"""

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import (
    DATA_V1_DIR,
    RAW_CACHE_DIR,
    ExactVerifiedMinHashIndex,
    cluster_pairs_union_find,
    get_policy_tier,
    load_config,
    normalize_for_hash,
    read_jsonl,
    write_jsonl,
)


def _ensure_testset_and_benchmarks_exist() -> None:
    test_path = DATA_V1_DIR / "test_500_full.jsonl"
    bench_dir = DATA_V1_DIR / "eval_benchmarks"
    if (
        not test_path.exists()
        or not (bench_dir / "ifeval_541.jsonl").exists()
        or not (bench_dir / "mmlu_500.jsonl").exists()
        or not (bench_dir / "arc_challenge_300.jsonl").exists()
    ):
        spec = importlib.util.spec_from_file_location(
            "testset_mod", Path(__file__).parent / "03_testset.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.build_testset()


def record_priority_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
    zero_loss = 1 if r.get("chosen_losses_in_pid", 0) == 0 else 0
    conf = r.get("selected_confidence")
    conf_score = conf if conf is not None else -1
    tier = get_policy_tier(r.get("chosen_policy"))
    return (-zero_loss, -conf_score, -tier, r["hash_pair"], r["pid"])


def main() -> None:
    cfg = load_config()
    _ensure_testset_and_benchmarks_exist()

    train_candidates = read_jsonl(RAW_CACHE_DIR / "canonical_train_pre_dedup.jsonl")
    test_500 = read_jsonl(DATA_V1_DIR / "test_500_full.jsonl")
    ifeval = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "ifeval_541.jsonl")
    mmlu = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "mmlu_500.jsonl")
    arc = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "arc_challenge_300.jsonl")

    scopes_report: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Step 1: Cross-set decontamination against test_500_full & benchmarks (§6.1 Step B)
    # -----------------------------------------------------------------------
    test_pids = {r["pid"] for r in test_500}
    test_hashes = {r["hash_prompt"] for r in test_500}
    test_index = ExactVerifiedMinHashIndex(num_perm=128, bands=32)
    for r in test_500:
        test_index.insert(r["pid"], normalize_for_hash(r["title"] + " " + r["post"]))

    bench_indices: Dict[str, ExactVerifiedMinHashIndex] = {}
    for b_name, b_data in [("ifeval", ifeval), ("mmlu", mmlu), ("arc_challenge", arc)]:
        idx_obj = ExactVerifiedMinHashIndex(num_perm=128, bands=32)
        for item in b_data:
            idx_obj.insert(str(item["id"]), item["norm_text"])
        bench_indices[b_name] = idx_obj

    # 1a. Pool vs test_500
    after_test_decontam = []
    test_matches = []
    for r in train_candidates:
        norm_p = normalize_for_hash(r["title"] + " " + r["post"])
        if r["pid"] in test_pids or r["hash_prompt"] in test_hashes:
            test_matches.append({"train_pid": r["pid"], "match_id": r["pid"], "jaccard": 1.0})
            continue
        hits = test_index.query_Text(norm_p, threshold=0.8)
        if hits:
            test_matches.append(
                {"train_pid": r["pid"], "match_id": hits[0][0], "jaccard": hits[0][1]}
            )
            continue
        after_test_decontam.append(r)

    scopes_report.append(
        {
            "name": "pool_vs_test_500",
            "left_n": len(train_candidates),
            "right_n": len(test_500),
            "matches": len(test_matches),
            "dropped_from_left": len(test_matches),
            "examples": test_matches[:10],
        }
    )

    # 1b. Pool vs IFEval / MMLU / ARC-Challenge
    curr_pool = after_test_decontam
    for b_name, b_data in [("ifeval", ifeval), ("mmlu", mmlu), ("arc_challenge", arc)]:
        next_pool = []
        b_matches = []
        b_idx = bench_indices[b_name]
        for r in curr_pool:
            norm_full = normalize_for_hash(
                r["title"] + " " + r["post"] + " " + r["chosen"]
            )
            hits = b_idx.query_Text(norm_full, threshold=0.8)
            if hits:
                b_matches.append(
                    {"train_pid": r["pid"], "benchmark_id": hits[0][0], "jaccard": hits[0][1]}
                )
            else:
                next_pool.append(r)
        scopes_report.append(
            {
                "name": f"pool_vs_{b_name}",
                "left_n": len(curr_pool),
                "right_n": len(b_data),
                "matches": len(b_matches),
                "dropped_from_left": len(b_matches),
                "examples": b_matches[:10],
            }
        )
        curr_pool = next_pool

    # -----------------------------------------------------------------------
    # Step 2: Internal Exact Deduplication on hash_prompt (§6.2)
    # -----------------------------------------------------------------------
    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in curr_pool:
        by_hash[r["hash_prompt"]].append(r)

    exact_kept = []
    exact_dropped_examples = []
    exact_dropped_count = 0
    for h_val, group in sorted(by_hash.items()):
        if len(group) == 1:
            exact_kept.append(group[0])
        else:
            sorted_g = sorted(group, key=record_priority_key)
            winner = sorted_g[0]
            exact_kept.append(winner)
            for loser in sorted_g[1:]:
                exact_dropped_count += 1
                if len(exact_dropped_examples) < 10:
                    exact_dropped_examples.append(
                        {
                            "kept_pid": winner["pid"],
                            "dropped_pid": loser["pid"],
                            "hash_prompt": h_val,
                        }
                    )

    scopes_report.append(
        {
            "name": "pool_internal_exact",
            "input_n": len(curr_pool),
            "output_n": len(exact_kept),
            "dropped": exact_dropped_count,
            "examples": exact_dropped_examples,
        }
    )

    # -----------------------------------------------------------------------
    # Step 3: Internal MinHash + Exact 5-gram Jaccard Deduplication (§6.3)
    # -----------------------------------------------------------------------
    pool_index = ExactVerifiedMinHashIndex(num_perm=128, bands=32)
    by_pid = {r["pid"]: r for r in exact_kept}
    for r in exact_kept:
        pool_index.insert(r["pid"], normalize_for_hash(r["title"] + " " + r["post"]))

    # Find all pairs >= 0.7 once, then filter for 0.7, 0.8, 0.9 sensitivity check
    pairs_07 = pool_index.find_all_internal_pairs(threshold=0.7)
    pairs_08 = [p for p in pairs_07 if p[2] >= 0.8]
    pairs_09 = [p for p in pairs_07 if p[2] >= 0.9]

    sensitivity_counts = {}
    for thresh, p_list in [(0.7, pairs_07), (0.8, pairs_08), (0.9, pairs_09)]:
        cl = cluster_pairs_union_find(by_pid.keys(), p_list)
        dropped_n = sum(len(m) - 1 for m in cl.values())
        sensitivity_counts[str(thresh)] = {
            "verified_pairs": len(p_list),
            "clusters": len(cl),
            "dropped_prompts": dropped_n,
        }

    # Apply main threshold 0.8
    clusters_08 = cluster_pairs_union_find(by_pid.keys(), pairs_08)
    minhash_dropped_pids = set()
    minhash_examples = []
    for _, members in sorted(clusters_08.items()):
        sorted_m = sorted((by_pid[m] for m in members), key=record_priority_key)
        winner = sorted_m[0]
        for loser in sorted_m[1:]:
            minhash_dropped_pids.add(loser["pid"])
            if len(minhash_examples) < 10:
                jac = next(
                    (
                        p[2]
                        for p in pairs_08
                        if {p[0], p[1]} == {winner["pid"], loser["pid"]}
                    ),
                    0.8,
                )
                minhash_examples.append(
                    {
                        "kept_pid": winner["pid"],
                        "dropped_pid": loser["pid"],
                        "jaccard": jac,
                        "subreddit": winner["subreddit"],
                    }
                )

    final_pool = [
        r for r in sorted(exact_kept, key=lambda x: x["pid"]) if r["pid"] not in minhash_dropped_pids
    ]

    if len(final_pool) < 10000:
        raise RuntimeError(
            f"[02_decontaminate_and_dedup] FATAL: final_pool has {len(final_pool)} < 10,000 prompts!"
        )

    scopes_report.append(
        {
            "name": "pool_internal_minhash_0.8",
            "input_n": len(exact_kept),
            "output_n": len(final_pool),
            "clusters": len(clusters_08),
            "dropped": len(minhash_dropped_pids),
            "sensitivity_thresholds": sensitivity_counts,
            "examples": minhash_examples,
        }
    )

    write_jsonl(DATA_V1_DIR / "canonical_pool.jsonl", final_pool)

    decontam_report = {
        "minhash": cfg["minhash"],
        "initial_canonical_train_n": len(train_candidates),
        "final_canonical_pool_n": len(final_pool),
        "surplus_above_10k": len(final_pool) - 10000,
        "scopes": scopes_report,
    }
    manifests_dir = DATA_V1_DIR / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(manifests_dir / "decontamination_report.json", "w", encoding="utf-8") as f:
        json.dump(decontam_report, f, indent=2, ensure_ascii=False)

    print(
        f"[02_decontaminate_and_dedup] {len(train_candidates)} candidates -> "
        f"{len(final_pool)} final canonical_pool.jsonl "
        f"(surplus above 10k: +{len(final_pool) - 10000}; "
        f"exact dropped: {exact_dropped_count}, MinHash 0.8 dropped: {len(minhash_dropped_pids)})"
    )


if __name__ == "__main__":
    main()
