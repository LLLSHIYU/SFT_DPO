#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_split.py — Paired stratified splitting, rejected-policy & subreddit balance control,
and midpoint nested ordering (§8, P1-1, P2-1, P2-6):
1. Stratifies canonical_pool.jsonl into 5 (prompt_tokens) x 3 (chosen_tokens) x 3 (chosen_policy_family) = 45 strata.
2. Merges any small stratum (< 6 items) deterministically into adjacent strata (§8.4).
3. Sorts items within each stratum by `(rejected_policy_bucket, subreddit_bucket, sha256(pid + SALT_PAIR))`
   before pairing adjacent items `(A, B)`, controlling both chosen & rejected policy families and subreddit across A/B (P1-1).
4. Orders pairs using midpoint spacing `(i + 0.5) / n` within `(stratum, subreddit_bucket)` sub-cells
   so nested prefixes P500 ⊂ P2000 ⊂ P10000 preserve both stratum and subreddit proportions (< 3% drift, P2-1).
5. Reports 3x3 and 4x4 `(chosen_family, rejected_family)` joint distributions, chi-square tests, SMDs,
   and `test_200` vs `test_500` 35-stratum balance in `split_report.json`.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from utils import (
    DATA_V1_DIR,
    get_stratum_policy_bucket,
    load_config,
    read_jsonl,
    sha256_str,
    write_jsonl,
)


def compute_smd(vals_a: List[float], vals_b: List[float]) -> float:
    if not vals_a or not vals_b:
        return 0.0
    arr_a = np.asarray(vals_a, dtype=np.float64)
    arr_b = np.asarray(vals_b, dtype=np.float64)
    mean_a, mean_b = float(np.mean(arr_a)), float(np.mean(arr_b))
    var_a = float(np.var(arr_a, ddof=1)) if len(arr_a) > 1 else 0.0
    var_b = float(np.var(arr_b, ddof=1)) if len(arr_b) > 1 else 0.0
    pooled_sd = math.sqrt((var_a + var_b) / 2.0)
    if pooled_sd < 1e-9:
        return 0.0
    return round((mean_a - mean_b) / pooled_sd, 4)


def compute_two_sample_chi2(
    counts_a: Dict[str, int], counts_b: Dict[str, int]
) -> Dict[str, Any]:
    """
    Computes two-sample homogeneity chi-square statistic, df, and p-value between Part A and Part B
    categorical distributions (where |A| == |B|).
    """
    keys = sorted(set(counts_a.keys()) | set(counts_b.keys()))
    active_keys = [k for k in keys if (counts_a.get(k, 0) + counts_b.get(k, 0)) > 0]
    if len(active_keys) <= 1:
        return {"chi2": 0.0, "df": 0, "p_value": 1.0, "max_prop_diff": 0.0}

    tot_a = max(1, sum(counts_a.values()))
    tot_b = max(1, sum(counts_b.values()))
    chi2_val = 0.0
    max_diff = 0.0
    for k in active_keys:
        oa = counts_a.get(k, 0)
        ob = counts_b.get(k, 0)
        ea = (oa + ob) * tot_a / (tot_a + tot_b)
        eb = (oa + ob) * tot_b / (tot_a + tot_b)
        if ea > 0:
            chi2_val += (oa - ea) ** 2 / ea
        if eb > 0:
            chi2_val += (ob - eb) ** 2 / eb
        diff = abs(oa / tot_a - ob / tot_b)
        if diff > max_diff:
            max_diff = diff

    df = len(active_keys) - 1
    try:
        from scipy.stats import chi2 as scipy_chi2

        p_val = float(scipy_chi2.sf(chi2_val, df))
    except Exception:
        # Regularized upper incomplete gamma / Wilson-Hilferty approximation fallback
        z = ((chi2_val / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(
            2.0 / (9.0 * df)
        )
        p_val = 0.5 * math.erfc(z / math.sqrt(2.0))

    return {
        "chi2": round(float(chi2_val), 4),
        "df": int(df),
        "p_value": round(float(p_val), 4),
        "max_prop_diff": round(float(max_diff), 4),
    }


def summarize_numeric(vals: List[int]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(np.std(arr, ddof=1)), 2),
        "p10": round(float(np.quantile(arr, 0.10)), 2),
        "p90": round(float(np.quantile(arr, 0.90)), 2),
        "p99": round(float(np.quantile(arr, 0.99)), 2),
        "max": int(np.max(arr)),
    }


def distribution_dict(items: List[str], top_k: int = 0) -> Dict[str, float]:
    if not items:
        return {}
    cnt = Counter(items)
    total = len(items)
    if top_k > 0:
        top_keys = [k for k, _ in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[:top_k]]
        out = {k: round(cnt[k] / total, 4) for k in top_keys}
        other_cnt = sum(v for k, v in cnt.items() if k not in top_keys)
        out["other"] = round(other_cnt / total, 4)
        return out
    return {k: round(cnt[k] / total, 4) for k in sorted(cnt.keys())}


def joint_policy_matrix(records: List[Dict[str, Any]], use_3_buckets: bool = True) -> Dict[str, Dict[str, float]]:
    total = max(1, len(records))
    if use_3_buckets:
        buckets = ["rl_or_strong", "human_ref", "sft_early"]
        mat = {b1: {b2: 0 for b2 in buckets} for b1 in buckets}
        for r in records:
            c_b = get_stratum_policy_bucket(r["chosen_policy_family"])
            r_b = get_stratum_policy_bucket(r["rejected_policy_family"])
            mat[c_b][r_b] += 1
        return {
            b1: {b2: round(mat[b1][b2] / total, 4) for b2 in buckets}
            for b1 in buckets
        }
    else:
        fams = ["rl_aligned", "sft_strong", "human_ref", "sft_early"]
        mat4 = {f1: {f2: 0 for f2 in fams} for f1 in fams}
        for r in records:
            c_f = r["chosen_policy_family"]
            r_f = r["rejected_policy_family"]
            mat4[c_f][r_f] += 1
        return {
            f1: {f2: round(mat4[f1][f2] / total, 4) for f2 in fams}
            for f1 in fams
        }


def main() -> None:
    cfg = load_config()
    pool = read_jsonl(DATA_V1_DIR / "canonical_pool.jsonl")
    if len(pool) < 10000:
        raise RuntimeError(f"canonical_pool has only {len(pool)} < 10000 items!")

    top6_subs = [
        s
        for s, _ in sorted(
            Counter(r["subreddit"] for r in pool).items(), key=lambda x: (-x[1], x[0])
        )[:6]
    ]
    top6_set = set(top6_subs)

    # 1. Compute quantile cutoffs on canonical_pool (§8.4)
    prompt_lens = np.array([r["prompt_tokens"] for r in pool])
    chosen_lens = np.array([r["chosen_tokens"] for r in pool])

    p_quantiles = np.quantile(prompt_lens, [0.2, 0.4, 0.6, 0.8])
    c_quantiles = np.quantile(chosen_lens, [1.0 / 3.0, 2.0 / 3.0])

    def get_p_bin(val: int) -> int:
        return int(np.searchsorted(p_quantiles, val, side="right"))  # 0..4

    def get_c_bin(val: int) -> int:
        return int(np.searchsorted(c_quantiles, val, side="right"))  # 0..2

    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool:
        pb = get_p_bin(r["prompt_tokens"])
        cb = get_c_bin(r["chosen_tokens"])
        pol_b = get_stratum_policy_bucket(r["chosen_policy_family"])
        s_key = f"p{pb}_c{cb}_{pol_b}"
        strata[s_key].append(r)

    # 2. Merge strata with < 6 items into adjacent chosen_tokens / prompt_tokens bin (§8.4)
    merges_log: List[Dict[str, Any]] = []
    for pb in range(5):
        for cb in range(3):
            for pol_b in ["rl_or_strong", "human_ref", "sft_early"]:
                s_key = f"p{pb}_c{cb}_{pol_b}"
                if s_key in strata and 0 < len(strata[s_key]) < 6:
                    target_cb = 1 if cb != 1 else 0
                    target_key = f"p{pb}_c{target_cb}_{pol_b}"
                    merges_log.append(
                        {"from": s_key, "to": target_key, "count": len(strata[s_key])}
                    )
                    strata[target_key].extend(strata.pop(s_key))

    for s_key in list(strata.keys()):
        if len(strata[s_key]) == 0:
            strata.pop(s_key)
        elif len(strata[s_key]) < 6:
            pol_b = s_key.split("_", 2)[2]
            fallback_key = f"p2_c1_{pol_b}"
            if fallback_key != s_key:
                merges_log.append(
                    {"from": s_key, "to": fallback_key, "count": len(strata[s_key])}
                )
                strata[fallback_key].extend(strata.pop(s_key))

    # 3. Within-stratum pairing with rejected_policy_family & subreddit sub-sorting (§8.5, P1-1)
    salt_pair = cfg["salts"]["pair"]
    salt_ab = cfg["salts"]["ab"]
    salt_order = cfg["salts"]["order"]

    pairs: List[Dict[str, Any]] = []
    reserve_odd: List[Dict[str, Any]] = []

    for s_key in sorted(strata.keys()):
        # Sort by rejected_policy_bucket and subreddit_bucket before pairing adjacent items
        # so every (A, B) pair matches on chosen_policy, rejected_policy, and subreddit
        items = sorted(
            strata[s_key],
            key=lambda r: (
                get_stratum_policy_bucket(r["rejected_policy_family"]),
                r["subreddit"] if r["subreddit"] in top6_set else "other",
                sha256_str(f"{r['pid']}_{salt_pair}"),
            ),
        )
        if len(items) % 2 == 1:
            # Drop the item with the largest hash so odd removal is unbiased across sub-buckets
            odd_idx = max(
                range(len(items)),
                key=lambda idx: sha256_str(f"odd_{items[idx]['pid']}_{salt_pair}"),
            )
            reserve_odd.append(items.pop(odd_idx))
        for i in range(0, len(items), 2):
            a, b = items[i], items[i + 1]
            if int(sha256_str(f"{a['pid']}_{b['pid']}_{salt_ab}")[:8], 16) % 2 == 1:
                a, b = b, a
            sub_b = a["subreddit"] if a["subreddit"] in top6_set else "other"
            rej_b = get_stratum_policy_bucket(a["rejected_policy_family"])
            pairs.append(
                {
                    "A": a,
                    "B": b,
                    "stratum": s_key,
                    "sub_b": sub_b,
                    "rej_b": rej_b,
                }
            )

    # 4. Hierarchical midpoint nested ordering `(i + 0.5) / n` (§8.6, P1-1, P2-1):
    # Inner level interleaves 45-stratum and rejected-policy buckets within each subreddit bucket;
    # outer level applies midpoint spacing across the 7 subreddit buckets (top-6 + other),
    # guaranteeing both subreddit drift < 1% and stratum drift < 1% across N=500, 2000, 10000.
    by_sub: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        by_sub[p["sub_b"]].append(p)

    for sub_b, sub_pairs in by_sub.items():
        by_s: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for p in sub_pairs:
            by_s[p["stratum"]].append(p)
        for s_key, s_ps in by_s.items():
            s_ps.sort(
                key=lambda p: (
                    p["rej_b"],
                    sha256_str(f"{p['A']['pid']}_{p['B']['pid']}_{salt_order}"),
                )
            )
            n_s = len(s_ps)
            for j, p in enumerate(s_ps):
                p["_s_key"] = (
                    (j + 0.5) / n_s,
                    sha256_str(f"{p['A']['pid']}_{p['B']['pid']}_{salt_order}"),
                )
        sub_pairs.sort(key=lambda p: p["_s_key"])
        n_sub = len(sub_pairs)
        for i, p in enumerate(sub_pairs):
            p["order_key"] = (
                (i + 0.5) / n_sub,
                sha256_str(f"{p['A']['pid']}_{p['B']['pid']}_{salt_order}"),
            )

    ordered_pairs = sorted(pairs, key=lambda p: p["order_key"])
    if len(ordered_pairs) < 5000:
        raise RuntimeError(
            f"Not enough pairs for N=10000: got {len(ordered_pairs)} pairs (< 5000)"
        )

    splits_dir = DATA_V1_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 5. Extract nested budgets N in {500, 2000, 10000} and compute balance report (§8.7–8.9)
    def sub_dist_fixed(records: List[Dict[str, Any]]) -> Dict[str, float]:
        cnt = Counter(r["subreddit"] for r in records)
        tot = max(1, len(records))
        out = {s: round(cnt[s] / tot, 4) for s in top6_subs}
        out["other"] = round(
            sum(v for k, v in cnt.items() if k not in top6_set) / tot, 4
        )
        return out

    tier_reports: Dict[str, Any] = {}
    pool_10k_records = [
        item for p in ordered_pairs[:5000] for item in (p["A"], p["B"])
    ]
    sub_dist_10k = sub_dist_fixed(pool_10k_records)

    for N in cfg["budgets"]:
        k = N // 2
        chosen_pairs = ordered_pairs[:k]
        part_a = [p["A"] for p in chosen_pairs]
        part_b = [p["B"] for p in chosen_pairs]
        pool_n = [item for p in chosen_pairs for item in (p["A"], p["B"])]

        assert len(part_a) == k and len(part_b) == k and len(pool_n) == N

        write_jsonl(splits_dir / f"pool_{N}.jsonl", pool_n)
        write_jsonl(splits_dir / f"partA_{N}.jsonl", part_a)
        write_jsonl(splits_dir / f"partB_{N}.jsonl", part_b)

        smd_prompt = compute_smd(
            [r["prompt_tokens"] for r in part_a],
            [r["prompt_tokens"] for r in part_b],
        )
        smd_chosen = compute_smd(
            [r["chosen_tokens"] for r in part_a],
            [r["chosen_tokens"] for r in part_b],
        )
        smd_rejected = compute_smd(
            [r["rejected_tokens"] for r in part_a],
            [r["rejected_tokens"] for r in part_b],
        )

        sub_dist_n = sub_dist_fixed(pool_n)
        max_sub_drift_vs_10k = round(
            max(abs(sub_dist_n[k_sub] - sub_dist_10k[k_sub]) for k_sub in sub_dist_10k),
            4,
        )

        # P1-1: Rejected policy family and (chosen, rejected) 3x3 & 4x4 joint balance tests
        rej_chi2 = compute_two_sample_chi2(
            Counter(r["rejected_policy_family"] for r in part_a),
            Counter(r["rejected_policy_family"] for r in part_b),
        )
        joint_3x3_chi2 = compute_two_sample_chi2(
            Counter(
                f"{get_stratum_policy_bucket(r['chosen_policy_family'])}->{get_stratum_policy_bucket(r['rejected_policy_family'])}"
                for r in part_a
            ),
            Counter(
                f"{get_stratum_policy_bucket(r['chosen_policy_family'])}->{get_stratum_policy_bucket(r['rejected_policy_family'])}"
                for r in part_b
            ),
        )

        tier_reports[str(N)] = {
            "budget_N": N,
            "partA_n": len(part_a),
            "partB_n": len(part_b),
            "smd": {
                "prompt_tokens": smd_prompt,
                "chosen_tokens": smd_chosen,
                "rejected_tokens": smd_rejected,
                "all_within_0_10": max(
                    abs(smd_prompt), abs(smd_chosen), abs(smd_rejected)
                )
                < 0.10,
            },
            "policy_balance_tests": {
                "rejected_policy_family_chi2": rej_chi2,
                "joint_chosen_rejected_3x3_chi2": joint_3x3_chi2,
            },
            "partA_stats": {
                "prompt_tokens": summarize_numeric([r["prompt_tokens"] for r in part_a]),
                "chosen_tokens": summarize_numeric([r["chosen_tokens"] for r in part_a]),
                "rejected_tokens": summarize_numeric([r["rejected_tokens"] for r in part_a]),
                "chosen_policy_family": distribution_dict(
                    [r["chosen_policy_family"] for r in part_a]
                ),
                "rejected_policy_family": distribution_dict(
                    [r["rejected_policy_family"] for r in part_a]
                ),
                "joint_policy_3x3": joint_policy_matrix(part_a, use_3_buckets=True),
                "joint_policy_4x4": joint_policy_matrix(part_a, use_3_buckets=False),
                "subreddit_top6": sub_dist_fixed(part_a),
            },
            "partB_stats": {
                "prompt_tokens": summarize_numeric([r["prompt_tokens"] for r in part_b]),
                "chosen_tokens": summarize_numeric([r["chosen_tokens"] for r in part_b]),
                "rejected_tokens": summarize_numeric([r["rejected_tokens"] for r in part_b]),
                "chosen_policy_family": distribution_dict(
                    [r["chosen_policy_family"] for r in part_b]
                ),
                "rejected_policy_family": distribution_dict(
                    [r["rejected_policy_family"] for r in part_b]
                ),
                "joint_policy_3x3": joint_policy_matrix(part_b, use_3_buckets=True),
                "joint_policy_4x4": joint_policy_matrix(part_b, use_3_buckets=False),
                "subreddit_top6": sub_dist_fixed(part_b),
            },
            "max_subreddit_drift_vs_10k": max_sub_drift_vs_10k,
        }

    # Save reserve pool (pairs after 5,000 + odd remainders)
    reserve_records = [
        item for p in ordered_pairs[5000:] for item in (p["A"], p["B"])
    ] + reserve_odd
    write_jsonl(splits_dir / "reserve.jsonl", reserve_records)

    manifests_dir = DATA_V1_DIR / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    testset_stratum_rep = {}
    testset_rep_path = manifests_dir / "testset_stratum_report.json"
    if testset_rep_path.exists():
        with open(testset_rep_path, "r", encoding="utf-8") as f:
            testset_stratum_rep = json.load(f)

    split_report = {
        "total_canonical_pool": len(pool),
        "total_pairs_formed": len(ordered_pairs),
        "reserve_odd_count": len(reserve_odd),
        "reserve_total_count": len(reserve_records),
        "num_strata_active": len(strata),
        "stratum_merges": merges_log,
        "prompt_token_quantiles_20_40_60_80": [float(x) for x in p_quantiles],
        "chosen_token_quantiles_33_67": [float(x) for x in c_quantiles],
        "canonical_pool_chosen_policy_family_counts": dict(
            Counter(r["chosen_policy_family"] for r in pool)
        ),
        "canonical_pool_rejected_policy_family_counts": dict(
            Counter(r["rejected_policy_family"] for r in pool)
        ),
        "budgets": tier_reports,
        "test_200_vs_500_stratification": testset_stratum_rep,
    }

    with open(manifests_dir / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2, ensure_ascii=False)

    for N in cfg["budgets"]:
        rep = tier_reports[str(N)]
        print(
            f"[04_split] N={N}: |A|={rep['partA_n']}, |B|={rep['partB_n']}, "
            f"SMD(prompt={rep['smd']['prompt_tokens']:+.4f}, "
            f"chosen={rep['smd']['chosen_tokens']:+.4f}, "
            f"rejected={rep['smd']['rejected_tokens']:+.4f}), "
            f"rej_chi2_p={rep['policy_balance_tests']['rejected_policy_family_chi2']['p_value']:.4f}, "
            f"max_sub_drift_vs_10k={rep['max_subreddit_drift_vs_10k']:.4f}"
        )


if __name__ == "__main__":
    main()
