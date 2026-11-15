#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_canonicalize.py — Row-level filtering, Condorcet one-pair-per-prompt canonicalization,
and exact Qwen2.5 token length filtering for train and validation splits (§5).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow.parquet as pq
from utils import (
    DATA_V1_DIR,
    RAW_CACHE_DIR,
    SYSTEM_PROMPT,
    QwenTokenizerWrapper,
    clean_subreddit,
    format_user_content,
    get_policy_family,
    get_policy_tier,
    is_deleted_or_empty,
    load_config,
    normalize_for_hash,
    normalize_for_training,
    sha256_str,
    word_count,
    write_jsonl,
)


def process_split(
    parquet_path: Path,
    split_name: str,
    tokenizer: QwenTokenizerWrapper,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = pq.read_table(parquet_path).to_pylist()
    raw_row_count = len(rows)

    drop_reasons = Counter()
    drop_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record_drop(reason: str, info_obj: Dict[str, Any], detail: str = "") -> None:
        drop_reasons[reason] += 1
        if len(drop_examples[reason]) < 5:
            drop_examples[reason].append(
                {
                    "pid": (info_obj or {}).get("id"),
                    "subreddit": (info_obj or {}).get("subreddit"),
                    "detail": detail[:160],
                }
            )

    min_post_w, max_post_w = cfg["filters"]["prompt_words"]
    min_chosen_w, max_chosen_w = cfg["filters"]["chosen_words"]
    min_rej_w, max_rej_w = cfg["filters"]["rejected_words"]

    valid_rows_by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stripped_prefix_count = 0

    for row in rows:
        batch = str(row.get("batch") or "")
        info = row.get("info") or {}

        # 1. Domain filter: exclude CNN/DailyMail and non-Reddit batches (§5.2)
        if "cnndm" in batch.lower() or info.get("site") is not None:
            record_drop("cnndm_or_external_site", info, f"batch={batch}, site={info.get('site')}")
            continue

        pid = info.get("id")
        if not pid or not isinstance(pid, str) or not pid.startswith("t3_"):
            record_drop("invalid_or_missing_pid", info, f"pid={pid}, batch={batch}")
            continue

        raw_sub = clean_subreddit(info.get("subreddit"))
        raw_title = info.get("title")
        raw_post = info.get("post")

        if not raw_sub or is_deleted_or_empty(raw_title) or is_deleted_or_empty(raw_post):
            record_drop("empty_or_deleted_post_title_sub", info, f"sub={raw_sub}")
            continue

        choice = row.get("choice")
        summaries = row.get("summaries") or []
        if choice not in (0, 1) or len(summaries) < 2:
            record_drop("invalid_choice_or_summaries", info, f"choice={choice}")
            continue

        s_chosen = summaries[choice] or {}
        s_rejected = summaries[1 - choice] or {}

        title, _ = normalize_for_training(raw_title, strip_tldr_prefix=False)
        post, _ = normalize_for_training(raw_post, strip_tldr_prefix=False)
        chosen, ch_stripped = normalize_for_training(
            s_chosen.get("text"), strip_tldr_prefix=True
        )
        rejected, rj_stripped = normalize_for_training(
            s_rejected.get("text"), strip_tldr_prefix=True
        )
        if ch_stripped or rj_stripped:
            stripped_prefix_count += 1

        if is_deleted_or_empty(chosen) or is_deleted_or_empty(rejected):
            record_drop("empty_or_deleted_summary", info)
            continue

        # Word count filter (§5.2)
        pw = word_count(post)
        cw = word_count(chosen)
        rw = word_count(rejected)
        if not (min_post_w <= pw <= max_post_w):
            record_drop("post_word_length_out_of_range", info, f"words={pw}")
            continue
        if not (min_chosen_w <= cw <= max_chosen_w):
            record_drop("chosen_word_length_out_of_range", info, f"words={cw}")
            continue
        if not (min_rej_w <= rw <= max_rej_w):
            record_drop("rejected_word_length_out_of_range", info, f"words={rw}")
            continue

        norm_chosen_hash_txt = normalize_for_hash(chosen)
        norm_rejected_hash_txt = normalize_for_hash(rejected)
        if not norm_chosen_hash_txt or not norm_rejected_hash_txt:
            record_drop("empty_after_hash_norm", info)
            continue
        if norm_chosen_hash_txt == norm_rejected_hash_txt:
            record_drop("identical_chosen_and_rejected", info, chosen[:80])
            continue

        chosen_pol = str(s_chosen.get("policy") or "unknown")
        rejected_pol = str(s_rejected.get("policy") or "unknown")
        extra = row.get("extra") or {}
        conf = extra.get("confidence")
        if conf is not None:
            try:
                conf = int(conf)
            except (ValueError, TypeError):
                conf = None

        row_hash = sha256_str(f"{pid}\x00{chosen}\x00{rejected}")
        valid_rows_by_pid[pid].append(
            {
                "pid": pid,
                "subreddit": raw_sub,
                "title": title,
                "post": post,
                "chosen": chosen,
                "rejected": rejected,
                "norm_chosen": norm_chosen_hash_txt,
                "norm_rejected": norm_rejected_hash_txt,
                "chosen_policy": chosen_pol,
                "rejected_policy": rejected_pol,
                "chosen_policy_family": get_policy_family(chosen_pol),
                "rejected_policy_family": get_policy_family(rejected_pol),
                "chosen_policy_tier": get_policy_tier(chosen_pol),
                "confidence": conf,
                "row_hash": row_hash,
                "batch": batch,
            }
        )

    unique_pids_after_basic = len(valid_rows_by_pid)

    # 2. Canonical pair selection per pid (§5.3: Condorcet 0-loss -> confidence -> policy tier -> hash)
    max_prompt_tok = cfg["filters"]["max_prompt_tokens"]
    max_chosen_tok = cfg["filters"]["max_chosen_tokens"]
    max_rej_tok = cfg["filters"]["max_rejected_tokens"]
    max_total_tok = cfg["filters"]["max_total_tokens"]

    canonical_records: List[Dict[str, Any]] = []
    condorcet_winner_count = 0
    confidence_available_count = 0
    policy_family_counter = Counter()

    for pid in sorted(valid_rows_by_pid.keys()):
        cands = valid_rows_by_pid[pid]
        wins = Counter()
        losses = Counter()
        for c in cands:
            wins[c["norm_chosen"]] += 1
            losses[c["norm_rejected"]] += 1

        def sort_key(c: Dict[str, Any]) -> Tuple[Any, ...]:
            ch_norm = c["norm_chosen"]
            zero_loss = 1 if losses[ch_norm] == 0 else 0
            conf_val = c["confidence"]
            high_conf = 1 if (conf_val is None or conf_val >= 3) else 0
            net_wins = wins[ch_norm] - losses[ch_norm]
            conf_score = conf_val if conf_val is not None else -1
            tier = c["chosen_policy_tier"]
            # Negate numeric priorities for ascending sort, then tie-break by row_hash ascending
            return (
                -zero_loss,
                -high_conf,
                -net_wins,
                -conf_score,
                -tier,
                c["row_hash"],
            )

        cands_sorted = sorted(cands, key=sort_key)
        best = cands_sorted[0]

        # Canonicalize post/title if multiple variants exist for same pid (§5.3 rule 4)
        user_content = format_user_content(best["subreddit"], best["title"], best["post"])

        # Tokenize using Qwen2.5-0.5B-Base exact BPE (§5.5)
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        dpo_toks = tokenizer.tokenize_dpo_pair(
            prompt_msgs,
            [{"role": "assistant", "content": best["chosen"]}],
            [{"role": "assistant", "content": best["rejected"]}],
        )
        prompt_tok_len = len(dpo_toks["prompt_input_ids"])
        chosen_tok_len = len(dpo_toks["chosen_input_ids"])
        rej_tok_len = len(dpo_toks["rejected_input_ids"])
        total_tok_len = prompt_tok_len + max(chosen_tok_len, rej_tok_len)

        if prompt_tok_len > max_prompt_tok:
            record_drop("prompt_tokens_exceeded", {"id": pid, "subreddit": best["subreddit"]}, f"tokens={prompt_tok_len}")
            continue
        if chosen_tok_len > max_chosen_tok:
            record_drop("chosen_tokens_exceeded", {"id": pid, "subreddit": best["subreddit"]}, f"tokens={chosen_tok_len}")
            continue
        if rej_tok_len > max_rej_tok:
            record_drop("rejected_tokens_exceeded", {"id": pid, "subreddit": best["subreddit"]}, f"tokens={rej_tok_len}")
            continue
        if total_tok_len > max_total_tok:
            record_drop("total_tokens_exceeded", {"id": pid, "subreddit": best["subreddit"]}, f"tokens={total_tok_len}")
            continue

        ch_losses = losses[best["norm_chosen"]]
        ch_wins = wins[best["norm_chosen"]]
        if ch_losses == 0:
            condorcet_winner_count += 1
        if best["confidence"] is not None:
            confidence_available_count += 1
        policy_family_counter[best["chosen_policy_family"]] += 1

        hash_prompt = sha256_str(normalize_for_hash(best["title"] + " " + best["post"]))
        hash_pair = sha256_str(best["chosen"] + "\x00" + best["rejected"])

        canonical_records.append(
            {
                "pid": pid,
                "subreddit": best["subreddit"],
                "title": best["title"],
                "post": best["post"],
                "user_content": user_content,
                "chosen": best["chosen"],
                "rejected": best["rejected"],
                "chosen_policy": best["chosen_policy"],
                "rejected_policy": best["rejected_policy"],
                "chosen_policy_family": best["chosen_policy_family"],
                "rejected_policy_family": best["rejected_policy_family"],
                "n_candidate_rows": len(cands),
                "chosen_wins_in_pid": ch_wins,
                "chosen_losses_in_pid": ch_losses,
                "selected_confidence": best["confidence"],
                "selected_source_row_hash": best["row_hash"],
                "prompt_tokens": prompt_tok_len,
                "chosen_tokens": chosen_tok_len,
                "rejected_tokens": rej_tok_len,
                "hash_prompt": hash_prompt,
                "hash_pair": hash_pair,
                "source_split": split_name,
            }
        )

    funnel_stats = {
        "split": split_name,
        "raw_rows": raw_row_count,
        "valid_candidate_rows": sum(len(v) for v in valid_rows_by_pid.values()),
        "unique_pids_after_basic_filters": unique_pids_after_basic,
        "unique_pids_after_token_filters": len(canonical_records),
        "condorcet_zero_loss_rate": round(
            condorcet_winner_count / max(1, len(canonical_records)), 4
        ),
        "confidence_available_rate": round(
            confidence_available_count / max(1, len(canonical_records)), 4
        ),
        "tldr_prefix_stripped_rows": stripped_prefix_count,
        "chosen_policy_family_distribution": dict(policy_family_counter),
        "drop_reasons": dict(drop_reasons),
        "drop_examples": dict(drop_examples),
    }
    return canonical_records, funnel_stats


def main() -> None:
    cfg = load_config()
    tokenizer = QwenTokenizerWrapper(RAW_CACHE_DIR / "qwen_tokenizer")

    print("[01_canonicalize] Processing comparisons/train...")
    train_records, train_stats = process_split(
        RAW_CACHE_DIR / "tldr_train.parquet", "train", tokenizer, cfg
    )
    train_out = RAW_CACHE_DIR / "canonical_train_pre_dedup.jsonl"
    write_jsonl(train_out, train_records)
    print(
        f"  train: {train_stats['raw_rows']} raw rows -> "
        f"{train_stats['unique_pids_after_token_filters']} canonical prompts "
        f"(Condorcet 0-loss rate: {train_stats['condorcet_zero_loss_rate']*100:.1f}%)"
    )

    print("[01_canonicalize] Processing comparisons/validation...")
    val_records, val_stats = process_split(
        RAW_CACHE_DIR / "tldr_val.parquet", "validation", tokenizer, cfg
    )
    val_out = RAW_CACHE_DIR / "canonical_val_pre_dedup.jsonl"
    write_jsonl(val_out, val_records)
    print(
        f"  validation: {val_stats['raw_rows']} raw rows -> "
        f"{val_stats['unique_pids_after_token_filters']} canonical Reddit prompts "
        f"(cnndm/site dropped: {val_stats['drop_reasons'].get('cnndm_or_external_site', 0)})"
    )

    manifests_dir = DATA_V1_DIR / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(manifests_dir / "canonicalize_funnel.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_stats, "validation": val_stats}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
