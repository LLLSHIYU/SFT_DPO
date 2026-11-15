#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_format_arms.py — Generate all canonical training arm files (G2, G3, G4_stage1/stage2, G5_stage1/stage2, G3′),
rendered ChatML & token-label alignment inspection examples (aligned byte-for-byte with
official `tokenizer.apply_chat_template`), and the 100-pair blind data quality audit sheet
with separated answer key (§9, §11, §12.4, §14.3, P0-4, P1-5, P2-7).
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from utils import (
    DATA_ENG_DIR,
    DATA_V1_DIR,
    RAW_CACHE_DIR,
    SYSTEM_PROMPT,
    QwenTokenizerWrapper,
    load_config,
    read_jsonl,
    sha256_str,
    write_jsonl,
)

FORBIDDEN_BLIND_AUDIT_FIELDS = {
    "dataset_preferred_hidden_key",
    "dataset_preferred_key",
    "chosen",
    "rejected",
    "chosen_policy",
    "rejected_policy",
    "chosen_policy_family",
    "rejected_policy_family",
    "confidence",
    "selected_confidence",
}


def make_sft_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    P0-4: Freeze `messages` (+ `assistant_only_loss=True`) as the single canonical training field
    for SFT to prevent any ambiguity between `messages` and `prompt`+`completion` in TRL SFTTrainer.
    """
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": r["user_content"]},
    ]
    comp_msgs = [{"role": "assistant", "content": r["chosen"]}]
    return {
        "pid": r["pid"],
        "messages": prompt_msgs + comp_msgs,
        "meta": {
            "primary_training_field": "messages",
            "loss_mode": "assistant_only_loss=True",
            "subreddit": r["subreddit"],
            "chosen_policy": r["chosen_policy"],
            "chosen_policy_family": r["chosen_policy_family"],
            "prompt_tokens": r["prompt_tokens"],
            "chosen_tokens": r["chosen_tokens"],
            "hash_prompt": r["hash_prompt"],
            "hash_pair": r["hash_pair"],
        },
    }


def make_dpo_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    P0-4: Freeze conversational lists `prompt`, `chosen`, `rejected` as the single canonical
    training schema for TRL DPOTrainer.
    """
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": r["user_content"]},
    ]
    return {
        "pid": r["pid"],
        "prompt": prompt_msgs,
        "chosen": [{"role": "assistant", "content": r["chosen"]}],
        "rejected": [{"role": "assistant", "content": r["rejected"]}],
        "meta": {
            "primary_training_fields": ["prompt", "chosen", "rejected"],
            "subreddit": r["subreddit"],
            "chosen_policy": r["chosen_policy"],
            "rejected_policy": r["rejected_policy"],
            "chosen_policy_family": r["chosen_policy_family"],
            "rejected_policy_family": r["rejected_policy_family"],
            "prompt_tokens": r["prompt_tokens"],
            "chosen_tokens": r["chosen_tokens"],
            "rejected_tokens": r["rejected_tokens"],
            "hash_prompt": r["hash_prompt"],
            "hash_pair": r["hash_pair"],
        },
    }


def main() -> None:
    cfg = load_config()
    tokenizer = QwenTokenizerWrapper(RAW_CACHE_DIR / "qwen_tokenizer")
    splits_dir = DATA_V1_DIR / "splits"
    arms_dir = DATA_V1_DIR / "arms"
    arms_dir.mkdir(parents=True, exist_ok=True)

    # P2-7: Remove duplicate short-alias arm files so only canonical stage-named files exist
    for N in cfg["budgets"]:
        for old_alias in [
            f"G4_sft_{N}.jsonl",
            f"G4_dpo_{N}.jsonl",
            f"G5_dpo_{N}.jsonl",
            f"G5_sft_{N}.jsonl",
        ]:
            alias_path = arms_dir / old_alias
            if alias_path.exists():
                alias_path.unlink()

    for N in cfg["budgets"]:
        pool_n = read_jsonl(splits_dir / f"pool_{N}.jsonl")
        part_a = read_jsonl(splits_dir / f"partA_{N}.jsonl")
        part_b = read_jsonl(splits_dir / f"partB_{N}.jsonl")

        g2_sft = [make_sft_record(r) for r in pool_n]
        g3_dpo = [make_dpo_record(r) for r in pool_n]
        g4_s1_sft = [make_sft_record(r) for r in part_a]
        g4_s2_dpo = [make_dpo_record(r) for r in part_b]
        g5_s1_dpo = [make_dpo_record(r) for r in part_b]
        g5_s2_sft = [make_sft_record(r) for r in part_a]

        write_jsonl(arms_dir / f"G2_sft_{N}.jsonl", g2_sft)
        write_jsonl(arms_dir / f"G3_dpo_{N}.jsonl", g3_dpo)
        write_jsonl(arms_dir / f"G4_stage1_sft_{N}.jsonl", g4_s1_sft)
        write_jsonl(arms_dir / f"G4_stage2_dpo_{N}.jsonl", g4_s2_dpo)
        write_jsonl(arms_dir / f"G5_stage1_dpo_{N}.jsonl", g5_s1_dpo)
        write_jsonl(arms_dir / f"G5_stage2_sft_{N}.jsonl", g5_s2_sft)

        if N == 500:
            write_jsonl(arms_dir / "G3prime_dpo_500.jsonl", g3_dpo)

    # Generate rendered_examples.json with token-by-token label alignment (§11.3 & §11.6, P0-4)
    g2_500 = read_jsonl(arms_dir / "G2_sft_500.jsonl")
    g3_500 = read_jsonl(arms_dir / "G3_dpo_500.jsonl")
    test_200 = read_jsonl(DATA_V1_DIR / "test_200_main.jsonl")

    sft_ex = g2_500[0]
    dpo_ex = g3_500[0]
    eval_ex = test_200[0]

    sft_ids, sft_labels = tokenizer.tokenize_sft_with_labels(sft_ex["messages"])
    dpo_toks = tokenizer.tokenize_dpo_pair(
        dpo_ex["prompt"], dpo_ex["chosen"], dpo_ex["rejected"]
    )
    sft_rendered_chatml = tokenizer.apply_chat_template(
        sft_ex["messages"], tokenize=False, add_generation_prompt=False
    )
    dpo_rendered_prompt = tokenizer.apply_chat_template(
        dpo_ex["prompt"], tokenize=False, add_generation_prompt=True
    )
    dpo_rendered_full_chosen = tokenizer.apply_chat_template(
        dpo_ex["prompt"] + dpo_ex["chosen"], tokenize=False, add_generation_prompt=False
    )

    assert sft_rendered_chatml == dpo_rendered_full_chosen, (
        "SFT rendered ChatML != DPO full chosen rendered ChatML!"
    )
    assert sft_ids == dpo_toks["full_chosen_input_ids"], (
        "SFT token IDs != DPO full_chosen_input_ids!"
    )

    alignment_preview = []
    active_indices = [i for i, lbl in enumerate(sft_labels) if lbl != -100]
    boundary_start = max(0, active_indices[0] - 6)
    for idx in range(boundary_start, len(sft_ids)):
        alignment_preview.append(
            {
                "pos": idx,
                "token_id": sft_ids[idx],
                "token_str": tokenizer.decode_token(sft_ids[idx]),
                "label": sft_labels[idx],
            }
        )

    rendered_examples = {
        "formatter_version": cfg["formatter_version"],
        "chat_template_source": "Qwen/Qwen2.5-0.5B-Base official apply_chat_template",
        "training_field_contract": {
            "sft": {"field": "messages", "assistant_only_loss": True},
            "dpo": {"fields": ["prompt", "chosen", "rejected"]},
        },
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "sft_example": {
            "pid": sft_ex["pid"],
            "rendered_chatml": sft_rendered_chatml,
            "byte_identical_to_apply_chat_template": sft_rendered_chatml
            == tokenizer.render_chatml(sft_ex["messages"], add_generation_prompt=False),
            "total_tokens": len(sft_ids),
            "supervised_assistant_tokens": len(active_indices),
            "label_alignment_around_assistant_turn": alignment_preview,
        },
        "dpo_example": {
            "pid": dpo_ex["pid"],
            "rendered_prompt": dpo_rendered_prompt,
            "rendered_chosen": f"{dpo_ex['chosen'][0]['content']}<|im_end|>\n",
            "rendered_rejected": f"{dpo_ex['rejected'][0]['content']}<|im_end|>\n",
            "sft_vs_dpo_chosen_token_ids_identical": sft_ids
            == dpo_toks["full_chosen_input_ids"],
        },
        "eval_prompt_example": {
            "pid": eval_ex["pid"],
            "rendered_generation_prompt": tokenizer.apply_chat_template(
                eval_ex["messages"], tokenize=False, add_generation_prompt=True
            ),
        },
    }

    manifests_dir = DATA_V1_DIR / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with open(manifests_dir / "rendered_examples.json", "w", encoding="utf-8") as f:
        json.dump(rendered_examples, f, indent=2, ensure_ascii=False)

    # P1-5: Generate truly blind 100-pair data label quality audit sheet + separate hidden key file (§14.3)
    pool_10k = read_jsonl(splits_dir / "pool_10000.jsonl")
    audit_sorted = sorted(
        pool_10k, key=lambda r: sha256_str(f"audit_100_v1.2_{r['pid']}")
    )[:100]
    audit_blind_records = []
    audit_key_records = []
    for idx, r in enumerate(audit_sorted):
        swap = int(sha256_str(f"audit_swap_{r['pid']}")[:8], 16) % 2 == 1
        summary_a = r["rejected"] if swap else r["chosen"]
        summary_b = r["chosen"] if swap else r["rejected"]
        dataset_preferred = "B" if swap else "A"

        blind_row = {
            "audit_id": idx,
            "pid": r["pid"],
            "subreddit": r["subreddit"],
            "title": r["title"],
            "post": r["post"],
            "summary_A": summary_a,
            "summary_B": summary_b,
        }
        assert FORBIDDEN_BLIND_AUDIT_FIELDS.isdisjoint(blind_row.keys()), (
            f"Leaked key field in blind audit row: {blind_row.keys()}"
        )
        audit_blind_records.append(blind_row)

        audit_key_records.append(
            {
                "audit_id": idx,
                "pid": r["pid"],
                "dataset_preferred_key": dataset_preferred,
                "chosen_policy": r["chosen_policy"],
                "rejected_policy": r["rejected_policy"],
                "chosen_policy_family": r["chosen_policy_family"],
                "rejected_policy_family": r["rejected_policy_family"],
                "confidence": r["selected_confidence"],
            }
        )

    write_jsonl(manifests_dir / "data_label_audit_100.jsonl", audit_blind_records)
    with open(manifests_dir / "data_label_audit_100_key.json", "w", encoding="utf-8") as f:
        json.dump(audit_key_records, f, indent=2, ensure_ascii=False)

    # Refresh rendered IFEval prompts using official tokenizer & ChatML template
    ifeval_script = DATA_ENG_DIR / "scripts" / "ifeval_harness.py"
    if ifeval_script.exists():
        subprocess.run(
            [sys.executable, str(ifeval_script), "--mode", "render"],
            check=True,
        )

    print(
        "[06_format_arms] Formatted 19 canonical arm JSONL files, "
        "rendered_examples.json, blind data_label_audit_100.jsonl + data_label_audit_100_key.json, "
        "and refreshed ifeval_541_rendered.jsonl."
    )


if __name__ == "__main__":
    main()
