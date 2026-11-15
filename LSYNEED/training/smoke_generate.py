"""Greedy generation smoke test for the eos/pad setup.

This does not score the model. It checks that generation is asked to stop on
<|im_end|> (151645) and <|endoftext|> (151643), using the first rows of the
held-out TL;DR test file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from common import (
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    assert_output_allowed,
    load_causal_lm,
    load_tokenizer,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate three TL;DR smoke samples.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def load_prompts(path: Path, count: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= count:
                break
    if len(rows) < count:
        raise ValueError(f"Need {count} test rows, found {len(rows)} in {path}")
    return rows


def main() -> int:
    args = parse_args()
    output_dir = args.output.resolve()
    assert_output_allowed(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "generate.log")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tokenizer = load_tokenizer(args.model.resolve())
    tokenizer.padding_side = "left"
    model = load_causal_lm(args.model.resolve())
    model.config.use_cache = True
    model.to("cuda")
    model.eval()

    rows = load_prompts(args.test_file.resolve(), args.num_samples)
    prompts = []
    for row in rows:
        messages = row["messages"]
        prompts.append([item for item in messages if item["role"] != "assistant"])

    encoded = tokenizer.apply_chat_template(
        prompts,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=[EOS_TOKEN_ID, PAD_TOKEN_ID],
            pad_token_id=PAD_TOKEN_ID,
        )

    reports = []
    stopped = 0
    for index, row in enumerate(rows):
        new_ids = generated[index, prompt_width:].tolist()
        while new_ids and new_ids[-1] == PAD_TOKEN_ID:
            new_ids.pop()
        hit_eos = EOS_TOKEN_ID in new_ids or PAD_TOKEN_ID in new_ids
        if hit_eos:
            stopped += 1
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        reports.append(
            {
                "pid": row.get("pid"),
                "order": row.get("order"),
                "new_token_count": len(new_ids),
                "stopped_on_eos": hit_eos,
                "text": text,
            }
        )
        logger.info("pid=%s stopped_on_eos=%s new_tokens=%s", row.get("pid"), hit_eos, len(new_ids))
    summary = {
        "samples": len(reports),
        "stopped_on_eos": stopped,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "reports": reports,
    }
    (output_dir / "generate_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if stopped != len(reports):
        raise RuntimeError(
            f"Only {stopped}/{len(reports)} samples stopped on eos within {args.max_new_tokens} tokens."
        )
    logger.info("all %s samples stopped on eos", stopped)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Generation smoke failed: {exc}", file=sys.stderr)
        raise
