"""Merge a LoRA adapter into the base weights and save a full model.

Reference: PEFT PeftModel.merge_and_unload
https://github.com/huggingface/peft/blob/main/src/peft/tuners/tuners_utils.py
TRL tells two-stage runs to merge before attaching a new adapter:
https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from peft import PeftModel

from common import (
    assert_output_allowed,
    load_causal_lm,
    load_tokenizer,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge one LoRA adapter into a base model.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = args.base.resolve()
    adapter_path = args.adapter.resolve()
    output_dir = args.output.resolve()
    assert_output_allowed(output_dir)
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "merge.log")

    tokenizer = load_tokenizer(base_path)
    base_model = load_causal_lm(base_path)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    weight_file = output_dir / "model.safetensors"
    if not weight_file.is_file() and not any(output_dir.glob("model-*.safetensors")):
        raise RuntimeError(f"Merged weights were not written to {output_dir}")
    report = {
        "base": str(base_path),
        "adapter": str(adapter_path),
        "output": str(output_dir),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    (output_dir / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("merged adapter into %s", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Merge failed: {exc}", file=sys.stderr)
        raise
