"""DPO entry point. Wraps TRL 1.16 DPOTrainer; it does not reimplement the loss.

Reference: https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py
The reference model is a second from_pretrained of the same starting checkpoint.
TRL rejects passing the policy object as its own reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_CACHE = Path(__file__).resolve().parents[1] / "training" / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE / "hub"))

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

from common import (
    assert_finite_loss,
    assert_output_allowed,
    build_lora_config,
    MonitorCallback,
    load_causal_lm,
    load_tokenizer,
    load_yaml,
    warmup_steps_from_config,
    require_cuda,
    require_keys,
    resolve_under_shared,
    setup_logging,
)

REQUIRED = [
    "model_path",
    "ref_model_path",
    "data_path",
    "output_dir",
    "seed",
    "learning_rate",
    "beta",
    "max_length",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DPO job with the locked LoRA recipe.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _check_preference_row(row: dict) -> None:
    for field in ("prompt", "chosen", "rejected"):
        value = row.get(field)
        if not isinstance(value, list) or not value or not isinstance(value[0], dict):
            raise ValueError(f"DPO field {field} must be a non-empty list of role/content dicts.")
        if "role" not in value[0] or "content" not in value[0]:
            raise ValueError(f"DPO field {field} items must contain role and content.")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    require_keys(config, REQUIRED, config_path)

    output_dir = resolve_under_shared(str(config["output_dir"]), config_path)
    assert_output_allowed(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "train.log")

    model_path = resolve_under_shared(str(config["model_path"]), config_path)
    ref_model_path = resolve_under_shared(str(config["ref_model_path"]), config_path)
    data_path = resolve_under_shared(str(config["data_path"]), config_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"DPO data not found: {data_path}")
    learning_rate = float(config["learning_rate"])
    beta = float(config["beta"])
    if learning_rate != 5e-6:
        raise ValueError(f"DPO learning rate must stay 5e-6, got {learning_rate}")
    if beta != 0.1:
        raise ValueError(f"DPO beta must stay 0.1, got {beta}")

    require_cuda(logger)
    tokenizer = load_tokenizer(model_path)
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    missing = [name for name in ("prompt", "chosen", "rejected") if name not in dataset.column_names]
    if missing:
        raise ValueError(f"{data_path} is missing DPO columns: {missing}")
    if len(dataset) < 1:
        raise ValueError(f"{data_path} is empty")
    dataset = dataset.select_columns(["prompt", "chosen", "rejected"])
    _check_preference_row(dataset[0])
    logger.info(
        "dpo row roles prompt=%s chosen=%s rejected=%s",
        [item["role"] for item in dataset[0]["prompt"]],
        [item["role"] for item in dataset[0]["chosen"]],
        [item["role"] for item in dataset[0]["rejected"]],
    )

    max_steps = int(config["max_steps"])
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    training_args = DPOConfig(
        output_dir=str(output_dir),
        seed=int(config["seed"]),
        learning_rate=learning_rate,
        beta=beta,
        loss_type=["sigmoid"],
        num_train_epochs=1,
        max_steps=max_steps,
        max_length=int(config["max_length"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        # Transformers 5: a warmup_steps value in [0, 1) is a ratio of total steps.
        # Smoke configs set 0 so the single step is not stuck at learning rate 0.
        warmup_steps=warmup_steps_from_config(config),
        max_grad_norm=1.0,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        disable_dropout=True,
        dataset_num_proc=None,
    )
    policy = load_causal_lm(model_path)
    reference = load_causal_lm(ref_model_path)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    if policy is reference:
        raise RuntimeError("Policy and reference must be two separately loaded models.")

    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(),
        callbacks=[MonitorCallback()],
    )
    result = trainer.train()
    if result.global_step < 1:
        raise RuntimeError(f"Training finished without a step. Expected {max_steps}.")
    if result.global_step != max_steps:
        logger.warning("global_step=%s expected=%s", result.global_step, max_steps)
    loss = assert_finite_loss(result, logger)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "task": "dpo",
        "global_step": result.global_step,
        "expected_steps": max_steps,
        "training_loss": loss,
        "model_path": str(model_path),
        "ref_model_path": str(ref_model_path),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("saved adapter to %s", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DPO failed: {exc}", file=sys.stderr)
        raise
