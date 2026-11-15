"""Shared helpers for the DSAI 4207 training scripts.

The training loop itself is the official TRL 1.16 API, not a custom trainer.
References:
- https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py
- https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py
- https://github.com/huggingface/trl/blob/main/docs/source/peft_integration.md
- https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback

from status_store import update_task

LSYNEED_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = LSYNEED_ROOT.parent

EOS_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dsai4207.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping, got {type(loaded).__name__}: {path}")
    return loaded


def require_keys(config: dict[str, Any], keys: list[str], config_path: Path) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing keys: {', '.join(missing)}")


def resolve_under_shared(raw_path: str, config_path: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    else:
        path = path.resolve()
    return path


def assert_output_allowed(output_dir: Path) -> None:
    """Allow LSYNEED and the shared 03_runs tree. Refuse the frozen data and docs trees."""
    resolved = output_dir.resolve()
    runs_root = (SHARED_ROOT / "03_runs").resolve()
    in_lsyneed = resolved == LSYNEED_ROOT or LSYNEED_ROOT in resolved.parents
    in_runs = resolved == runs_root or runs_root in resolved.parents
    if not in_lsyneed and not in_runs:
        raise ValueError(f"Refusing to write outside LSYNEED or 03_runs: {resolved}")
    if in_lsyneed:
        forbidden = {"01_data", "00_docs", "02_code", "04_results", "05_handoff", "03_runs"}
        if forbidden.intersection(resolved.parts):
            raise ValueError(f"Refusing to write into a shared project tree: {resolved}")


def require_cuda(logger: logging.Logger) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to run a CPU training step.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16. The locked recipe uses bf16.")
    logger.info("cuda device=%s", torch.cuda.get_device_name(0))


def build_lora_config() -> LoraConfig:
    """LoRA settings locked by the course proposal. Constructor matches PEFT's LoraConfig."""
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(LORA_TARGET_MODULES),
    )


def load_tokenizer(model_path: Path) -> PreTrainedTokenizerBase:
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if EOS_TOKEN not in tokenizer.get_vocab() or PAD_TOKEN not in tokenizer.get_vocab():
        raise ValueError("Tokenizer vocabulary is missing <|im_end|> or <|endoftext|>.")
    tokenizer.eos_token = EOS_TOKEN
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.padding_side = "right"
    if tokenizer.eos_token_id != EOS_TOKEN_ID or tokenizer.pad_token_id != PAD_TOKEN_ID:
        raise ValueError(
            f"Unexpected special token ids: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}."
        )
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("pad_token_id and eos_token_id must be different.")
    return tokenizer


def load_causal_lm(model_path: Path) -> PreTrainedModel:
    """Load the local Base weights in bf16.

    TRL 1.16 defaults a path-loaded model to float32 unless dtype is set.
    Transformers 5 uses `dtype`; older builds still accept `torch_dtype`.
    """
    model_path = Path(model_path)
    kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **kwargs)
    if model.dtype != torch.bfloat16:
        raise RuntimeError(f"Model dtype is {model.dtype}, expected bfloat16.")
    model.config.use_cache = False
    return model


def warmup_steps_from_config(config: dict[str, Any]) -> float:
    """Transformers 5 reads a fraction in [0, 1) as a ratio of total steps."""
    value = float(config.get("warmup_steps", 0.03))
    if value < 0 or (value > 1 and not value.is_integer()):
        raise ValueError(f"warmup_steps must be a step count or a ratio in [0, 1), got {value}")
    return value


def assert_finite_loss(train_result: Any, logger: logging.Logger) -> float:
    loss = getattr(train_result, "training_loss", None)
    if loss is None or not math.isfinite(float(loss)):
        raise RuntimeError(f"Training loss is not finite: {loss}")
    logger.info("training_loss=%.6f global_step=%s", float(loss), train_result.global_step)
    return float(loss)


_QWEN25_OFFICIAL_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
_QWEN25_BASE_SYSTEM = "You are a helpful assistant."


def install_sft_training_chat_template(tokenizer: PreTrainedTokenizerBase) -> str:
    """Point the tokenizer at TRL's Qwen2.5 training template and return the original.

    SFTTrainer calls ``get_training_chat_template`` when ``assistant_only_loss`` is
    set. The stock Qwen2.5 template has no ``{% generation %}`` markers, so the
    assistant mask is empty. TRL ships the patched template in
    ``trl/chat_templates/qwen2_5_training.jinja``. This checkpoint matches that
    family except the default system sentence, so the exact-string lookup raises.
    The fallback keeps this checkpoint's sentence and otherwise uses TRL's patch.
    """
    from trl.chat_template_utils import (
        get_training_chat_template,
        has_generation_markers,
        is_chat_template_stop_token_trained,
        qwen2_5_chat_template,
        qwen2_5_training_chat_template,
    )

    original = tokenizer.chat_template
    if not isinstance(original, str):
        raise RuntimeError("Qwen2.5-0.5B is expected to ship one chat template string.")
    if has_generation_markers(original):
        patched = original
    else:
        try:
            patched = get_training_chat_template(tokenizer)
        except ValueError:
            patched = _qwen25_training_template(original, qwen2_5_chat_template, qwen2_5_training_chat_template)
        if patched is None:
            patched = original

    tokenizer.chat_template = patched
    if not has_generation_markers(tokenizer.chat_template):
        tokenizer.chat_template = original
        raise RuntimeError("Training chat template is missing {% generation %} markers.")
    if not is_chat_template_stop_token_trained(tokenizer):
        tokenizer.chat_template = original
        raise RuntimeError("Training chat template leaves <|im_end|> outside the assistant loss mask.")
    return original


def _qwen25_training_template(model_template: str, official: str, official_training: str) -> str:
    if _QWEN25_OFFICIAL_SYSTEM not in official or _QWEN25_OFFICIAL_SYSTEM not in official_training:
        raise RuntimeError("TRL's Qwen2.5 templates no longer contain the expected default system prompt.")
    if official.replace(_QWEN25_OFFICIAL_SYSTEM, _QWEN25_BASE_SYSTEM) != model_template:
        raise RuntimeError(
            "This tokenizer chat template is not TRL's Qwen2.5 template and is not the known "
            "Qwen2.5-0.5B variant. Refusing to invent a training template."
        )
    return official_training.replace(_QWEN25_OFFICIAL_SYSTEM, _QWEN25_BASE_SYSTEM)


def _as_id_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def log_sft_label_preview(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]], logger: logging.Logger) -> None:
    """Print one (token, label) alignment before training. Protocol section 11.3."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
    )
    input_ids = _as_id_list(rendered["input_ids"])
    mask_key = "assistant_masks" if "assistant_masks" in rendered else "assistant_tokens_mask"
    if mask_key not in rendered:
        raise RuntimeError(f"Chat template did not return an assistant mask. Keys: {sorted(rendered)}")
    mask = _as_id_list(rendered[mask_key])
    if len(mask) != len(input_ids):
        raise RuntimeError("Assistant mask length does not match input ids.")
    supervised = [token_id for token_id, flag in zip(input_ids, mask) if flag == 1]
    if not supervised:
        raise RuntimeError("Assistant mask is all zeros, so assistant_only_loss would learn nothing.")
    if len(supervised) == len(input_ids):
        raise RuntimeError("Assistant mask is all ones, so the prompt would be included in the loss.")
    if EOS_TOKEN_ID not in supervised:
        raise RuntimeError("<|im_end|> is not inside the assistant loss mask.")
    preview = []
    for token_id, flag in list(zip(input_ids, mask))[:24]:
        label = token_id if flag == 1 else -100
        preview.append(f"{token_id}:{tokenizer.decode([token_id])!r}:label={label}")
    logger.info("sft label preview: %s", " | ".join(preview))
    logger.info("supervised_tokens=%s final_supervised=%s", len(supervised), supervised[-1])


class MonitorCallback(TrainerCallback):
    """Push each logged step into the monitor snapshot. Failures here must not stop training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        task_id = os.environ.get("MONITOR_TASK_ID")
        if not task_id or not logs or "loss" not in logs:
            return
        step = int(state.global_step)
        if step < 1:
            return
        started = float(os.environ.get("MONITOR_STARTED_EPOCH", "0"))
        elapsed = max(time.time() - started, 0.0)
        try:
            update_task(
                task_id,
                status="running",
                steps_done=step,
                loss=float(logs["loss"]),
                step_seconds=elapsed / step,
                message=f"第 {step} 步，loss {float(logs['loss']):.4f}",
            )
        except Exception:
            return
