"""Shared I/O, model loading, and tokenizer utilities for evaluation scripts."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DATA = ROOT / "SUMMARY_DATA"
IF_DATA = ROOT / "IF_DATA"
# Summary exams and TL;DR prompts live in SUMMARY_DATA. IFEval lives in IF_DATA.
DATA_V1 = SUMMARY_DATA
RUNS_DIR = ROOT / "03_runs"
RESULTS_DIR = ROOT / "04_results"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B"
SYSTEM_PROMPT = (
    "You are a helpful assistant who is an expert at summarizing content into "
    "TL;DR sentences. Summarize the following text"
)
EOS_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def item_id(row: dict[str, Any], index: int) -> str:
    for key in ("pid", "id", "key", "order"):
        if key in row:
            return str(row[key])
    return str(index)


def tldr_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(row.get("messages"), list):
        return row["messages"]
    if row.get("user_content"):
        user = str(row["user_content"])
    else:
        user = (
            f"Subreddit: r/{row.get('subreddit', 'unknown')}\n"
            f"Title: {row.get('title', '')}\n\n{row.get('post', '')}"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def configure_tokenizer(tokenizer: Any) -> Any:
    """Apply protocol token IDs and fail fast if tokenizer assets disagree."""
    if tokenizer.convert_tokens_to_ids(EOS_TOKEN) != EOS_TOKEN_ID:
        raise ValueError(
            f"Tokenizer {tokenizer.name_or_path} has unexpected {EOS_TOKEN} ID; "
            f"expected {EOS_TOKEN_ID}. Use the frozen project tokenizer revision."
        )
    if tokenizer.convert_tokens_to_ids(PAD_TOKEN) != PAD_TOKEN_ID:
        raise ValueError(
            f"Tokenizer {tokenizer.name_or_path} has unexpected {PAD_TOKEN} ID; "
            f"expected {PAD_TOKEN_ID}."
        )
    tokenizer.eos_token = EOS_TOKEN
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    base_model: str = DEFAULT_BASE_MODEL,
    adapter: str | None = None,
    dtype: str = "auto",
) -> tuple[Any, Any, Any]:
    """Load a full HF model or a PEFT adapter on the shared base checkpoint."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model evaluation requires torch and transformers; install "
            "PAN_Evaluation/requirements-eval.txt first."
        ) from exc

    tokenizer_source = base_model if adapter or not model_name_or_path else model_name_or_path
    tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    )
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    if adapter:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto"
        )
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT adapter loading requires the peft package") from exc
        model = PeftModel.from_pretrained(model, adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto"
        )
    model.eval()
    return model, tokenizer, torch


def word_count(text: str) -> int:
    return len(text.split())


def logistic_fit_at_zero(y: list[float], x: list[float], max_iter: int = 80) -> tuple[float, float]:
    """Fit fractional-logit outcome ~ log-length difference and return p(x=0), slope."""
    if len(y) != len(x) or not y:
        raise ValueError("The outcome and length arrays must have equal non-zero length")
    # Ridge-stabilized Newton fit for a two-parameter fractional-logit model.
    # Fractional outcomes encode both order-swapped calls (0.5 means disagreement/tie).
    b0, b1 = 0.0, 0.0
    ridge = 1e-6
    for _ in range(max_iter):
        probs = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, b0 + b1 * xi)))) for xi in x]
        weights = [max(p * (1.0 - p), 1e-8) for p in probs]
        h00 = sum(weights) + ridge
        h01 = sum(w * xi for w, xi in zip(weights, x))
        h11 = sum(w * xi * xi for w, xi in zip(weights, x)) + ridge
        g0 = sum(yi - p for yi, p in zip(y, probs)) - ridge * b0
        g1 = sum(xi * (yi - p) for yi, p, xi in zip(y, probs, x)) - ridge * b1
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            raise ValueError("Cannot fit length adjustment: covariate has no variation")
        delta0 = (g0 * h11 - g1 * h01) / det
        delta1 = (g1 * h00 - g0 * h01) / det
        nb0 = max(-30.0, min(30.0, b0 + delta0))
        nb1 = max(-30.0, min(30.0, b1 + delta1))
        if not math.isfinite(nb0 + nb1):
            raise ValueError("Logistic fit diverged; inspect judge labels and lengths")
        if max(abs(nb0 - b0), abs(nb1 - b1)) < 1e-8:
            b0, b1 = nb0, nb1
            break
        b0, b1 = nb0, nb1
    p0 = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, b0))))
    return p0, b1
