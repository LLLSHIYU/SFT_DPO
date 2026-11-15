"""Likelihood-score the frozen MMLU-500 and ARC-Challenge-300 subsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import DATA_V1, DEFAULT_BASE_MODEL, item_id, load_model_and_tokenizer, read_jsonl, write_csv
from eval_monitor import report


def choices_for(row: dict[str, Any]) -> tuple[list[str], list[str], int]:
    choices = row["choices"]
    if isinstance(choices, dict):
        texts = list(choices["text"])
        labels = list(choices.get("label", [chr(65 + i) for i in range(len(texts))]))
    else:
        texts = [str(value) for value in choices]
        labels = [str(x) for x in row.get("choice_labels", [chr(65 + i) for i in range(len(texts))])]
    if len(texts) != len(labels) or not texts:
        raise ValueError(f"Invalid choices for {item_id(row, 0)}")
    answer = row.get("answer_index")
    if answer is None:
        answer_label = str(row.get("answerKey", row.get("answer_letter", ""))).upper()
        if answer_label not in labels:
            raise ValueError(f"Unknown gold answer {answer_label!r} for {item_id(row, 0)}")
        answer = labels.index(answer_label)
    answer = int(answer)
    if not 0 <= answer < len(texts):
        raise ValueError(f"Gold answer index out of range for {item_id(row, 0)}")
    return texts, labels, answer


def benchmark_prompt(row: dict[str, Any], texts: list[str], labels: list[str], tokenizer: Any) -> list[int]:
    question = str(row.get("question", ""))
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    content = f"{question}\n{options}\nAnswer:"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def continuation_score(model: Any, torch: Any, prompt_ids: list[int], completion_ids: list[int], device: str) -> float:
    if not completion_ids:
        raise ValueError("Candidate continuation tokenized to an empty sequence")
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]
        start = len(prompt_ids) - 1
        token_logits = logits[start: start + len(completion_ids)].float()
        targets = torch.tensor(completion_ids, dtype=torch.long, device=token_logits.device)
        log_probs = torch.log_softmax(token_logits, dim=-1)
        return float(log_probs.gather(1, targets[:, None]).sum().item())


def single_token_labels(tokenizer: Any, labels: list[str], item: str) -> list[list[int]]:
    """Prefer a leading space. Digits such as 1–4 are one token only without that space."""
    for prefix in (" ", ""):
        encoded = [tokenizer.encode(f"{prefix}{label}", add_special_tokens=False) for label in labels]
        if encoded and all(len(ids) == 1 for ids in encoded):
            return encoded
    rendered = [tokenizer.encode(f" {label}", add_special_tokens=False) for label in labels]
    raise ValueError(
        f"Answer labels for {item} are not single tokens: "
        + ", ".join(f"{label}={ids}" for label, ids in zip(labels, rendered))
    )


def score_item(row: dict[str, Any], benchmark: str, model: Any, tokenizer: Any, torch: Any, device: str, index: int) -> dict[str, Any]:
    texts, labels, gold_index = choices_for(row)
    prompt_ids = list(benchmark_prompt(row, texts, labels, tokenizer))
    label_scores: list[float] = []
    text_scores: list[float] = []
    text_token_counts: list[int] = []
    label_token_ids = single_token_labels(tokenizer, labels, f"{benchmark} item {item_id(row, index)}")
    for label_ids, text in zip(label_token_ids, texts):
        label_scores.append(continuation_score(model, torch, prompt_ids, label_ids, device))
        text_ids = tokenizer.encode(f" {text}", add_special_tokens=False)
        text_token_counts.append(len(text_ids))
        text_scores.append(continuation_score(model, torch, prompt_ids, text_ids, device) / len(text_ids))

    letter_prediction = max(range(len(labels)), key=label_scores.__getitem__)
    text_prediction = max(range(len(labels)), key=text_scores.__getitem__)
    gold_label = labels[gold_index]
    return {
        "item_id": item_id(row, index),
        "benchmark": benchmark,
        "order": row.get("order", index),
        "subject": row.get("subject", ""),
        "gold_index": gold_index,
        "gold_label": gold_label,
        "letter_prediction": labels[letter_prediction],
        "letter_correct": int(letter_prediction == gold_index),
        "letter_logprob": label_scores[gold_index],
        "letter_scores_json": json.dumps(dict(zip(labels, label_scores))),
        "text_prediction": labels[text_prediction],
        "text_correct": int(text_prediction == gold_index),
        "text_norm_logprob": text_scores[gold_index],
        "text_scores_json": json.dumps(dict(zip(labels, text_scores))),
        "choice_token_counts_json": json.dumps(dict(zip(labels, text_token_counts))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("mmlu", "arc"), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter")
    parser.add_argument("--arm", default="model")
    parser.add_argument("--budget", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--device", help="Optional input device override")
    args = parser.parse_args()

    data_path = args.data or (
        DATA_V1 / "eval_benchmarks" / ("mmlu_500.jsonl" if args.benchmark == "mmlu" else "arc_challenge_300.jsonl")
    )
    model, tokenizer, torch = load_model_and_tokenizer(
        args.model, base_model=args.base_model, adapter=args.adapter, dtype=args.dtype
    )
    device = args.device or str(next(model.parameters()).device)
    rows = read_jsonl(data_path)
    scored = []
    report(0, len(rows), "正在算选项概率")
    for index, row in enumerate(rows, 1):
        scored.append(score_item(row, args.benchmark, model, tokenizer, torch, device, index - 1))
        if index == len(rows) or index % 25 == 0:
            report(index, len(rows), "正在算选项概率")
    model_id = args.adapter or args.model
    for row in scored:
        row.update({"arm": args.arm, "budget": args.budget, "seed": args.seed, "model": model_id, "data_version": "v1.0"})
    output = args.output.resolve()
    write_csv(output, scored)
    summary_path = output.with_suffix(".summary.json")
    summary = {
        "benchmark": args.benchmark,
        "num_items": len(scored),
        "arm": args.arm,
        "budget": args.budget,
        "seed": args.seed,
        "model": model_id,
        "data": str(data_path),
        "letter_accuracy": sum(row["letter_correct"] for row in scored) / len(scored),
        "text_accuracy": sum(row["text_correct"] for row in scored) / len(scored),
        "mean_gold_letter_logprob": sum(row["letter_logprob"] for row in scored) / len(scored),
        "mean_gold_text_norm_logprob": sum(row["text_norm_logprob"] for row in scored) / len(scored),
        "data_version": "v1.0",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scored {len(scored)} items; row-level scores: {output}; summary: {summary_path}")


if __name__ == "__main__":
    main()
