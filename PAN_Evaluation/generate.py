"""Generate deterministic TL;DR or IFEval outputs for one checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import DATA_V1, DEFAULT_BASE_MODEL, IF_DATA, item_id, load_model_and_tokenizer, read_jsonl, tldr_messages, write_jsonl
from eval_monitor import report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("tldr", "ifeval"), required=True)
    parser.add_argument("--data", type=Path, help="JSONL input; defaults to frozen v1.0 file")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Full HF model/checkpoint path")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model used with --adapter")
    parser.add_argument("--adapter", help="PEFT adapter directory; omit for full model checkpoints")
    parser.add_argument("--arm", default="model")
    parser.add_argument("--budget", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, help="Defaults: TL;DR=128, IFEval=1280")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--device", help="Optional input device override, e.g. cuda:0 or mps")
    parser.add_argument("--no-resume", action="store_true", help="Replace output instead of resuming")
    return parser


def prompt_messages(row: dict[str, Any], task: str) -> list[dict[str, str]]:
    if task == "tldr":
        return tldr_messages(row)
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"IFEval item {item_id(row, 0)} has no messages list")
    return messages


def main() -> None:
    args = build_parser().parse_args()
    data_path = args.data or (
        DATA_V1 / "test_200_main.jsonl" if args.task == "tldr"
        else IF_DATA / "eval_benchmarks" / "ifeval_541.jsonl"
    )
    max_new_tokens = args.max_new_tokens or (128 if args.task == "tldr" else 1280)
    if args.batch_size < 1 or max_new_tokens < 1:
        raise ValueError("--batch-size and --max-new-tokens must be positive")

    records = read_jsonl(data_path)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if output_path.exists() and not args.no_resume:
        completed = {str(row["item_id"]) for row in read_jsonl(output_path) if "item_id" in row}
    elif args.no_resume:
        output_path.unlink(missing_ok=True)

    pending_rows: list[tuple[int, dict[str, Any], str]] = []
    for index, row in enumerate(records):
        key = str(row.get("key", item_id(row, index))) if args.task == "ifeval" else item_id(row, index)
        if key in completed:
            continue
        pending_rows.append((index, row, key))
    if not pending_rows:
        report(len(records), len(records), "已有全部生成结果")
        print(f"Saved generations to {output_path}; source dataset: {data_path}")
        return

    model, tokenizer, torch = load_model_and_tokenizer(
        args.model, base_model=args.base_model, adapter=args.adapter, dtype=args.dtype
    )
    prompts: list[str] = []
    pending: list[tuple[int, dict[str, Any], str]] = []
    for index, row, key in pending_rows:
        messages = prompt_messages(row, args.task)
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
        pending.append((index, row, key))

    report(len(completed), len(records), "正在生成")
    device = args.device or str(next(model.parameters()).device)
    eos_ids = [151645, 151643]
    for offset in range(0, len(pending), args.batch_size):
        batch_pending = pending[offset: offset + args.batch_size]
        batch_prompts = prompts[offset: offset + args.batch_size]
        encoded = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=False,
            add_special_tokens=False,
        )
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=151643,
                use_cache=True,
            )
        continuation = generated[:, encoded["input_ids"].shape[1]:]
        rows_out = []
        for local_index, ((source_index, source, key), token_ids) in enumerate(zip(batch_pending, continuation)):
            response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            token_count = int((token_ids != tokenizer.pad_token_id).sum().item())
            rows_out.append({
                "item_id": key,
                "task": args.task,
                "arm": args.arm,
                "budget": args.budget,
                "seed": args.seed,
                "data_version": "v1.0",
                "source_index": source_index,
                "response": response,
                "prompt_tokens": int(prompt_lengths[local_index]),
                "completion_tokens": token_count,
                "max_new_tokens": max_new_tokens,
                "hit_token_limit": token_count >= max_new_tokens,
                "model": args.adapter or args.model,
            })
        write_jsonl(output_path, rows_out, append=True)
        finished = len(completed) + min(offset + len(rows_out), len(pending))
        report(finished, len(records), "正在生成")
        print(f"Generated {finished}/{len(records)}", flush=True)

    print(f"Saved generations to {output_path}; source dataset: {data_path}")


if __name__ == "__main__":
    main()
