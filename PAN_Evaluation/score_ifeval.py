"""Score IFEval generations with the official bundled Google Research checker."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from common import IF_DATA, ROOT, read_jsonl, write_csv
from eval_monitor import note


def load_official_checker(source: Path | None, archive: Path | None) -> tuple[Any, tempfile.TemporaryDirectory[str] | None]:
    candidates = []
    if source:
        candidates.append(source.resolve())
    candidates.extend([
        ROOT / "02_code" / "DSAI4207_code" / "data_engineering" / "scripts",
        ROOT / "02_code" / "data_engineering" / "scripts",
        ROOT / "data_engineering" / "scripts",
    ])
    for candidate in candidates:
        if (candidate / "instruction_following_eval" / "instructions_registry.py").exists():
            sys.path.insert(0, str(candidate))
            from instruction_following_eval import instructions_registry
            return instructions_registry, None

    archive = archive or (ROOT / "02_code" / "DSAI4207_code_20261021.zip")
    if not archive.exists():
        raise FileNotFoundError(
            "Could not locate the bundled official IFEval checker. Pass --ifeval-source "
            "(directory containing instruction_following_eval) or --code-zip."
        )
    temp = tempfile.TemporaryDirectory(prefix="dsai4207_ifeval_")
    with zipfile.ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if name.startswith("data_engineering/scripts/instruction_following_eval/")]
        if not names:
            temp.cleanup()
            raise FileNotFoundError(f"No official checker package found in {archive}")
        zf.extractall(temp.name, names)
    source_root = Path(temp.name) / "data_engineering" / "scripts"
    sys.path.insert(0, str(source_root))
    from instruction_following_eval import instructions_registry
    return instructions_registry, temp


def loose_candidates(response: str) -> list[str]:
    text = response.strip()
    lines = text.split("\n")
    no_first = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
    no_last = "\n".join(lines[:-1]).strip() if len(lines) > 1 else text
    no_both = "\n".join(lines[1:-1]).strip() if len(lines) > 2 else text
    variants = [text, text.replace("*", ""), no_first, no_last, no_both]
    variants.extend(value.replace("*", "") for value in (no_first, no_last, no_both))
    return list(dict.fromkeys(variants))


def check_instruction(registry: Any, instruction_id: str, response: str, kwargs: dict[str, Any]) -> bool:
    instruction_cls = registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)
    clean_kwargs = {key: value for key, value in (kwargs or {}).items() if value is not None}
    import inspect
    accepted = inspect.signature(instruction.build_description).parameters
    instruction.build_description(**{key: value for key, value in clean_kwargs.items() if key in accepted})
    return bool(response.strip() and instruction.check_following(response))


def score(bench: list[dict[str, Any]], generations: list[dict[str, Any]], registry: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {str(row.get("item_id", row.get("key"))): str(row.get("response", "")) for row in generations}
    results = []
    strict_prompts = loose_prompts = strict_instructions = loose_instructions = 0
    instruction_count = 0
    note("正在用官方规则打分")
    for index, row in enumerate(bench):
        key = str(row["key"])
        if key not in by_key:
            raise ValueError(f"Missing IFEval response for key={key}; check generation coverage")
        response = by_key[key]
        instruction_ids = row["instruction_id_list"]
        kwargs_list = row["kwargs"]
        if len(instruction_ids) != len(kwargs_list):
            raise ValueError(f"Mismatched instruction IDs/kwargs for key={key}")
        strict_each = [
            check_instruction(registry, instruction_id, response, kwargs)
            for instruction_id, kwargs in zip(instruction_ids, kwargs_list)
        ]
        variants = loose_candidates(response)
        loose_each = [
            strict or any(check_instruction(registry, instruction_id, variant, kwargs) for variant in variants)
            for instruction_id, kwargs, strict in zip(instruction_ids, kwargs_list, strict_each)
        ]
        strict_prompt = bool(strict_each) and all(strict_each)
        loose_prompt = bool(loose_each) and all(loose_each)
        strict_prompts += int(strict_prompt)
        loose_prompts += int(loose_prompt)
        strict_instructions += sum(strict_each)
        loose_instructions += sum(loose_each)
        instruction_count += len(strict_each)
        results.append({
            "item_id": key,
            "order": row.get("order", index),
            "arm": generations[0].get("arm", "") if generations else "",
            "budget": generations[0].get("budget", "") if generations else "",
            "seed": generations[0].get("seed", "") if generations else "",
            "strict_prompt_pass": int(strict_prompt),
            "loose_prompt_pass": int(loose_prompt),
            "strict_instruction_passes": sum(strict_each),
            "loose_instruction_passes": sum(loose_each),
            "instruction_count": len(strict_each),
            "response_words": len(response.split()),
        })
    n = len(bench)
    summary = {
        "checker": "official",
        "num_prompts": n,
        "num_instructions": instruction_count,
        "prompt_level_strict_acc": strict_prompts / n if n else 0.0,
        "prompt_level_loose_acc": loose_prompts / n if n else 0.0,
        "instruction_level_strict_acc": strict_instructions / instruction_count if instruction_count else 0.0,
        "instruction_level_loose_acc": loose_instructions / instruction_count if instruction_count else 0.0,
        "arm": results[0]["arm"] if results else "",
        "budget": results[0]["budget"] if results else "",
        "seed": results[0]["seed"] if results else "",
        "data_version": "v1.0",
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=IF_DATA / "eval_benchmarks" / "ifeval_541.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Per-prompt CSV; JSON summary is saved beside it")
    parser.add_argument("--ifeval-source", type=Path, help="Directory containing instruction_following_eval")
    parser.add_argument("--code-zip", type=Path, help="Project code snapshot ZIP (default: 02_code snapshot)")
    args = parser.parse_args()

    registry, temp = load_official_checker(args.ifeval_source, args.code_zip)
    try:
        per_prompt, summary = score(read_jsonl(args.benchmark), read_jsonl(args.responses), registry)
        write_csv(args.output, per_prompt)
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved per-prompt scores to {args.output}; summary to {summary_path}")
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    main()
