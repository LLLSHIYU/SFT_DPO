"""Pairwise GPT-4o-mini judging with the preregistered A/B position swap."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from common import DATA_V1, read_jsonl, word_count, write_jsonl

JUDGE_PROMPT = """You are an impartial evaluator of summaries of a Reddit post.
Compare the two candidate summaries using only the supplied post context. Prefer factual accuracy, coverage of the central situation, and clarity. Do not reward length by itself; do not infer facts not in the post. Ignore style preferences unless they harm clarity. Treat A and B symmetrically.
Return only a JSON object with exactly these fields: {\"winner\": \"A\" or \"B\" or \"TIE\", \"reason\": a brief explanation}. Use TIE when neither summary is meaningfully better.

POST CONTEXT:\n{context}\n\nSUMMARY A:\n{summary_a}\n\nSUMMARY B:\n{summary_b}"""


def response_text(row: dict[str, Any]) -> str:
    return str(row.get("response", row.get("summary", ""))).strip()


def judge_once(client: Any, model: str, context: str, summary_a: str, summary_b: str, retries: int) -> dict[str, str]:
    messages = [{"role": "user", "content": JUDGE_PROMPT.format(
        context=context, summary_a=summary_a, summary_b=summary_b
    )}]
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = completion.choices[0].message.content or "{}"
            result = json.loads(raw)
            winner = str(result.get("winner", "")).upper().strip()
            if winner not in {"A", "B", "TIE"}:
                raise ValueError(f"Judge returned invalid winner {winner!r}")
            return {"winner": winner, "reason": str(result.get("reason", "")), "raw": raw}
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Unreachable judge retry state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate model TL;DR generations")
    parser.add_argument("--baseline", type=Path, required=True, help="M0 generations on the identical test set")
    parser.add_argument("--test-data", type=Path, default=DATA_V1 / "test_200_main.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Raw paired-judge JSONL output")
    parser.add_argument("--arm", required=True, help="e.g. G4")
    parser.add_argument("--budget", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Small smoke test only")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY in your shell before judging; do not put it in a file.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai from requirements-eval.txt") from exc

    test_rows = read_jsonl(args.test_data)
    candidate_rows = read_jsonl(args.candidate)
    baseline_rows = read_jsonl(args.baseline)
    candidate_by_id = {str(row["item_id"]): row for row in candidate_rows}
    baseline_by_id = {str(row["item_id"]): row for row in baseline_rows}
    data_by_id = {str(row.get("pid", row.get("id", row.get("order", i)))): row for i, row in enumerate(test_rows)}
    common_ids = set(candidate_by_id) & set(baseline_by_id) & set(data_by_id)
    expected_ids = list(data_by_id)
    missing = [key for key in expected_ids if key not in common_ids]
    if missing:
        raise ValueError(f"Generation coverage mismatch; {len(missing)} test IDs missing (first: {missing[:5]})")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume:
        output.unlink(missing_ok=True)
    done = {str(row["item_id"]) for row in read_jsonl(output)} if output.exists() else set()
    ids = [key for key in expected_ids if key not in done]
    if args.limit is not None:
        ids = ids[:args.limit]
    client = OpenAI()
    for index, key in enumerate(ids, 1):
        source = data_by_id[key]
        context = str(source.get("user_content") or (
            f"Subreddit: r/{source.get('subreddit', 'unknown')}\n"
            f"Title: {source.get('title', '')}\n\n{source.get('post', '')}"
        ))
        candidate = response_text(candidate_by_id[key])
        baseline = response_text(baseline_by_id[key])
        # Two independent calls use opposite positions; map each answer back to candidate/baseline.
        first = judge_once(client, args.judge_model, context, candidate, baseline, args.retries)
        second = judge_once(client, args.judge_model, context, baseline, candidate, args.retries)
        candidate_votes = [
            1.0 if first["winner"] == "A" else 0.0 if first["winner"] == "B" else 0.5,
            1.0 if second["winner"] == "B" else 0.0 if second["winner"] == "A" else 0.5,
        ]
        record = {
            "item_id": key,
            "order": source.get("order", index - 1),
            "arm": args.arm,
            "budget": args.budget,
            "seed": args.seed,
            "data_version": "v1.0",
            "judge_model": args.judge_model,
            "candidate_wins": sum(candidate_votes) / len(candidate_votes),
            "candidate_length_words": word_count(candidate),
            "baseline_length_words": word_count(baseline),
            "log_length_delta": math.log1p(word_count(candidate)) - math.log1p(word_count(baseline)),
            "candidate_response": candidate,
            "baseline_response": baseline,
            "candidate_first_position": first["winner"],
            "baseline_first_position": second["winner"],
            "first_reason": first["reason"],
            "second_reason": second["reason"],
            "first_raw": first["raw"],
            "second_raw": second["raw"],
        }
        write_jsonl(output, [record], append=True)
        print(f"Judged {index}/{len(ids)} pending pairs (total {len(done) + index})", flush=True)
    print(f"Saved raw swapped-position judgments to {output}")


if __name__ == "__main__":
    main()
