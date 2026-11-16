"""Pairwise DeepSeek judging using the Balancing the Budget summarization prompt.

The system message and the user prompt follow Appendix A.2.1 of Raghavendra, Kang,
and Ritter. The judge is DeepSeek, not the paper's GPT-4o. Rank 1 is the better
summary. The returned dictionary is parsed with json.loads and is never executed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from common import DATA_V1, SYSTEM_PROMPT, read_jsonl, word_count, write_jsonl

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

JUDGE_USER_PROMPT = """Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Precise as in it captures the most important points in the post. Concise as in it does not include unimportant or irrelevant details. It should be a good TL;DR of the post - so a good summary should be no more than 2-3 sentences.

Example:

Post: I [18M] know her [15F] now over a Year (since I know my friend [17M]). We haven't really talked to much I saw her from time to time when I was at his place over the weekend. About 2 months ago we came a little closer (I was at his place again and we played on his Xbox One when he went to bed early so we played Dance Central [Dancing Game] until 3am). Since then we've talked more often, she hang out with us and 2 Weeks ago we began texting. Now I would really like to ask her out on a date. But I still haven't talked to my friend about this... I mean she is still his sister, wouldn't it be wierd, plus she is a little young in my opinion(she's her age way ahead in mind)."
Summary A: I'd like to date the sister of my best friend. But I don't know if she's to young and don't how to start all of this.
Summary B: You've known a 15-year-old girl for over a year through your 17-year-old friend. Recently, you've been texting and getting closer. You're considering asking her out on a date, but you're unsure if it's appropriate since she's your friend's sister and she's younger than you expect.
In this example, summary A is better since it is more concise and captures the most important points in the post.
Post: {post}
Summary A: { 'model': 'model_1', 'summary': {output_1} }
Summary B: { 'model': 'model_2', 'summary': {output_2} }
Now please rank the models by the quality of their summaries, so that the model with rank 1 has the best summary. Then return a dictionary of the form
{
"reasoning": short justification of ranking,
"model_1": rank_1,
"model_2": rank_2
}
Your response must be a valid json dictionary and should contain nothing else because we will directly execute it in Python."""


def response_text(row: dict[str, Any]) -> str:
    return str(row.get("response", row.get("summary", ""))).strip()


def render_user_prompt(post: str, output_1: str, output_2: str) -> str:
    return (
        JUDGE_USER_PROMPT.replace("{post}", post)
        .replace("{output_1}", output_1)
        .replace("{output_2}", output_2)
    )


def parse_ranks(raw: str) -> dict[str, Any]:
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Judge response is not a JSON object")
    rank_1 = int(result["model_1"])
    rank_2 = int(result["model_2"])
    if rank_1 < 1 or rank_2 < 1:
        raise ValueError(f"Judge returned invalid ranks {rank_1}, {rank_2}")
    return {
        "model_1": rank_1,
        "model_2": rank_2,
        "reasoning": str(result.get("reasoning", "")),
        "raw": raw,
    }


def report_progress(done_count: int) -> None:
    task_id = os.environ.get("EVAL_MONITOR_TASK")
    if not task_id:
        return
    try:
        from eval_monitor import update_task
        update_task(
            task_id,
            status="running",
            steps_done=done_count,
            message=f"已判 {done_count} / 200 题",
        )
    except Exception as exc:
        print(f"Monitor update skipped: {exc}", flush=True)


def model_1_score(ranks: dict[str, Any]) -> float:
    """Rank 1 is best. Equal ranks are a tie."""
    if ranks["model_1"] == ranks["model_2"]:
        return 0.5
    return 1.0 if ranks["model_1"] < ranks["model_2"] else 0.0


def judge_once(
    client: Any,
    model: str,
    post: str,
    output_1: str,
    output_2: str,
    retries: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(post, output_1, output_2)},
    ]
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = completion.choices[0].message.content or "{}"
            return parse_ranks(raw)
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
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--judge-base-url", default=DEEPSEEK_BASE_URL)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Small smoke test only")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("Set DEEPSEEK_API_KEY in your shell before judging; do not put it in a file.")
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
    client = OpenAI(api_key=api_key, base_url=args.judge_base_url)
    for index, key in enumerate(ids, 1):
        source = data_by_id[key]
        post = str(source.get("user_content") or (
            f"Subreddit: r/{source.get('subreddit', 'unknown')}\n"
            f"Title: {source.get('title', '')}\n\n{source.get('post', '')}"
        ))
        candidate = response_text(candidate_by_id[key])
        baseline = response_text(baseline_by_id[key])
        # Two calls swap which summary is model_1, matching the paper's anonymous labels.
        first = judge_once(client, args.judge_model, post, candidate, baseline, args.retries)
        second = judge_once(client, args.judge_model, post, baseline, candidate, args.retries)
        candidate_votes = [model_1_score(first), 1.0 - model_1_score(second)]
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
            "first_model_1_rank": first["model_1"],
            "first_model_2_rank": first["model_2"],
            "second_model_1_rank": second["model_1"],
            "second_model_2_rank": second["model_2"],
            "first_reasoning": first["reasoning"],
            "second_reasoning": second["reasoning"],
            "first_raw": first["raw"],
            "second_raw": second["raw"],
        }
        write_jsonl(output, [record], append=True)
        report_progress(len(done) + index)
        print(f"Judged {index}/{len(ids)} pending pairs (total {len(done) + index})", flush=True)
    print(f"Saved raw swapped-position judgments to {output}")


if __name__ == "__main__":
    main()
