"""Create a blinded human-judge sheet and compute Cohen's kappa after annotation."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from common import DATA_V1, read_jsonl, write_csv

LABELS = {"A", "B", "TIE"}


def kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Cohen's kappa needs equal-length, non-empty ratings")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    categories = sorted(LABELS)
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(left_counts[c] * right_counts[c] for c in categories) / (n * n)
    return 1.0 if expected == 1.0 and observed == 1.0 else (0.0 if expected == 1.0 else (observed - expected) / (1.0 - expected))


def context_for(row: dict[str, Any]) -> str:
    if row.get("user_content"):
        return str(row["user_content"])
    return (
        f"Subreddit: r/{row.get('subreddit', 'unknown')}\n"
        f"Title: {row.get('title', '')}\n\n{row.get('post', '')}"
    )


def make_sheet(args: argparse.Namespace) -> None:
    judgments = [row for path in args.judgments for row in read_jsonl(path)]
    if len(judgments) < args.n:
        raise ValueError(f"Requested {args.n} pairs but only {len(judgments)} judgments are available")
    tests = read_jsonl(args.test_data)
    tests_by_id = {str(row.get("pid", row.get("id", row.get("order", i)))): row for i, row in enumerate(tests)}
    rng = random.Random(args.seed)
    selected = rng.sample(judgments, args.n)
    sheet, key = [], []
    for i, judged in enumerate(selected, 1):
        item_id = str(judged["item_id"])
        if item_id not in tests_by_id:
            raise ValueError(f"No source context for item_id={item_id}")
        candidate_first = rng.choice((True, False))
        summary_a = str(judged["candidate_response"] if candidate_first else judged["baseline_response"])
        summary_b = str(judged["baseline_response"] if candidate_first else judged["candidate_response"])
        vote = float(judged["candidate_wins"])
        judge_candidate_label = "A" if vote > 0.5 else "B" if vote < 0.5 else "TIE"
        if not candidate_first and judge_candidate_label in {"A", "B"}:
            judge_candidate_label = "B" if judge_candidate_label == "A" else "A"
        audit_id = f"H{i:03d}"
        source = tests_by_id[item_id]
        sheet.append({
            "audit_id": audit_id,
            "context": context_for(source),
            "summary_A": summary_a,
            "summary_B": summary_b,
            "annotator_1": "",
            "annotator_2": "",
            "annotator_3": "",
        })
        key.append({
            "audit_id": audit_id,
            "item_id": item_id,
            "arm": judged.get("arm", ""),
            "budget": judged.get("budget", ""),
            "seed": judged.get("seed", ""),
            "candidate_side": "A" if candidate_first else "B",
            "judge_label": judge_candidate_label,
            "judge_candidate_win_fraction": vote,
        })
    write_csv(args.sheet, sheet)
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Blinded annotation sheet: {args.sheet}\nRestricted mapping key: {args.key}")


def score_annotations(args: argparse.Namespace) -> None:
    key = json.loads(args.key.read_text(encoding="utf-8"))
    key_by_id = {str(row["audit_id"]): row for row in key}
    with args.sheet.open(encoding="utf-8-sig", newline="") as handle:
        sheet = list(csv.DictReader(handle))
    annotators = [name.strip() for name in args.annotators.split(",") if name.strip()]
    for row in sheet:
        for name in annotators:
            rating = row.get(name, "").strip().upper()
            if rating not in LABELS:
                raise ValueError(f"Invalid or missing rating in {name}, {row.get('audit_id')}: {rating!r}")
    results = {"n_pairs": len(sheet), "annotators": annotators, "human_vs_judge": {}, "human_pairwise": {}}
    for name in annotators:
        human = [row[name].strip().upper() for row in sheet]
        judge = [key_by_id[row["audit_id"]]["judge_label"] for row in sheet]
        results["human_vs_judge"][name] = {"cohen_kappa": kappa(human, judge)}
    for i, left_name in enumerate(annotators):
        for right_name in annotators[i + 1:]:
            results["human_pairwise"][f"{left_name}__{right_name}"] = {
                "cohen_kappa": kappa(
                    [row[left_name].strip().upper() for row in sheet],
                    [row[right_name].strip().upper() for row in sheet],
                )
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="Create randomized, blinded annotations from judged output pairs")
    sample.add_argument("--judgments", type=Path, nargs="+", required=True)
    sample.add_argument("--test-data", type=Path, default=DATA_V1 / "test_200_main.jsonl")
    sample.add_argument("--n", type=int, default=100)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--sheet", type=Path, required=True)
    sample.add_argument("--key", type=Path, required=True, help="Keep this mapping separate from the blinded sheet")
    sample.set_defaults(func=make_sheet)
    agreement = sub.add_parser("agreement", help="Compute human/judge and inter-rater Cohen's kappa")
    agreement.add_argument("--sheet", type=Path, required=True, help="Annotated CSV sheet")
    agreement.add_argument("--key", type=Path, required=True)
    agreement.add_argument("--annotators", default="annotator_1,annotator_2,annotator_3")
    agreement.add_argument("--output", type=Path, required=True)
    agreement.set_defaults(func=score_annotations)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
