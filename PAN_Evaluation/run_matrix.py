"""Run generation, capability scoring, IFEval, and LCWR judging from a model manifest."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from common import DEFAULT_BASE_MODEL, ROOT, SUMMARY_DATA
from eval_monitor import update_task

SCRIPT_DIR = Path(__file__).resolve().parent
TLDR_ROWS = 200
SENSITIVITY_ROWS = 500
IFEVAL_ROWS = 541


def monitor_id(entry: dict[str, Any], kind: str) -> str:
    arm = str(entry["arm"])
    if arm == "M0":
        return f"m0-{kind}"
    return f"{arm.lower()}-{entry.get('budget', '')}-档-{kind}"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def done_task_ids() -> set[str]:
    override = os.environ.get("EVAL_STATUS")
    if not override:
        raise RuntimeError("--resume needs EVAL_STATUS pointing at this seed's evaluation board.")
    path = Path(override)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation board not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {task["id"] for task in payload.get("tasks", []) if task.get("status") == "done"}


def skip_finished(done_ids: set[str], task_id: str, ready: bool) -> bool:
    if task_id not in done_ids:
        return False
    if not ready:
        print(f"\n# {task_id} is marked done, but its output is incomplete; running it again", flush=True)
        return False
    print(f"\n# skip {task_id}", flush=True)
    return True


def run(command: list[str], dry_run: bool, task_id: str | None = None, mark: str | None = None) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    if task_id:
        env["EVAL_MONITOR_TASK"] = task_id
    if task_id and mark in {"start", "both"}:
        update_task(task_id, status="running", steps_done=0, started_at=now(), message="正在开始")
    try:
        subprocess.run(command, cwd=ROOT, check=True, env=env)
    except subprocess.CalledProcessError:
        if task_id:
            update_task(task_id, status="failed", finished_at=now(), message="这一项失败了，后面的评测没有继续")
        raise
    if task_id and mark in {"finish", "both"}:
        update_task(task_id, status="done", finished_at=now(), message="完成")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list of model/run entries")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--judge-base-url", default="https://api.deepseek.com")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Also generate TL;DR summaries on test_500_full.jsonl",
    )
    parser.add_argument("--skip-capability", action="store_true")
    parser.add_argument("--skip-ifeval", action="store_true")
    parser.add_argument("--skip-tldr", action="store_true", help="Skip TL;DR generation. Used for the IF track.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks already marked done on the EVAL_STATUS board when their output is complete",
    )
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    entries: list[dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a non-empty JSON list")
    by_arm: dict[str, dict[str, Any]] = {}
    for entry in entries:
        for required in ("run_id", "arm"):
            if not entry.get(required):
                raise ValueError(f"Each manifest entry must contain {required!r}")
        if entry["run_id"] in {".", ".."} or "/" in entry["run_id"]:
            raise ValueError(f"Unsafe run_id: {entry['run_id']!r}")
        if entry["arm"] == "M0":
            if by_arm.get("M0"):
                raise ValueError("Manifest must contain exactly one M0 baseline entry")
            by_arm["M0"] = entry
    if not args.skip_judge and "M0" not in by_arm:
        raise ValueError("Manifest must include the shared M0 baseline checkpoint")

    finished = done_task_ids() if args.resume else set()
    generated_by_run: dict[str, dict[str, Path]] = {}
    for entry in entries:
        run_dir = Path(entry.get("run_dir", f"03_runs/{entry['run_id']}"))
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        eval_dir = run_dir / "eval"
        if not args.dry_run:
            eval_dir.mkdir(parents=True, exist_ok=True)
        model = str(entry.get("model", args.base_model))
        adapter = entry.get("adapter")
        common = ["--model", model, "--base-model", args.base_model, "--arm", str(entry["arm"]),
                  "--budget", str(entry.get("budget", "")), "--seed", str(entry.get("seed", "")),
                  "--batch-size", str(args.batch_size), "--dtype", args.dtype]
        if adapter:
            common.extend(["--adapter", str(adapter)])
        safe_id = str(entry["run_id"]).replace(" ", "_")
        tldr_path = eval_dir / f"{safe_id}.tldr.jsonl"
        ifeval_path = eval_dir / f"{safe_id}.ifeval.jsonl"
        generated_by_run[str(entry["run_id"])] = {"tldr": tldr_path, "ifeval": ifeval_path, "eval_dir": eval_dir}
        lcwr_id = monitor_id(entry, "lcwr")
        if not args.skip_tldr and not (args.resume and skip_finished(finished, lcwr_id, jsonl_rows(tldr_path) >= TLDR_ROWS)):
            run([sys.executable, str(SCRIPT_DIR / "generate.py"), "--task", "tldr", *common, "--output", str(tldr_path)],
                args.dry_run, lcwr_id, "both")
        if args.sensitivity:
            sens_path = eval_dir / f"{safe_id}.tldr500.jsonl"
            sens_id = monitor_id(entry, "sens")
            if not (args.resume and skip_finished(finished, sens_id, jsonl_rows(sens_path) >= SENSITIVITY_ROWS)):
                run([sys.executable, str(SCRIPT_DIR / "generate.py"), "--task", "tldr",
                     "--data", str(SUMMARY_DATA / "test_500_full.jsonl"),
                     *common, "--output", str(sens_path)],
                    args.dry_run, sens_id, "both")
        if not args.skip_ifeval:
            ifeval_id = monitor_id(entry, "ifeval")
            ifeval_scores = eval_dir / f"{safe_id}.ifeval.csv"
            ifeval_ready = (
                jsonl_rows(ifeval_path) >= IFEVAL_ROWS and ifeval_scores.with_suffix(".summary.json").is_file()
            )
            if not (args.resume and skip_finished(finished, ifeval_id, ifeval_ready)):
                run([sys.executable, str(SCRIPT_DIR / "generate.py"), "--task", "ifeval", *common, "--output", str(ifeval_path)],
                    args.dry_run, ifeval_id, "start")
                run([sys.executable, str(SCRIPT_DIR / "score_ifeval.py"), "--responses", str(ifeval_path), "--output", str(ifeval_scores)],
                    args.dry_run, ifeval_id, "finish")
        if not args.skip_capability:
            for benchmark, filename, kind in (("mmlu", "mmlu_500", "mmlu"), ("arc", "arc_challenge_300", "arc")):
                score_path = eval_dir / f"{safe_id}.{filename}.csv"
                kind_id = monitor_id(entry, kind)
                if args.resume and skip_finished(finished, kind_id, score_path.with_suffix(".summary.json").is_file()):
                    continue
                score_common = ["--benchmark", benchmark, "--model", model, "--base-model", args.base_model,
                                "--arm", str(entry["arm"]), "--budget", str(entry.get("budget", "")),
                                "--seed", str(entry.get("seed", "")), "--output", str(score_path), "--dtype", args.dtype]
                if adapter:
                    score_common.extend(["--adapter", str(adapter)])
                run([sys.executable, str(SCRIPT_DIR / "score_capability.py"), *score_common],
                    args.dry_run, kind_id, "both")

    if not args.skip_judge:
        baseline = by_arm["M0"]
        baseline_paths = generated_by_run[str(baseline["run_id"])]
        for entry in entries:
            if entry["arm"] == "M0":
                continue
            run_dir = Path(entry.get("run_dir", f"03_runs/{entry['run_id']}"))
            if not run_dir.is_absolute():
                run_dir = ROOT / run_dir
            safe_id = str(entry["run_id"]).replace(" ", "_")
            judge_path = run_dir / "eval" / f"lcwr_{safe_id}.jsonl"
            command = [sys.executable, str(SCRIPT_DIR / "judge_lcwr.py"),
                       "--candidate", str(generated_by_run[str(entry["run_id"])]["tldr"]),
                       "--baseline", str(baseline_paths["tldr"]),
                       "--output", str(judge_path), "--arm", str(entry["arm"]),
                       "--budget", str(entry.get("budget", "")), "--seed", str(entry.get("seed", "")),
                       "--judge-model", args.judge_model, "--judge-base-url", args.judge_base_url]
            run(command, args.dry_run)

    print("\nEvaluation matrix commands completed.")
    print("Then run analyze.py over 03_runs to create LCWR/OA tables and figures.")


if __name__ == "__main__":
    main()
