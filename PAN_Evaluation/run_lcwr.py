"""Judge every summary final model against M0 with the DeepSeek LCWR judge.

Uses the three summary manifests. G4/G5 stage-1 checkpoints and the IF track are
not included. The API key is read from DEEPSEEK_API_KEY and is never written here.
A finished eval/lcwr_<run_id>.jsonl is skipped. A partial file is continued by
judge_lcwr.py.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from common import RESULTS_DIR, ROOT, SUMMARY_DATA, read_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFESTS = (
    SCRIPT_DIR / "models.json",
    SCRIPT_DIR / "models_seed1337.json",
    SCRIPT_DIR / "models_seed2024.json",
)
BASELINE_RUN_ID = "M0_warmup_seed42_20261105"
EXPECTED_ROWS = 200
JUDGE_STATUS = ROOT / "LSYNEED" / "monitor" / "lcwr_judge_status.json"
# Two DeepSeek calls took about 11 seconds per item in the smoke test.
SECONDS_PER_ITEM = 11


def keep_awake() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)


def allow_sleep() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must be a non-empty JSON list")
    return payload


def run_dir_of(entry: dict[str, Any]) -> Path:
    run_dir = Path(entry.get("run_dir", f"03_runs/{entry['run_id']}"))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return run_dir


def tldr_path(entry: dict[str, Any]) -> Path:
    run_id = str(entry["run_id"])
    return run_dir_of(entry) / "eval" / f"{run_id}.tldr.jsonl"


def judgment_path(entry: dict[str, Any]) -> Path:
    run_id = str(entry["run_id"])
    return run_dir_of(entry) / "eval" / f"lcwr_{run_id}.jsonl"


def judged_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row["item_id"]) for row in read_jsonl(path)}


def collect_entries(manifests: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        for entry in load_manifest(manifest):
            run_id = str(entry.get("run_id", ""))
            arm = str(entry.get("arm", ""))
            if not run_id or not arm:
                raise ValueError(f"{manifest} has an entry without run_id or arm")
            if run_id in seen:
                raise ValueError(f"Duplicate run_id {run_id}")
            seen.add(run_id)
            if arm == "M0":
                if baseline is not None:
                    raise ValueError("Expected exactly one M0 baseline")
                baseline = entry
                continue
            candidates.append(entry)
    if baseline is None:
        raise ValueError(f"No M0 entry found; expected {BASELINE_RUN_ID}")
    if str(baseline["run_id"]) != BASELINE_RUN_ID:
        raise ValueError(f"Baseline run_id is {baseline['run_id']!r}, expected {BASELINE_RUN_ID}")
    if not candidates:
        raise ValueError("No candidate models found")
    return baseline, candidates


def monitor_task(entry: dict[str, Any], done: int, previous: dict[str, Any] | None) -> dict[str, Any]:
    previous = previous or {}
    finished = done >= EXPECTED_ROWS
    task = {
        "id": str(entry["run_id"]),
        "order": 0,
        "group": f"种子 {entry.get('seed', '')}",
        "title": f"{entry['arm']} · {entry.get('budget', '')}",
        "kind": "JUDGE",
        "budget": str(entry.get("budget", "")),
        "models": [str(entry["arm"])],
        "item_count": EXPECTED_ROWS,
        "steps_total": EXPECTED_ROWS,
        "steps_done": EXPECTED_ROWS if finished else done,
        "status": "done" if finished else "waiting",
        "estimate_seconds": SECONDS_PER_ITEM * EXPECTED_ROWS,
        "data_file": "SUMMARY_DATA/test_200_main.jsonl",
        "output_dir": str(judgment_path(entry).parent.relative_to(ROOT)).replace("\\", "/") + "/",
        "message": "完成" if finished else (f"已判 {done} / {EXPECTED_ROWS} 题" if done else "还没开始。"),
    }
    if previous.get("started_at"):
        task["started_at"] = previous["started_at"]
    if finished and previous.get("finished_at"):
        task["finished_at"] = previous["finished_at"]
    elif finished:
        task["finished_at"] = now()
    return task


def write_board(candidates: list[dict[str, Any]]) -> None:
    previous: dict[str, dict[str, Any]] = {}
    if JUDGE_STATUS.is_file():
        payload = json.loads(JUDGE_STATUS.read_text(encoding="utf-8"))
        previous = {str(task["id"]): task for task in payload.get("tasks", [])}
    tasks = []
    for order, entry in enumerate(candidates, 1):
        task = monitor_task(entry, len(judged_ids(judgment_path(entry))), previous.get(str(entry["run_id"])))
        task["order"] = order
        tasks.append(task)
    done = sum(task["status"] == "done" for task in tasks)
    if done == len(tasks):
        phase, headline = "done", "36 个模型的 LCWR 胜率都算完了"
    elif done:
        phase, headline = "running", f"LCWR 胜率裁判进行中：{done} / {len(tasks)} 个模型已完成"
    else:
        phase, headline = "not_started", "LCWR 胜率裁判还没开始"
    board = {
        "phase": phase,
        "headline": headline,
        "waiting_headline": "LCWR 胜率裁判还没开始",
        "done_headline": "36 个模型的 LCWR 胜率都算完了",
        "gpu": "DeepSeek",
        "estimate_note": "36 个摘要最终模型对种子 42 的 M0，每题对调两次。上面评测表里的 TL;DR 摘要已经写好，这里才是胜率。",
        "updated_at": now(),
        "tasks": tasks,
    }
    JUDGE_STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = JUDGE_STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, JUDGE_STATUS)


def capability_summaries(entries: list[dict[str, Any]]) -> list[Path]:
    names = ("mmlu_500.summary.json", "arc_challenge_300.summary.json")
    found: list[Path] = []
    for entry in entries:
        run_id = str(entry["run_id"])
        eval_dir = run_dir_of(entry) / "eval"
        for name in names:
            path = eval_dir / f"{run_id}.{name}"
            if path.is_file():
                found.append(path)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", dest="manifests")
    parser.add_argument("--test-data", type=Path, default=SUMMARY_DATA / "test_200_main.jsonl")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--judge-base-url", default="https://api.deepseek.com")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Judge only the first N pending items of each model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    manifests = args.manifests or list(DEFAULT_MANIFESTS)
    manifests = [path if path.is_absolute() else ROOT / path for path in manifests]
    baseline, candidates = collect_entries(manifests)
    baseline_tldr = tldr_path(baseline)
    test_data = args.test_data if args.test_data.is_absolute() else ROOT / args.test_data

    missing: list[str] = []
    if not test_data.is_file():
        missing.append(f"test data missing: {test_data}")
    if not baseline_tldr.is_file():
        missing.append(f"M0 summaries missing: {baseline_tldr}")
    jobs: list[dict[str, Any]] = []
    already_done = 0
    for entry in candidates:
        source = tldr_path(entry)
        output = judgment_path(entry)
        if not source.is_file():
            missing.append(f"candidate summaries missing: {source}")
            continue
        done = len(judged_ids(output))
        if args.limit is None and done >= EXPECTED_ROWS:
            already_done += 1
            print(f"skip {entry['run_id']} ({done} judgments)", flush=True)
            continue
        jobs.append({"entry": entry, "source": source, "output": output, "done": done})
    if missing:
        raise FileNotFoundError("\n".join(missing))
    write_board(candidates)
    os.environ["EVAL_STATUS"] = str(JUDGE_STATUS)

    pending_calls = sum(max(EXPECTED_ROWS - int(job["done"]), 0) for job in jobs) * 2
    if args.limit is not None:
        pending_calls = len(jobs) * args.limit * 2
    print(
        f"{len(candidates)} summary models vs M0; {already_done} already finished; "
        f"{len(jobs)} to judge; about {pending_calls} DeepSeek calls.",
        flush=True,
    )
    if not jobs and (args.skip_analyze or args.dry_run or args.limit is not None):
        print("Nothing left to judge.")
        return
    if not args.dry_run and not os.environ.get("DEEPSEEK_API_KEY") and jobs:
        raise EnvironmentError("Set DEEPSEEK_API_KEY in the shell before judging. Do not put it in a file.")

    keep_awake()
    try:
        for index, job in enumerate(jobs, 1):
            entry = job["entry"]
            command = [
                sys.executable,
                str(SCRIPT_DIR / "judge_lcwr.py"),
                "--candidate", str(job["source"]),
                "--baseline", str(baseline_tldr),
                "--test-data", str(test_data),
                "--output", str(job["output"]),
                "--arm", str(entry["arm"]),
                "--budget", str(entry.get("budget", "")),
                "--seed", str(entry.get("seed", "")),
                "--judge-model", args.judge_model,
                "--judge-base-url", args.judge_base_url,
                "--retries", str(args.retries),
            ]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            print(f"\n[{index}/{len(jobs)}] {entry['run_id']} ({job['done']} already judged)", flush=True)
            print("$ " + " ".join(command), flush=True)
            if args.dry_run:
                continue
            from eval_monitor import update_task
            task_id = str(entry["run_id"])
            env = os.environ.copy()
            env["EVAL_STATUS"] = str(JUDGE_STATUS)
            env["EVAL_MONITOR_TASK"] = task_id
            env["PYTHONUNBUFFERED"] = "1"
            update_task(task_id, status="running", steps_done=int(job["done"]), started_at=now(), message="正在调用 DeepSeek")
            try:
                subprocess.run(command, cwd=SCRIPT_DIR, check=True, env=env)
            except subprocess.CalledProcessError:
                update_task(task_id, status="failed", finished_at=now(), message="这一项失败了，后面的模型没有继续")
                raise
            finished = len(judged_ids(job["output"]))
            if finished >= EXPECTED_ROWS:
                update_task(task_id, status="done", finished_at=now(), message="完成")
            else:
                update_task(
                    task_id,
                    status="waiting",
                    steps_done=finished,
                    message=f"已判 {finished} / {EXPECTED_ROWS} 题，还没跑完",
                )
    finally:
        allow_sleep()

    if args.dry_run or args.limit is not None or args.skip_analyze:
        return
    unfinished = [entry["run_id"] for entry in candidates if len(judged_ids(judgment_path(entry))) < EXPECTED_ROWS]
    if unfinished:
        raise RuntimeError(f"{len(unfinished)} models are still short of {EXPECTED_ROWS} judgments: {unfinished[:5]}")
    summaries = capability_summaries([baseline, *candidates])
    command = [
        sys.executable,
        str(SCRIPT_DIR / "analyze.py"),
        "--judgments",
        *[str(judgment_path(entry)) for entry in candidates],
        "--input-dir", str(SCRIPT_DIR),
        "--output-dir", str(RESULTS_DIR),
        "--capability-summaries",
        *[str(path) for path in summaries],
    ]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


if __name__ == "__main__":
    main()
