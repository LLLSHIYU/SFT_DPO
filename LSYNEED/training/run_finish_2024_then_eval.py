"""Obsolete. The old runs were deleted. Start the new queue with run_all.py.

Finish seed 2024 training, then evaluate seed 1337, then seed 2024.

Seed 42 training and seed 42 evaluation status are left untouched.
M0 is not retrained and is not evaluated again.
The interrupted G3 10000 job for seed 2024 has no saved checkpoint, so that
job starts over. Tasks that already have metrics and merged weights are skipped.

Prepare the manifests and monitor boards without starting work:

    python run_finish_2024_then_eval.py --prepare
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SHARED = Path(r"C:\Users\liush\Desktop\DSAI4207_shared")
PYTHON = SHARED / "02_code" / "DSAI4207_code" / ".venv" / "Scripts" / "python.exe"
TRAIN = SHARED / "LSYNEED" / "training" / "run_seeds_1337_2024.py"
MATRIX = SHARED / "PAN_Evaluation" / "run_matrix.py"
RUNS = SHARED / "03_runs"
MONITOR = SHARED / "LSYNEED" / "monitor"
SEED42_EVAL = MONITOR / "eval_status.json"
SEEDS_STATUS = MONITOR / "seeds_status.json"
LOG = MONITOR / "finish_queue.log"
PAN = SHARED / "PAN_Evaluation"

BUDGETS = ("500", "2000", "10000")
FINAL_MODELS = (
    ("G2", "G2_sft_{budget}"),
    ("G3", "G3_dpo_{budget}"),
    ("G4", "G4_stage2_dpo_{budget}"),
    ("G5", "G5_stage2_sft_{budget}"),
)


def keep_awake() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)


def allow_sleep() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def log(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_dir(seed: int, folder: str) -> Path:
    if seed == 42:
        return RUNS / f"{folder}_seed42_20261105"
    return RUNS / f"{folder}_seed{seed}"


def merged_weights(seed: int, folder: str) -> Path:
    return run_dir(seed, folder) / "checkpoint" / "merged" / "model.safetensors"


def missing_final_weights(seed: int) -> list[str]:
    missing = []
    for _arm, pattern in FINAL_MODELS:
        for budget in BUDGETS:
            weights = merged_weights(seed, pattern.format(budget=budget))
            if not weights.is_file():
                missing.append(str(weights.relative_to(SHARED)))
    return missing


def manifest_path(seed: int) -> Path:
    return PAN / f"models_seed{seed}.json"


def eval_status_path(seed: int) -> Path:
    return MONITOR / f"eval_status_{seed}.json"


def write_manifest(seed: int) -> None:
    entries = []
    for arm, pattern in FINAL_MODELS:
        for budget in BUDGETS:
            folder = pattern.format(budget=budget)
            directory = run_dir(seed, folder)
            relative = directory.relative_to(SHARED).as_posix()
            entries.append({
                "run_id": directory.name,
                "run_dir": relative,
                "arm": arm,
                "budget": budget,
                "seed": str(seed),
                "model": f"{relative}/checkpoint/merged",
            })
    manifest_path(seed).write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_eval_board(seed: int) -> None:
    path = eval_status_path(seed)
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if any(item.get("status") != "waiting" for item in current.get("tasks", [])):
            raise SystemExit(f"{path.name} already has started tasks; not rebuilding it.")
    source = json.loads(SEED42_EVAL.read_text(encoding="utf-8"))
    tasks = []
    order = 1
    for task in source["tasks"]:
        if task.get("group") == "M0":
            continue
        item = dict(task)
        item.pop("started_at", None)
        item.pop("finished_at", None)
        item["order"] = order
        item["steps_done"] = 0
        item["status"] = "waiting"
        item["message"] = "还没开始。"
        item["output_dir"] = str(item["output_dir"]).replace("_seed42_20261105", f"_seed{seed}")
        tasks.append(item)
        order += 1
    if len(tasks) != 60:
        raise SystemExit(f"Expected 60 evaluation tasks for seed {seed}, got {len(tasks)}.")
    weight = sum(int(item["estimate_seconds"]) for item in tasks)
    hours = round(weight / 3600, 1)
    document = {
        "phase": "not_started",
        "headline": f"种子 {seed} 评测还没开始",
        "waiting_headline": f"种子 {seed} 评测还没开始",
        "done_headline": f"种子 {seed} 的 {len(tasks)} 项评测都跑完了",
        "gpu": "RTX 4060 Laptop",
        "estimate_note": (
            f"12 个最终模型：G2、G3、G4、G5 各自的 500、2000、10000 档。"
            f"每个模型 5 项评测，一共 {len(tasks)} 项。M0 只在种子 42 评测里。"
            f"时间表沿用种子 42 同一项的预计，合计约 {hours} 小时。不叫 GPT 裁判。"
        ),
        "updated_at": None,
        "tasks": tasks,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mark_interrupted_job_waiting() -> None:
    """The stopped G3 10000 run did not save metrics. The next start retrains it."""
    payload = json.loads(SEEDS_STATUS.read_text(encoding="utf-8"))
    target = next(item for item in payload["tasks"] if item["id"] == "g3-10000-seed2024")
    weights = merged_weights(2024, "G3_dpo_10000")
    metrics = RUNS / "G3_dpo_10000_seed2024" / "checkpoint" / "adapter" / "metrics.json"
    if weights.is_file() and metrics.is_file():
        return
    if target.get("status") == "done":
        raise SystemExit("G3 10000 for seed 2024 is marked done but the merged weights are missing.")
    target["status"] = "waiting"
    target["steps_done"] = 0
    target["loss"] = None
    target.pop("started_at", None)
    target["message"] = "上次手动停掉，没有存盘。开始后从这一项重新训练。"
    payload["phase"] = "running"
    payload["headline"] = "种子 2024 还没训完：从 G3 一万档重新开始"
    payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    temporary = SEEDS_STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, SEEDS_STATUS)


def prepare() -> None:
    write_manifest(1337)
    write_manifest(2024)
    write_eval_board(1337)
    write_eval_board(2024)
    mark_interrupted_job_waiting()
    print(f"manifests: {manifest_path(1337).name}, {manifest_path(2024).name}")
    print(f"boards: {eval_status_path(1337).name}, {eval_status_path(2024).name}")


def python_commands() -> str:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Select-Object -ExpandProperty CommandLine",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout


def assert_queue_idle() -> None:
    commands = python_commands()
    blockers = ("run_seeds_1337_2024.py", "run_finish_2024_then_eval.py", "train_sft.py", "train_dpo.py", "run_matrix.py")
    own = "run_finish_2024_then_eval.py"
    hits = [line.strip() for line in commands.splitlines() if any(name in line for name in blockers)]
    hits = [line for line in hits if own not in line or "train_" in line or "run_matrix.py" in line or "run_seeds_" in line]
    if hits:
        raise SystemExit("Another training or evaluation process is already running:\n" + "\n".join(hits))


def run_step(command: list[str], extra_env: dict[str, str] | None = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(command, cwd=str(SHARED), env=env)
    return completed.returncode


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--prepare":
        prepare()
        return 0
    assert_queue_idle()
    keep_awake()
    try:
        log("step 1: finish seed 2024 training; seed 1337 completed tasks are skipped")
        code = run_step([str(PYTHON), str(TRAIN)])
        if code != 0:
            log(f"training exited {code}; evaluations not started")
            return code
        missing_2024 = missing_final_weights(2024)
        if missing_2024:
            log("seed 2024 training finished without these merged weights: " + ", ".join(missing_2024))
            return 1
        missing_1337 = missing_final_weights(1337)
        if missing_1337:
            log("seed 1337 is missing merged weights: " + ", ".join(missing_1337))
            return 1
        for seed in (1337, 2024):
            log(f"step: evaluate seed {seed}")
            code = run_step(
                [str(PYTHON), str(MATRIX), "--manifest", str(manifest_path(seed)), "--skip-judge", "--sensitivity"],
                {"EVAL_STATUS": str(eval_status_path(seed))},
            )
            if code != 0:
                log(f"seed {seed} evaluation exited {code}")
                return code
        log("seed 2024 training, seed 1337 evaluation, and seed 2024 evaluation are done")
        return 0
    finally:
        allow_sleep()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"queue failed: {exc}")
        raise
