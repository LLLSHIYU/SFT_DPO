"""Obsolete. The old runs were deleted. Start the new queue with run_all.py.

Resume seed 2024 evaluation from the paused board.

Seed 42 and seed 1337 are not retrained and are not evaluated again.
Completed seed 2024 tasks stay as they are. IFEval generation continues
from the partial jsonl. This script does not rebuild the monitor board.
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
MATRIX = SHARED / "PAN_Evaluation" / "run_matrix.py"
MANIFEST = SHARED / "PAN_Evaluation" / "models_seed2024.json"
BOARD = SHARED / "LSYNEED" / "monitor" / "eval_status_2024.json"
RUNS = SHARED / "03_runs"
LOG = SHARED / "LSYNEED" / "monitor" / "resume_2024_eval.log"

FINAL_FOLDERS = (
    "G2_sft_500",
    "G2_sft_2000",
    "G2_sft_10000",
    "G3_dpo_500",
    "G3_dpo_2000",
    "G3_dpo_10000",
    "G4_stage2_dpo_500",
    "G4_stage2_dpo_2000",
    "G4_stage2_dpo_10000",
    "G5_stage2_sft_500",
    "G5_stage2_sft_2000",
    "G5_stage2_sft_10000",
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
    blockers = (
        "run_seeds_1337_2024.py",
        "run_finish_2024_then_eval.py",
        "run_resume_2024_eval.py",
        "train_sft.py",
        "train_dpo.py",
        "run_matrix.py",
    )
    hits = [line.strip() for line in python_commands().splitlines() if any(name in line for name in blockers)]
    hits = [line for line in hits if "run_resume_2024_eval.py" not in line or "run_matrix.py" in line]
    if hits:
        raise SystemExit("Another training or evaluation process is already running:\n" + "\n".join(hits))


def assert_ready() -> tuple[int, int]:
    missing = [
        folder for folder in FINAL_FOLDERS
        if not (RUNS / f"{folder}_seed2024" / "checkpoint" / "merged" / "model.safetensors").is_file()
    ]
    if missing:
        raise SystemExit("Seed 2024 is missing merged weights: " + ", ".join(missing))
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing manifest: {MANIFEST}")
    if not BOARD.is_file():
        raise SystemExit(f"Missing evaluation board: {BOARD}")
    payload = json.loads(BOARD.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    if len(tasks) != 60:
        raise SystemExit(f"Expected 60 evaluation tasks, found {len(tasks)}.")
    done = sum(task.get("status") == "done" for task in tasks)
    if done == len(tasks):
        raise SystemExit("Seed 2024 evaluation is already complete.")
    return done, len(tasks)


def main() -> int:
    assert_queue_idle()
    done, total = assert_ready()
    keep_awake()
    try:
        log(f"resume seed 2024 evaluation: {done} / {total} already done")
        completed = subprocess.run(
            [
                str(PYTHON),
                str(MATRIX),
                "--manifest",
                str(MANIFEST),
                "--skip-judge",
                "--sensitivity",
                "--resume",
            ],
            cwd=str(SHARED),
            env={**os.environ, "EVAL_STATUS": str(BOARD)},
        )
        if completed.returncode != 0:
            log(f"seed 2024 evaluation exited {completed.returncode}")
            return completed.returncode
        log("seed 2024 evaluation is done")
        return 0
    finally:
        allow_sleep()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"resume failed: {exc}")
        raise
