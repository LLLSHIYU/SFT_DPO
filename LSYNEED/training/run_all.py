"""Run the full queue once the user says start.

Order: summary seed 42 train, summary seed 42 eval, summary 1337 train, summary 1337 eval,
summary 2024 train, summary 2024 eval, then the same six steps for IF.
IF training starts from the summary M0 and does not train another warmup.
Summary eval is LCWR, MMLU, and ARC. IF eval is IFEval, MMLU, and ARC.
Already finished jobs are skipped.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
from datetime import datetime
from pathlib import Path

SHARED = Path(r"C:\Users\liush\Desktop\DSAI4207_shared")
PYTHON = SHARED / "02_code" / "DSAI4207_code" / ".venv" / "Scripts" / "python.exe"
TRAIN = SHARED / "LSYNEED" / "training"
EVAL = SHARED / "PAN_Evaluation" / "run_matrix.py"
BASE = SHARED / "LSYNEED" / "models" / "Qwen2.5-0.5B"
MONITOR = SHARED / "LSYNEED" / "monitor"
LOG = MONITOR / "run_all.log"


def log(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def keep_awake() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)


def allow_sleep() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def run(command: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("MONITOR_STATUS", None)
    env.pop("MONITOR_TASK_ID", None)
    if extra_env:
        env.update(extra_env)
    log("$ " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(TRAIN),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    with LOG.open("a", encoding="utf-8") as handle:
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        log(f"stopped, exit {code}")
        raise SystemExit(code)


def eval_command(manifest: str, status_name: str, skip: list[str]) -> tuple[list[str], dict[str, str]]:
    command = [
        str(PYTHON),
        str(EVAL),
        "--manifest",
        str(SHARED / "PAN_Evaluation" / manifest),
        "--base-model",
        str(BASE),
        "--skip-judge",
        "--resume",
        *skip,
    ]
    return command, {"EVAL_STATUS": str(MONITOR / status_name)}


def phases() -> list[tuple[str, list[str], dict[str, str]]]:
    summary_eval = ["--skip-ifeval"]
    if_eval = ["--skip-tldr"]
    items = [
        ("摘要 种子42 训练", [str(PYTHON), str(TRAIN / "run_seed42.py")], {}),
        ("摘要 种子42 评测", *eval_command("models.json", "eval_status.json", summary_eval)),
        ("摘要 种子1337 训练", [str(PYTHON), str(TRAIN / "run_seeds_1337_2024.py"), "--seed", "1337"], {}),
        ("摘要 种子1337 评测", *eval_command("models_seed1337.json", "eval_status_1337.json", summary_eval)),
        ("摘要 种子2024 训练", [str(PYTHON), str(TRAIN / "run_seeds_1337_2024.py"), "--seed", "2024"], {}),
        ("摘要 种子2024 评测", *eval_command("models_seed2024.json", "eval_status_2024.json", summary_eval)),
        ("IF 种子42 训练", [str(PYTHON), str(TRAIN / "run_if.py"), "--seed", "42"], {}),
        ("IF 种子42 评测", *eval_command("models_if_seed42.json", "if_eval_status.json", if_eval)),
        ("IF 种子1337 训练", [str(PYTHON), str(TRAIN / "run_if.py"), "--seed", "1337"], {}),
        ("IF 种子1337 评测", *eval_command("models_if_seed1337.json", "if_eval_status_1337.json", if_eval)),
        ("IF 种子2024 训练", [str(PYTHON), str(TRAIN / "run_if.py"), "--seed", "2024"], {}),
        ("IF 种子2024 评测", *eval_command("models_if_seed2024.json", "if_eval_status_2024.json", if_eval)),
    ]
    return [(name, command, env) for name, command, env in items]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run summary training and eval, then IF training and eval.")
    parser.add_argument("--check", action="store_true", help="Check files and boards, then exit without using the GPU.")
    args = parser.parse_args()
    prepare = [str(PYTHON), str(TRAIN / "run_if.py"), "--prepare"]
    if args.check:
        run(prepare)
        for name, command, _env in phases():
            print(f"phase {name}: {' '.join(command)}", flush=True)
        print("ready", flush=True)
        return 0
    keep_awake()
    try:
        run(prepare)
        for name, command, env in phases():
            log(f"phase start: {name}")
            run(command, env)
            log(f"phase done: {name}")
        log("all phases finished")
    finally:
        allow_sleep()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"queue failed: {exc}")
        raise
