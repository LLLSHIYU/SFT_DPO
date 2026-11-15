"""Wait until the seed-42 evaluation is fully done, then train seeds 1337 and 2024.

A failed or still-running evaluation does not start training. The wait continues,
so a later successful resume can still hand off to the training queue.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SHARED = Path(r"C:\Users\liush\Desktop\DSAI4207_shared")
PYTHON = SHARED / "02_code" / "DSAI4207_code" / ".venv" / "Scripts" / "python.exe"
EVAL_STATUS = SHARED / "LSYNEED" / "monitor" / "eval_status.json"
LOG = SHARED / "LSYNEED" / "monitor" / "seeds_queue.log"
TRAIN = SHARED / "LSYNEED" / "training" / "run_seeds_1337_2024.py"


def log(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def evaluation_done() -> bool:
    payload = json.loads(EVAL_STATUS.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") or []
    return bool(tasks) and all(item.get("status") == "done" for item in tasks)


def main() -> int:
    log("waiting for all 65 evaluation tasks to finish")
    announced_failure = False
    while True:
        try:
            if evaluation_done():
                break
            payload = json.loads(EVAL_STATUS.read_text(encoding="utf-8"))
            if payload.get("phase") == "failed" and not announced_failure:
                log("evaluation is failed; training stays waiting until every task is done")
                announced_failure = True
            elif payload.get("phase") != "failed":
                announced_failure = False
        except (OSError, json.JSONDecodeError) as exc:
            log(f"could not read eval status, still waiting: {exc}")
        time.sleep(20)
    log("evaluation is done; starting seeds 1337 then 2024")
    completed = subprocess.run([str(PYTHON), str(TRAIN)], cwd=str(TRAIN.parent))
    log(f"training queue exited {completed.returncode}")
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"watcher failed: {exc}")
        raise
