"""Read and write the monitor snapshot. No training imports, so the runner stays off the GPU."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

def status_path() -> Path:
    override = os.environ.get("MONITOR_STATUS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "monitor" / "status.json"


def load_status() -> dict:
    return json.loads(status_path().read_text(encoding="utf-8"))


def update_task(task_id: str, **fields) -> None:
    path = status_path()
    if not path.is_file():
        return
    payload = load_status()
    target = None
    for task in payload["tasks"]:
        if task["id"] == task_id:
            target = task
            task.update(fields)
            break
    if target is None:
        return
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    payload["updated_at"] = now
    status = target.get("status")
    if status == "running":
        payload["phase"] = "running"
        payload["headline"] = f"正在跑：{target['group']} {target['title']} · {target['budget']}"
    elif status == "merging":
        payload["phase"] = "running"
        payload["headline"] = f"正在合并：{target['group']} {target['title']} · {target['budget']}"
    elif status == "failed":
        payload["phase"] = "failed"
        payload["headline"] = f"停在：{target['group']} {target['title']}"
    elif status == "done" and all(item["status"] == "done" for item in payload["tasks"]):
        payload["phase"] = "done"
        payload["headline"] = payload.get("done_headline") or "种子 42 的 19 个任务都跑完了"
    temporary = path.with_suffix(".json.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error = None
    for _ in range(8):
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise OSError(f"Could not update {path}") from last_error
