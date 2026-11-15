"""Update the evaluation rows on the local monitor page."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from common import ROOT


def status_path() -> Path:
    override = os.environ.get("EVAL_STATUS")
    if override:
        return Path(override)
    return ROOT / "LSYNEED" / "monitor" / "eval_status.json"


def update_task(task_id: str | None, **fields) -> None:
    path = status_path()
    if not task_id or not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = next((task for task in payload["tasks"] if task["id"] == task_id), None)
    if target is None:
        return
    if fields.get("status") == "done":
        fields["steps_done"] = target.get("steps_total", target.get("item_count", 1))
    if fields.get("started_at") and target.get("started_at"):
        fields.pop("started_at")
    target.update(fields)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    payload["updated_at"] = now
    tasks = payload["tasks"]
    status = target.get("status")
    if status == "running":
        payload["phase"] = "running"
        payload["headline"] = (
            f"正在评测：{target['group']} · {target['title']}"
            f"（{target.get('steps_done', 0)} / {target.get('steps_total', 0)}）"
        )
    elif status == "failed":
        payload["phase"] = "failed"
        payload["headline"] = f"停在：{target['group']} · {target['title']}"
    elif all(item["status"] == "done" for item in tasks):
        payload["phase"] = "done"
        payload["headline"] = payload.get("done_headline") or "种子 42 的 65 项评测都跑完了"
    else:
        done = sum(item["status"] == "done" for item in tasks)
        if any(item["status"] == "running" for item in tasks):
            payload["phase"] = "running"
        elif any(item["status"] == "failed" for item in tasks):
            payload["phase"] = "failed"
        elif done:
            payload["phase"] = "running"
            payload["headline"] = f"评测进行中：{done} / {len(tasks)} 项已完成"
        else:
            payload["phase"] = "not_started"
            payload["headline"] = payload.get("waiting_headline") or "种子 42 的评测还没开始"
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
    print(f"Could not update the monitor: {last_error}", flush=True)


def note(message: str) -> None:
    task_id = os.environ.get("EVAL_MONITOR_TASK")
    try:
        update_task(task_id, status="running", message=message)
    except Exception as exc:
        print(f"Monitor update skipped: {exc}", flush=True)


def report(steps_done: int, steps_total: int, message: str | None = None) -> None:
    task_id = os.environ.get("EVAL_MONITOR_TASK")
    fields = {"steps_done": steps_done, "steps_total": steps_total, "status": "running"}
    if message:
        fields["message"] = message
    try:
        update_task(task_id, **fields)
    except Exception as exc:
        print(f"Monitor update skipped: {exc}", flush=True)
