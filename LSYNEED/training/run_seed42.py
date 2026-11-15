"""Run seed 42: M0, then G2, G3, G4, and G5 at 500, 2000, and 10000.

The monitor snapshot is updated before each task, on every training step, and after the merge.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from status_store import update_task

SHARED = Path(r"C:\Users\liush\Desktop\DSAI4207_shared")
PYTHON = SHARED / "02_code" / "DSAI4207_code" / ".venv" / "Scripts" / "python.exe"
TRAIN = SHARED / "LSYNEED" / "training"
BASE = SHARED / "LSYNEED" / "models" / "Qwen2.5-0.5B"
DATA = SHARED / "SUMMARY_DATA"
CONFIGS = TRAIN / "configs" / "seed42"
PIPELINE_LOG = SHARED / "LSYNEED" / "monitor" / "pipeline.log"

# id, kind, data relative to v1.0, max_length, weights id, reference id
JOBS = [
    ("m0", "sft", "warmup_5k.jsonl", 2048, None, None),
    ("g2-500", "sft", "arms/G2_sft_500.jsonl", 1024, "m0", None),
    ("g2-2000", "sft", "arms/G2_sft_2000.jsonl", 1024, "m0", None),
    ("g2-10000", "sft", "arms/G2_sft_10000.jsonl", 1024, "m0", None),
    ("g3-500", "dpo", "arms/G3_dpo_500.jsonl", 1024, "m0", "m0"),
    ("g3-2000", "dpo", "arms/G3_dpo_2000.jsonl", 1024, "m0", "m0"),
    ("g3-10000", "dpo", "arms/G3_dpo_10000.jsonl", 1024, "m0", "m0"),
    ("g4-500-s1", "sft", "arms/G4_stage1_sft_500.jsonl", 1024, "m0", None),
    ("g4-500-s2", "dpo", "arms/G4_stage2_dpo_500.jsonl", 1024, "g4-500-s1", "g4-500-s1"),
    ("g4-2000-s1", "sft", "arms/G4_stage1_sft_2000.jsonl", 1024, "m0", None),
    ("g4-2000-s2", "dpo", "arms/G4_stage2_dpo_2000.jsonl", 1024, "g4-2000-s1", "g4-2000-s1"),
    ("g4-10000-s1", "sft", "arms/G4_stage1_sft_10000.jsonl", 1024, "m0", None),
    ("g4-10000-s2", "dpo", "arms/G4_stage2_dpo_10000.jsonl", 1024, "g4-10000-s1", "g4-10000-s1"),
    ("g5-500-s1", "dpo", "arms/G5_stage1_dpo_500.jsonl", 1024, "m0", "m0"),
    ("g5-500-s2", "sft", "arms/G5_stage2_sft_500.jsonl", 1024, "g5-500-s1", None),
    ("g5-2000-s1", "dpo", "arms/G5_stage1_dpo_2000.jsonl", 1024, "m0", "m0"),
    ("g5-2000-s2", "sft", "arms/G5_stage2_sft_2000.jsonl", 1024, "g5-2000-s1", None),
    ("g5-10000-s1", "dpo", "arms/G5_stage1_dpo_10000.jsonl", 1024, "m0", "m0"),
    ("g5-10000-s2", "sft", "arms/G5_stage2_sft_10000.jsonl", 1024, "g5-10000-s1", None),
]


def keep_awake() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)


def allow_sleep() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def log(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def has_weights(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or any(path.glob("model-*.safetensors"))


def run_dir_for(task: dict) -> Path:
    return SHARED / task["output_dir"]


def merged_dir(task: dict) -> Path:
    return run_dir_for(task) / "checkpoint" / "merged"


def adapter_dir(task: dict) -> Path:
    return run_dir_for(task) / "checkpoint" / "adapter"


def weights_for(task_id: str | None, tasks: dict[str, dict]) -> Path:
    if task_id is None:
        return BASE
    path = merged_dir(tasks[task_id])
    if not has_weights(path):
        raise FileNotFoundError(f"Merged weights missing for {task_id}: {path}")
    return path


def write_config(task: dict, kind: str, data_rel: str, max_length: int, model: Path, reference: Path | None) -> Path:
    adapter = adapter_dir(task)
    adapter.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": model.as_posix(),
        "data_path": (DATA / data_rel).as_posix(),
        "output_dir": adapter.as_posix(),
        "seed": 42,
        "learning_rate": 0.0001 if kind == "sft" else 0.000005,
        "max_length": max_length,
        "max_steps": int(task["steps_total"]),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "warmup_steps": 0.03,
    }
    if kind == "dpo":
        payload["ref_model_path"] = reference.as_posix()
        payload["beta"] = 0.1
    CONFIGS.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS / f"{task['id']}.yaml"
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    config_path.write_text(text, encoding="utf-8")
    shutil.copyfile(config_path, run_dir_for(task) / "config.yaml")
    return config_path


def run_process(script: Path, args: list[str], task_id: str | None) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if task_id:
        env["MONITOR_TASK_ID"] = task_id
        env["MONITOR_STARTED_EPOCH"] = str(time.time())
    command = [str(PYTHON), str(script), *args]
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
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
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        return process.wait()


def write_readme(task: dict, started: float, loss: float | None, steps: int) -> None:
    hours = (time.time() - started) / 3600
    loss_text = "n/a" if loss is None else f"{loss:.4f}"
    text = (
        f"{task['group']} {task['title']}，预算 {task['budget']}，种子 42，数据 SUMMARY_DATA。"
        f"耗时 {hours:.2f} 小时，最终 loss {loss_text}，实际步数 {steps}。"
        f"显卡 RTX 4060 Laptop 8GB。\n"
    )
    (run_dir_for(task) / "README.md").write_text(text, encoding="utf-8")


def train_and_merge(task: dict, kind: str, data_rel: str, max_length: int, model: Path, reference: Path | None) -> None:
    run_dir = run_dir_for(task)
    adapter = adapter_dir(task)
    merged = merged_dir(task)
    metrics_path = adapter / "metrics.json"
    if metrics_path.is_file() and has_weights(merged):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        update_task(
            task["id"],
            status="done",
            steps_done=int(metrics.get("global_step", task["steps_total"])),
            loss=metrics.get("training_loss"),
            message="已完成，跳过",
            finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        log(f"skip {task['id']}")
        return

    started = time.time()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    if not metrics_path.is_file():
        update_task(task["id"], status="running", steps_done=0, loss=None, started_at=started_at, message="正在加载模型和数据")
        config_path = write_config(task, kind, data_rel, max_length, model, reference)
        script = TRAIN / ("train_sft.py" if kind == "sft" else "train_dpo.py")
        code = run_process(script, ["--config", str(config_path)], task["id"])
        if code != 0 or not metrics_path.is_file():
            update_task(task["id"], status="failed", message=f"训练进程退出码 {code}。日志在 LSYNEED/monitor/pipeline.log")
            raise SystemExit(code or 1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    update_task(task["id"], status="merging", steps_done=int(metrics["global_step"]), loss=metrics["training_loss"], message="正在合并 LoRA")
    if not has_weights(merged):
        code = run_process(
            TRAIN / "merge_lora.py",
            ["--base", str(model), "--adapter", str(adapter), "--output", str(merged)],
            None,
        )
        if code != 0 or not has_weights(merged):
            update_task(task["id"], status="failed", message=f"合并进程退出码 {code}")
            raise SystemExit(code or 1)
    log_path = adapter / "train.log"
    if log_path.is_file():
        shutil.copyfile(log_path, run_dir / "train.log")
    write_readme(task, started, float(metrics["training_loss"]), int(metrics["global_step"]))
    update_task(
        task["id"],
        status="done",
        steps_done=int(metrics["global_step"]),
        loss=float(metrics["training_loss"]),
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        message=f"完成，{metrics['global_step']} 步，loss {float(metrics['training_loss']):.4f}",
    )
    log(f"done {task['id']} steps={metrics['global_step']} loss={metrics['training_loss']}")


def main() -> int:
    keep_awake()
    tasks = {item["id"]: item for item in json.loads((SHARED / "LSYNEED" / "monitor" / "status.json").read_text(encoding="utf-8"))["tasks"]}
    if list(tasks) != [job[0] for job in JOBS]:
        raise SystemExit("Monitor tasks and runner jobs are out of order.")
    log("seed 42 queue started")
    try:
        for task_id, kind, data_rel, max_length, start_id, ref_id in JOBS:
            task = tasks[task_id]
            model = weights_for(start_id, tasks)
            reference = weights_for(ref_id, tasks) if ref_id else None
            train_and_merge(task, kind, data_rel, max_length, model, reference)
    finally:
        allow_sleep()
    log("seed 42 queue finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"queue failed: {exc}")
        raise
