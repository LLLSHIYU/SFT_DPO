"""Train seeds 1337 and 2024 after seed 42.

Each seed runs G2, G3, G4, and G5 at 500, 2000, and 10000. M0 is not retrained.
Stage 1 and the single-stage arms start from seed 42's merged M0. G4 and G5 stage 2
start from that same seed's own stage 1. Progress is written to seeds_status.json.
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
M0 = SHARED / "03_runs" / "M0_warmup_seed42_20261105" / "checkpoint" / "merged"
DATA = SHARED / "SUMMARY_DATA"
STATUS = SHARED / "LSYNEED" / "monitor" / "seeds_status.json"
PIPELINE_LOG = SHARED / "LSYNEED" / "monitor" / "seeds_pipeline.log"
SEEDS = (1337, 2024)

# short id, kind, data relative to v1.0, max_length, start id, reference id, folder
JOBS = [
    ("g2-500", "sft", "arms/G2_sft_500.jsonl", 1024, "m0", None, "G2_sft_500"),
    ("g2-2000", "sft", "arms/G2_sft_2000.jsonl", 1024, "m0", None, "G2_sft_2000"),
    ("g2-10000", "sft", "arms/G2_sft_10000.jsonl", 1024, "m0", None, "G2_sft_10000"),
    ("g3-500", "dpo", "arms/G3_dpo_500.jsonl", 1024, "m0", "m0", "G3_dpo_500"),
    ("g3-2000", "dpo", "arms/G3_dpo_2000.jsonl", 1024, "m0", "m0", "G3_dpo_2000"),
    ("g3-10000", "dpo", "arms/G3_dpo_10000.jsonl", 1024, "m0", "m0", "G3_dpo_10000"),
    ("g4-500-s1", "sft", "arms/G4_stage1_sft_500.jsonl", 1024, "m0", None, "G4_stage1_sft_500"),
    ("g4-500-s2", "dpo", "arms/G4_stage2_dpo_500.jsonl", 1024, "g4-500-s1", "g4-500-s1", "G4_stage2_dpo_500"),
    ("g4-2000-s1", "sft", "arms/G4_stage1_sft_2000.jsonl", 1024, "m0", None, "G4_stage1_sft_2000"),
    ("g4-2000-s2", "dpo", "arms/G4_stage2_dpo_2000.jsonl", 1024, "g4-2000-s1", "g4-2000-s1", "G4_stage2_dpo_2000"),
    ("g4-10000-s1", "sft", "arms/G4_stage1_sft_10000.jsonl", 1024, "m0", None, "G4_stage1_sft_10000"),
    ("g4-10000-s2", "dpo", "arms/G4_stage2_dpo_10000.jsonl", 1024, "g4-10000-s1", "g4-10000-s1", "G4_stage2_dpo_10000"),
    ("g5-500-s1", "dpo", "arms/G5_stage1_dpo_500.jsonl", 1024, "m0", "m0", "G5_stage1_dpo_500"),
    ("g5-500-s2", "sft", "arms/G5_stage2_sft_500.jsonl", 1024, "g5-500-s1", None, "G5_stage2_sft_500"),
    ("g5-2000-s1", "dpo", "arms/G5_stage1_dpo_2000.jsonl", 1024, "m0", "m0", "G5_stage1_dpo_2000"),
    ("g5-2000-s2", "sft", "arms/G5_stage2_sft_2000.jsonl", 1024, "g5-2000-s1", None, "G5_stage2_sft_2000"),
    ("g5-10000-s1", "dpo", "arms/G5_stage1_dpo_10000.jsonl", 1024, "m0", "m0", "G5_stage1_dpo_10000"),
    ("g5-10000-s2", "sft", "arms/G5_stage2_sft_10000.jsonl", 1024, "g5-10000-s1", None, "G5_stage2_sft_10000"),
]

TITLES = {
    "g2": ("G2", "只做 SFT", "SFT"),
    "g3": ("G3", "只做 DPO", "DPO"),
    "g4-s1": ("G4", "第一阶段 SFT", "SFT"),
    "g4-s2": ("G4", "第二阶段 DPO", "DPO"),
    "g5-s1": ("G5", "第一阶段 DPO", "DPO"),
    "g5-s2": ("G5", "第二阶段 SFT", "SFT"),
}


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


def task_id(short: str, seed: int) -> str:
    return f"{short}-seed{seed}"


def title_key(short: str) -> str:
    arm = short.split("-")[0]
    if arm in {"g2", "g3"}:
        return arm
    return f"{arm}-{short.rsplit('-', 1)[-1]}"


def has_weights(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or any(path.glob("model-*.safetensors"))


def run_dir_for(task: dict) -> Path:
    return SHARED / task["output_dir"]


def merged_dir(task: dict) -> Path:
    return run_dir_for(task) / "checkpoint" / "merged"


def adapter_dir(task: dict) -> Path:
    return run_dir_for(task) / "checkpoint" / "adapter"


def weights_for(ref: str | None, seed: int, tasks: dict[str, dict]) -> Path:
    if ref is None:
        return BASE
    if ref == "m0":
        if not has_weights(M0):
            raise FileNotFoundError(f"Seed 42 M0 merged weights missing: {M0}")
        return M0
    path = merged_dir(tasks[task_id(ref, seed)])
    if not has_weights(path):
        raise FileNotFoundError(f"Merged weights missing for {task_id(ref, seed)}: {path}")
    return path


def write_config(task: dict, seed: int, kind: str, data_rel: str, max_length: int, model: Path, reference: Path | None) -> Path:
    adapter = adapter_dir(task)
    adapter.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": model.as_posix(),
        "data_path": (DATA / data_rel).as_posix(),
        "output_dir": adapter.as_posix(),
        "seed": seed,
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
    config_dir = TRAIN / "configs" / f"seed{seed}"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{task['id']}.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copyfile(config_path, run_dir_for(task) / "config.yaml")
    return config_path


def run_process(script: Path, args: list[str], task_id_value: str | None) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["MONITOR_STATUS"] = str(STATUS)
    if task_id_value:
        env["MONITOR_TASK_ID"] = task_id_value
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


def write_readme(task: dict, seed: int, started: float, loss: float | None, steps: int) -> None:
    hours = (time.time() - started) / 3600
    loss_text = "n/a" if loss is None else f"{loss:.4f}"
    text = (
        f"{task['group']} {task['title']}，预算 {task['budget']}，种子 {seed}，数据 SUMMARY_DATA。"
        f"耗时 {hours:.2f} 小时，最终 loss {loss_text}，实际步数 {steps}。"
        f"显卡 RTX 4060 Laptop 8GB。\n"
    )
    (run_dir_for(task) / "README.md").write_text(text, encoding="utf-8")


def train_and_merge(task: dict, seed: int, kind: str, data_rel: str, max_length: int, model: Path, reference: Path | None) -> None:
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
        config_path = write_config(task, seed, kind, data_rel, max_length, model, reference)
        script = TRAIN / ("train_sft.py" if kind == "sft" else "train_dpo.py")
        code = run_process(script, ["--config", str(config_path)], task["id"])
        if code != 0 or not metrics_path.is_file():
            update_task(task["id"], status="failed", message=f"训练进程退出码 {code}。日志在 LSYNEED/monitor/seeds_pipeline.log")
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
    write_readme(task, seed, started, float(metrics["training_loss"]), int(metrics["global_step"]))
    update_task(
        task["id"],
        status="done",
        steps_done=int(metrics["global_step"]),
        loss=float(metrics["training_loss"]),
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        message=f"完成，{metrics['global_step']} 步，loss {float(metrics['training_loss']):.4f}",
    )
    log(f"done {task['id']} steps={metrics['global_step']} loss={metrics['training_loss']}")


def write_waiting_status() -> None:
    """Create the waiting snapshot. Refuses to overwrite a queue that has already started."""
    if STATUS.is_file():
        current = json.loads(STATUS.read_text(encoding="utf-8"))
        if any(item.get("status") != "waiting" for item in current.get("tasks", [])):
            raise SystemExit("seeds_status.json already has started tasks; not rebuilding it.")
    done = {item["id"]: item for item in json.loads((STATUS.parent / "status.json").read_text(encoding="utf-8"))["tasks"]}
    tasks = []
    order = 1
    for seed in SEEDS:
        for short, _kind, data_rel, _max_length, _start, _ref, folder in JOBS:
            base = done[short]
            start = datetime.fromisoformat(base["started_at"])
            finish = datetime.fromisoformat(base["finished_at"])
            seconds = max(1, round((finish - start).total_seconds()))
            arm, title, kind_label = TITLES[title_key(short)]
            tasks.append({
                "id": task_id(short, seed),
                "order": order,
                "group": f"种子 {seed}",
                "arm": arm,
                "title": f"{arm} {title}",
                "budget": base["budget"],
                "kind": kind_label,
                "data_file": data_rel,
                "rows": base["rows"],
                "avg_tokens": base["avg_tokens"],
                "steps_total": base["steps_total"],
                "steps_done": 0,
                "estimate_seconds": seconds,
                "status": "waiting",
                "loss": None,
                "message": "等种子 42 的评测全部完成后自动开始。从种子 42 的 M0 出发；G4、G5 的第二阶段用这一颗种子自己的第一阶段。",
                "output_dir": f"03_runs/{folder}_seed{seed}",
            })
            order += 1
    one = sum(item["estimate_seconds"] for item in tasks if item["group"] == "种子 1337")
    total = sum(item["estimate_seconds"] for item in tasks)
    document = {
        "phase": "not_started",
        "headline": "种子 1337 和 2024 还没开始",
        "done_headline": "种子 1337 和 2024 的 36 个任务都跑完了",
        "gpu": "RTX 4060 Laptop",
        "estimate_note": (
            f"每个种子 18 项：G2、G3、G4、G5 的 500、2000、10000 档。G4、G5 的一万档各两个阶段。"
            f"M0 不重跑。一颗种子约 {round(one / 3600, 1)} 小时，两颗合计约 {round(total / 3600, 1)} 小时。"
            "评测 65 项全部完成后自动开始，先 1337，再 2024。"
        ),
        "updated_at": None,
        "tasks": tasks,
    }
    STATUS.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"tasks={len(tasks)} one_seed_seconds={one} total_seconds={total}")


def expected_ids() -> list[str]:
    return [task_id(short, seed) for seed in SEEDS for short, *_rest in JOBS]


def main(only_seed: int | None = None) -> int:
    os.environ["MONITOR_STATUS"] = str(STATUS)
    seeds = (only_seed,) if only_seed else SEEDS
    keep_awake()
    tasks = {item["id"]: item for item in json.loads(STATUS.read_text(encoding="utf-8"))["tasks"]}
    if list(tasks) != expected_ids():
        raise SystemExit("Monitor tasks and runner jobs are out of order.")
    log(f"seed {only_seed} queue started" if only_seed else "seeds 1337 and 2024 queue started")
    try:
        for seed in seeds:
            for short, kind, data_rel, max_length, start_id, ref_id, _folder in JOBS:
                task = tasks[task_id(short, seed)]
                model = weights_for(start_id, seed, tasks)
                reference = weights_for(ref_id, seed, tasks) if ref_id else None
                train_and_merge(task, seed, kind, data_rel, max_length, model, reference)
    finally:
        allow_sleep()
    log(f"seed {only_seed} queue finished" if only_seed else "seeds 1337 and 2024 queue finished")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--write-status":
        write_waiting_status()
    else:
        import argparse

        parser = argparse.ArgumentParser(description="Train summary seeds 1337 and 2024, or one of them.")
        parser.add_argument("--seed", type=int, choices=SEEDS)
        args = parser.parse_args()
        try:
            raise SystemExit(main(args.seed))
        except Exception as exc:
            log(f"queue failed: {exc}")
            raise
