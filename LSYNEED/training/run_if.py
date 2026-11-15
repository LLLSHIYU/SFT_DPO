"""Train the IF arms from the summary M0. There is no second warmup.

Seed 42 is 18 jobs. Seeds 1337 and 2024 are 18 jobs each. G4 and G5 stage 2
start from that seed's own stage 1. Progress is written to if_status.json
and if_seeds_status.json. This script does not start the GPU unless --seed is set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from run_seeds_1337_2024 import JOBS, TITLES, has_weights, title_key
from status_store import update_task

SHARED = Path(r"C:\Users\liush\Desktop\DSAI4207_shared")
PYTHON = SHARED / "02_code" / "DSAI4207_code" / ".venv" / "Scripts" / "python.exe"
TRAIN = SHARED / "LSYNEED" / "training"
M0 = SHARED / "03_runs" / "M0_warmup_seed42_20261105" / "checkpoint" / "merged"
DATA = SHARED / "IF_DATA"
SUMMARY = SHARED / "SUMMARY_DATA"
MONITOR = SHARED / "LSYNEED" / "monitor"
STATUS_42 = MONITOR / "if_status.json"
STATUS_LATER = MONITOR / "if_seeds_status.json"
PIPELINE_LOG = MONITOR / "if_pipeline.log"
SEEDS = (42, 1337, 2024)
LATER = (1337, 2024)

FINALS = [
    ("G2", "500", "G2_sft_500"),
    ("G2", "2000", "G2_sft_2000"),
    ("G2", "10000", "G2_sft_10000"),
    ("G3", "500", "G3_dpo_500"),
    ("G3", "2000", "G3_dpo_2000"),
    ("G3", "10000", "G3_dpo_10000"),
    ("G4", "500", "G4_stage2_dpo_500"),
    ("G4", "2000", "G4_stage2_dpo_2000"),
    ("G4", "10000", "G4_stage2_dpo_10000"),
    ("G5", "500", "G5_stage2_sft_500"),
    ("G5", "2000", "G5_stage2_sft_2000"),
    ("G5", "10000", "G5_stage2_sft_10000"),
]
EVAL_KINDS = [
    ("ifeval", "IFEval", "IFEval", 541, "IF_DATA/eval_benchmarks/ifeval_541.jsonl"),
    ("mmlu", "MMLU", "MMLU", 500, "SUMMARY_DATA/eval_benchmarks/mmlu_500.jsonl"),
    ("arc", "ARC", "ARC 字母 / ARC 全文", 300, "SUMMARY_DATA/eval_benchmarks/arc_challenge_300.jsonl"),
]


def log(message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def run_name(folder: str, seed: int) -> str:
    return f"IF_{folder}_seed{seed}"


def task_id(short: str, seed: int) -> str:
    return short if seed == 42 else f"{short}-seed{seed}"


def status_path(seed: int) -> Path:
    return STATUS_42 if seed == 42 else STATUS_LATER


def load_tasks(seed: int) -> dict[str, dict]:
    payload = json.loads(status_path(seed).read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["tasks"]}


def expected_ids(seed: int) -> list[str]:
    shorts = [short for short, *_rest in JOBS]
    if seed == 42:
        return shorts
    return [f"{short}-seed{item}" for item in LATER for short in shorts]


def run_dir_for(task: dict) -> Path:
    return SHARED / task["output_dir"]


def merged_dir(task: dict) -> Path:
    return run_dir_for(task) / "checkpoint" / "merged"


def adapter_dir(task: dict) -> Path:
    return run_dir_for(task) / "checkpoint" / "adapter"


def weights_for(ref: str | None, seed: int, tasks: dict[str, dict]) -> Path:
    if ref == "m0":
        if not has_weights(M0):
            raise FileNotFoundError(f"Summary M0 merged weights missing: {M0}")
        return M0
    path = merged_dir(tasks[task_id(ref or "", seed)])
    if not has_weights(path):
        raise FileNotFoundError(f"Merged weights missing for {task_id(ref or '', seed)}: {path}")
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
    config_dir = TRAIN / "configs" / "if" / f"seed{seed}"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{task['id']}.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copyfile(config_path, run_dir_for(task) / "config.yaml")
    return config_path


def run_process(script: Path, args: list[str], task_id_value: str | None, status: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["MONITOR_STATUS"] = str(status)
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


def train_and_merge(task: dict, seed: int, kind: str, data_rel: str, max_length: int, model: Path, reference: Path | None) -> None:
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
        code = run_process(script, ["--config", str(config_path)], task["id"], status_path(seed))
        if code != 0 or not metrics_path.is_file():
            update_task(task["id"], status="failed", message=f"训练进程退出码 {code}。日志在 LSYNEED/monitor/if_pipeline.log")
            raise SystemExit(code or 1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    update_task(task["id"], status="merging", steps_done=int(metrics["global_step"]), loss=metrics["training_loss"], message="正在合并 LoRA")
    if not has_weights(merged):
        code = run_process(
            TRAIN / "merge_lora.py",
            ["--base", str(model), "--adapter", str(adapter), "--output", str(merged)],
            None,
            status_path(seed),
        )
        if code != 0 or not has_weights(merged):
            update_task(task["id"], status="failed", message=f"合并进程退出码 {code}")
            raise SystemExit(code or 1)
    log_path = adapter / "train.log"
    if log_path.is_file():
        shutil.copyfile(log_path, run_dir_for(task) / "train.log")
    hours = (time.time() - started) / 3600
    (run_dir_for(task) / "README.md").write_text(
        f"{task['group']} {task['title']}，预算 {task['budget']}，种子 {seed}，数据 IF_DATA。"
        f"耗时 {hours:.2f} 小时，最终 loss {float(metrics['training_loss']):.4f}，实际步数 {metrics['global_step']}。"
        "从摘要 M0 出发。\n",
        encoding="utf-8",
    )
    update_task(
        task["id"],
        status="done",
        steps_done=int(metrics["global_step"]),
        loss=float(metrics["training_loss"]),
        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        message=f"完成，{metrics['global_step']} 步，loss {float(metrics['training_loss']):.4f}",
    )
    log(f"done {task['id']} steps={metrics['global_step']} loss={metrics['training_loss']}")


def train_seed(seed: int) -> int:
    os.environ["MONITOR_STATUS"] = str(status_path(seed))
    tasks = load_tasks(seed)
    if list(tasks) != expected_ids(seed):
        raise SystemExit(f"IF monitor tasks for seed {seed} do not match the runner.")
    log(f"IF seed {seed} queue started")
    for short, kind, data_rel, max_length, start_id, ref_id, _folder in JOBS:
        task = tasks[task_id(short, seed)]
        model = weights_for(start_id, seed, tasks)
        reference = weights_for(ref_id, seed, tasks) if ref_id else None
        train_and_merge(task, seed, kind, data_rel, max_length, model, reference)
    log(f"IF seed {seed} queue finished")
    return 0


def board_is_fresh(path: Path) -> bool:
    if not path.is_file():
        return True
    payload = json.loads(path.read_text(encoding="utf-8"))
    return all(item.get("status") == "waiting" for item in payload.get("tasks", []))


def write_training_boards() -> None:
    summary = {item["id"]: item for item in json.loads((MONITOR / "status.json").read_text(encoding="utf-8"))["tasks"]}
    seed42_tasks = []
    later_tasks = []
    for index, (short, _kind, data_rel, _max_length, _start, _ref, folder) in enumerate(JOBS, 1):
        base = summary[short]
        data_path = DATA / data_rel
        rows = count_lines(data_path)
        steps = (rows + 15) // 16
        if steps != int(base["steps_total"]) or rows != int(base["rows"]):
            raise SystemExit(f"{data_path.name} has {rows} rows ({steps} steps), summary board expects {base['rows']} rows and {base['steps_total']} steps.")
        arm, title, kind_label = TITLES[title_key(short)]
        common = {
            "arm": arm,
            "budget": base["budget"],
            "kind": kind_label,
            "dataset": "IF_DATA",
            "data_file": data_rel.replace("\\", "/"),
            "rows": rows,
            "avg_tokens": base["avg_tokens"],
            "steps_total": steps,
            "steps_done": 0,
            "estimate_seconds": base["estimate_seconds"],
            "status": "waiting",
            "loss": None,
        }
        seed42_tasks.append({
            "id": short,
            "order": index,
            "group": arm,
            "title": title,
            "message": "还没开始。从摘要的 M0 出发，不再另训通用 SFT。",
            "output_dir": f"03_runs/{run_name(folder, 42)}",
            **common,
        })
    later_order = 1
    for seed in LATER:
        for short, _kind, data_rel, _max_length, _start, _ref, folder in JOBS:
            base = summary[short]
            arm, title, kind_label = TITLES[title_key(short)]
            later_tasks.append({
                "id": task_id(short, seed),
                "order": later_order,
                "group": f"种子 {seed}",
                "arm": arm,
                "title": f"{arm} {title}",
                "budget": base["budget"],
                "kind": kind_label,
                "dataset": "IF_DATA",
                "data_file": data_rel.replace("\\", "/"),
                "rows": int(base["rows"]),
                "avg_tokens": base["avg_tokens"],
                "steps_total": int(base["steps_total"]),
                "steps_done": 0,
                "estimate_seconds": base["estimate_seconds"],
                "status": "waiting",
                "loss": None,
                "message": "还没开始。从摘要的 M0 出发。G4、G5 的第二阶段用这一颗种子自己的第一阶段。",
                "output_dir": f"03_runs/{run_name(folder, seed)}",
            })
            later_order += 1
    documents = [
        (STATUS_42, {
            "phase": "not_started",
            "headline": "还没开始",
            "done_headline": "IF 种子 42 的 18 个训练都跑完了",
            "gpu": "RTX 4060 Laptop",
            "estimate_note": "18 项：G2、G3、G4、G5 的 500、2000、10000 档。G4 和 G5 各有两个阶段。不另训 M0，从摘要的 M0 出发。",
            "updated_at": None,
            "tasks": seed42_tasks,
        }),
        (STATUS_LATER, {
            "phase": "not_started",
            "headline": "还没开始",
            "done_headline": "IF 种子 1337 和 2024 的 36 个任务都跑完了",
            "gpu": "RTX 4060 Laptop",
            "estimate_note": "每个种子 18 项。不另训 M0。种子 42 的 IF 评测结束后开始 1337，1337 评完再开始 2024。",
            "updated_at": None,
            "tasks": later_tasks,
        }),
    ]
    for path, document in documents:
        if not board_is_fresh(path):
            print(f"keep {path.name}")
            continue
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.name} tasks={len(document['tasks'])}")


def manifest_for(seed: int) -> list[dict]:
    entries = []
    if seed == 42:
        entries.append({
            "run_id": "M0_warmup_seed42_20261105",
            "run_dir": "03_runs/M0_warmup_seed42_20261105",
            "arm": "M0",
            "budget": "0",
            "seed": "42",
            "model": "03_runs/M0_warmup_seed42_20261105/checkpoint/merged",
        })
    for arm, budget, folder in FINALS:
        name = run_name(folder, seed)
        entries.append({
            "run_id": name,
            "run_dir": f"03_runs/{name}",
            "arm": arm,
            "budget": budget,
            "seed": str(seed),
            "model": f"03_runs/{name}/checkpoint/merged",
        })
    return entries


def eval_task(entry: dict, kind: str, label: str, title: str, items: int, data_file: str, order: int, estimate: int) -> dict:
    arm = str(entry["arm"])
    budget = str(entry["budget"])
    group = "M0" if arm == "M0" else f"{arm} {budget} 档"
    task_key = f"m0-{kind}" if arm == "M0" else f"{arm.lower()}-{budget}-档-{kind}"
    return {
        "id": task_key,
        "order": order,
        "group": group,
        "title": title,
        "kind": label,
        "budget": "M0" if arm == "M0" else budget,
        "models": [arm],
        "item_count": items,
        "steps_total": items,
        "steps_done": 0,
        "status": "waiting",
        "estimate_seconds": estimate,
        "data_file": data_file,
        "output_dir": f"{entry['run_dir']}/eval/",
        "message": "还没开始。",
    }


def write_eval_boards() -> None:
    estimates = {}
    for name in ("eval_status.json", "eval_status_1337.json", "eval_status_2024.json"):
        payload = json.loads((MONITOR / name).read_text(encoding="utf-8"))
        for task in payload["tasks"]:
            estimates.setdefault(task["id"], int(task["estimate_seconds"]))
    targets = {
        42: MONITOR / "if_eval_status.json",
        1337: MONITOR / "if_eval_status_1337.json",
        2024: MONITOR / "if_eval_status_2024.json",
    }
    manifests = SHARED / "PAN_Evaluation"
    for seed, path in targets.items():
        entries = manifest_for(seed)
        manifest_path = manifests / f"models_if_seed{seed}.json"
        manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not board_is_fresh(path):
            print(f"keep {path.name}")
            continue
        tasks = []
        order = 1
        for entry in entries:
            for kind, label, title, items, data_file in EVAL_KINDS:
                guess = estimates.get(eval_task(entry, kind, label, title, items, data_file, order, 0)["id"], items)
                tasks.append(eval_task(entry, kind, label, title, items, data_file, order, guess))
                order += 1
        count = len(tasks)
        document = {
            "phase": "not_started",
            "headline": "还没开始",
            "waiting_headline": "还没开始",
            "done_headline": f"IF 种子 {seed} 的 {count} 项评测都跑完了",
            "gpu": "RTX 4060 Laptop",
            "estimate_note": "每个模型做 IFEval、MMLU、ARC。ARC 字母和 ARC 全文是同一次评测。MMLU 和 ARC 用摘要文件夹里的题。",
            "updated_at": None,
            "tasks": tasks,
        }
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.name} tasks={count}")


def check_files() -> None:
    required = [
        SUMMARY / "warmup_5k.jsonl",
        SUMMARY / "test_200_main.jsonl",
        SUMMARY / "eval_benchmarks" / "mmlu_500.jsonl",
        SUMMARY / "eval_benchmarks" / "arc_challenge_300.jsonl",
        DATA / "eval_benchmarks" / "ifeval_541.jsonl",
        SHARED / "LSYNEED" / "models" / "Qwen2.5-0.5B" / "model.safetensors",
        PYTHON,
    ]
    for _short, _kind, data_rel, *_rest in JOBS:
        required.append(SUMMARY / data_rel)
        required.append(DATA / data_rel)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing files:\n" + "\n".join(missing))
    print(f"files ok, jobs={len(JOBS)}")


def prepare() -> None:
    check_files()
    write_training_boards()
    write_eval_boards()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or train one IF seed.")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.check or args.prepare:
        prepare() if args.prepare else check_files()
        return 0
    if args.seed is None:
        raise SystemExit("Pass --seed 42, 1337, or 2024. The full order is run_all.py.")
    return train_seed(args.seed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"IF queue failed: {exc}")
        raise
