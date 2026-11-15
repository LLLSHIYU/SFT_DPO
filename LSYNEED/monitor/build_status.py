"""Write the pre-training monitor snapshot. Does not start training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\liush\Desktop\DSAI4207_shared")
DATA = ROOT / "01_data" / "v1.0"
OUT = Path(__file__).resolve().parent / "status.json"

# One smoke SFT step on this 4060: 3.003s for 368 tokens, one example.
SFT_SECONDS_PER_TOKEN = 3.003 / 368
# DPO scores the pair on both the training model and the frozen reference.
DPO_FACTOR = 2.2
MERGE_SECONDS = 45
DATE = "20261105"


def steps_for(rows: int) -> int:
    return (rows + 15) // 16


def average_tokens(path: Path, kind: str) -> tuple[int, float]:
    total = 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "total_tokens" in row:
                tokens = int(row["total_tokens"])
            else:
                meta = row["meta"]
                tokens = int(meta["prompt_tokens"]) + int(meta["chosen_tokens"])
                if kind == "DPO":
                    tokens += int(meta["rejected_tokens"])
            total += tokens
            count += 1
    if count == 0:
        raise RuntimeError(f"No rows in {path}")
    return count, total / count


def estimate_seconds(kind: str, avg_tokens: float, step_count: int) -> int:
    if kind == "SFT":
        per_step = 16 * avg_tokens * SFT_SECONDS_PER_TOKEN
    else:
        per_step = DPO_FACTOR * 16 * avg_tokens * SFT_SECONDS_PER_TOKEN
    return round(per_step * step_count + MERGE_SECONDS)


def task(
    order: int,
    task_id: str,
    group: str,
    title: str,
    budget: str,
    kind: str,
    data_rel: str,
    output_name: str,
    token_source: Path | None = None,
) -> dict:
    path = DATA / data_rel
    rows, avg = average_tokens(path if token_source is None else token_source, kind)
    if token_source is not None:
        rows, _ = average_tokens(path, kind)
    step_count = steps_for(rows)
    return {
        "id": task_id,
        "order": order,
        "group": group,
        "title": title,
        "budget": budget,
        "kind": kind,
        "data_file": data_rel.replace("\\", "/"),
        "rows": rows,
        "avg_tokens": round(avg, 1),
        "steps_total": step_count,
        "steps_done": 0,
        "estimate_seconds": estimate_seconds(kind, avg, step_count),
        "step_seconds": None,
        "status": "waiting",
        "loss": None,
        "message": "还没开始",
        "started_at": None,
        "finished_at": None,
        "output_dir": f"03_runs/{output_name}_{DATE}",
    }


def main() -> None:
    g2_2000 = DATA / "arms" / "G2_sft_2000.jsonl"
    g2_10000 = DATA / "arms" / "G2_sft_10000.jsonl"
    g3_500 = DATA / "arms" / "G3_dpo_500.jsonl"
    g4_s1_500 = DATA / "arms" / "G4_stage1_sft_500.jsonl"
    g4_s2_500 = DATA / "arms" / "G4_stage2_dpo_500.jsonl"
    jobs = [
        task(1, "m0", "M0", "通用 SFT", "5000", "SFT", "warmup_5k.jsonl", "M0_warmup_seed42"),
        task(2, "g2-500", "G2", "只做 SFT", "500", "SFT", "arms/G2_sft_500.jsonl", "G2_sft_500_seed42"),
        task(3, "g2-2000", "G2", "只做 SFT", "2000", "SFT", "arms/G2_sft_2000.jsonl", "G2_sft_2000_seed42", g2_2000),
        task(4, "g2-10000", "G2", "只做 SFT", "10000", "SFT", "arms/G2_sft_10000.jsonl", "G2_sft_10000_seed42", g2_10000),
        task(5, "g3-500", "G3", "只做 DPO", "500", "DPO", "arms/G3_dpo_500.jsonl", "G3_dpo_500_seed42"),
        task(6, "g3-2000", "G3", "只做 DPO", "2000", "DPO", "arms/G3_dpo_2000.jsonl", "G3_dpo_2000_seed42", g3_500),
        task(7, "g3-10000", "G3", "只做 DPO", "10000", "DPO", "arms/G3_dpo_10000.jsonl", "G3_dpo_10000_seed42", g3_500),
        task(8, "g4-500-s1", "G4", "第一阶段 SFT", "500", "SFT", "arms/G4_stage1_sft_500.jsonl", "G4_stage1_sft_500_seed42"),
        task(9, "g4-500-s2", "G4", "第二阶段 DPO", "500", "DPO", "arms/G4_stage2_dpo_500.jsonl", "G4_stage2_dpo_500_seed42"),
        task(10, "g4-2000-s1", "G4", "第一阶段 SFT", "2000", "SFT", "arms/G4_stage1_sft_2000.jsonl", "G4_stage1_sft_2000_seed42", g4_s1_500),
        task(11, "g4-2000-s2", "G4", "第二阶段 DPO", "2000", "DPO", "arms/G4_stage2_dpo_2000.jsonl", "G4_stage2_dpo_2000_seed42", g4_s2_500),
        task(12, "g4-10000-s1", "G4", "第一阶段 SFT", "10000", "SFT", "arms/G4_stage1_sft_10000.jsonl", "G4_stage1_sft_10000_seed42", g4_s1_500),
        task(13, "g4-10000-s2", "G4", "第二阶段 DPO", "10000", "DPO", "arms/G4_stage2_dpo_10000.jsonl", "G4_stage2_dpo_10000_seed42", g4_s2_500),
        task(14, "g5-500-s1", "G5", "第一阶段 DPO", "500", "DPO", "arms/G5_stage1_dpo_500.jsonl", "G5_stage1_dpo_500_seed42"),
        task(15, "g5-500-s2", "G5", "第二阶段 SFT", "500", "SFT", "arms/G5_stage2_sft_500.jsonl", "G5_stage2_sft_500_seed42"),
        task(16, "g5-2000-s1", "G5", "第一阶段 DPO", "2000", "DPO", "arms/G5_stage1_dpo_2000.jsonl", "G5_stage1_dpo_2000_seed42", g4_s2_500),
        task(17, "g5-2000-s2", "G5", "第二阶段 SFT", "2000", "SFT", "arms/G5_stage2_sft_2000.jsonl", "G5_stage2_sft_2000_seed42", g4_s1_500),
        task(18, "g5-10000-s1", "G5", "第一阶段 DPO", "10000", "DPO", "arms/G5_stage1_dpo_10000.jsonl", "G5_stage1_dpo_10000_seed42", g4_s2_500),
        task(19, "g5-10000-s2", "G5", "第二阶段 SFT", "10000", "SFT", "arms/G5_stage2_sft_10000.jsonl", "G5_stage2_sft_10000_seed42", g4_s1_500),
    ]
    payload = {
        "version": 1,
        "phase": "not_started",
        "headline": "训练还没开始",
        "seed": 42,
        "gpu": "NVIDIA GeForce RTX 4060 Laptop · 8GB",
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "estimate_note": (
            "开跑前的估计。SFT 按这张 4060 通电时的 3.003 秒 / 368 token，"
            "再乘上每步 16 条和该文件的平均长度。DPO 按同等长度 SFT 的 2.2 倍，"
            "因为要同时算训练模型和冻结的参考模型。每个任务另加 45 秒合并。"
            "真正开跑后，用该任务已经走完的步数重算剩余时间。"
        ),
        "tasks": jobs,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tasks={len(jobs)} seconds={sum(item['estimate_seconds'] for item in jobs)} -> {OUT}")


if __name__ == "__main__":
    main()
