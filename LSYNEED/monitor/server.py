"""Local monitor page. It serves the page files and the budget charts.

It does not start training. charts.json is built from eval summary files
that already exist on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SHARED = Path(__file__).resolve().parents[2]
RUNS = SHARED / "03_runs"
RESULTS = SHARED / "04_results"
BUDGETS = ("500", "2000", "10000")
ARMS = (
    ("G2", "G2_sft_{budget}"),
    ("G3", "G3_dpo_{budget}"),
    ("G4", "G4_stage2_dpo_{budget}"),
    ("G5", "G5_stage2_sft_{budget}"),
)
# Ratios are read in stage order. G8 is DPO then SFT at 1:3, so its SFT file
# is the larger one and lives in the 3:1 package. G9 is the reverse.
EXTRA_ARMS = (
    ("G6", "1:3", "G4_stage2_dpo_{budget}", "G6_stage2_dpo_{budget}"),
    ("G7", "3:1", "G4_stage2_dpo_{budget}", "G7_stage2_dpo_{budget}"),
    ("G8", "3:1", "G5_stage2_sft_{budget}", "G8_stage2_sft_{budget}"),
    ("G9", "1:3", "G5_stage2_sft_{budget}", "G9_stage2_sft_{budget}"),
)


def percent(value: float) -> float:
    return round(value * 1000) / 10


def run_dir(seed: str, folder: str) -> Path:
    if seed == "42":
        return RUNS / f"{folder}_seed42_20261105"
    return RUNS / f"{folder}_seed{seed}"


def read_scores(directory: Path, run_id: str) -> list[float] | None:
    paths = {
        "mmlu": directory / "eval" / f"{run_id}.mmlu_500.summary.json",
        "arc": directory / "eval" / f"{run_id}.arc_challenge_300.summary.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    mmlu = json.loads(paths["mmlu"].read_text(encoding="utf-8"))
    arc = json.loads(paths["arc"].read_text(encoding="utf-8"))
    return [
        percent(mmlu["letter_accuracy"]),
        percent(arc["letter_accuracy"]),
        percent(arc["text_accuracy"]),
    ]


def m0_scores() -> list[float] | None:
    folder = "M0_warmup_seed42_20261105"
    return read_scores(RUNS / folder, folder)


def normalize_ratio(value: str) -> str:
    text = value.strip().replace(" ", "")
    return {
        "": "1:1",
        "1:1": "1:1",
        "1/1": "1:1",
        "1_1": "1:1",
        "1:3": "1:3",
        "1/3": "1:3",
        "1_3": "1:3",
        "3:1": "3:1",
        "3/1": "3:1",
        "3_1": "3:1",
    }.get(text, "")


def canonical_arm(arm: str, ratio_field: str) -> str | None:
    """Map a results-table arm onto G2–G9. A blank ratio stays on that arm."""
    name = arm.strip()
    ratio = normalize_ratio(ratio_field)
    if ratio == "1:1" and name not in {"G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"}:
        for separator in ("_", "-", " "):
            if separator not in name:
                continue
            head, tail = name.split(separator, 1)
            parsed = normalize_ratio(tail)
            if parsed:
                name, ratio = head, parsed
                break
    if name in {"G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"} and ratio in {"", "1:1"}:
        return name
    return {
        ("G4", "1:3"): "G6",
        ("G4", "3:1"): "G7",
        ("G5", "3:1"): "G8",
        ("G5", "1:3"): "G9",
    }.get((name, ratio))


def load_lcwr() -> dict[tuple[str, str, str], float]:
    """Length-adjusted win rate vs M0, keyed by arm, budget, and seed."""
    found: dict[tuple[str, str, str], float] = {}
    for table in sorted(RESULTS.glob("lcwr*.csv")):
        with table.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                arm = canonical_arm(row.get("arm", ""), row.get("ratio", "") or "")
                seed = row.get("seed", "")
                if arm is None or seed in {"", "MEAN"} or not row.get("lcwr"):
                    continue
                found[(arm, row.get("budget", ""), seed)] = float(row["lcwr"])
    return found


def find_extra_dir(
    runs: list[Path],
    task: str,
    seed: str,
    ratio: str,
    source_folder: str,
    named_folder: str,
) -> Path | None:
    named = run_dir(seed, named_folder) if task == "summary" else if_run_dir(seed, named_folder)
    if named.is_dir():
        return named
    marker = "ratio_1_3" if ratio == "1:3" else "ratio_3_1"
    short = "_1_3" if ratio == "1:3" else "_3_1"
    spoken = "1to3" if ratio == "1:3" else "3to1"
    hits = []
    for path in runs:
        name = path.name
        if f"seed{seed}" not in name:
            continue
        if task == "summary" and name.startswith("IF_"):
            continue
        if task == "if" and not name.startswith("IF_"):
            continue
        named_hit = named_folder in name
        ratio_hit = source_folder in name and (marker in name or short in name or spoken in name)
        if named_hit or ratio_hit:
            hits.append(path)
    if not hits:
        return None
    hits.sort(key=lambda path: (not (path / "eval").is_dir(), len(path.name), path.name))
    return hits[0]


def extra_summary_row(
    runs: list[Path],
    seed: str,
    budget: str,
    lcwr: dict[tuple[str, str, str], float],
    arm: str,
    ratio: str,
    source_pattern: str,
    named_pattern: str,
) -> list[float | None]:
    directory = find_extra_dir(
        runs,
        "summary",
        seed,
        ratio,
        source_pattern.format(budget=budget),
        named_pattern.format(budget=budget),
    )
    scores = None if directory is None else read_scores(directory, directory.name)
    win = lcwr.get((arm, budget, seed))
    win_cell = None if win is None else percent(win)
    return [win_cell, *(scores or [None, None, None])]


def extra_if_row(
    runs: list[Path],
    seed: str,
    budget: str,
    arm: str,
    ratio: str,
    source_pattern: str,
    named_pattern: str,
) -> list[float | None]:
    directory = find_extra_dir(
        runs,
        "if",
        seed,
        ratio,
        source_pattern.format(budget=budget),
        named_pattern.format(budget=budget),
    )
    if directory is None:
        return [None, None, None, None]
    return read_if_scores(directory, directory.name) or [None, None, None, None]


def seed_charts(
    seed: str,
    baseline: list[float] | None,
    lcwr: dict[tuple[str, str, str], float],
    runs: list[Path],
) -> dict:
    budgets = []
    for budget in BUDGETS:
        rows = {}
        for arm, pattern in ARMS:
            folder = pattern.format(budget=budget)
            directory = run_dir(seed, folder)
            scores = read_scores(directory, directory.name)
            if scores is None:
                return {"ready": False}
            win = lcwr.get((arm, budget, seed))
            rows[arm] = [None if win is None else percent(win), *scores]
        for arm, ratio, source_pattern, named_pattern in EXTRA_ARMS:
            rows[arm] = extra_summary_row(
                runs, seed, budget, lcwr, arm, ratio, source_pattern, named_pattern
            )
        if baseline is not None:
            # LCWR is a win rate against M0, so M0 itself is the 50% tie line.
            rows["M0"] = [50.0 if lcwr else None, *baseline]
        budgets.append({"title": f"{budget} 档", "rows": rows})
    return {"ready": True, "budgets": budgets}


def if_run_dir(seed: str, folder: str) -> Path:
    return RUNS / f"IF_{folder}_seed{seed}"


def read_if_scores(directory: Path, run_id: str) -> list[float] | None:
    paths = {
        "ifeval": directory / "eval" / f"{run_id}.ifeval.summary.json",
        "mmlu": directory / "eval" / f"{run_id}.mmlu_500.summary.json",
        "arc": directory / "eval" / f"{run_id}.arc_challenge_300.summary.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    ifeval = json.loads(paths["ifeval"].read_text(encoding="utf-8"))
    mmlu = json.loads(paths["mmlu"].read_text(encoding="utf-8"))
    arc = json.loads(paths["arc"].read_text(encoding="utf-8"))
    return [
        percent(ifeval["prompt_level_strict_acc"]),
        percent(mmlu["letter_accuracy"]),
        percent(arc["letter_accuracy"]),
        percent(arc["text_accuracy"]),
    ]


def if_m0_scores() -> list[float] | None:
    folder = "M0_warmup_seed42_20261105"
    return read_if_scores(RUNS / folder, folder)


def if_seed_charts(seed: str, baseline: list[float] | None, runs: list[Path]) -> dict:
    budgets = []
    for budget in BUDGETS:
        rows = {}
        for arm, pattern in ARMS:
            folder = pattern.format(budget=budget)
            directory = if_run_dir(seed, folder)
            scores = read_if_scores(directory, directory.name)
            if scores is None:
                return {"ready": False}
            rows[arm] = scores
        for arm, ratio, source_pattern, named_pattern in EXTRA_ARMS:
            rows[arm] = extra_if_row(runs, seed, budget, arm, ratio, source_pattern, named_pattern)
        if baseline is not None:
            rows["M0"] = baseline
        budgets.append({"title": f"{budget} 档", "rows": rows})
    return {"ready": True, "budgets": budgets}




def build_charts() -> dict:
    baseline = m0_scores()
    lcwr = load_lcwr()
    runs = [path for path in RUNS.iterdir() if path.is_dir()] if RUNS.is_dir() else []
    charts = {seed: seed_charts(seed, baseline, lcwr, runs) for seed in ("42", "1337", "2024")}
    if_baseline = if_m0_scores()
    charts["if42"] = if_seed_charts("42", if_baseline, runs)
    charts["if1337"] = if_seed_charts("1337", if_baseline, runs)
    charts["if2024"] = if_seed_charts("2024", if_baseline, runs)
    return charts


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/charts.json":
            body = json.dumps(build_charts(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt: str, *args) -> None:
        if args and str(args[1]) != "200":
            super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the training monitor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    handler = partial(Handler, directory=str(directory))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"monitor http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
