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


def load_lcwr() -> dict[tuple[str, str, str], float]:
    """Length-adjusted win rate vs M0, keyed by arm, budget, and seed."""
    tables = sorted(RESULTS.glob("lcwr_v1.0_*.csv"))
    if not tables:
        return {}
    found: dict[tuple[str, str, str], float] = {}
    with tables[-1].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            seed = row.get("seed", "")
            arm = row.get("arm", "")
            if not arm or seed in {"", "MEAN"} or not row.get("lcwr"):
                continue
            found[(arm, row["budget"], seed)] = float(row["lcwr"])
    return found


def seed_charts(seed: str, baseline: list[float] | None, lcwr: dict[tuple[str, str, str], float]) -> dict:
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


def if_seed_charts(seed: str, baseline: list[float] | None) -> dict:
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
        if baseline is not None:
            rows["M0"] = baseline
        budgets.append({"title": f"{budget} 档", "rows": rows})
    return {"ready": True, "budgets": budgets}


def build_charts() -> dict:
    baseline = m0_scores()
    lcwr = load_lcwr()
    charts = {seed: seed_charts(seed, baseline, lcwr) for seed in ("42", "1337", "2024")}
    if_baseline = if_m0_scores()
    charts["if42"] = if_seed_charts("42", if_baseline)
    charts["if1337"] = if_seed_charts("1337", if_baseline)
    charts["if2024"] = if_seed_charts("2024", if_baseline)
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
