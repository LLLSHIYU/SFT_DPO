"""LCWR length adjustment, bootstrap CIs, paired McNemar tests, OA curve and figures."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import RESULTS_DIR, logistic_fit_at_zero, read_jsonl, write_csv


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def fit_records(records: list[dict[str, Any]]) -> tuple[float, float]:
    y = [float(row["candidate_wins"]) for row in records]
    x = [float(row["log_length_delta"]) for row in records]
    if max(x, default=0.0) - min(x, default=0.0) < 1e-12:
        return sum(y) / len(y), 0.0
    try:
        return logistic_fit_at_zero(y, x)
    except ValueError:
        # If both systems always use the same length, length cannot confound the estimate.
        if max(x, default=0.0) - min(x, default=0.0) < 1e-12:
            return sum(y) / len(y), 0.0
        raise


def bootstrap_lcwr(records: list[dict[str, Any]], n_boot: int, rng: random.Random) -> tuple[float, float]:
    n = len(records)
    estimates = []
    for _ in range(n_boot):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        estimates.append(fit_records(sample)[0])
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def binary_vote(value: float) -> int | None:
    if value > 0.5:
        return 1
    if value < 0.5:
        return 0
    return None


def plot_results(lcwr_rows: list[dict[str, Any]], oa_rows: list[dict[str, Any]], capability_summaries: list[Path], out_dir: Path, date_tag: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plot generation requires matplotlib; install requirements-eval.txt") from exc

    aggregate = [row for row in lcwr_rows if row.get("seed") == "MEAN"]
    if not aggregate:
        aggregate = lcwr_rows
    oa_by_budget = {str(row["budget"]): row for row in oa_rows}
    budgets = [budget for budget in ("500", "2000", "10000") if budget in oa_by_budget]
    if budgets:
        x = [int(value) for value in budgets]
        means = [oa_by_budget[value]["oa_lcwr"] for value in budgets]
        low = [means[i] - oa_by_budget[value]["ci_low"] for i, value in enumerate(budgets)]
        high = [oa_by_budget[value]["ci_high"] - means[i] for i, value in enumerate(budgets)]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(x, means, yerr=[low, high], marker="o", capsize=4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks(x, labels=budgets)
        ax.set_xlabel("Unique prompt budget N")
        ax.set_ylabel("OA(N) = LCWR(G4) − LCWR(G5)")
        ax.set_title("Order-attributable LCWR gap (95% prompt bootstrap CI)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_oa_curve_v1.0_{date_tag}.pdf")
        plt.close(fig)

    summaries = []
    for path in capability_summaries:
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    m0_by_benchmark: dict[str, dict[str, Any]] = {}
    for row in summaries:
        by_key[(str(row.get("budget", "")), str(row.get("seed", "")), str(row.get("benchmark", "")))].append(row)
        if row.get("arm") == "M0":
            m0_by_benchmark[str(row.get("benchmark", ""))] = row
    points: dict[str, list[dict[str, Any]]] = {"mmlu": [], "arc": []}
    for lcwr in lcwr_rows:
        if lcwr.get("seed") == "MEAN" or lcwr.get("arm") == "M0":
            continue
        budget, seed = str(lcwr.get("budget", "")), str(lcwr.get("seed", ""))
        for benchmark in ("mmlu", "arc"):
            current = next((x for x in by_key[(budget, seed, benchmark)] if x.get("arm") == lcwr.get("arm")), None)
            baseline = next((x for x in by_key[(budget, seed, benchmark)] if x.get("arm") == "M0"), m0_by_benchmark.get(benchmark))
            if current and baseline:
                metric = "mean_gold_letter_logprob" if benchmark == "mmlu" else "mean_gold_text_norm_logprob"
                points[benchmark].append({
                    "gain_lcwr": float(lcwr["lcwr"]),
                    "capability_tax": float(baseline[metric]) - float(current[metric]),
                    "arm": lcwr["arm"], "budget": budget, "seed": seed,
                })
    if any(points.values()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        metric_titles = {
            "mmlu": "MMLU single-token gold-label log-prob tax",
            "arc": "ARC length-normalized gold-choice log-prob tax",
        }
        for ax, benchmark in zip(axes, ("mmlu", "arc")):
            for arm in sorted({str(point["arm"]) for point in points[benchmark]}):
                selected = [point for point in points[benchmark] if point["arm"] == arm]
                ax.scatter([p["capability_tax"] for p in selected], [p["gain_lcwr"] for p in selected], label=arm, alpha=0.8)
                for point in selected:
                    ax.annotate(f"{point['budget']}/{point['seed']}", (point["capability_tax"], point["gain_lcwr"]), fontsize=6, alpha=0.65)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
            ax.set_xlabel(f"Capability tax (M0 − model; {benchmark.upper()} log-prob)")
            ax.set_ylabel("LCWR vs M0 (adjusted to equal log length)")
            ax.set_title(metric_titles[benchmark])
            ax.grid(alpha=0.2)
            if points[benchmark]:
                ax.legend()
        fig.suptitle("Gain–tax trade-off")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_gain_tax_v1.0_{date_tag}.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, nargs="+", help="Raw swapped-position judgment JSONL files")
    parser.add_argument("--input-dir", type=Path, default=Path("03_runs"), help="Recursively finds eval/lcwr*.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--capability-summaries", type=Path, nargs="*", default=[])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4207)
    args = parser.parse_args()

    paths = args.judgments or sorted(args.input_dir.rglob("lcwr*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No LCWR judgment JSONL files found under {args.input_dir}")
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    rng = random.Random(args.seed)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in paths:
        records = read_jsonl(path)
        if not records:
            continue
        first = records[0]
        key = (str(first.get("arm", "unknown")), str(first.get("budget", "")), str(first.get("seed", "")))
        ids = [str(row["item_id"]) for row in records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate item_id in {path}")
        if any((str(row.get("arm")), str(row.get("budget")), str(row.get("seed"))) != key for row in records):
            raise ValueError(f"Mixed experiment metadata in {path}")
        if key in grouped:
            raise ValueError(f"Duplicate judgment set for arm/budget/seed {key}; specify one file per run")
        grouped[key] = records

    lcwr_rows: list[dict[str, Any]] = []
    fit_by_key = {}
    for key, records in sorted(grouped.items()):
        point, slope = fit_records(records)
        low, high = bootstrap_lcwr(records, args.bootstrap, rng)
        fit_by_key[key] = (point, records)
        lcwr_rows.append({
            "arm": key[0], "budget": key[1], "seed": key[2], "n_prompts": len(records),
            "lcwr": point, "ci_low": low, "ci_high": high, "length_logit_slope": slope,
            "mean_unadjusted_win": sum(float(row["candidate_wins"]) for row in records) / len(records),
            "judge_model": records[0].get("judge_model", ""), "data_version": "v1.0",
        })

    # Seed-level summaries (mean ± sample SD; 10k single-seed results remain descriptive).
    for arm, budget in sorted({(key[0], key[1]) for key in fit_by_key}):
        rows = [row for row in lcwr_rows if row["arm"] == arm and row["budget"] == budget]
        values = [float(row["lcwr"]) for row in rows]
        sd = math.sqrt(sum((value - sum(values) / len(values)) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
        if len(values) > 1:
            run_records = {
                str(row["seed"]): {str(item["item_id"]): item for item in fit_by_key[(arm, budget, str(row["seed"]))][1]}
                for row in rows
            }
            id_sets = [set(items) for items in run_records.values()]
            if any(ids != id_sets[0] for ids in id_sets[1:]):
                raise ValueError(f"Seeds do not share identical test IDs for arm={arm}, budget={budget}")
            shared_ids = sorted(id_sets[0])
            mean_boot = []
            for _ in range(args.bootstrap):
                sampled_ids = [shared_ids[rng.randrange(len(shared_ids))] for _ in shared_ids]
                estimates = [
                    fit_records([run_records[seed][item] for item in sampled_ids])[0]
                    for seed in sorted(run_records)
                ]
                mean_boot.append(sum(estimates) / len(estimates))
            mean_low, mean_high = percentile(mean_boot, 0.025), percentile(mean_boot, 0.975)
        else:
            mean_low, mean_high = float(rows[0]["ci_low"]), float(rows[0]["ci_high"])
        lcwr_rows.append({
            "arm": arm, "budget": budget, "seed": "MEAN", "n_seeds": len(values),
            "lcwr": sum(values) / len(values), "seed_sd": sd,
            "ci_low": mean_low, "ci_high": mean_high,
            "data_version": "v1.0",
        })

    oa_rows = []
    for budget in sorted({key[1] for key in fit_by_key}, key=lambda value: int(value or 0)):
        seeds = sorted({key[2] for key in fit_by_key if key[1] == budget and key[0] in {"G4", "G5"}})
        paired = []
        seed_oa = []
        for seed in seeds:
            k4, k5 = ("G4", budget, seed), ("G5", budget, seed)
            if k4 not in fit_by_key or k5 not in fit_by_key:
                continue
            value4, rec4 = fit_by_key[k4]
            value5, rec5 = fit_by_key[k5]
            by4 = {str(row["item_id"]): row for row in rec4}
            by5 = {str(row["item_id"]): row for row in rec5}
            shared = sorted(set(by4) & set(by5))
            if not shared:
                continue
            pair_rows = [(by4[item], by5[item]) for item in shared]
            seed_oa.append(value4 - value5)
            paired.append((seed, pair_rows))

        if not paired:
            continue
        # Prompt-cluster bootstrap, paired across G4/G5 and averaged across seeds.
        boot = []
        for _ in range(args.bootstrap):
            effects = []
            for _, pair_rows in paired:
                sampled = [pair_rows[rng.randrange(len(pair_rows))] for _ in range(len(pair_rows))]
                records4 = [pair[0] for pair in sampled]
                records5 = [pair[1] for pair in sampled]
                effects.append(fit_records(records4)[0] - fit_records(records5)[0])
            boot.append(sum(effects) / len(effects))
        mean_oa = sum(seed_oa) / len(seed_oa)
        # Aggregate paired G4/G5 majority outcomes across shared prompts for exact McNemar.
        mcnemar_by_seed = {}
        total_paired = 0
        for seed, pair_rows in paired:
            g4_win_g5_loss = g4_loss_g5_win = 0
            for row4, row5 in pair_rows:
                v4, v5 = binary_vote(float(row4["candidate_wins"])), binary_vote(float(row5["candidate_wins"]))
                if v4 is not None and v5 is not None:
                    if v4 == 1 and v5 == 0:
                        g4_win_g5_loss += 1
                    elif v4 == 0 and v5 == 1:
                        g4_loss_g5_win += 1
            total_paired += len(pair_rows)
            mcnemar_by_seed[seed] = {
                "g4_win_g5_loss": g4_win_g5_loss,
                "g4_loss_g5_win": g4_loss_g5_win,
                "exact_p": exact_mcnemar_p(g4_win_g5_loss, g4_loss_g5_win),
                "n_paired_prompts": len(pair_rows),
            }
        oa_rows.append({
            "budget": budget, "n_seeds": len(paired), "n_paired_prompts_per_seed": total_paired,
            "oa_lcwr": mean_oa, "ci_low": percentile(boot, 0.025), "ci_high": percentile(boot, 0.975),
            "mcnemar_by_seed_json": json.dumps(mcnemar_by_seed),
            "seed_oa_json": json.dumps(seed_oa), "interpretation": "descriptive_single_seed" if len(paired) == 1 else "paired_seed_analysis",
            "data_version": "v1.0",
        })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = dt.date.today().strftime("%Y%m%d")
    lcwr_path = output_dir / f"lcwr_v1.0_{date_tag}.csv"
    oa_path = output_dir / f"oa_v1.0_{date_tag}.csv"
    write_csv(lcwr_path, lcwr_rows)
    if oa_rows:
        write_csv(oa_path, oa_rows)
    cap_paths = list(args.capability_summaries)
    if not cap_paths:
        cap_paths = sorted(args.input_dir.rglob("*.summary.json"))
    capability_rows = []
    if eval_rows := sorted(args.input_dir.rglob("*.ifeval.summary.json")):
        ifeval_rows = [json.loads(path.read_text(encoding="utf-8")) for path in eval_rows]
        write_csv(output_dir / f"ifeval_v1.0_{date_tag}.csv", ifeval_rows)
    else:
        ifeval_rows = []
    for path in cap_paths:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "benchmark" in summary:
            capability_rows.append(summary)
    if capability_rows:
        write_csv(output_dir / f"capability_tax_v1.0_{date_tag}.csv", capability_rows)
    plot_results(lcwr_rows, oa_rows, cap_paths, output_dir, date_tag)
    print(f"LCWR table: {lcwr_path}")
    if oa_rows:
        print(f"OA/paired McNemar table: {oa_path}")
    print(f"Bootstrap resamples: {args.bootstrap}; seed: {args.seed}")


if __name__ == "__main__":
    main()
