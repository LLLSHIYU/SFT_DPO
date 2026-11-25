"""Write one workbook a teammate can chart without opening 03_runs.

The numbers come from the eval summary JSON files and the LCWR/OA tables
already produced by analyze.py. Proportions stay in 0-1; Excel shows them
as percentages.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from common import RESULTS_DIR, RUNS_DIR

RUN_RE = re.compile(
    r"^(?P<prefix>IF_)?(?P<arm>M0|G2|G3|G4|G5|G6|G7|G8|G9)_"
    r"(?:(?P<stage>stage1|stage2)_)?"
    r"(?P<method>warmup|sft|dpo)_"
    r"(?:(?P<budget>500|2000|10000)_)?"
    r"seed(?P<seed>42|1337|2024)(?:_\d+)?$"
)
ARM_ORDER = {"M0": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, "G7": 6, "G8": 7, "G9": 8}
RATIO = {
    "M0": "", "G2": "", "G3": "",
    "G4": "1:1", "G5": "1:1",
    "G6": "1:3", "G7": "3:1",
    "G8": "1:3", "G9": "3:1",
}
SEQUENCE = {
    "G4": "SFT→DPO", "G5": "DPO→SFT",
    "G6": "SFT→DPO", "G7": "SFT→DPO",
    "G8": "DPO→SFT", "G9": "DPO→SFT",
}
ORDER_PAIRS = (
    ("1:1", "G4", "G5"),
    ("1:3", "G6", "G8"),
    ("3:1", "G7", "G9"),
)
SEED_ORDER = {"42": 0, "1337": 1, "2024": 2, "平均": 3}
TASK_ORDER = {"两边共用": 0, "摘要": 1, "指令跟随": 2}
STAGE_ORDER = {"最终": 0, "第一阶段": 1}
METHOD_LABEL = {
    ("M0", "warmup", "最终"): "只做 warmup",
    ("G2", "sft", "最终"): "只做 SFT",
    ("G3", "dpo", "最终"): "只做 DPO",
    ("G4", "sft", "第一阶段"): "第一阶段 SFT",
    ("G4", "dpo", "最终"): "先 SFT 再 DPO",
    ("G5", "dpo", "第一阶段"): "第一阶段 DPO",
    ("G5", "sft", "最终"): "先 DPO 再 SFT",
    ("G6", "sft", "第一阶段"): "第一阶段 SFT",
    ("G6", "dpo", "最终"): "先 SFT 再 DPO",
    ("G7", "sft", "第一阶段"): "第一阶段 SFT",
    ("G7", "dpo", "最终"): "先 SFT 再 DPO",
    ("G8", "dpo", "第一阶段"): "第一阶段 DPO",
    ("G8", "sft", "最终"): "先 DPO 再 SFT",
    ("G9", "dpo", "第一阶段"): "第一阶段 DPO",
    ("G9", "sft", "最终"): "先 DPO 再 SFT",
}
CHART_METRICS = (
    ("lcwr", "LCWR", "lcwr_low", "lcwr_high", "lcwr_n"),
    ("lcwr_unadjusted", "LCWR未调整", None, None, "lcwr_n"),
    ("mmlu_letter", "MMLU字母", None, None, "mmlu_n"),
    ("mmlu_text", "MMLU全文", None, None, "mmlu_n"),
    ("arc_letter", "ARC字母", None, None, "arc_n"),
    ("arc_text", "ARC全文", None, None, "arc_n"),
    ("ifeval_prompt_strict", "IFEval题目严格", None, None, "ifeval_n"),
    ("ifeval_prompt_loose", "IFEval题目宽松", None, None, "ifeval_n"),
    ("ifeval_inst_strict", "IFEval指令严格", None, None, "ifeval_inst_n"),
    ("ifeval_inst_loose", "IFEval指令宽松", None, None, "ifeval_inst_n"),
)
METRIC_ORDER = {name: index for index, (_, name, *_) in enumerate(CHART_METRICS)}
HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True)
NOTE_FONT = Font(name="Calibri", size=12)


def latest_csv(pattern: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} under {RESULTS_DIR}")
    return matches[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_summaries() -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for directory in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        eval_dir = directory / "eval"
        if not eval_dir.is_dir():
            continue
        bucket: dict[str, Any] = {}
        for kind, suffix in (
            ("mmlu", "mmlu_500.summary.json"),
            ("arc", "arc_challenge_300.summary.json"),
            ("ifeval", "ifeval.summary.json"),
        ):
            path = eval_dir / f"{directory.name}.{suffix}"
            if path.is_file():
                bucket[kind] = json.loads(path.read_text(encoding="utf-8"))
        if bucket:
            found[directory.name] = bucket
    return found


def parse_run(run_id: str) -> dict[str, Any]:
    match = RUN_RE.fullmatch(run_id)
    if match is None:
        raise ValueError(f"Unrecognized run id: {run_id}")
    arm = match.group("arm")
    raw_stage = match.group("stage")
    method = match.group("method")
    if arm in {"M0", "G2", "G3"}:
        stage = "最终"
    elif raw_stage == "stage1":
        stage = "第一阶段"
    elif raw_stage == "stage2":
        stage = "最终"
    else:
        raise ValueError(f"Missing stage in {run_id}")
    budget = 0 if arm == "M0" else int(match.group("budget"))
    label = METHOD_LABEL.get((arm, method, stage))
    if label is None:
        raise ValueError(f"No training label for {run_id}")
    if arm == "M0":
        task = "两边共用"
    elif match.group("prefix"):
        task = "指令跟随"
    else:
        task = "摘要"
    return {
        "run_id": run_id,
        "task": task,
        "arm": arm,
        "stage": stage,
        "method": label,
        "budget": budget,
        "seed": match.group("seed"),
        "ratio": RATIO[arm],
        "sequence": SEQUENCE.get(arm, ""),
    }


def optional_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    return float(text)


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def build_models(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    lcwr_rows = read_csv(latest_csv("lcwr_v1.0_*.csv"))
    lcwr = {
        (row["arm"], int(row["budget"]), row["seed"]): row
        for row in lcwr_rows
        if row.get("seed") not in {"", "MEAN"}
    }
    models = []
    for run_id, files in summaries.items():
        meta = parse_run(run_id)
        mmlu = files.get("mmlu") or {}
        arc = files.get("arc") or {}
        ifeval = files.get("ifeval") or {}
        judged = lcwr.get((meta["arm"], meta["budget"], meta["seed"]))
        row = {
            **meta,
            "mmlu_letter": mmlu.get("letter_accuracy"),
            "mmlu_text": mmlu.get("text_accuracy"),
            "mmlu_letter_logprob": mmlu.get("mean_gold_letter_logprob"),
            "mmlu_text_logprob": mmlu.get("mean_gold_text_norm_logprob"),
            "mmlu_n": mmlu.get("num_items"),
            "arc_letter": arc.get("letter_accuracy"),
            "arc_text": arc.get("text_accuracy"),
            "arc_letter_logprob": arc.get("mean_gold_letter_logprob"),
            "arc_text_logprob": arc.get("mean_gold_text_norm_logprob"),
            "arc_n": arc.get("num_items"),
            "ifeval_prompt_strict": ifeval.get("prompt_level_strict_acc"),
            "ifeval_prompt_loose": ifeval.get("prompt_level_loose_acc"),
            "ifeval_inst_strict": ifeval.get("instruction_level_strict_acc"),
            "ifeval_inst_loose": ifeval.get("instruction_level_loose_acc"),
            "ifeval_n": ifeval.get("num_prompts"),
            "ifeval_inst_n": ifeval.get("num_instructions"),
            "lcwr": None,
            "lcwr_low": None,
            "lcwr_high": None,
            "lcwr_unadjusted": None,
            "lcwr_slope": None,
            "lcwr_n": None,
            "lcwr_source": "",
            "judge": "",
        }
        if meta["arm"] == "M0" and meta["stage"] == "最终":
            row.update(lcwr=0.5, lcwr_n=200, lcwr_source="参照线")
        elif meta["task"] == "摘要" and meta["stage"] == "最终" and judged is not None:
            row.update(
                lcwr=float(judged["lcwr"]),
                lcwr_low=float(judged["ci_low"]),
                lcwr_high=float(judged["ci_high"]),
                lcwr_unadjusted=float(judged["mean_unadjusted_win"]),
                lcwr_slope=float(judged["length_logit_slope"]),
                lcwr_n=int(judged["n_prompts"]),
                lcwr_source="裁判",
                judge=judged.get("judge_model", ""),
            )
        models.append(row)
    return models


def chart_tasks(model: dict[str, Any]) -> tuple[str, ...]:
    if model["task"] == "两边共用":
        return ("摘要", "指令跟随")
    return (model["task"],)


def chart_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model, metric, and seed, plus a three-seed mean row."""
    points: list[dict[str, Any]] = []
    for model in models:
        for task in chart_tasks(model):
            for key, label, low_key, high_key, n_key in CHART_METRICS:
                value = model.get(key)
                if value is None:
                    continue
                if task == "指令跟随" and key.startswith("lcwr"):
                    continue
                if task == "摘要" and key.startswith("ifeval"):
                    continue
                points.append({
                    "task": task,
                    "arm": model["arm"],
                    "stage": model["stage"],
                    "method": model["method"],
                    "ratio": model["ratio"],
                    "sequence": model["sequence"],
                    "budget": model["budget"],
                    "seed": model["seed"],
                    "metric": label,
                    "metric_key": key,
                    "value": value,
                    "low": model.get(low_key) if low_key else None,
                    "high": model.get(high_key) if high_key else None,
                    "sd": None,
                    "n_seeds": 1,
                    "n_items": model.get(n_key),
                })

    lcwr_mean = {
        (row["arm"], int(row["budget"])): row
        for row in read_csv(latest_csv("lcwr_v1.0_*.csv"))
        if row.get("seed") == "MEAN"
    }
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for point in points:
        group = (point["task"], point["arm"], point["stage"], point["budget"], point["metric"])
        grouped.setdefault(group, []).append(point)
    for group, members in grouped.items():
        task, arm, stage, budget, metric = group
        values = [float(member["value"]) for member in members]
        published = lcwr_mean.get((arm, budget)) if metric == "LCWR" and task == "摘要" and stage == "最终" else None
        points.append({
            "task": task,
            "arm": arm,
            "stage": stage,
            "method": members[0]["method"],
            "ratio": members[0]["ratio"],
            "sequence": members[0]["sequence"],
            "budget": budget,
            "seed": "平均",
            "metric": metric,
            "metric_key": members[0]["metric_key"],
            "value": float(published["lcwr"]) if published else sum(values) / len(values),
            "low": float(published["ci_low"]) if published else None,
            "high": float(published["ci_high"]) if published else None,
            "sd": float(published["seed_sd"]) if published else sample_sd(values),
            "n_seeds": int(published["n_seeds"]) if published else len(values),
            "n_items": members[0]["n_items"],
        })
    return points


def oa_rows() -> list[dict[str, Any]]:
    lcwr = {
        (row["arm"], int(row["budget"]), row["seed"]): float(row["lcwr"])
        for row in read_csv(latest_csv("lcwr_v1.0_*.csv"))
        if row.get("seed") not in {"", "MEAN"}
    }
    published = {
        int(row["budget"]): row
        for row in read_csv(latest_csv("oa_v1.0_*.csv"))
    }
    seeds = ("42", "1337", "2024")
    rows = []
    for ratio, first, second in ORDER_PAIRS:
        for budget in (500, 2000, 10000):
            for seed in seeds:
                left = lcwr[(first, budget, seed)]
                right = lcwr[(second, budget, seed)]
                test = None
                if ratio == "1:1":
                    test = json.loads(published[budget]["mcnemar_by_seed_json"])[seed]
                rows.append({
                    "ratio": ratio,
                    "budget": budget,
                    "seed": seed,
                    "first_arm": first,
                    "second_arm": second,
                    "first": left,
                    "second": right,
                    "oa": left - right,
                    "low": None,
                    "high": None,
                    "first_win_second_loss": None if test is None else test["g4_win_g5_loss"],
                    "first_loss_second_win": None if test is None else test["g4_loss_g5_win"],
                    "p": None if test is None else test["exact_p"],
                    "n_prompts": 200,
                })
            lefts = [lcwr[(first, budget, seed)] for seed in seeds]
            rights = [lcwr[(second, budget, seed)] for seed in seeds]
            official = published[budget] if ratio == "1:1" else None
            rows.append({
                "ratio": ratio,
                "budget": budget,
                "seed": "平均",
                "first_arm": first,
                "second_arm": second,
                "first": sum(lefts) / len(lefts),
                "second": sum(rights) / len(rights),
                "oa": float(official["oa_lcwr"]) if official else sum(lefts) / len(lefts) - sum(rights) / len(rights),
                "low": float(official["ci_low"]) if official else None,
                "high": float(official["ci_high"]) if official else None,
                "first_win_second_loss": None,
                "first_loss_second_win": None,
                "p": None,
                "n_prompts": 200,
            })
    return rows


def sort_chart(row: dict[str, Any]) -> tuple:
    return (
        TASK_ORDER[row["task"]],
        STAGE_ORDER[row["stage"]],
        ARM_ORDER[row["arm"]],
        row["budget"],
        SEED_ORDER[str(row["seed"])],
        METRIC_ORDER[row["metric"]],
    )


def write_sheet(book: Workbook, title: str, headers: list[str], rows: list[list[Any]], widths: list[int]) -> None:
    sheet = book.create_sheet(title)
    sheet.append(headers)
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in rows:
        sheet.append(row)
    sheet.auto_filter.ref = sheet.dimensions
    sheet.freeze_panes = "A2"
    sheet.row_dimensions[1].height = 32
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{sheet.max_row}"
    for index, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    percent_headers = {
        "LCWR", "LCWR下限", "LCWR上限", "LCWR未调整",
        "MMLU字母", "MMLU全文", "ARC字母", "ARC全文",
        "IFEval题目严格", "IFEval题目宽松", "IFEval指令严格", "IFEval指令宽松",
        "数值", "区间下限", "区间上限", "种子标准差",
        "先SFT再DPO胜率", "先DPO再SFT胜率", "顺序差",
    }
    logprob_headers = {"MMLU字母对数概率", "MMLU全文对数概率", "ARC字母对数概率", "ARC全文对数概率", "LCWR长度斜率"}
    for column, header in enumerate(headers, 1):
        if header in percent_headers:
            number_format = "0.0%"
        elif header in logprob_headers:
            number_format = "0.000"
        elif header == "McNemar精确p":
            number_format = "0.0000"
        else:
            continue
        for cell in sheet.iter_cols(min_col=column, max_col=column, min_row=2, max_row=sheet.max_row):
            for item in cell:
                if isinstance(item.value, float):
                    item.number_format = number_format


def notes() -> list[str]:
    return [
        "这份表汇总了 v1.0 已经算完的评测，日期 2026-11-25。基座是 Qwen2.5-0.5B。三种子是 42、1337、2024。数据量是 500、2000、10000。",
        "打开「画图」即可作图。准确率和胜率存在单元格里的是 0 到 1 的比例，Excel 按百分比显示。监控页面上的 74.6，就是这里的 74.6%。",
        "画某一个种子：在「画图」里把「种子」筛成 42、1337 或 2024。画三种子平均：把「种子」筛成「平均」。这两类行不要放进同一张图，否则每个模型会被算两次。",
        "再筛「任务」「阶段」「指标」。摘要图用任务=摘要、阶段=最终，指标是 LCWR、MMLU字母、ARC字母、ARC全文。指令跟随图用任务=指令跟随、阶段=最终，指标是 IFEval题目严格、MMLU字母、ARC字母、ARC全文。",
        "组别：M0 只做 warmup。G2 只做 SFT。G3 只做 DPO。G4 到 G9 都是两阶段，表里的最终模型是第二阶段。比例按阶段顺序：G4 是 SFT→DPO 的 1:1，G5 是 DPO→SFT 的 1:1，G6 是 SFT→DPO 的 1:3，G7 是 SFT→DPO 的 3:1，G8 是 DPO→SFT 的 1:3，G9 是 DPO→SFT 的 3:1。",
        "M0 的 MMLU 和 ARC 只有一份，摘要和指令跟随两张图共用。",
        "LCWR 是摘要最终模型相对种子 42 的 M0 的胜率。DeepSeek 当裁判，同一题前后对调各问一次，再按摘要字数差调整。图上用的是调整后的 LCWR。LCWR未调整是 200 题胜负的直接平均。M0 的 LCWR 写成 50%，表示和自己打平，不是另一次裁判，所以没有置信区间。G2 到 G9 的 72 个摘要最终模型都已裁判。",
        "LCWR 的区间是 1000 次 bootstrap、随机种子 4207 的 95% 区间。单种子区间来自这 200 道题；平均行的区间是三颗种子一起重抽样后的平均值。其他指标没有区间。平均行的种子标准差是三颗种子点估计的样本标准差。",
        "IFEval 用官方 checker。题目准确率看整道题是否通过，指令准确率看每一条约束。严格和宽松是两套判定。指令跟随模型没有做 LCWR。",
        "题数：LCWR 200，MMLU 500，ARC-Challenge 300，IFEval 541 题、834 条指令。",
        "「顺序差」是同一数据量、同一比例下，先 SFT 再 DPO 的 LCWR 减去先 DPO 再 SFT 的 LCWR。1:1 是 G4 减 G5，1:3 是 G6 减 G8，3:1 是 G7 减 G9。正数表示先做 SFT 的胜率更高。只有 1:1 有配对照的 bootstrap 区间和 McNemar 精确 p。McNemar 只数双方都分出胜负、而且胜负相反的题，平局不计入。平均行的 p 值留空。",
        "「各模型」是每个有汇总分数的训练结果一行，右边还有对数概率。「画图」把这些数拆成了长表。",
        "第一阶段的中间模型没有单独评测，所以不在表里。扩展敏感性摘要和摘要轨道的 IFEval 没有跑。逐题的生成和裁判原文在 03_runs，没有放进这份表。",
    ]


def main() -> None:
    summaries = load_summaries()
    models = build_models(summaries)
    judged = [row for row in models if row["lcwr_source"] == "裁判"]
    if len(judged) != 72:
        raise RuntimeError(f"Expected 72 judged LCWR models, found {len(judged)}")
    chart = sorted(chart_rows(models), key=sort_chart)
    order = oa_rows()

    book = Workbook()
    note_sheet = book.active
    note_sheet.title = "说明"
    note_sheet.sheet_properties.tabColor = "1F4E79"
    for index, line in enumerate(notes(), 1):
        note_sheet.cell(index, 1, line).font = NOTE_FONT
        note_sheet.cell(index, 1).alignment = Alignment(wrap_text=True, vertical="center")
        note_sheet.row_dimensions[index].height = 36
    note_sheet.column_dimensions["A"].width = 140

    write_sheet(
        book,
        "画图",
        ["任务", "组别", "比例", "阶段顺序", "阶段", "训练方式", "数据量", "种子", "种子数", "指标", "数值", "区间下限", "区间上限", "种子标准差", "题数"],
        [[
            row["task"], row["arm"], row["ratio"], row["sequence"], row["stage"], row["method"], row["budget"], row["seed"], row["n_seeds"],
            row["metric"], row["value"], row["low"], row["high"], row["sd"], row["n_items"],
        ] for row in chart],
        [12, 10, 10, 14, 12, 18, 12, 10, 10, 18, 12, 12, 12, 14, 10],
    )
    model_headers = [
        "运行编号", "任务", "组别", "比例", "阶段顺序", "阶段", "训练方式", "数据量", "种子",
        "LCWR", "LCWR下限", "LCWR上限", "LCWR未调整", "LCWR长度斜率", "LCWR来源",
        "MMLU字母", "MMLU全文", "MMLU字母对数概率", "MMLU全文对数概率",
        "ARC字母", "ARC全文", "ARC字母对数概率", "ARC全文对数概率",
        "IFEval题目严格", "IFEval题目宽松", "IFEval指令严格", "IFEval指令宽松",
        "LCWR题数", "MMLU题数", "ARC题数", "IFEval题数", "IFEval指令数", "裁判",
    ]
    model_keys = [
        "run_id", "task", "arm", "ratio", "sequence", "stage", "method", "budget", "seed",
        "lcwr", "lcwr_low", "lcwr_high", "lcwr_unadjusted", "lcwr_slope", "lcwr_source",
        "mmlu_letter", "mmlu_text", "mmlu_letter_logprob", "mmlu_text_logprob",
        "arc_letter", "arc_text", "arc_letter_logprob", "arc_text_logprob",
        "ifeval_prompt_strict", "ifeval_prompt_loose", "ifeval_inst_strict", "ifeval_inst_loose",
        "lcwr_n", "mmlu_n", "arc_n", "ifeval_n", "ifeval_inst_n", "judge",
    ]
    ordered_models = sorted(models, key=lambda row: (
        TASK_ORDER[row["task"]], STAGE_ORDER[row["stage"]], ARM_ORDER[row["arm"]], row["budget"], SEED_ORDER[row["seed"]]
    ))
    write_sheet(
        book,
        "各模型",
        model_headers,
        [[row[key] for key in model_keys] for row in ordered_models],
        [42, 12, 10, 10, 14, 12, 18, 10, 10, 12, 12, 12, 14, 14, 12, 12, 12, 18, 18, 12, 12, 18, 18, 16, 16, 16, 16, 12, 12, 12, 12, 14, 16],
    )
    write_sheet(
        book,
        "顺序差",
        ["比例", "数据量", "种子", "先SFT再DPO", "先DPO再SFT", "先SFT再DPO胜率", "先DPO再SFT胜率", "顺序差", "区间下限", "区间上限", "先SFT赢且先DPO输", "先SFT输且先DPO赢", "McNemar精确p", "配对数"],
        [[
            row["ratio"], row["budget"], row["seed"], row["first_arm"], row["second_arm"],
            row["first"], row["second"], row["oa"], row["low"], row["high"],
            row["first_win_second_loss"], row["first_loss_second_win"], row["p"], row["n_prompts"],
        ] for row in order],
        [10, 12, 10, 14, 14, 16, 16, 12, 12, 12, 18, 18, 16, 12],
    )
    output = RESULTS_DIR / "eval_results_v1.0_20261125.xlsx"
    book.save(output)
    tasks = {row["task"] for row in models}
    print(f"wrote {output}")
    print(f"models {len(models)} tasks {sorted(tasks)} chart_rows {len(chart)} oa_rows {len(order)}")
    print("stages", sorted({(row["task"], row["stage"]) for row in models}))


if __name__ == "__main__":
    main()
