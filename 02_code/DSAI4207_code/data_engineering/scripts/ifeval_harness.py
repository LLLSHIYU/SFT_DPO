#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ifeval_harness.py — IFEval (Zhou et al., 2023) Prompt Renderer & Official Rule Checker Harness
Owner: PAN Haoren (24104216d) · DSAI 4207 Data Engineering & IFEval Harness (§12.4, P1-2)

Prioritizes the official Google Research `instruction_following_eval` package and records
`"checker": "official"` (or `"fallback"` only when `--allow-fallback` is explicitly passed for debug).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import DATA_V1_DIR, RAW_CACHE_DIR, QwenTokenizerWrapper, read_jsonl, write_jsonl  # noqa: E402


def _has_official_ifeval() -> bool:
    try:
        from instruction_following_eval import instructions_registry  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def _strip_loose_variants(resp: str) -> List[str]:
    """Generates official IFEval loose evaluation candidates (removing first/last lines & markdown asterisks)."""
    r = resp.strip()
    r_no_md = r.replace("*", "")
    lines = r.split("\n")
    r_no_first = "\n".join(lines[1:]).strip() if len(lines) > 1 else r
    r_no_last = "\n".join(lines[:-1]).strip() if len(lines) > 1 else r
    r_no_both = "\n".join(lines[1:-1]).strip() if len(lines) > 2 else r
    variants = [r, r_no_md, r_no_first, r_no_last, r_no_both]
    variants.extend([v.replace("*", "") for v in [r_no_first, r_no_last, r_no_both]])
    seen = []
    for v in variants:
        if v not in seen:
            seen.append(v)
    return seen


def _compare_relation(val: float, target: float, relation: Optional[str]) -> bool:
    rel = (relation or "at least").lower()
    if "less than" in rel:
        return val < target
    if "at most" in rel:
        return val <= target
    return val >= target


def check_single_instruction(
    inst_id: str, resp: str, kw: Dict[str, Any], allow_fallback: bool = False
) -> Tuple[bool, str]:
    """
    Programmatic checker covering the 25 official IFEval instruction types.
    Uses Google Research's `instruction_following_eval` package by default (P1-2).
    Returns (is_followed, checker_used).
    """
    try:
        from instruction_following_eval import instructions_registry  # type: ignore

        cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        inst = cls(inst_id)
        clean_kw = {k: v for k, v in (kw or {}).items() if v is not None}
        # Filter clean_kw to accepted parameters of build_description
        import inspect

        sig = inspect.signature(inst.build_description)
        accepted_kw = {k: v for k, v in clean_kw.items() if k in sig.parameters}
        inst.build_description(**accepted_kw)
        return bool(resp.strip() and inst.check_following(resp)), "official"
    except ImportError as e:
        if not allow_fallback:
            raise RuntimeError(
                "Official `instruction_following_eval` package is not installed! "
                "Run 00_download.py or pass --allow-fallback for debug only."
            ) from e

    text = resp.strip()
    if not text:
        return False, "fallback"

    if inst_id == "keywords:existence":
        keywords = kw.get("keywords") or []
        return all(re.search(re.escape(k), text, re.IGNORECASE) for k in keywords), "fallback"
    if inst_id == "keywords:frequency":
        keyword = kw.get("keyword") or ""
        freq = int(kw.get("frequency") or 1)
        cnt = len(re.findall(re.escape(keyword), text, re.IGNORECASE))
        return _compare_relation(cnt, freq, kw.get("relation")), "fallback"
    if inst_id == "keywords:forbidden_words":
        forbidden = kw.get("forbidden_words") or []
        return not any(re.search(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE) for w in forbidden), "fallback"
    if inst_id == "keywords:letter_frequency":
        letter = (kw.get("letter") or "").lower()
        let_freq = int(kw.get("let_frequency") or 1)
        cnt = text.lower().count(letter)
        return _compare_relation(cnt, let_freq, kw.get("let_relation")), "fallback"
    if inst_id == "length_constraints:number_words":
        num_words = int(kw.get("num_words") or 1)
        cnt = len(re.findall(r"\w+", text))
        return _compare_relation(cnt, num_words, kw.get("relation")), "fallback"
    if inst_id == "punctuation:no_comma":
        return ("," not in text), "fallback"
    return True, "fallback"


def score_ifeval_responses(
    bench_records: List[Dict[str, Any]],
    response_records: List[Dict[str, Any]],
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    resp_by_key = {}
    resp_by_prompt = {}
    for r in response_records:
        if "key" in r:
            resp_by_key[int(r["key"])] = str(r.get("response") or "")
        if "prompt" in r:
            resp_by_prompt[str(r["prompt"])] = str(r.get("response") or "")

    prompt_strict_list = []
    prompt_loose_list = []
    inst_strict_list = []
    inst_loose_list = []
    checker_Backend = "official" if _has_official_ifeval() else "fallback"

    for item in bench_records:
        key = int(item["key"])
        resp = resp_by_key.get(key, resp_by_prompt.get(item["prompt"], ""))
        inst_ids = item["instruction_id_list"]
        kwargs_list = item["kwargs"]

        s_res = []
        l_res = []
        loose_cands = _strip_loose_variants(resp)
        for inst_id, kw in zip(inst_ids, kwargs_list):
            kw = kw or {}
            ok_strict, checker_used = check_single_instruction(
                inst_id, resp, kw, allow_fallback=allow_fallback
            )
            checker_Backend = checker_used
            ok_loose = ok_strict or any(
                check_single_instruction(inst_id, cand, kw, allow_fallback=allow_fallback)[0]
                for cand in loose_cands
            )
            s_res.append(ok_strict)
            l_res.append(ok_loose)
            inst_strict_list.append(ok_strict)
            inst_loose_list.append(ok_loose)

        prompt_strict_list.append(all(s_res))
        prompt_loose_list.append(all(l_res))

    return {
        "checker": checker_Backend,
        "num_prompts": len(bench_records),
        "num_instructions": len(inst_strict_list),
        "prompt_level_strict_acc": round(
            sum(prompt_strict_list) / max(1, len(prompt_strict_list)), 4
        ),
        "prompt_level_loose_acc": round(
            sum(prompt_loose_list) / max(1, len(prompt_loose_list)), 4
        ),
        "instruction_level_strict_acc": round(
            sum(inst_strict_list) / max(1, len(inst_strict_list)), 4
        ),
        "instruction_level_loose_acc": round(
            sum(inst_loose_list) / max(1, len(inst_loose_list)), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DSAI 4207 IFEval Harness")
    parser.add_argument("--mode", choices=["render", "score"], required=True)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DATA_V1_DIR / "eval_benchmarks" / "ifeval_541.jsonl",
    )
    parser.add_argument("--output", type=Path, help="Output path for rendered prompts")
    parser.add_argument("--responses", type=Path, help="Model responses JSONL for scoring")
    parser.add_argument("--report", type=Path, help="Output JSON report path for scoring")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow fallback rule engine if official instruction_following_eval is absent (debug only).",
    )
    args = parser.parse_args()

    bench = read_jsonl(args.benchmark)
    if args.mode == "render":
        tokenizer = QwenTokenizerWrapper(RAW_CACHE_DIR / "qwen_tokenizer")
        rendered = []
        for item in bench:
            chatml_prompt = tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=True
            )
            rendered.append({**item, "rendered_prompt": chatml_prompt})
        out_path = args.output or (
            DATA_V1_DIR / "eval_benchmarks" / "ifeval_541_rendered.jsonl"
        )
        write_jsonl(out_path, rendered)
        print(f"[ifeval_harness] Rendered {len(rendered)} official IFEval prompts to {out_path}")
    else:
        if not args.responses:
            raise ValueError("--responses is required when --mode=score")
        resps = read_jsonl(args.responses)
        metrics = score_ifeval_responses(
            bench, resps, allow_fallback=args.allow_fallback
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
