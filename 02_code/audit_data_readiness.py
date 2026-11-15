#!/usr/bin/env python3
"""Read-only audit of the frozen v1.0 export; Python standard library only.

Writes a new JSON audit, never regenerates or modifies frozen data/manifests.
Near-duplicate checks use exact 5-word sets with a complete prefix filter,
independent of the export's probabilistic MinHash candidate retrieval.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_data/v1.0"


def load(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines()]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def norm(text):
    text = re.sub(r"https?://\S+|www\.\S+", "", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def distribution(rows, field):
    counts = Counter(r[field] for r in rows)
    return {k: {"count": counts[k], "proportion": counts[k] / len(rows)}
            for k in sorted(counts)}


def smd(a, b, field):
    aa, bb = [r[field] for r in a], [r[field] for r in b]
    sd = math.sqrt((statistics.variance(aa) + statistics.variance(bb)) / 2)
    return (statistics.mean(aa) - statistics.mean(bb)) / sd if sd else 0


class ExactPrefixIndex:
    """Lossless candidate filter for exact Jaccard >= 4/5, no MinHash."""

    def __init__(self):
        self.sets, self.ids, self.postings, self.exact = [], [], defaultdict(list), defaultdict(list)

    @staticmethod
    def shingles(text):
        words = norm(text).split()
        return {" ".join(words[i:i + 5]) for i in range(len(words) - 4)}

    @staticmethod
    def prefix(shingles):
        # A common, deterministic total order for all documents. Include the
        # original string as a tiebreak, so hash collisions cannot lose matches.
        ordered = sorted(shingles, key=lambda s: (hashlib.sha1(s.encode()).digest(), s))
        return ordered[:len(ordered) - (4 * len(ordered) + 4) // 5 + 1]

    def insert(self, label, text):
        ss = self.shingles(text)
        idx = len(self.sets)
        self.sets.append(ss)
        self.ids.append(label)
        self.exact[norm(text)].append(idx)
        if ss:
            for shingle in self.prefix(ss):
                self.postings[shingle].append(idx)

    def query(self, text):
        ss = self.shingles(text)
        candidates = set(self.exact.get(norm(text), []))
        if ss:
            for shingle in self.prefix(ss):
                candidates.update(self.postings.get(shingle, []))
        hits = []
        for idx in sorted(candidates):
            other = self.sets[idx]
            if not ss or not other:
                if norm(text) in self.exact and idx in self.exact[norm(text)]:
                    hits.append({"reference": self.ids[idx], "jaccard": 1.0})
                continue
            if 5 * min(len(ss), len(other)) < 4 * max(len(ss), len(other)):
                continue
            intersection = len(ss & other)
            union = len(ss) + len(other) - intersection
            if 5 * intersection >= 4 * union:
                hits.append({"reference": self.ids[idx], "jaccard": intersection / union})
        return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "04_results/data_readiness_audit_20261104.json")
    parser.add_argument("--skip-near", action="store_true")
    args = parser.parse_args()
    out = {"audit_date": "2026-11-04", "checks": {}, "failures": [], "limitations": [
        "No official tokenizer assets in this shared export: stored token lengths checked, not independently retokenized.",
        "No raw source parquets: cannot independently reconstruct preference labels/canonicalization.",
        "No model training or original TRL integration tests executed.",
        "Full canonical-pool internal near-duplicate scan not repeated."]}

    def check(name, condition):
        out["checks"][name] = bool(condition)
        if not condition:
            out["failures"].append(name)

    manifest = json.loads((DATA / "manifests/manifest.json").read_text())
    bad_hashes = []
    line_counts = {}
    for entry in manifest["files"]:
        path = DATA / entry["path"]
        if not path.exists():
            bad_hashes.append(entry["path"] + ":missing")
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != entry["bytes"]:
            bad_hashes.append(entry["path"])
        if path.suffix == ".jsonl":
            rows = [json.loads(x) for x in raw.splitlines()]
            line_counts[entry["path"]] = len(rows)
            check("line_count:" + entry["path"], len(rows) == entry["n"])
    out["manifest"] = {"tracked_files": len(manifest["files"]), "bad_hashes": bad_hashes, "jsonl_counts": line_counts}
    check("manifest_hashes", not bad_hashes)
    handoff_bad, handoff_count = [], 0
    for line in (ROOT / "05_handoff/CHECKSUMS.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = ROOT / name.lstrip("*")
        handoff_count += 1
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            handoff_bad.append(name)
    out["handoff_checksums"] = {"files": handoff_count, "mismatches": handoff_bad}
    check("handoff_checksums", not handoff_bad)
    check("recorded_double_run_equal", (DATA / "manifests/run_1_sha256.txt").read_bytes() == (DATA / "manifests/run_2_sha256.txt").read_bytes())

    pool, test, main_test, warmup = map(load, ["canonical_pool.jsonl", "test_500_full.jsonl", "test_200_main.jsonl", "warmup_5k.jsonl"])
    canonical = {r["pid"]: r for r in pool}
    check("canonical_unique_ids", len(canonical) == len(pool) == 14733 and len({r["hash_prompt"] for r in pool}) == len(pool))
    check("valid_pairs_and_metadata", all(r["pid"].startswith("t3_") and all(r[k] for k in ["title", "subreddit", "post", "chosen", "rejected"]) and norm(r["chosen"]) != norm(r["rejected"]) for r in pool))
    check("canonical_hash_fields", all(r["hash_prompt"] == digest(norm(r["title"] + " " + r["post"])) and r["hash_pair"] == digest(r["chosen"] + "\0" + r["rejected"]) for r in pool))
    check("stored_token_limits", all(0 < r["prompt_tokens"] <= 768 and 0 < r["chosen_tokens"] <= 160 and 0 < r["rejected_tokens"] <= 200 and r["prompt_tokens"] + max(r["chosen_tokens"], r["rejected_tokens"]) <= 1024 for r in pool))
    check("test_prefix", len(test) == 500 and len(main_test) == 200 and main_test == test[:200] and len({r["pid"] for r in test}) == 500 and len({r["hash_prompt"] for r in test}) == 500)
    check("test_pool_id_isolation", set(canonical).isdisjoint(r["pid"] for r in test))
    check("test_pool_hash_isolation", {r["hash_prompt"] for r in pool}.isdisjoint(r["hash_prompt"] for r in test))
    out["canonical_quality"] = {
        "zero_observed_losses": sum(r["chosen_losses_in_pid"] == 0 for r in pool),
        "confidence_missing": sum(r["selected_confidence"] is None for r in pool),
        "confidence_below_3": sum(r["selected_confidence"] is not None and r["selected_confidence"] < 3 for r in pool),
        "chosen_policy": distribution(pool, "chosen_policy_family"),
        "rejected_policy": distribution(pool, "rejected_policy_family"),
        "stored_token_stats": {k: {"mean": statistics.mean(r[k] for r in pool), "max": max(r[k] for r in pool)} for k in ["prompt_tokens", "chosen_tokens", "rejected_tokens"]},
        "chosen_longer_count": sum(r["chosen_tokens"] > r["rejected_tokens"] for r in pool)}
    out["embedded_summary_scan"] = {}
    for name, rows, target in [("canonical_pool", pool, "chosen"), ("test_main", main_test, "reference"), ("test_full", test, "reference")]:
        out["embedded_summary_scan"][name] = {
            "tldr_marker_in_post": sum(bool(re.search(r"\btl\s*[;:/]?\s*dr\b|\btldr\b", r["post"], re.I)) for r in rows),
            "entire_target_is_post_substring": sum(r[target].lower() in r["post"].lower() for r in rows)}

    out["budgets"], nested = {}, {}
    for n in [500, 2000, 10000]:
        p, a, b = [load(f"splits/{name}_{n}.jsonl") for name in ["pool", "partA", "partB"]]
        ids = [set(r["pid"] for r in rows) for rows in [p, a, b]]
        nested[n] = ids
        check(f"partition:{n}", len(p) == n and len(a) == len(b) == n // 2 and len(ids[0]) == n and ids[1].isdisjoint(ids[2]) and ids[1] | ids[2] == ids[0])
        check(f"canonical_content:{n}", all({k: r[k] for k in canonical[r["pid"]]} == canonical[r["pid"]] for r in p))
        g2, g3, g4s, g4d, g5d, g5s = [load(f"arms/{name}_{n}.jsonl") for name in ["G2_sft", "G3_dpo", "G4_stage1_sft", "G4_stage2_dpo", "G5_stage1_dpo", "G5_stage2_sft"]]
        check(f"mirror_bytes_sft:{n}", (DATA / f"arms/G4_stage1_sft_{n}.jsonl").read_bytes() == (DATA / f"arms/G5_stage2_sft_{n}.jsonl").read_bytes())
        check(f"mirror_bytes_dpo:{n}", (DATA / f"arms/G4_stage2_dpo_{n}.jsonl").read_bytes() == (DATA / f"arms/G5_stage1_dpo_{n}.jsonl").read_bytes())
        for arm_name, rows, source, kind in [("G2", g2, p, "sft"), ("G3", g3, p, "dpo"), ("G4s", g4s, a, "sft"), ("G4d", g4d, b, "dpo"), ("G5d", g5d, b, "dpo"), ("G5s", g5s, a, "sft")]:
            valid = len(rows) == len(source)
            for r, src in zip(rows, source):
                valid &= r["pid"] == src["pid"] and r["meta"]["hash_pair"] == src["hash_pair"]
                if kind == "sft":
                    valid &= [m["role"] for m in r["messages"]] == ["system", "user", "assistant"] and r["messages"][1]["content"] == src["user_content"] and r["messages"][2]["content"] == src["chosen"]
                else:
                    valid &= r["prompt"] == [{"role": "system", "content": g2[0]["messages"][0]["content"]}, {"role": "user", "content": src["user_content"]}] and r["chosen"] == [{"role": "assistant", "content": src["chosen"]}] and r["rejected"] == [{"role": "assistant", "content": src["rejected"]}]
            check(f"arm_content:{n}:{arm_name}", valid)
        out["budgets"][str(n)] = {"partA": len(a), "partB": len(b), "smd_stored_lengths": {f: smd(a, b, f) for f in ["prompt_tokens", "chosen_tokens", "rejected_tokens"]}, "chosen_policy": distribution(p, "chosen_policy_family"), "rejected_policy": distribution(p, "rejected_policy_family")}
        out["budgets"][str(n)]["confidence_missing_proportion"] = sum(r["selected_confidence"] is None for r in p) / n
        out["budgets"][str(n)]["confidence_below_3_proportion"] = sum(r["selected_confidence"] is not None and r["selected_confidence"] < 3 for r in p) / n
        out["budgets"][str(n)]["zero_observed_losses_proportion"] = sum(r["chosen_losses_in_pid"] == 0 for r in p) / n
    check("nested_pool_A_B", all(nested[500][i] < nested[2000][i] < nested[10000][i] for i in range(3)))
    reserve = load("splits/reserve.jsonl")
    check("reserve_partition", len(reserve) == 4733 and nested[10000][0].isdisjoint(r["pid"] for r in reserve) and nested[10000][0] | {r["pid"] for r in reserve} == set(canonical))
    check("cold_probe_same_data", load("arms/G3prime_dpo_500.jsonl") == load("arms/G3_dpo_500.jsonl"))
    user_re = re.compile(r"\b(summar(y|ies|ize|ise|ized|ised|izing|ising|ization|isation)|tl[;/]?dr|condense|recap|shorten|gist|synops[ie]s|key points|main points|in a nutshell|brief overview|abstract of)\b", re.I)
    asst_re = re.compile(r"\b(tl[;/]?dr|in summary|to summarize|here is a summary)\b", re.I)
    violations = [r["prompt_id"] for r in warmup if r["category"] == "Summarize" or r["total_tokens"] > 2048 or any((asst_re if m["role"] == "assistant" else user_re).search(m["content"]) for m in r["messages"])]
    check("warmup_count_and_regex", len(warmup) == 5000 and not violations)
    check("warmup_messages", all(any(m["role"] == "assistant" and m["content"].strip() for m in r["messages"]) for r in warmup) and len({r["prompt_id"] for r in warmup}) == 5000 and len({norm(" ".join(m["content"] for m in r["messages"])) for r in warmup}) == 5000)
    out["warmup"] = {"regex_violations": violations, "categories": dict(Counter(r["category"] for r in warmup)), "max_stored_tokens": max(r["total_tokens"] for r in warmup), "multi_turn_rows": sum(len(r["messages"]) > 2 for r in warmup)}
    mmlu, arc, ifeval = map(load, ["eval_benchmarks/mmlu_500.jsonl", "eval_benchmarks/arc_challenge_300.jsonl", "eval_benchmarks/ifeval_541.jsonl"])
    check("mmlu_schema", len(mmlu) == 500 and len({r["subject"] for r in mmlu}) == 57 and all(len(r["choices"]) == 4 and r["answer_letter"] == "ABCD"[r["answer_index"]] for r in mmlu))
    check("arc_schema", len(arc) == 300 and all(len(r["choice_labels"]) == len(r["choices"]) and r["answerKey"] in r["choice_labels"] for r in arc))
    check("ifeval_schema", len(ifeval) == 541 and all(len(r["instruction_id_list"]) == len(r["kwargs"]) > 0 and r["messages"] == [{"role": "user", "content": r["prompt"]}] and r["max_new_tokens"] == 1280 for r in ifeval))
    out["benchmarks"] = {"mmlu": len(mmlu), "mmlu_subjects": len({r["subject"] for r in mmlu}), "arc": len(arc), "ifeval": len(ifeval)}
    for label, rows in [("mmlu", mmlu), ("arc", arc), ("ifeval", ifeval)]:
        check(f"benchmark_unique_ids:{label}", len({r["id"] for r in rows}) == len(rows))
    fields = Counter(tuple(sorted(r.keys())) for r in load("arms/G2_sft_10000.jsonl"))
    out["sft_schema_variants"] = [{"keys": list(k), "count": v} for k, v in fields.items()]
    out["methodological_findings"] = [
        "Within-budget A/B balance does not imply cross-budget data composition balance.",
        "Rejected sft_early proportion is 0.008 at N=500 and 0.4258 at N=10000; scale attribution is confounded.",
        "Original SFT smoke test does not set assistant_only_loss=True or assert actual trainer labels.",
        "Protocol anchors G4 DPO reference to stage-1 output and G5 DPO reference to M0; this is a recipe comparison with changing reference."]

    if not args.skip_near:
        print("Checking exact cross-set Jaccard with complete prefix filtering...", flush=True)
        idx = ExactPrefixIndex()
        for r in test:
            idx.insert("test:" + r["pid"], r["title"] + " " + r["post"])
        for label, rows in [("mmlu", mmlu), ("arc", arc), ("ifeval", ifeval)]:
            for r in rows:
                idx.insert(label + ":" + r["id"], r["norm_text"])
        leaks = []
        for r in pool:
            for hit in idx.query(r["title"] + " " + r["post"]):
                leaks.append({"query": r["pid"], **hit})
        for r in warmup:
            # Additional prompt-only check, stricter than relying on full-turn
            # similarity alone when assistant responses dilute an overlap.
            text = " ".join(m["content"] for m in r["messages"] if m["role"] in ["system", "user"])
            for hit in idx.query(text):
                leaks.append({"query": "warmup:" + r["prompt_id"], **hit})
        out["exact_cross_set_jaccard"] = {"threshold": 0.8, "method": "lossless prefix filter + exact full 5-word shingle Jaccard", "reference_n": len(idx.ids), "query_n": len(pool) + len(warmup), "hits": leaks}
        check("exact_cross_set_jaccard_no_leaks", not leaks)
        full_warmup_hits = []
        warmup_idx = ExactPrefixIndex()
        for r in warmup:
            text = " ".join(m["content"] for m in r["messages"])
            for hit in idx.query(text):
                full_warmup_hits.append({"query": "warmup:" + r["prompt_id"], **hit})
            warmup_idx.insert("warmup:" + r["prompt_id"], text)
        # Reproduce original warmup-vs-task scope (title + post + chosen/reference).
        task_warmup_hits = []
        for label, rows, target in [("pool", pool, "chosen"), ("test", test, "reference")]:
            for r in rows:
                text = r["title"] + " " + r["post"] + " " + r[target]
                for hit in warmup_idx.query(text):
                    task_warmup_hits.append({"query": label + ":" + r["pid"], **hit})
        out["exact_warmup_full_turn_jaccard"] = {"threshold": 0.8, "warmup_vs_test_benchmarks_hits": full_warmup_hits, "warmup_vs_task_full_text_hits": task_warmup_hits}
        check("exact_full_turn_warmup_no_leaks", not full_warmup_hits and not task_warmup_hits)
    out["summary"] = {"checks": len(out["checks"]), "passed": sum(out["checks"].values()), "failed": len(out["failures"])}
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **out["summary"], "failures": out["failures"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
