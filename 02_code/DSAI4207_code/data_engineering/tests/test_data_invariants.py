#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_invariants.py — Comprehensive automated test suite verifying:
- Invariants I1–I10 from DATA_PROTOCOL.md v1.2-final on data/v1.0/
- P1-3 Authenticity & Integration tests:
  * test_tokenizer_authentic (official Qwen2.5-0.5B-Base vocab_size, len(tokenizer), probe token IDs, apply_chat_template parity)
  * test_benchmark_authentic (official MMLU 14,042 rows/57 subjects -> 500, ARC-Challenge 1,172 -> 300, IFEval 541 -> 541)
  * test_trl_smoke (1-step smoke test of TRL SFTTrainer and DPOTrainer on 1 real sample from data/v1.0/arms/)
  * test_test200_stratification (35-stratum balance between test_200_main and test_500_full)
  * test_blind_audit_100 (blindness of data_label_audit_100.jsonl and separation of data_label_audit_100_key.json)
  * test_rejected_policy_balance (3x3 joint policy & rejected policy family chi-square balance across Part A / Part B)
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "data_engineering" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import DATA_V1_DIR, RAW_CACHE_DIR, QwenTokenizerWrapper, read_jsonl, sha256_file  # noqa: E402


class TestDataInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = DATA_V1_DIR / "manifests" / "manifest.json"
        if not cls.manifest_path.exists():
            raise RuntimeError("manifest.json not found! Run scripts/run_all.py first.")
        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)
        with open(RAW_CACHE_DIR / "sources_lock.json", "r", encoding="utf-8") as f:
            cls.sources_lock = json.load(f)
        cls.tokenizer = QwenTokenizerWrapper(RAW_CACHE_DIR / "qwen_tokenizer")

    def test_invariants_i1_to_i9_and_token_mirror(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "validate_mod", SCRIPTS_DIR / "07_validate.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        summary = mod.verify_all_invariants()
        self.assertGreaterEqual(summary["canonical_pool_n"], 10000)
        self.assertEqual(summary["test_main_n"], 200)
        self.assertEqual(summary["test_full_n"], 500)
        self.assertEqual(summary["warmup_n"], 5000)

    def test_manifest_file_hashes_and_line_counts(self) -> None:
        self.assertFalse(self.manifest.get("synthetic", True))
        # Verify R1/P2-5 valid commit SHAs, R3/P3-1 complete env versions, and R4/P3-2 tokenizer repo mapping
        srcs = self.manifest["sources"]
        self.assertEqual(srcs["tldr"]["revision"], "b8f7d168b6f4e95b2a92e84768bd6c955bed2f29")
        self.assertEqual(srcs["warmup"]["revision"], "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b")
        self.assertEqual(srcs["benchmarks"]["mmlu"]["revision"], "c30699e8356da336a370243923dbaf21066bb9fe")
        self.assertEqual(srcs["benchmarks"]["arc_challenge"]["revision"], "210d026faf9955653af8916fad021475a3f00453")
        self.assertEqual(srcs["benchmarks"]["ifeval"]["revision"], "966cd89545d6b6acfd7638bc708b98261ca58e84")
        self.assertEqual(srcs["tokenizer"]["hub_repo"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(srcs["tokenizer"]["revision_for_repo"], "Qwen/Qwen2.5-0.5B")
        env = self.manifest.get("env", {})
        for req_pkg in ("python", "pyarrow", "pyyaml", "transformers", "tokenizers", "trl", "torch", "datasets", "scipy"):
            self.assertIn(req_pkg, env, f"Missing {req_pkg} in manifest['env']")
        files = self.manifest.get("files", [])
        self.assertGreaterEqual(len(files), 35)
        for entry in files:
            fpath = DATA_V1_DIR / entry["path"]
            self.assertTrue(fpath.exists(), f"Missing tracked file: {fpath}")
            self.assertEqual(
                sha256_file(fpath),
                entry["sha256"],
                f"SHA256 mismatch for {entry['path']}",
            )
            if fpath.suffix == ".jsonl":
                rows = read_jsonl(fpath)
                self.assertEqual(
                    len(rows), entry["n"], f"Line count mismatch for {entry['path']}"
                )

    def test_smd_and_rejected_policy_balance(self) -> None:
        with open(DATA_V1_DIR / "manifests" / "split_report.json", "r", encoding="utf-8") as f:
            split_rep = json.load(f)
        for budget_str in ("500", "2000", "10000"):
            b_rep = split_rep["budgets"][budget_str]
            smd_info = b_rep["smd"]
            self.assertLess(abs(smd_info["prompt_tokens"]), 0.10)
            self.assertLess(abs(smd_info["chosen_tokens"]), 0.10)
            self.assertLess(abs(smd_info["rejected_tokens"]), 0.10)
            self.assertTrue(smd_info["all_within_0_10"])
            # P2-1: Subreddit drift vs 10k must be < 3% (0.03)
            self.assertLess(b_rep["max_subreddit_drift_vs_10k"], 0.03)
            # P1-1: Rejected policy family & 3x3 joint matrix chi2 balance must not show imbalance
            pol_tests = b_rep["policy_balance_tests"]
            self.assertGreater(pol_tests["rejected_policy_family_chi2"]["p_value"], 0.05)
            self.assertGreater(pol_tests["joint_chosen_rejected_3x3_chi2"]["p_value"], 0.05)

    def test_tokenizer_authentic(self) -> None:
        """P0-1 & P1-3: Verify official Qwen/Qwen2.5-0.5B-Base tokenizer & byte-exact apply_chat_template."""
        self.assertFalse(self.sources_lock.get("synthetic", True))
        tok_meta = self.sources_lock["tokenizer"]
        self.assertEqual(self.tokenizer.vocab_size, 151643)
        self.assertEqual(len(self.tokenizer), 151665)
        self.assertEqual(tok_meta["vocab_size"], 151643)
        self.assertEqual(tok_meta["len_tokenizer"], 151665)
        self.assertEqual(self.tokenizer.eos_token_id, 151645)
        self.assertEqual(self.tokenizer.pad_token_id, 151643)

        # Verify single-token IDs for common English words ("hello", " the") match official Qwen2.5 vocab
        hello_ids = self.tokenizer.encode("hello")
        the_ids = self.tokenizer.encode(" the")
        self.assertEqual(len(hello_ids), 1)
        self.assertEqual(len(the_ids), 1)
        self.assertEqual(hello_ids, [14990])
        self.assertEqual(the_ids, [279])

        # Verify byte-exact parity with official AutoTokenizer.apply_chat_template
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(
            str(RAW_CACHE_DIR / "qwen_tokenizer"), use_fast=True, local_files_only=True
        )
        g2_500 = read_jsonl(DATA_V1_DIR / "arms" / "G2_sft_500.jsonl")
        sample_msgs = g2_500[0]["messages"]
        self.assertEqual(
            self.tokenizer.render_chatml(sample_msgs, add_generation_prompt=False),
            hf_tok.apply_chat_template(sample_msgs, tokenize=False, add_generation_prompt=False),
        )
        self.assertEqual(
            self.tokenizer.render_chatml(sample_msgs[:2], add_generation_prompt=True),
            hf_tok.apply_chat_template(sample_msgs[:2], tokenize=False, add_generation_prompt=True),
        )

    def test_benchmark_authentic(self) -> None:
        """P0-2 & P1-3: Verify official MMLU (14,042/57), ARC-Challenge (1,172), and IFEval (541)."""
        bench_cache = RAW_CACHE_DIR / "benchmarks"
        mmlu_src = pq.read_table(bench_cache / "mmlu_all_test.parquet")
        arc_src = pq.read_table(bench_cache / "arc_challenge_test.parquet")
        ifeval_src = pq.read_table(bench_cache / "ifeval_train.parquet")

        self.assertEqual(mmlu_src.num_rows, 14042)
        self.assertEqual(len(set(mmlu_src.column("subject").to_pylist())), 57)
        self.assertEqual(arc_src.num_rows, 1172)
        self.assertEqual(ifeval_src.num_rows, 541)

        mmlu_500 = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "mmlu_500.jsonl")
        arc_300 = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "arc_challenge_300.jsonl")
        ifeval_541 = read_jsonl(DATA_V1_DIR / "eval_benchmarks" / "ifeval_541.jsonl")

        self.assertEqual(len(mmlu_500), 500)
        self.assertEqual(len({r["subject"] for r in mmlu_500}), 57)
        self.assertEqual(len(arc_300), 300)
        self.assertEqual(len(ifeval_541), 541)

        for r in mmlu_500[:20]:
            self.assertNotIn("phenomena #", r["question"])
            self.assertFalse(r["choices"][0].startswith("Option A for "))
        for r in arc_300[:20]:
            self.assertNotIn("controlled scientific experiment #", r["question"])
            self.assertFalse(r["choices"][0].startswith("Hypothesis A for experiment"))
        for r in ifeval_541[:20]:
            self.assertFalse(r["prompt"].startswith("[IFEval #"))
            self.assertTrue(len(r["instruction_id_list"]) >= 1)

    def test_test200_stratification(self) -> None:
        """P1-3 & P2-6: Verify 35-stratum distribution drift between test_200_main and test_500_full is < 2%."""
        with open(DATA_V1_DIR / "manifests" / "testset_stratum_report.json", "r", encoding="utf-8") as f:
            rep = json.load(f)
        self.assertEqual(rep["num_strata_configured"], 35)
        self.assertLess(rep["max_abs_stratum_prop_drift_200_vs_500"], 0.02)
        self.assertLessEqual(rep["max_stratum_count_error_vs_exact_quota"], 1.0)

    def test_blind_audit_100(self) -> None:
        """P1-5: Verify 100-pair audit sheet is strictly blind and key file is separate."""
        audit_rows = read_jsonl(DATA_V1_DIR / "manifests" / "data_label_audit_100.jsonl")
        with open(DATA_V1_DIR / "manifests" / "data_label_audit_100_key.json", "r", encoding="utf-8") as f:
            key_rows = json.load(f)
        self.assertEqual(len(audit_rows), 100)
        self.assertEqual(len(key_rows), 100)
        allowed_keys = {"audit_id", "pid", "subreddit", "title", "post", "summary_A", "summary_B"}
        for r in audit_rows:
            self.assertEqual(set(r.keys()), allowed_keys)

    def test_trl_smoke(self) -> None:
        """P0-4 & P1-3: Run 1-step TRL SFTTrainer and DPOTrainer smoke test on real formatted arm records."""
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

        tok = AutoTokenizer.from_pretrained(
            str(RAW_CACHE_DIR / "qwen_tokenizer"), use_fast=True, local_files_only=True
        )
        tok.eos_token = "<|im_end|>"
        tok.pad_token = "<|endoftext|>"
        tok.padding_side = "right"

        tiny_cfg = Qwen2Config(
            vocab_size=151936,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            eos_token_id=151645,
            pad_token_id=151643,
        )
        torch.manual_seed(42)
        model_sft = Qwen2ForCausalLM(tiny_cfg)
        sft_sample = read_jsonl(DATA_V1_DIR / "arms" / "G2_sft_500.jsonl")[0]
        dpo_sample = read_jsonl(DATA_V1_DIR / "arms" / "G3_dpo_500.jsonl")[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            sft_ds = Dataset.from_list([{"messages": sft_sample["messages"]}])
            sft_args = SFTConfig(
                output_dir=str(Path(tmpdir) / "sft_out"),
                max_steps=1,
                per_device_train_batch_size=1,
                learning_rate=1e-4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
                max_length=1024,
            )
            sft_trainer = SFTTrainer(
                model=model_sft,
                args=sft_args,
                train_dataset=sft_ds,
                processing_class=tok,
            )
            sft_res = sft_trainer.train()
            self.assertEqual(sft_res.global_step, 1)

            model_dpo = Qwen2ForCausalLM(tiny_cfg)
            ref_dpo = Qwen2ForCausalLM(tiny_cfg)
            dpo_ds = Dataset.from_list(
                [
                    {
                        "prompt": dpo_sample["prompt"],
                        "chosen": dpo_sample["chosen"],
                        "rejected": dpo_sample["rejected"],
                    }
                ]
            )
            dpo_args = DPOConfig(
                output_dir=str(Path(tmpdir) / "dpo_out"),
                max_steps=1,
                per_device_train_batch_size=1,
                learning_rate=1e-4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
                max_length=1024,
            )
            dpo_trainer = DPOTrainer(
                model=model_dpo,
                ref_model=ref_dpo,
                args=dpo_args,
                train_dataset=dpo_ds,
                processing_class=tok,
            )
            dpo_res = dpo_trainer.train()
            self.assertEqual(dpo_res.global_step, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
