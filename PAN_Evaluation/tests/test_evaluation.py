import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import exact_mcnemar_p, fit_records, main as analyze_main
from common import write_jsonl
from human_audit import kappa
from score_capability import choices_for


class CapabilityTests(unittest.TestCase):
    def test_mmlu_answer_index_and_letter(self):
        texts, labels, answer = choices_for({
            "id": "mmlu_x_0",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 3,
            "answer_letter": "D",
        })
        self.assertEqual(len(texts), 4)
        self.assertEqual(labels, ["A", "B", "C", "D"])
        self.assertEqual(answer, 3)

    def test_arc_answer_key(self):
        texts, labels, answer = choices_for({
            "id": "Mercury_1",
            "choices": ["x", "y"],
            "choice_labels": ["A", "B"],
            "answerKey": "B",
        })
        self.assertEqual(answer, 1)


class StatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_is_two_sided(self):
        self.assertAlmostEqual(exact_mcnemar_p(10, 0), 2 / 2**10)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)

    def test_lcwr_at_equal_lengths_matches_mean_win(self):
        records = [
            {"candidate_wins": 1.0, "log_length_delta": 0.0},
            {"candidate_wins": 0.0, "log_length_delta": 0.0},
            {"candidate_wins": 0.5, "log_length_delta": 0.0},
        ]
        estimate, slope = fit_records(records)
        self.assertAlmostEqual(estimate, 0.5)
        self.assertEqual(slope, 0.0)

    def test_lcwr_adjusts_on_varied_lengths(self):
        records = [
            {"candidate_wins": 1.0, "log_length_delta": -1.0},
            {"candidate_wins": 0.0, "log_length_delta": 1.0},
            {"candidate_wins": 1.0, "log_length_delta": -0.5},
            {"candidate_wins": 0.0, "log_length_delta": 0.5},
        ]
        estimate, slope = fit_records(records)
        self.assertTrue(0.0 < estimate < 1.0)
        self.assertLess(slope, 0.0)

    def test_cohen_kappa_perfect_and_disagreement(self):
        ratings = ["A", "B", "TIE", "A"]
        self.assertEqual(kappa(ratings, ratings), 1.0)
        self.assertLess(kappa(ratings, ["B", "A", "TIE", "B"]), 0.0)


class AnalysisIntegrationTests(unittest.TestCase):
    def test_analysis_writes_lcwr_and_oa_tables(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is optional in the test environment")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / "run" / "eval"
            run_dir.mkdir(parents=True)
            for seed in ("42", "1337"):
                for arm, outcomes in (("G4", [1.0, 1.0, 0.0, 0.0]), ("G5", [1.0, 0.0, 0.0, 0.0])):
                    rows = [
                        {
                            "item_id": f"p{i}", "arm": arm, "budget": "500", "seed": seed,
                            "candidate_wins": outcome, "log_length_delta": delta,
                            "judge_model": "test",
                        }
                        for i, (outcome, delta) in enumerate(zip(outcomes, (-1.0, -0.3, 0.3, 1.0)))
                    ]
                    write_jsonl(run_dir / f"lcwr_{arm}_{seed}.jsonl", rows)
            out_dir = root / "results"
            with patch.object(sys, "argv", [
                "analyze.py", "--input-dir", str(root / "runs"), "--output-dir", str(out_dir),
                "--bootstrap", "5",
            ]):
                analyze_main()
            self.assertEqual(len(list(out_dir.glob("lcwr_v1.0_*.csv"))), 1)
            self.assertEqual(len(list(out_dir.glob("oa_v1.0_*.csv"))), 1)


if __name__ == "__main__":
    unittest.main()
