#!/usr/bin/env python3

import json
import math
import statistics
import tempfile
from pathlib import Path

from evaluate_mqm_parallel import compute_mqm_score, compute_method_stats


class TestRecorder:
    def __init__(self):
        self.failures = []
        self.passes = 0

    def check_equal(self, actual, expected, label):
        if actual == expected:
            self.passes += 1
            print(f"PASS: {label}")
        else:
            self.failures.append(f"{label}: expected {expected}, got {actual}")
            print(f"FAIL: {label}: expected {expected}, got {actual}")

    def check_close(self, actual, expected, label, rel_tol=1e-9, abs_tol=1e-9):
        if math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            self.passes += 1
            print(f"PASS: {label}")
        else:
            self.failures.append(f"{label}: expected {expected}, got {actual}")
            print(f"FAIL: {label}: expected {expected}, got {actual}")

    def section(self, name):
        print(f"\n=== {name} ===")

    def summary(self):
        print("\n=== TEST SUMMARY ===")
        print(f"Passed checks: {self.passes}")
        print(f"Failed checks: {len(self.failures)}")
        if self.failures:
            print("\nFailed details:")
            for failure in self.failures:
                print(f"- {failure}")
            return 1
        print("ALL TESTS PASSED")
        return 0


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_single_run_score(tmpdir: Path, tr: TestRecorder):
    tr.section("single-run score test")

    test_run_score = {
        "method": "zero",
        "run": "1",
        "items": [
            {
                "id": 1,
                "issues": [
                    {
                        "severity": "major",
                        "category": "accuracy",
                        "justification": "Test major issue."
                    }
                ]
            },
            {
                "id": 2,
                "issues": [
                    {
                        "severity": "minor",
                        "category": "style",
                        "justification": "Test minor issue."
                    }
                ]
            }
        ]
    }

    path = tmpdir / "test_run_score.json"
    write_json(path, test_run_score)

    data = load_json(path)
    score = compute_mqm_score(data, data)

    tr.check_equal(score["counts"]["critical"], 0, "single-run critical count")
    tr.check_equal(score["counts"]["major"], 1, "single-run major count")
    tr.check_equal(score["counts"]["minor"], 1, "single-run minor count")
    tr.check_equal(score["total_points"], 6, "single-run total points")
    tr.check_equal(score["meaning_units"], 2, "single-run meaning units")
    tr.check_close(score["penalty_per_unit"], 3.0, "single-run penalty per unit")
    tr.check_close(score["major_equiv_per_unit"], 0.6, "single-run major-equiv per unit")


def test_aggregation(tmpdir: Path, tr: TestRecorder):
    tr.section("aggregation test")

    a_run_1 = {
        "method": "A",
        "run": "1",
        "items": [
            {
                "id": 1,
                "issues": [
                    {
                        "severity": "major",
                        "category": "accuracy",
                        "justification": "x"
                    }
                ]
            },
            {
                "id": 2,
                "issues": []
            }
        ]
    }

    a_run_2 = {
        "method": "A",
        "run": "2",
        "items": [
            {
                "id": 1,
                "issues": [
                    {
                        "severity": "major",
                        "category": "accuracy",
                        "justification": "x"
                    }
                ]
            },
            {
                "id": 2,
                "issues": [
                    {
                        "severity": "major",
                        "category": "style",
                        "justification": "x"
                    }
                ]
            }
        ]
    }

    b_run_1 = {
        "method": "B",
        "run": "1",
        "items": [
            {
                "id": 1,
                "issues": []
            },
            {
                "id": 2,
                "issues": []
            }
        ]
    }

    b_run_2 = {
        "method": "B",
        "run": "2",
        "items": [
            {
                "id": 1,
                "issues": [
                    {
                        "severity": "minor",
                        "category": "fluency",
                        "justification": "x"
                    }
                ]
            },
            {
                "id": 2,
                "issues": [
                    {
                        "severity": "minor",
                        "category": "style",
                        "justification": "x"
                    }
                ]
            }
        ]
    }

    files = {
        "A_run_1.json": a_run_1,
        "A_run_2.json": a_run_2,
        "B_run_1.json": b_run_1,
        "B_run_2.json": b_run_2,
    }

    for filename, data in files.items():
        write_json(tmpdir / filename, data)

    scores_by_method = {}
    all_run_scores = []

    for filename in ["A_run_1.json", "A_run_2.json", "B_run_1.json", "B_run_2.json"]:
        data = load_json(tmpdir / filename)
        score = compute_mqm_score(data, data)
        method = data["method"]
        scores_by_method.setdefault(method, []).append(score["major_equiv_per_unit"])
        all_run_scores.append(score["major_equiv_per_unit"])

    tr.check_close(scores_by_method["A"][0], 0.5, "A run 1 score")
    tr.check_close(scores_by_method["A"][1], 1.0, "A run 2 score")
    tr.check_close(scores_by_method["B"][0], 0.0, "B run 1 score")
    tr.check_close(scores_by_method["B"][1], 0.2, "B run 2 score")

    a_stats = compute_method_stats(scores_by_method["A"])
    b_stats = compute_method_stats(scores_by_method["B"])

    tr.check_close(a_stats["mean_major_equiv_per_unit"], 0.75, "A mean")
    tr.check_close(a_stats["run_sd"], 0.3535533905932738, "A run SD")
    tr.check_close(a_stats["se_mean"], 0.25, "A SE")
    tr.check_close(a_stats["ci_95_half_width"], 0.49, "A 95% CI half-width")
    tr.check_equal(a_stats["n_runs"], 2, "A n_runs")

    tr.check_close(b_stats["mean_major_equiv_per_unit"], 0.1, "B mean")
    tr.check_close(b_stats["run_sd"], 0.1414213562373095, "B run SD")
    tr.check_close(b_stats["se_mean"], 0.1, "B SE")
    tr.check_close(b_stats["ci_95_half_width"], 0.196, "B 95% CI half-width")
    tr.check_equal(b_stats["n_runs"], 2, "B n_runs")

    method_means = [
        a_stats["mean_major_equiv_per_unit"],
        b_stats["mean_major_equiv_per_unit"],
    ]
    overall_major_mean = statistics.mean(method_means)
    method_sd = statistics.stdev(method_means)
    se_method = method_sd / math.sqrt(len(method_means))
    ci_95_half_width = 1.96 * se_method
    ci_lower = overall_major_mean - ci_95_half_width
    ci_upper = overall_major_mean + ci_95_half_width
    overall_run_sd = statistics.stdev(all_run_scores)

    tr.check_close(overall_major_mean, 0.425, "overall mean across methods")
    tr.check_close(method_sd, 0.4596194077712559, "method-level SD")
    tr.check_close(se_method, 0.325, "method-level SE")
    tr.check_close(ci_95_half_width, 0.637, "overall 95% CI half-width")
    tr.check_close(ci_lower, -0.212, "overall CI lower")
    tr.check_close(ci_upper, 1.062, "overall CI upper")
    tr.check_close(overall_run_sd, 0.43493294502332963, "overall run-level SD")


def test_no_issue_handling(tmpdir: Path, tr: TestRecorder):
    tr.section("no-issue handling test")

    test_no_issue = {
        "method": "zero",
        "run": "1",
        "items": [
            {
                "id": 1,
                "issues": [
                    {
                        "severity": "major",
                        "category": "no-issue",
                        "justification": "This should not count."
                    }
                ]
            },
            {
                "id": 2,
                "issues": [
                    {
                        "severity": "minor",
                        "category": "style",
                        "justification": "This should count."
                    }
                ]
            },
            {
                "id": 3,
                "issues": []
            }
        ]
    }

    path = tmpdir / "test_no_issue.json"
    write_json(path, test_no_issue)

    data = load_json(path)
    score = compute_mqm_score(data, data)

    tr.check_equal(score["counts"]["critical"], 0, "no-issue critical count")
    tr.check_equal(score["counts"]["major"], 0, "no-issue major count")
    tr.check_equal(score["counts"]["minor"], 1, "no-issue minor count")
    tr.check_equal(score["total_points"], 1, "no-issue total points")
    tr.check_equal(score["meaning_units"], 3, "no-issue meaning units")
    tr.check_close(score["penalty_per_unit"], 1 / 3, "no-issue penalty per unit")
    tr.check_close(score["major_equiv_per_unit"], 1 / 15, "no-issue major-equiv per unit")


def main():
    tr = TestRecorder()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        test_single_run_score(tmpdir, tr)
        test_aggregation(tmpdir, tr)
        test_no_issue_handling(tmpdir, tr)

    raise SystemExit(tr.summary())


if __name__ == "__main__":
    main()