#!/usr/bin/env python3

import json
import math
import statistics
import sys
import tempfile
import types
from pathlib import Path

# Stub optional API modules so the evaluator helpers can be imported in a test environment
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

if "google" not in sys.modules:
    google_stub = types.ModuleType("google")
    genai_stub = types.ModuleType("google.genai")
    types_stub = types.ModuleType("google.genai.types")
    genai_stub.types = types_stub
    google_stub.genai = genai_stub
    sys.modules["google"] = google_stub
    sys.modules["google.genai"] = genai_stub
    sys.modules["google.genai.types"] = types_stub

from evaluate_mqm_parallel import (
    RATES,
    compute_mqm_score,
    collect_shared_items,
    build_translation_tasks,
    accumulate_usage,
    summarize_translation,
    summarize_method,
    summarize_overall,
    pooled_sd_from_groups,
)


class TestRecorder:
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self):
        self.failures = []
        self.passes = 0

    def check_equal(self, actual, expected, label):
        if actual == expected:
            self.passes += 1
            print(f"{self.GREEN}PASS{self.RESET}: {label}")
        else:
            msg = f"{label}: expected {expected}, got {actual}"
            self.failures.append(msg)
            print(f"{self.RED}FAIL{self.RESET}: {msg}")

    def check_close(self, actual, expected, label, rel_tol=1e-9, abs_tol=1e-9):
        if math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            self.passes += 1
            print(f"{self.GREEN}PASS{self.RESET}: {label}")
        else:
            msg = f"{label}: expected {expected}, got {actual}"
            self.failures.append(msg)
            print(f"{self.RED}FAIL{self.RESET}: {msg}")

    def section(self, name):
        print(f"\n{self.BOLD}=== {name} ==={self.RESET}")

    def summary(self):
        print(f"\n{self.BOLD}=== TEST SUMMARY ==={self.RESET}")
        print(f"Passed checks: {self.passes}")
        print(f"Failed checks: {len(self.failures)}")
        if self.failures:
            print(f"\n{self.YELLOW}Failed details:{self.RESET}")
            for failure in self.failures:
                print(f"{self.RED}- {failure}{self.RESET}")
            return 1
        print(f"{self.GREEN}ALL TESTS PASSED{self.RESET}")
        return 0


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_usage_aggregation(tr: TestRecorder):
    tr.section("usage aggregation and task discovery")

    input = {
        "title": "MINIMAL TEST",
        "model": "gpt-5.2",
        "items": [
            {
                "id": 1,
                "character": "Bunsha",
                "original": {"rus": "Меня опять терзают смутные сомнения…"},
                "reference": {"eng": "Vague doubts haunt me once again..."},
                "analysis": "Mock-elevated theatrical tone matters.",
                "segment_number": [569],
                "translations": {
                    "eng": {
                        "zero": {
                            "1": "I’m tormented by vague doubts again…",
                            "2": "Vague doubts plague me again..."
                        },
                        "characters": {
                            "1": "Once again I’m tormented by vague misgivings...",
                            "2": "I’m plagued by vague doubts again..."
                        }
                    }
                }
            },
            {
                "id": 2,
                "character": "Ivan the Terrible",
                "original": {"rus": "Оставь меня, старушка, я в печали"},
                "reference": {"eng": "Leave me, old woman, I am in sorrow."},
                "analysis": "Archaic elevated register matters.",
                "segment_number": [597],
                "translations": {
                    "eng": {
                        "zero": {
                            "1": "Leave me, old woman, I’m in sorrow.",
                            "2": "Leave me alone, old woman, I’m sad."
                        },
                        "characters": {
                            "1": "Leave me, old woman, I am in sorrow.",
                            "2": "Go away, old woman, I’m grieving."
                        }
                    }
                }
            }
        ]
    }

    shared_items = collect_shared_items(input)
    tasks = build_translation_tasks(shared_items)

    tr.check_equal(len(tasks), 4, "number of discovered translation tasks")
    tr.check_equal(tasks[0]["method"], "characters", "first task method sorted")
    tr.check_equal(tasks[0]["run"], "1", "first task run sorted")
    tr.check_equal(len(tasks[0]["items"]), 2, "items copied into task")

    fake_usage = {
        ("zero", "1"): {"input_tokens": 1000, "output_tokens": 100, "cost_total": 0.00315},
        ("zero", "2"): {"input_tokens": 1100, "output_tokens": 120, "cost_total": 0.00361},
        ("characters", "1"): {"input_tokens": 900, "output_tokens": 80, "cost_total": 0.002695},
        ("characters", "2"): {"input_tokens": 950, "output_tokens": 90, "cost_total": 0.0029225},
    }

    fake_results = [fake_usage[(task["method"], task["run"])] for task in tasks]
    totals = accumulate_usage(fake_results)

    tr.check_equal(totals["total_input_tokens"], 3950, "aggregated input tokens")
    tr.check_equal(totals["total_output_tokens"], 390, "aggregated output tokens")
    tr.check_close(totals["total_cost"], 0.0123775, "aggregated cost total")

    expected_cost = (
        1000 * RATES["gpt-5.2"]["input"] + 100 * RATES["gpt-5.2"]["output"]
    )
    tr.check_close(expected_cost, 0.00315, "pricing table unchanged for gpt-5.2 sample")



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



def test_no_issues(tmpdir: Path, tr: TestRecorder):
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



def test_translation_level_stats(tr: TestRecorder):
    tr.section("translation-level evaluator-repeat statistics")

    eval_scores = [0.4, 0.6, 0.5]
    stats = summarize_translation(eval_scores)

    expected_mean = statistics.mean(eval_scores)
    expected_sd = statistics.stdev(eval_scores)
    expected_se = expected_sd / math.sqrt(len(eval_scores))
    expected_ci = 1.96 * expected_se

    tr.check_close(stats["mean_major_equiv_per_unit"], expected_mean, "translation mean across eval runs")
    tr.check_close(stats["eval_run_sd"], expected_sd, "translation eval-run SD")
    tr.check_close(stats["eval_noise_se"], expected_se, "translation evaluator noise SE")
    tr.check_close(stats["ci_95_half_width"], expected_ci, "translation 95% CI half-width")
    tr.check_equal(stats["n_eval_runs"], 3, "translation n_eval_runs")
    tr.check_equal(stats["eval_scores"], eval_scores, "translation scores preserved")

def test_method_level_stats(tr: TestRecorder):
    tr.section("method-level statistics with separate T and E")

    # Two translations (T=2), each with three evaluator repeats (E=3)
    run1_scores = [0.4, 0.6, 0.5]
    run2_scores = [0.7, 0.8, 0.9]

    per_translation = {
        "1": summarize_translation(run1_scores),
        "2": summarize_translation(run2_scores),
    }
    method_stats = summarize_method(per_translation)

    mean1 = statistics.mean(run1_scores)
    mean2 = statistics.mean(run2_scores)
    translation_means = [mean1, mean2]
    expected_method_mean = statistics.mean(translation_means)
    expected_translation_sd = statistics.stdev(translation_means)
    expected_se_method = expected_translation_sd / math.sqrt(2)
    expected_ci = 1.96 * expected_se_method
    expected_pooled_sd = pooled_sd_from_groups([run1_scores, run2_scores])
    expected_avg_eval_noise = expected_pooled_sd / math.sqrt(3)

    tr.check_equal(method_stats["num_translations"], 2, "method T")
    tr.check_equal(method_stats["eval_runs_per_translation"], 3, "method common E")
    tr.check_equal(method_stats["observed_eval_runs_per_translation"], [3], "method observed E values")
    tr.check_close(method_stats["mean_major_equiv_per_unit"], expected_method_mean, "method mean over translation means")
    tr.check_close(method_stats["translation_sd"], expected_translation_sd, "translation SD across translation means")
    tr.check_close(method_stats["se_method"], expected_se_method, "method SE from T")
    tr.check_close(method_stats["ci_95_half_width"], expected_ci, "method 95% CI half-width")
    tr.check_close(method_stats["ci_95_lower"], expected_method_mean - expected_ci, "method CI lower")
    tr.check_close(method_stats["ci_95_upper"], expected_method_mean + expected_ci, "method CI upper")
    tr.check_close(method_stats["pooled_eval_run_sd"], expected_pooled_sd, "pooled eval-run SD")
    tr.check_close(method_stats["avg_eval_noise"], expected_avg_eval_noise, "average evaluator noise from pooled SD")
    tr.check_equal(method_stats["translation_means"], translation_means, "translation means preserved")



def test_stats_across_methods(tr: TestRecorder):
    tr.section("overall statistics across methods")

    method_a = summarize_method({
        "1": summarize_translation([0.4, 0.6, 0.5]),
        "2": summarize_translation([0.7, 0.8, 0.9]),
    })
    method_b = summarize_method({
        "1": summarize_translation([0.0, 0.1, 0.2]),
        "2": summarize_translation([0.2, 0.3, 0.4]),
    })

    overall = summarize_overall({"A": method_a, "B": method_b})

    method_means = [method_a["mean_major_equiv_per_unit"], method_b["mean_major_equiv_per_unit"]]
    expected_mean = statistics.mean(method_means)
    expected_sd = statistics.stdev(method_means)
    expected_se = expected_sd / math.sqrt(2)
    expected_ci = 1.96 * expected_se

    tr.check_equal(overall["num_methods"], 2, "overall number of methods")
    tr.check_close(overall["mean_major_equiv_per_unit"], expected_mean, "overall mean across methods")
    tr.check_close(overall["method_sd"], expected_sd, "overall method SD")
    tr.check_close(overall["se_method_across_methods"], expected_se, "overall SE across methods")
    tr.check_close(overall["ci_95_half_width"], expected_ci, "overall CI half-width")
    tr.check_close(overall["ci_95_lower"], expected_mean - expected_ci, "overall CI lower")
    tr.check_close(overall["ci_95_upper"], expected_mean + expected_ci, "overall CI upper")
    tr.check_equal(overall["method_means"], method_means, "overall method means preserved")



def main():
    tr = TestRecorder()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        test_single_run_score(tmpdir, tr)
        test_no_issues(tmpdir, tr)
        test_usage_aggregation(tr)
        test_translation_level_stats(tr)
        test_method_level_stats(tr)
        test_stats_across_methods(tr)

    raise SystemExit(tr.summary())


if __name__ == "__main__":
    main()
