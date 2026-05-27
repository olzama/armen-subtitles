import math
import statistics
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

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


# ---------------------------------------------------------------------------
# MQM scoring
# ---------------------------------------------------------------------------

def test_single_run_score():
    data = {
        "method": "zero", "run": "1",
        "items": [
            {"id": 1, "issues": [{"severity": "major", "category": "accuracy",
                                   "justification": "Test major issue."}]},
            {"id": 2, "issues": [{"severity": "minor", "category": "style",
                                   "justification": "Test minor issue."}]},
        ],
    }
    score = compute_mqm_score(data, data)
    assert score["counts"]["major"] == 1
    assert score["counts"]["minor"] == 1
    assert score["total_points"] == 6
    assert score["meaning_units"] == 2
    assert score["penalty_per_unit"] == pytest.approx(3.0)
    assert score["major_equiv_per_unit"] == pytest.approx(0.6)


def test_no_issue_category_excluded():
    data = {
        "method": "zero", "run": "1",
        "items": [
            {"id": 1, "issues": [{"severity": "major", "category": "no-issue",
                                   "justification": "Should not count."}]},
            {"id": 2, "issues": [{"severity": "minor", "category": "style",
                                   "justification": "Should count."}]},
            {"id": 3, "issues": []},
        ],
    }
    score = compute_mqm_score(data, data)
    assert score["counts"]["major"] == 0
    assert score["counts"]["minor"] == 1
    assert score["total_points"] == 1
    assert score["meaning_units"] == 3
    assert score["penalty_per_unit"] == pytest.approx(1 / 3)
    assert score["major_equiv_per_unit"] == pytest.approx(1 / 15)


# ---------------------------------------------------------------------------
# Task discovery and usage aggregation
# ---------------------------------------------------------------------------

def test_task_discovery(two_item_translation_data):
    shared_items = collect_shared_items(two_item_translation_data, "rus", "eng")
    tasks, _ = build_translation_tasks(shared_items)

    assert len(tasks) == 4
    assert tasks[0]["method"] == "characters"
    assert tasks[0]["run"] == "1"
    assert len(tasks[0]["items"]) == 2


def test_usage_aggregation(two_item_translation_data):
    shared_items = collect_shared_items(two_item_translation_data, "rus", "eng")
    tasks, _ = build_translation_tasks(shared_items)

    fake_results = [
        {"input_tokens": 1000, "output_tokens": 100, "cost_total": 0.00315},
        {"input_tokens": 1100, "output_tokens": 120, "cost_total": 0.00361},
        {"input_tokens":  900, "output_tokens":  80, "cost_total": 0.002695},
        {"input_tokens":  950, "output_tokens":  90, "cost_total": 0.0029225},
    ]
    totals = accumulate_usage(fake_results)

    assert totals["total_input_tokens"] == 3950
    assert totals["total_output_tokens"] == 390
    assert totals["total_cost"] == pytest.approx(0.0123775)


def test_rates_gpt52_unchanged():
    expected_cost = 1000 * RATES["gpt-5.2"]["input"] + 100 * RATES["gpt-5.2"]["output"]
    assert expected_cost == pytest.approx(0.00315)


# ---------------------------------------------------------------------------
# Translation-level statistics
# ---------------------------------------------------------------------------

def test_translation_level_stats():
    scores = [0.4, 0.6, 0.5]
    stats = summarize_translation(scores)

    assert stats["mean_major_equiv_per_unit"] == pytest.approx(statistics.mean(scores))
    assert stats["eval_run_sd"] == pytest.approx(statistics.stdev(scores))
    expected_se = statistics.stdev(scores) / math.sqrt(len(scores))
    assert stats["eval_noise_se"] == pytest.approx(expected_se)
    assert stats["ci_95_half_width"] == pytest.approx(1.96 * expected_se)
    assert stats["n_eval_runs"] == 3
    assert stats["eval_scores"] == scores


# ---------------------------------------------------------------------------
# Method-level statistics
# ---------------------------------------------------------------------------

def test_method_level_stats():
    run1_scores = [0.4, 0.6, 0.5]
    run2_scores = [0.7, 0.8, 0.9]
    per_translation = {
        "1": summarize_translation(run1_scores),
        "2": summarize_translation(run2_scores),
    }
    stats = summarize_method(per_translation)

    means = [statistics.mean(run1_scores), statistics.mean(run2_scores)]
    expected_mean = statistics.mean(means)
    expected_sd = statistics.stdev(means)
    expected_se = expected_sd / math.sqrt(2)
    expected_ci = 1.96 * expected_se
    expected_pooled_sd = pooled_sd_from_groups([run1_scores, run2_scores])

    assert stats["num_translations"] == 2
    assert stats["eval_runs_per_translation"] == 3
    assert stats["observed_eval_runs_per_translation"] == [3]
    assert stats["mean_major_equiv_per_unit"] == pytest.approx(expected_mean)
    assert stats["translation_sd"] == pytest.approx(expected_sd)
    assert stats["se_method"] == pytest.approx(expected_se)
    assert stats["ci_95_half_width"] == pytest.approx(expected_ci)
    assert stats["ci_95_lower"] == pytest.approx(expected_mean - expected_ci)
    assert stats["ci_95_upper"] == pytest.approx(expected_mean + expected_ci)
    assert stats["pooled_eval_run_sd"] == pytest.approx(expected_pooled_sd)
    assert stats["avg_eval_noise"] == pytest.approx(expected_pooled_sd / math.sqrt(3))
    assert stats["translation_means"] == means


# ---------------------------------------------------------------------------
# Overall cross-method statistics
# ---------------------------------------------------------------------------

def test_stats_across_methods():
    method_a = summarize_method({
        "1": summarize_translation([0.4, 0.6, 0.5]),
        "2": summarize_translation([0.7, 0.8, 0.9]),
    })
    method_b = summarize_method({
        "1": summarize_translation([0.0, 0.1, 0.2]),
        "2": summarize_translation([0.2, 0.3, 0.4]),
    })
    overall = summarize_overall({"A": method_a, "B": method_b})

    means = [method_a["mean_major_equiv_per_unit"], method_b["mean_major_equiv_per_unit"]]
    expected_mean = statistics.mean(means)
    expected_sd = statistics.stdev(means)
    expected_se = expected_sd / math.sqrt(2)
    expected_ci = 1.96 * expected_se

    assert overall["num_methods"] == 2
    assert overall["mean_major_equiv_per_unit"] == pytest.approx(expected_mean)
    assert overall["method_sd"] == pytest.approx(expected_sd)
    assert overall["se_method_across_methods"] == pytest.approx(expected_se)
    assert overall["ci_95_half_width"] == pytest.approx(expected_ci)
    assert overall["ci_95_lower"] == pytest.approx(expected_mean - expected_ci)
    assert overall["ci_95_upper"] == pytest.approx(expected_mean + expected_ci)
    assert overall["method_means"] == means
