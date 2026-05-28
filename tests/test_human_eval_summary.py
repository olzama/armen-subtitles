import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from compute_human_eval_summary import (
    _pearson,
    _rank,
    _spearman,
    _krippendorff_alpha_interval,
    compute_iaa,
    compute_within_annotator_reliability,
    records_to_method_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(evaluator_id, item_id, score, method="zero", run="1",
                film="test-film", trans_model="gpt-5.2",
                is_repeat=False, auto_filled=False):
    return {
        "evaluator_id":   evaluator_id,
        "evaluator_meta": {},
        "film":           film,
        "trans_model":    trans_model,
        "item_id":        item_id,
        "method":         method,
        "run":            run,
        "is_repeat":      is_repeat,
        "auto_filled":    auto_filled,
        "score":          score,
        "issues":         [],
    }


# ---------------------------------------------------------------------------
# _pearson
# ---------------------------------------------------------------------------

def test_pearson_perfect_positive():
    assert _pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    assert _pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_too_short():
    assert _pearson([1], [1]) is None


def test_pearson_constant_x_returns_none():
    # Zero variance in x → denominator is 0
    assert _pearson([1, 1, 1], [1, 2, 3]) is None


# ---------------------------------------------------------------------------
# _rank
# ---------------------------------------------------------------------------

def test_rank_no_ties():
    assert _rank([3, 1, 2]) == pytest.approx([3.0, 1.0, 2.0])


def test_rank_with_ties():
    # [1, 1, 3]: the two 1s share ranks 1 and 2, so each gets 1.5
    assert _rank([1, 1, 3]) == pytest.approx([1.5, 1.5, 3.0])


# ---------------------------------------------------------------------------
# _spearman
# ---------------------------------------------------------------------------

def test_spearman_perfect_positive():
    assert _spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    assert _spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_too_short():
    assert _spearman([1], [1]) is None


def test_spearman_with_ties():
    # [1, 1, 2] → ranks [1.5, 1.5, 3]; [3, 4, 5] → ranks [1, 2, 3]
    # Pearson on those ranks: mx=2, my=2
    # num = (1.5−2)(1−2) + (1.5−2)(2−2) + (3−2)(3−2) = 0.5 + 0 + 1 = 1.5
    # den = sqrt(1.5 * 2) = sqrt(3)
    expected = 1.5 / math.sqrt(3)
    assert _spearman([1, 1, 2], [3, 4, 5]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _krippendorff_alpha_interval
# ---------------------------------------------------------------------------

def test_krippendorff_perfect_agreement():
    # All raters agree → observed disagreement = 0 → alpha = 1.0
    assert _krippendorff_alpha_interval([[1, 1], [2, 2], [3, 3]]) == pytest.approx(1.0)


def test_krippendorff_no_pairable_units():
    # Every unit has only one rating → d_o_den stays 0 → None
    assert _krippendorff_alpha_interval([[1], [2], [3]]) is None


def test_krippendorff_opposite_ratings():
    # Two raters, two units, completely reversed:
    # unit1: [0, 1]  unit2: [1, 0]
    # d_o = ((0-1)^2 + (1-0)^2) / 2 = 1
    # pairable = [0, 1, 1, 0]; d_e = 4/6 = 2/3
    # alpha = 1 − 1 / (2/3) = −0.5
    assert _krippendorff_alpha_interval([[0, 1], [1, 0]]) == pytest.approx(-0.5)


def test_krippendorff_all_identical_values():
    # Every rating is the same value → d_e = 0, d_o = 0 → alpha = 1.0
    assert _krippendorff_alpha_interval([[2, 2], [2, 2]]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_iaa
# ---------------------------------------------------------------------------

def test_iaa_no_shared_items():
    # Two evaluators rate completely different items
    records = [
        make_record("a", item_id=1, score=0.5),
        make_record("b", item_id=2, score=0.3),
    ]
    result = compute_iaa(records)
    assert result["num_shared_items"] == 0
    assert "note" in result


def test_iaa_empty_records():
    result = compute_iaa([])
    assert result["num_shared_items"] == 0


def test_iaa_perfect_agreement():
    # Both evaluators rate the same 3 items with identical scores
    scores = [0.2, 0.5, 0.8]
    records = [make_record(eid, item_id=i + 1, score=s)
               for eid in ("a", "b")
               for i, s in enumerate(scores)]
    result = compute_iaa(records)
    assert result["num_shared_items"] == 3
    assert result["krippendorff_alpha_interval"] == pytest.approx(1.0)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)


def test_iaa_fewer_than_two_shared_items_skips_pairwise():
    # Only 1 item in common: not enough for correlation (needs >= 2)
    records = [
        make_record("a", item_id=1, score=0.5),
        make_record("b", item_id=1, score=0.5),
    ]
    result = compute_iaa(records)
    assert result["num_shared_items"] == 1
    assert result["pairwise"] == []
    # alpha is still computed (only needs >= 1 pairable unit)
    assert result["krippendorff_alpha_interval"] == pytest.approx(1.0)


def test_iaa_top_level_pearson_only_when_one_pair():
    # With exactly one evaluator pair, pearson_r and spearman_rho are
    # promoted to the top level for convenience
    records = [make_record(eid, item_id=i + 1, score=float(i))
               for eid in ("a", "b")
               for i in range(3)]
    result = compute_iaa(records)
    assert "pearson_r" in result
    assert "spearman_rho" in result


# ---------------------------------------------------------------------------
# compute_within_annotator_reliability
# ---------------------------------------------------------------------------

def test_within_annotator_no_repeats():
    records = [make_record("a", item_id=i, score=0.5) for i in range(5)]
    assert compute_within_annotator_reliability(records) == []


def test_within_annotator_perfect_consistency():
    # Evaluator rates 3 items first, then repeats them with the same scores
    first = [make_record("a", item_id=i, score=float(i), is_repeat=False)
             for i in range(1, 4)]
    repeat = [make_record("a", item_id=i, score=float(i), is_repeat=True)
              for i in range(1, 4)]
    result = compute_within_annotator_reliability(first + repeat)
    assert len(result) == 1
    assert result[0]["evaluator_id"] == "a"
    assert result[0]["num_repeated_items"] == 3
    assert result[0]["pearson_r"] == pytest.approx(1.0)
    assert result[0]["spearman_rho"] == pytest.approx(1.0)


def test_within_annotator_only_one_repeated_item_excluded():
    # Only 1 shared repeat item: not enough for correlation (needs >= 2)
    first = [make_record("a", item_id=1, score=0.5, is_repeat=False)]
    repeat = [make_record("a", item_id=1, score=0.6, is_repeat=True)]
    assert compute_within_annotator_reliability(first + repeat) == []


def test_within_annotator_auto_filled_ignored():
    # Auto-filled items must not contribute even if is_repeat=False
    first = [make_record("a", item_id=i, score=float(i), auto_filled=True)
             for i in range(1, 4)]
    repeat = [make_record("a", item_id=i, score=float(i), is_repeat=True)
              for i in range(1, 4)]
    assert compute_within_annotator_reliability(first + repeat) == []


# ---------------------------------------------------------------------------
# records_to_method_data
# ---------------------------------------------------------------------------

def test_records_to_method_data_structure():
    records = [
        make_record("a", item_id=1, score=0.4, method="zero", run="1"),
        make_record("a", item_id=2, score=0.6, method="zero", run="1"),
        make_record("a", item_id=1, score=0.2, method="summary", run="1"),
    ]
    result = records_to_method_data(records)
    assert set(result.keys()) == {"zero", "summary"}
    assert "1" in result["zero"]
    # Each entry is a list of {"eval_run": int, "value": float}
    entries = result["zero"]["1"]
    assert len(entries) == 2
    assert all("eval_run" in e and "value" in e for e in entries)
    values = {e["value"] for e in entries}
    assert values == {0.4, 0.6}


def test_records_to_method_data_excludes_repeats_and_autofilled():
    records = [
        make_record("a", item_id=1, score=0.5, method="zero", run="1"),
        make_record("a", item_id=2, score=0.9, method="zero", run="1", is_repeat=True),
        make_record("a", item_id=3, score=0.7, method="zero", run="1", auto_filled=True),
    ]
    result = records_to_method_data(records)
    # Only the non-repeat, non-auto-filled item should appear
    assert len(result["zero"]["1"]) == 1
    assert result["zero"]["1"][0]["value"] == pytest.approx(0.5)
