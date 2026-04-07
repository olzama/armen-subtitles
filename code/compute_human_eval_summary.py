#!/usr/bin/env python3
"""
Compute MQM scores and interannotator agreement from human evaluation files.

Usage:
    python compute_human_eval_summary.py <eval_dir>

    <eval_dir> is a directory containing one or more human_*.json files
    (produced by the web evaluation tool / evaluate_human.py session export).

Output:
    <eval_dir>/human_eval_summary.json
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Reuse scoring and aggregation logic from existing modules
from evaluate_human import score_issues, SEVERITY_WEIGHTS
from aggregate_mqm import (
    compute_method_stats,
    compute_overall_across_methods,
    build_ranking,
)


# ─────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────

def load_human_file(path):
    """
    Load one human_*.json and return a list of scored judgment records.

    Each record:
        evaluator_id, evaluator_meta, film, trans_model, item_id, method, run,
        is_repeat, auto_filled, score (major_equiv_per_unit), issues
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    evaluator_id = data["evaluator_id"]
    evaluator_meta = data.get("evaluator_meta", {})
    tasks = data["tasks"]
    judgments = data["judgments"]          # str(index) -> {issues, ...}
    auto_filled_set = set(data.get("auto_filled_set", []))

    records = []
    for idx, task in enumerate(tasks):
        judgment = judgments.get(str(idx))
        if judgment is None:
            continue
        issues = judgment.get("issues", [])
        records.append({
            "evaluator_id":   evaluator_id,
            "evaluator_meta": evaluator_meta,
            "film":           task["film"],
            "trans_model":    task["trans_model"],
            "item_id":        task["item_id"],
            "method":         task["method"],
            "run":            str(task["run"]),
            "is_repeat":      task.get("is_repeat", False),
            "auto_filled":    idx in auto_filled_set,
            "score":          score_issues(issues),
            "issues":         issues,
        })
    return records


# ─────────────────────────────────────────────
# RESHAPE FOR aggregate_mqm COMPATIBILITY
# ─────────────────────────────────────────────

def records_to_method_data(records):
    """
    Convert human records into the dict expected by compute_method_stats:

        method -> run_id -> [{"eval_run": int, "value": float}, ...]

    Each run is treated as one "translation" (the unit of comparison, same as
    in auto eval).  Each (item, annotator) score within that run is treated as
    one "eval_run" observation — compute_method_stats will average them to get
    the per-run mean, then compute run-to-run SD/SE for the method ±.

    This makes the ± directly comparable to the auto-eval ±: both reflect
    how much quality varies across runs of the same method.
    """
    # method -> run_id -> list of scores (one per item × annotator)
    raw = defaultdict(lambda: defaultdict(list))

    for r in records:
        if r["auto_filled"] or r["is_repeat"]:
            continue
        raw[r["method"]][r["run"]].append(r["score"])

    method_data = {}
    for method, run_dict in raw.items():
        method_data[method] = {}
        for run_id, scores in run_dict.items():
            method_data[method][run_id] = [
                {"eval_run": i + 1, "value": score}
                for i, score in enumerate(scores)
            ]

    return method_data


# ─────────────────────────────────────────────
# INTERANNOTATOR AGREEMENT
# ─────────────────────────────────────────────

def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / den if den > 0 else None


def _rank(values):
    """Return rank list (average ties)."""
    sorted_vals = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j + 1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys)) if len(xs) >= 2 else None


def _krippendorff_alpha_interval(ratings_by_unit):
    """
    Krippendorff's alpha (interval scale).
    ratings_by_unit: list of lists; units with < 2 ratings are ignored.
    """
    pairable = []
    d_o_num, d_o_den = 0.0, 0

    for unit_ratings in ratings_by_unit:
        if len(unit_ratings) < 2:
            continue
        pairable.extend(unit_ratings)
        n_u = len(unit_ratings)
        for i in range(n_u):
            for j in range(i + 1, n_u):
                d_o_num += (unit_ratings[i] - unit_ratings[j]) ** 2
                d_o_den += 1

    if d_o_den == 0:
        return None

    d_o = d_o_num / d_o_den

    n_all = len(pairable)
    d_e_num, d_e_den = 0.0, 0
    for i in range(n_all):
        for j in range(i + 1, n_all):
            d_e_num += (pairable[i] - pairable[j]) ** 2
            d_e_den += 1

    if d_e_den == 0:
        return None
    d_e = d_e_num / d_e_den
    if d_e == 0.0:
        return 1.0 if d_o == 0.0 else None

    return 1.0 - d_o / d_e


def compute_iaa(records):
    """Compute interannotator agreement on shared (non-auto-filled, non-repeat) items."""
    # item_key -> {evaluator_id: score}
    item_scores = defaultdict(dict)
    for r in records:
        key = (r["film"], r["trans_model"], r["item_id"], r["method"], r["run"])
        item_scores[key][r["evaluator_id"]] = r["score"]

    overlapping = {k: v for k, v in item_scores.items() if len(v) >= 2}
    num_shared = len(overlapping)

    if num_shared == 0:
        return {
            "num_shared_items": 0,
            "note": "No items evaluated by more than one annotator.",
        }

    evaluator_ids = sorted({eid for v in overlapping.values() for eid in v})

    pairwise = []
    for i, e1 in enumerate(evaluator_ids):
        for e2 in evaluator_ids[i + 1:]:
            shared = [k for k, v in overlapping.items() if e1 in v and e2 in v]
            if len(shared) < 2:
                continue
            xs = [overlapping[k][e1] for k in shared]
            ys = [overlapping[k][e2] for k in shared]
            pairwise.append({
                "evaluator_1":      e1,
                "evaluator_2":      e2,
                "num_shared_items": len(shared),
                "pearson_r":        _pearson(xs, ys),
                "spearman_rho":     _spearman(xs, ys),
            })

    ratings_by_unit = [list(v.values()) for v in overlapping.values()]
    alpha = _krippendorff_alpha_interval(ratings_by_unit)

    result = {
        "num_shared_items":              num_shared,
        "krippendorff_alpha_interval":   alpha,
        "pairwise":                      pairwise,
    }
    if len(pairwise) == 1:
        result["pearson_r"] = pairwise[0]["pearson_r"]
        result["spearman_rho"] = pairwise[0]["spearman_rho"]

    return result


# ─────────────────────────────────────────────
# WITHIN-ANNOTATOR RELIABILITY (is_repeat items)
# ─────────────────────────────────────────────

def compute_within_annotator_reliability(all_records):
    """
    For items where the same evaluator judged the same translation twice
    (is_repeat=True), compute first-vs-repeat score correlations.
    """
    first = defaultdict(dict)   # evaluator_id -> item_key -> score
    repeat = defaultdict(dict)

    for r in all_records:
        if r["auto_filled"]:
            continue
        key = (r["film"], r["trans_model"], r["item_id"], r["method"], r["run"])
        eid = r["evaluator_id"]
        target = repeat if r["is_repeat"] else first
        target[eid][key] = r["score"]

    results = []
    for eid in sorted(set(list(first.keys()) + list(repeat.keys()))):
        shared = [k for k in first[eid] if k in repeat[eid]]
        if len(shared) < 2:
            continue
        xs = [first[eid][k] for k in shared]
        ys = [repeat[eid][k] for k in shared]
        results.append({
            "evaluator_id":       eid,
            "num_repeated_items": len(shared),
            "pearson_r":          _pearson(xs, ys),
            "spearman_rho":       _spearman(xs, ys),
        })
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute MQM scores and IAA from human evaluation files."
    )
    parser.add_argument("eval_dir", help="Directory containing human_*.json files")
    parser.add_argument(
        "--films", nargs="+", metavar="FILM",
        help="Only include records for these film(s); all others are excluded",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"Error: not a directory: {eval_dir}")
        sys.exit(1)

    film_filter = set(args.films) if args.films else None

    human_files = sorted(eval_dir.glob("human_*.json"))
    if not human_files:
        print(f"No human_*.json files found in {eval_dir}")
        sys.exit(1)

    print(f"Found {len(human_files)} human evaluation file(s):")
    all_records = []
    evaluator_meta_map = {}
    for path in human_files:
        records = load_human_file(path)
        if film_filter is not None:
            records = [r for r in records if r["film"] in film_filter]
        all_records.extend(records)
        if records:
            eid = records[0]["evaluator_id"]
            evaluator_meta_map[eid] = records[0]["evaluator_meta"]
        suffix = f" (filtered to: {', '.join(sorted(film_filter))})" if film_filter else ""
        print(f"  {path.name}: {len(records)} judgments{suffix}")

    total = len(all_records)
    auto_filled_n = sum(1 for r in all_records if r["auto_filled"])
    repeat_n = sum(1 for r in all_records if r["is_repeat"] and not r["auto_filled"])
    usable_n = sum(1 for r in all_records if not r["auto_filled"] and not r["is_repeat"])
    evaluator_ids = sorted(evaluator_meta_map.keys())

    print(f"\nEvaluators: {evaluator_ids}")
    print(f"Judgments — total: {total}, auto-filled: {auto_filled_n}, repeats: {repeat_n}, usable: {usable_n}")

    # Build method_data in the format aggregate_mqm expects
    method_data = records_to_method_data(all_records)

    # Per-method stats (reusing aggregate_mqm logic)
    _NOISE_FIELDS = {"avg_eval_noise", "pooled_eval_run_sd",
                     "eval_runs_per_translation", "observed_eval_runs_per_translation"}
    method_stats = {}
    for method, trans_data in sorted(method_data.items()):
        method_stats[method] = compute_method_stats(trans_data, method_name=method)

    overall = compute_overall_across_methods(method_stats)

    dataset_names = sorted({r["film"] for r in all_records})
    dataset_name = dataset_names[0] if len(dataset_names) == 1 else dataset_names

    ranking = build_ranking(method_stats, dataset_name)

    # Remove auto-eval-specific noise fields after ranking is built (build_ranking
    # reads them). With human eval, eval-noise metrics are 0 and misleading.
    for stats in method_stats.values():
        for field in _NOISE_FIELDS:
            stats.pop(field, None)
        for t_stats in stats.get("per_translation", {}).values():
            for field in _NOISE_FIELDS:
                t_stats.pop(field, None)

    # Usable records for IAA
    usable_records = [r for r in all_records if not r["auto_filled"] and not r["is_repeat"]]
    iaa = compute_iaa(usable_records)
    within_reliability = compute_within_annotator_reliability(all_records)

    # Per-evaluator summary
    per_evaluator = {}
    for eid in evaluator_ids:
        scores = [r["score"] for r in usable_records if r["evaluator_id"] == eid]
        n = len(scores)
        m = statistics.mean(scores) if scores else None
        sd = statistics.stdev(scores) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 0 else 0.0
        hw = 1.96 * se
        per_evaluator[eid] = {
            "meta":                      evaluator_meta_map.get(eid, {}),
            "num_usable_judgments":      n,
            "mean_major_equiv_per_unit": m,
            "ci_95_lower":               (m - hw) if m is not None else None,
            "ci_95_upper":               (m + hw) if m is not None else None,
            "ci_95_half_width":          hw if m is not None else None,
        }

    summary = {
        "dataset_name":           dataset_name,
        "evaluator_type":         "human",
        "evaluators":             evaluator_ids,
        "num_evaluators":         len(evaluator_ids),
        "total_judgments":        total,
        "auto_filled_judgments":  auto_filled_n,
        "repeat_judgments":       repeat_n,
        "usable_judgments":       usable_n,
        "num_methods":            len(method_stats),
        "overall_across_methods": overall,
        "per_method":             method_stats,
        "ranking":                ranking,
        "per_evaluator":          per_evaluator,
        "interannotator_agreement":       iaa,
        "within_annotator_reliability":   within_reliability,
    }

    out_path = eval_dir / "summary_human.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    print("\nMethod ranking (lower = fewer errors):")
    print("  (± = 95% CI from run-to-run variability; same meaning as in auto eval)")
    for entry in ranking:
        m = entry["mean_major_equiv_per_unit"]
        hw = entry["ci_95_half_width"]
        n = entry["num_translations"]
        print(f"  {entry['method']:20s}  {m:.4f} ± {hw:.4f}  (n={n} runs)")

    if within_reliability:
        print("\nWithin-annotator reliability (first vs. repeat judgments):")
        for r in within_reliability:
            p = f"{r['pearson_r']:.3f}" if r["pearson_r"] is not None else "n/a"
            s = f"{r['spearman_rho']:.3f}" if r["spearman_rho"] is not None else "n/a"
            print(f"  {r['evaluator_id']}: n={r['num_repeated_items']} repeated items, "
                  f"Pearson r={p}, Spearman ρ={s}")
    else:
        print("\nWithin-annotator reliability: no repeated items found.")


if __name__ == "__main__":
    main()
