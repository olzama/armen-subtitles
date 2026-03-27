import os
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
import sys


def load_eval(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        dataset = data.get("dataset_name", "")
        summary = data["summary"]
        usage = data.get("usage", {})

        return {
            "dataset": dataset,
            "method": data["method"],
            "translation": str(data["run"]),
            "eval_run": int(data["eval_run"]),
            "value": summary["major_equiv_per_unit"],
            "path": str(path),
            "evaluation_model": data.get("evaluator"),
            "evaluated_translation_model": data.get("translator"),
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
        }
    except KeyError as e:
        raise ValueError(f"{path} missing key: {e}")


def collect_runs_from_method_subfolders(parent_dir):
    """
    Traverse:
        parent_dir/
            method_1/
                *.json
            method_2/
                *.json
    and collect raw eval runs only.
    """
    parent = Path(parent_dir)
    if not parent.is_dir():
        raise ValueError(f"Not a directory: {parent_dir}")

    runs = {}
    skipped = []

    for method_dir in sorted(parent.iterdir()):
        if not method_dir.is_dir():
            continue

        for path in sorted(method_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".json":
                continue

            lower_name = path.name.lower()
            if "aggregated_summary" in lower_name or "method_comparison" in lower_name:
                continue

            try:
                r = load_eval(path)
            except Exception as e:
                skipped.append((str(path), str(e)))
                continue

            key = (r["method"], r["translation"], r["eval_run"])

            if key in runs:
                print(
                    f"WARNING: duplicate eval run for key {key}; "
                    f"keeping later file:\n"
                    f"  old: {runs[key]['path']}\n"
                    f"  new: {r['path']}"
                )

            runs[key] = r

    if not runs:
        raise ValueError("No valid raw evaluation JSON files found in method subfolders.")

    if skipped:
        print("\nSkipped files:")
        for path, err in skipped:
            print(f"  {path}: {err}")

    return runs


def structure_runs(runs):
    """
    method -> translation -> list of {"eval_run": int, "value": float}
    """
    data = defaultdict(lambda: defaultdict(list))

    for (_, _, _), record in runs.items():
        method = record["method"]
        translation = record["translation"]
        data[method][translation].append({
            "eval_run": record["eval_run"],
            "value": record["value"],
        })
        #print(f"Added record for method={method}, translation={translation}, eval_run={record['eval_run']}")

    for method in data:
        for translation in data[method]:
            data[method][translation].sort(key=lambda x: x["eval_run"])

    return data


def _compute_single_translation_stats(run_records):
    """Stats for one translation across its eval runs."""
    run_records = sorted(run_records, key=lambda x: x["eval_run"])
    eval_scores = [r["value"] for r in run_records]
    n = len(eval_scores)
    sd = statistics.stdev(eval_scores) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    return {
        "mean_major_equiv_per_unit": statistics.mean(eval_scores),
        "eval_run_sd": sd,
        "eval_noise_se": se,
        "ci_95_half_width": 1.96 * se,
        "n_eval_runs": n,
        "eval_scores": eval_scores,
        "eval_run_ids": [r["eval_run"] for r in run_records],
    }


def compute_method_stats(method_data, method_name="?"):
    """
    method_data: translation -> list of {"eval_run": int, "value": float}

    Output schema matches the reference aggregated_summary.json per-method block.
    """
    per_translation = {}
    translation_means = []
    all_runs = []
    method_eval_run_ids = set()
    observed_eval_runs_per_translation = []

    for translation, run_records in sorted(
        method_data.items(),
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])
    ):
        if not run_records:
            continue

        t_stats = _compute_single_translation_stats(run_records)
        eval_run_ids = t_stats.pop("eval_run_ids")
        per_translation[str(translation)] = t_stats

        translation_means.append(t_stats["mean_major_equiv_per_unit"])
        all_runs.extend(t_stats["eval_scores"])
        method_eval_run_ids.update(eval_run_ids)
        observed_eval_runs_per_translation.append(t_stats["n_eval_runs"])

    num_translations = len(translation_means)
    if num_translations == 0:
        raise ValueError("Method has no usable translations.")
    print(f"Computed stats for method '{method_name}' with {num_translations} translations "
          f"and eval run IDs: {sorted(method_eval_run_ids)}")

    # Number of distinct eval-run IDs observed for this method.
    eval_runs_per_translation = len(method_eval_run_ids)
    mean_major_equiv_per_unit = statistics.mean(translation_means)
    translation_sd = statistics.stdev(translation_means) if num_translations > 1 else 0.0
    se_method = translation_sd / math.sqrt(num_translations) if num_translations > 1 else 0.0
    ci_95_half_width = 1.96 * se_method
    pooled_eval_run_sd = statistics.stdev(all_runs) if len(all_runs) > 1 else 0.0
    avg_eval_noise = (
        pooled_eval_run_sd / math.sqrt(eval_runs_per_translation)
        if eval_runs_per_translation > 0 else 0.0
    )

    return {
        "num_translations": num_translations,
        "eval_runs_per_translation": eval_runs_per_translation,
        "observed_eval_runs_per_translation": sorted(set(observed_eval_runs_per_translation)),
        "mean_major_equiv_per_unit": mean_major_equiv_per_unit,
        "translation_sd": translation_sd,
        "se_method": se_method,
        "ci_95_half_width": ci_95_half_width,
        "ci_95_lower": mean_major_equiv_per_unit - ci_95_half_width,
        "ci_95_upper": mean_major_equiv_per_unit + ci_95_half_width,
        "pooled_eval_run_sd": pooled_eval_run_sd,
        "avg_eval_noise": avg_eval_noise,
        "translation_means": translation_means,
        "per_translation": per_translation,
    }


def compute_overall_across_methods(method_stats):
    method_means = [
        stats["mean_major_equiv_per_unit"]
        for _, stats in sorted(method_stats.items())
    ]

    num_methods = len(method_means)
    if num_methods == 0:
        raise ValueError("No methods available for across-method summary.")

    mean_major_equiv_per_unit = statistics.mean(method_means)
    method_sd = statistics.stdev(method_means) if num_methods > 1 else 0.0
    se_method_across_methods = (
        method_sd / math.sqrt(num_methods)
        if num_methods > 1 else 0.0
    )
    ci_95_half_width = 1.96 * se_method_across_methods

    return {
        "mean_major_equiv_per_unit": mean_major_equiv_per_unit,
        "ci_95_lower": mean_major_equiv_per_unit - ci_95_half_width,
        "ci_95_upper": mean_major_equiv_per_unit + ci_95_half_width,
        "ci_95_half_width": ci_95_half_width,
        "method_sd": method_sd,
        "se_method_across_methods": se_method_across_methods,
    }


def compute_method_comparison(method_stats, dataset_name):
    """
    Output matches method_comparison.json exactly:
    {
      "methods": [
        {...}, ...
      ]
    }
    """
    methods = []

    for method, stats in sorted(
        method_stats.items(),
        key=lambda kv: kv[1]["mean_major_equiv_per_unit"]
    ):
        methods.append({
            "method": method,
            "num_translations": stats["num_translations"],
            "eval_runs_per_translation": stats["eval_runs_per_translation"],
            "mean_major_equiv_per_unit": stats["mean_major_equiv_per_unit"],
            "ci_95_half_width": stats["ci_95_half_width"],
            "translation_sd": stats["translation_sd"],
            "pooled_eval_run_sd": stats["pooled_eval_run_sd"],
            "avg_eval_noise": stats["avg_eval_noise"],
        })

    return {"dataset_name": dataset_name, "methods": methods}


def build_ranking(method_stats, dataset_name):
    ranked = sorted(
        method_stats.items(),
        key=lambda kv: kv[1]["mean_major_equiv_per_unit"]
    )

    return [
        {
            "dataset": dataset_name,
            "method": method,
            "num_translations": stats["num_translations"],
            "eval_runs_per_translation": stats["eval_runs_per_translation"],
            "mean_major_equiv_per_unit": stats["mean_major_equiv_per_unit"],
            "ci_95_half_width": stats["ci_95_half_width"],
            "translation_sd": stats["translation_sd"],
            "pooled_eval_run_sd": stats["pooled_eval_run_sd"],
            "avg_eval_noise": stats["avg_eval_noise"],
        }
        for method, stats in ranked
    ]


def summarize_backend(runs):
    evaluation_models = sorted({
        r["evaluation_model"] for r in runs.values()
        if r.get("evaluation_model")
    })
    evaluated_translation_models = sorted({
        r["evaluated_translation_model"] for r in runs.values()
        if r.get("evaluated_translation_model")
    })

    return {
        "evaluation_model": evaluation_models[0] if len(evaluation_models) == 1 else (
            evaluation_models if evaluation_models else None
        ),
        "evaluated_translation_model": evaluated_translation_models[0] if len(evaluated_translation_models) == 1 else (
            evaluated_translation_models if evaluated_translation_models else None
        ),
    }


def summarize_usage(runs):
    total_input_tokens = sum(r.get("input_tokens", 0) for r in runs.values())
    total_output_tokens = sum(r.get("output_tokens", 0) for r in runs.values())
    total_cost_usd = sum(r.get("cost_usd", 0.0) for r in runs.values())

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": total_cost_usd,
    }


def infer_requested_eval_runs_per_translation(method_stats):
    observed = set()
    for stats in method_stats.values():
        observed.add(stats["eval_runs_per_translation"])

    if len(observed) == 1:
        return next(iter(observed))
    return max(observed)


def main():
    if len(sys.argv) != 3:
        print("Usage: python aggregate_mqm.py <film_name> <eval_model>")
        print("  parent eval dir: output/films/<film_name>/eval/<eval_model>/")
        sys.exit(1)

    film_name = sys.argv[1]
    eval_model = sys.argv[2]
    parent_eval_dir = str(Path("output/films") / film_name / "eval" / eval_model)
    dataset_name = film_name

    runs = collect_runs_from_method_subfolders(parent_eval_dir)
    structured = structure_runs(runs)

    method_stats = {}
    for method, method_data in sorted(structured.items()):
        method_stats[method] = compute_method_stats(method_data, method_name=method)

    aggregated_summary = {
        "dataset_name": dataset_name,
        "backend": summarize_backend(runs),
        "requested_eval_runs_per_translation": infer_requested_eval_runs_per_translation(method_stats),
        "num_methods": len(method_stats),
        "total_successful_eval_runs": len(runs),
        "overall_across_methods": compute_overall_across_methods(method_stats),
        "per_method": method_stats,
        "ranking": build_ranking(method_stats, dataset_name),
        "usage": summarize_usage(runs),
    }

    aggregated_out_path = Path(parent_eval_dir) / "merged_summary.json"
    with open(aggregated_out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_summary, f, indent=2, ensure_ascii=False)

    method_comparison = compute_method_comparison(method_stats, dataset_name)
    comparison_out_path = Path(parent_eval_dir) / "method_comparison.json"
    with open(comparison_out_path, "w", encoding="utf-8") as f:
        json.dump(method_comparison, f, indent=2, ensure_ascii=False)

    print(f"Saved aggregated summary to: {aggregated_out_path}")
    print(f"Saved method comparison to: {comparison_out_path}")


if __name__ == "__main__":
    main()