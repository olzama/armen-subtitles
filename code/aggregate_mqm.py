import os
import sys
import json
import math
import statistics
from collections import defaultdict


def load_major_equiv_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        return data["summary"]["major_equiv_per_unit"]
    except KeyError:
        raise ValueError(f"Missing major_equiv_per_unit in {path}")


def group_by_translation(folder):
    groups = defaultdict(list)

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".json"):
            continue

        # Expected format: translation.txt_eval_1.json
        base = fname.split("_eval_")[0]
        full_path = os.path.join(folder, fname)

        val = load_major_equiv_from_file(full_path)
        groups[base].append(val)

    return groups


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python aggregate_mqm_results.py <evaluation_folder>")
        sys.exit(1)

    eval_folder = sys.argv[1]

    groups = group_by_translation(eval_folder)

    if not groups:
        raise ValueError("No evaluation JSON files found.")

    translation_means = []
    all_runs = []

    for translation, runs in groups.items():
        if not runs:
            continue

        mean_val = statistics.mean(runs)
        translation_means.append(mean_val)
        all_runs.extend(runs)

    T = len(translation_means)
    E = len(all_runs) // T if T > 0 else 0

    overall_mean = statistics.mean(translation_means)

    between_translation_sd = (
        statistics.stdev(translation_means)
        if T > 1 else 0.0
    )

    se_method = (
        between_translation_sd / math.sqrt(T)
        if T > 1 else 0.0
    )

    ci_half_width = 1.96 * se_method

    ci_lower = overall_mean - ci_half_width
    ci_upper = overall_mean + ci_half_width

    run_sd = (
        statistics.stdev(all_runs)
        if len(all_runs) > 1 else 0.0
    )

    print("\n=== FINAL MQM RESULTS (MAJOR-EQUIV PER UNIT) ===")
    print(f"Translations (T): {T}")
    print(f"Evaluation runs per translation (E): {E}")

    print("\n--- Method-Level Result ---")
    print(f"Mean major-equiv per unit: {overall_mean:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"95% CI half-width: ±{ci_half_width:.4f}")
    print(f"Between-translation SD: {between_translation_sd:.4f}")

    print("\n--- Evaluation Noise (Diagnostic Only) ---")
    print(f"Run-level SD: {run_sd:.4f}")