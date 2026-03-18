#!/usr/bin/env python3

import sys
import json
import math
from pathlib import Path


Z_95 = 1.96


def sample_sd(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_eval_runs_per_translation(method_info):
    if "eval_runs_per_translation" in method_info:
        return float(method_info["eval_runs_per_translation"])

    observed = method_info.get("observed_eval_runs_per_translation", [])
    if observed:
        return float(sum(observed) / len(observed))

    per_translation = method_info.get("per_translation", {})
    if per_translation:
        counts = []
        for _, tinfo in per_translation.items():
            if "n_eval_runs" in tinfo:
                counts.append(float(tinfo["n_eval_runs"]))
            elif "eval_scores" in tinfo:
                counts.append(float(len(tinfo["eval_scores"])))
        if counts:
            return sum(counts) / len(counts)

    return None


def get_translation_sd(method_info):
    if "translation_sd" in method_info:
        return float(method_info["translation_sd"])

    means = method_info.get("translation_means")
    if means is not None:
        return sample_sd([float(x) for x in means])

    per_translation = method_info.get("per_translation", {})
    if per_translation:
        vals = []
        for _, tinfo in sorted(per_translation.items(), key=lambda kv: str(kv[0])):
            if "mean_major_equiv_per_unit" in tinfo:
                vals.append(float(tinfo["mean_major_equiv_per_unit"]))
        if vals:
            return sample_sd(vals)

    return None


def get_pooled_eval_run_sd(method_info):
    if "pooled_eval_run_sd" in method_info:
        return float(method_info["pooled_eval_run_sd"])

    per_translation = method_info.get("per_translation", {})
    if per_translation:
        within_vars = []
        total_df = 0
        for _, tinfo in per_translation.items():
            scores = tinfo.get("eval_scores")
            if scores is None:
                continue
            scores = [float(x) for x in scores]
            n = len(scores)
            if n < 2:
                continue
            sd = sample_sd(scores)
            within_vars.append((n - 1) * (sd ** 2))
            total_df += (n - 1)

        if total_df > 0:
            return math.sqrt(sum(within_vars) / total_df)

    return None


def get_num_translations(method_info):
    if "num_translations" in method_info:
        return int(method_info["num_translations"])

    means = method_info.get("translation_means")
    if means is not None:
        return len(means)

    per_translation = method_info.get("per_translation", {})
    if per_translation:
        return len(per_translation)

    return None


def ci_halfwidth_for_diff(var_translation, var_evaluation, T, E):
    """
    95% CI half-width for a difference between two method means:
        1.96 * sqrt(2 * (var_translation / T + var_evaluation / (T * E)))
    """
    if T <= 0 or E <= 0:
        return None
    var_method = (var_translation / T) + (var_evaluation / (T * E))
    return Z_95 * math.sqrt(2.0 * var_method)


def required_E_for_delta(var_translation, var_evaluation, T, delta):
    """
    Solve:
        1.96 * sqrt(2 * (var_translation/T + var_evaluation/(T*E))) <= delta
    for E, keeping T fixed.
    """
    if T <= 0 or delta <= 0:
        return None, "invalid"

    target_var = (delta / Z_95) ** 2 / 2.0
    fixed_term = var_translation / T

    if target_var <= fixed_term:
        return None, "increase_T"

    req_E = var_evaluation / (T * (target_var - fixed_term))
    return max(1, math.ceil(req_E)), "ok"


def required_T_for_delta(var_translation, var_evaluation, E, delta):
    """
    Solve:
        1.96 * sqrt(2 * (var_translation/T + var_evaluation/(T*E))) <= delta
    for T, keeping E fixed.

    Since:
        var_translation/T + var_evaluation/(T*E)
      = (var_translation + var_evaluation/E) / T
    """
    if E <= 0 or delta <= 0:
        return None

    target_var = (delta / Z_95) ** 2 / 2.0
    req_T = (var_translation + (var_evaluation / E)) / target_var
    return max(1, math.ceil(req_T))


def classify_priority(var_translation, var_evaluation, T, E, ratio_threshold=1.5):
    """
    Compare current contributions to method-mean variance:
        translation component = var_translation / T
        evaluation component  = var_evaluation / (T * E)
    """
    trans_component = var_translation / T if T > 0 else float("inf")
    eval_component = var_evaluation / (T * E) if T > 0 and E > 0 else float("inf")

    if eval_component == 0 and trans_component == 0:
        return "stable", trans_component, eval_component

    if trans_component > ratio_threshold * eval_component:
        return "increase_T", trans_component, eval_component

    if eval_component > ratio_threshold * trans_component:
        return "increase_E", trans_component, eval_component

    return "increase_both", trans_component, eval_component


def marginal_gain_in_ci(var_translation, var_evaluation, T, E):
    current = ci_halfwidth_for_diff(var_translation, var_evaluation, T, E)
    next_T = ci_halfwidth_for_diff(var_translation, var_evaluation, T + 1, E)
    next_E = ci_halfwidth_for_diff(var_translation, var_evaluation, T, E + 1)

    gain_T = current - next_T if current is not None and next_T is not None else None
    gain_E = current - next_E if current is not None and next_E is not None else None

    return current, gain_T, gain_E


def extract_method_stats(method_name, method_info):
    T = get_num_translations(method_info)
    E = get_eval_runs_per_translation(method_info)
    translation_sd = get_translation_sd(method_info)
    pooled_eval_run_sd = get_pooled_eval_run_sd(method_info)

    missing = []
    if T is None:
        missing.append("num_translations / translation_means / per_translation")
    if E is None:
        missing.append("eval_runs_per_translation / observed_eval_runs_per_translation / per_translation")
    if translation_sd is None:
        missing.append("translation_sd / translation_means / per_translation")
    if pooled_eval_run_sd is None:
        missing.append("pooled_eval_run_sd / per_translation.eval_scores")

    if missing:
        raise ValueError(
            f"Method '{method_name}' is missing required information for variance decomposition: "
            + ", ".join(missing)
        )

    var_translation = translation_sd ** 2
    var_evaluation = pooled_eval_run_sd ** 2

    return {
        "method": method_name,
        "T": int(T),
        "E": float(E),
        "translation_sd": translation_sd,
        "pooled_eval_run_sd": pooled_eval_run_sd,
        "var_translation": var_translation,
        "var_evaluation": var_evaluation,
    }


def print_method_report(stats, delta):
    method = stats["method"]
    T = stats["T"]
    E = stats["E"]
    var_translation = stats["var_translation"]
    var_evaluation = stats["var_evaluation"]

    priority, trans_component, eval_component = classify_priority(
        var_translation, var_evaluation, T, E
    )
    current_ci, gain_T, gain_E = marginal_gain_in_ci(var_translation, var_evaluation, T, E)
    req_E, req_E_status = required_E_for_delta(var_translation, var_evaluation, T, delta)
    req_T = required_T_for_delta(var_translation, var_evaluation, E, delta)

    print(f"\nMETHOD: {method}")
    print(f"  T = {T}")
    if abs(E - round(E)) < 1e-9:
        print(f"  E = {int(round(E))}")
    else:
        print(f"  E = {E:.3f}")

    print(f"  translation_sd = {stats['translation_sd']:.6f}")
    print(f"  pooled_eval_run_sd = {stats['pooled_eval_run_sd']:.6f}")
    print(f"  translation variance component per method mean = {trans_component:.6f}")
    print(f"  evaluation variance component per method mean  = {eval_component:.6f}")

    if current_ci is not None:
        print(f"  current sensitivity (95% CI half-width for method difference) = {current_ci:.6f}")

    if gain_T is not None:
        print(f"  one more translation (T -> {T + 1}) would reduce sensitivity half-width by {gain_T:.6f}")
    if gain_E is not None:
        next_E_str = f"{int(round(E)) + 1}" if abs(E - round(E)) < 1e-9 else f"{E + 1:.3f}"
        print(f"  one more eval run    (E -> {next_E_str}) would reduce sensitivity half-width by {gain_E:.6f}")

    print(f"  target sensitivity = {delta:.6f}")

    if current_ci is not None and current_ci <= delta:
        print("  conclusion: current setup already meets target; no increase needed")
        return

    if req_E_status == "increase_T":
        print("  keeping T fixed: impossible to reach target by increasing E alone")
    elif req_E_status == "ok":
        print(f"  required E at current T = {req_E}")

    if req_T is not None:
        print(f"  required T at current E = {req_T}")

    if priority == "increase_T":
        print("  conclusion: current setup does not meet target; increase T first")
    elif priority == "increase_E":
        print("  conclusion: current setup does not meet target; increase E first")
    elif priority == "increase_both":
        if gain_T is not None and gain_E is not None:
            if gain_T > gain_E:
                print("  conclusion: current setup does not meet target; both matter, but T helps more")
            elif gain_E > gain_T:
                print("  conclusion: current setup does not meet target; both matter, but E helps more")
            else:
                print("  conclusion: current setup does not meet target; T and E help similarly")
        else:
            print("  conclusion: current setup does not meet target; increase both")
    else:
        print("  conclusion: current setup looks stable")

def main():
    if len(sys.argv) < 3:
        print("Usage: python variance.py <aggregated_summary.json> <delta>")
        sys.exit(1)

    summary_path = Path(sys.argv[1])
    delta = float(sys.argv[2])

    data = load_summary(summary_path)

    if "per_method" not in data or not isinstance(data["per_method"], dict):
        raise ValueError("Input JSON must contain a top-level 'per_method' object.")

    print(f"Loaded summary: {summary_path}")
    backend = data.get("backend", {})
    if backend:
        eval_model = backend.get("evaluation_model", "unknown")
        trans_model = backend.get("evaluated_translation_model", "unknown")
        print(f"  evaluation model: {eval_model}")
        print(f"  evaluated translation model: {trans_model}")

    print(f"  methods: {len(data['per_method'])}")
    print(f"  target detectable delta: {delta:.6f}")

    collected = []
    for method_name, method_info in data["per_method"].items():
        stats = extract_method_stats(method_name, method_info)
        collected.append(stats)

    for stats in collected:
        print_method_report(stats, delta)

    print("\nSUMMARY TABLE")
    print("method\tpriority\tT\tE\ttranslation_sd\tpooled_eval_run_sd\tcurrent_ci_halfwidth")
    for stats in collected:
        priority, _, _ = classify_priority(
            stats["var_translation"],
            stats["var_evaluation"],
            stats["T"],
            stats["E"],
        )
        current_ci, _, _ = marginal_gain_in_ci(
            stats["var_translation"],
            stats["var_evaluation"],
            stats["T"],
            stats["E"],
        )
        print(
            f"{stats['method']}\t{priority}\t{stats['T']}\t"
            f"{int(round(stats['E'])) if abs(stats['E'] - round(stats['E'])) < 1e-9 else f'{stats['E']:.3f}'}\t"
            f"{stats['translation_sd']:.6f}\t"
            f"{stats['pooled_eval_run_sd']:.6f}\t"
            f"{current_ci:.6f}"
        )


if __name__ == "__main__":
    main()