#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from google import genai
from google.genai import types as gtypes


# =========================
# COST RATES (Current 2026)
# =========================
RATES = {
    "gpt-5.2": {"input": 1.75 / 1_000_000, "output": 14.00 / 1_000_000},
    "gpt-5-mini": {"input": 0.25 / 1_000_000, "output": 2.00 / 1_000_000},
    "gemini-2.5-flash": {"input": 0.30 / 1_000_000, "output": 2.5 / 1_000_000},
    "gemini-3-flash-preview": {"input": 0.50 / 1_000_000, "output": 3.0 / 1_000_000},
    "gemini-3-flash-lite-preview": {"input": 0.25 / 1_000_000, "output": 1.5 / 1_000_000},
}

Z_95 = 1.96


# =========================
# MQM SCORING LOGIC
# =========================

def compute_mqm_score(mqm_json, approved_json=None):
    """Calculate MQM penalty points and normalized statistics."""
    severity_weights = {
        "critical": 10,
        "major": 5,
        "minor": 1,
    }

    counts = {s: 0 for s in severity_weights}

    for item in mqm_json.get("items", []):
        for issue in item.get("issues", []):
            severity = issue.get("severity", "").lower()
            if severity in counts and issue.get("category") != "no-issue":
                counts[severity] += 1

    total_points = sum(counts[s] * severity_weights[s] for s in counts)

    reference_json = approved_json if isinstance(approved_json, dict) and "items" in approved_json else mqm_json
    num_units = len(reference_json.get("items", [])) if isinstance(reference_json, dict) else 0

    penalty_per_unit = 0.0
    major_equiv_per_unit = 0.0
    interpretation = "No units found"

    if num_units > 0:
        penalty_per_unit = total_points / num_units
        major_equiv_per_unit = penalty_per_unit / severity_weights["major"]
        interpretation = (
            f"{penalty_per_unit:.2f} MQM penalty points per meaning unit "
            f"(≈ {major_equiv_per_unit:.2f} major issues per unit; "
            f"weights: minor=1, major=5, critical=10)"
        )

    return {
        "counts": counts,
        "total_points": total_points,
        "meaning_units": num_units,
        "penalty_per_unit": penalty_per_unit,
        "major_equiv_per_unit": major_equiv_per_unit,
        "interpretation": interpretation,
    }


# =========================
# API WRAPPERS
# =========================

def call_gpt_mqm(content, client, model_name):
    response = client.chat.completions.create(
        model=model_name,
        temperature=0 if model_name == "gpt-5.2" else 1.0,
        top_p=1,
        messages=[
            {"role": "system", "content": "Expert in literary translation quality assessment using MQM framework."},
            {"role": "user", "content": content},
        ],
    )
    raw_text = response.choices[0].message.content.strip()
    usage = response.usage
    in_tokens = usage.prompt_tokens
    out_tokens = usage.completion_tokens
    cost = (in_tokens * RATES[model_name]["input"]) + (out_tokens * RATES[model_name]["output"])
    return raw_text, in_tokens, out_tokens, cost


def call_gemini_mqm(content, client, model_name):
    response = client.models.generate_content(
        model=model_name,
        config=gtypes.GenerateContentConfig(
            system_instruction="Expert in literary translation quality assessment using MQM framework.",
            temperature=0.0,
        ),
        contents=content,
    )
    raw_text = response.text.strip()
    usage = response.usage_metadata
    in_tokens = usage.prompt_token_count
    out_tokens = usage.candidates_token_count + (usage.thoughts_token_count or 0)
    cost = (in_tokens * RATES[model_name]["input"]) + (out_tokens * RATES[model_name]["output"])
    return raw_text, in_tokens, out_tokens, cost


# =========================
# JSON REPAIR / KEYS
# =========================

def load_openai_key():
    for candidate in ("./GreenAI-API", "./GreenAI-API-key.txt"):
        p = Path(candidate)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Could not find OpenAI API key file: ./GreenAI-API or ./GreenAI-API-key.txt")


def load_gemini_key():
    for candidate in ("./gemini-personal-API-key.txt", "./gemini-API-key.txt"):
        p = Path(candidate)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Could not find Gemini API key file.")


def fix_invalid_json(raw_str, fixer_client):
    prompt = (
        "The following string is supposed to be a JSON object but may have formatting issues. "
        "Please fix it so that it can be parsed as valid JSON. Return the valid JSON ONLY.\n\n"
        f"{raw_str}"
    )
    response = fixer_client.chat.completions.create(
        model="gpt-5.2",
        temperature=0,
        messages=[
            {"role": "system", "content": "Expert in correcting JSON formatting issues."},
            {"role": "user", "content": prompt},
        ],
    )
    fixed_str = response.choices[0].message.content.strip()
    in_t = response.usage.prompt_tokens
    out_t = response.usage.completion_tokens
    cost = (in_t * RATES["gpt-5.2"]["input"]) + (out_t * RATES["gpt-5.2"]["output"])
    return fixed_str, in_t, out_t, cost


# =========================
# INPUT / TASK PREPARATION
# =========================

def load_enriched_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "items" not in data or not isinstance(data["items"], list):
        raise ValueError("Input JSON must be an object with a top-level 'items' list.")
    return data


def collect_shared_items(data):
    shared_items = []
    for item in data["items"]:
        item_id = item.get("id")
        character = item.get("character")
        original = item.get("original", {})
        reference = item.get("reference", {})
        analysis = item.get("analysis")

        if item_id is None:
            raise ValueError("Every item must have an 'id'.")
        if not isinstance(original, dict) or "rus" not in original:
            raise ValueError(f"Item {item_id} is missing original.rus.")
        if not isinstance(reference, dict) or "eng" not in reference:
            raise ValueError(f"Item {item_id} is missing reference.eng.")
        if analysis is None:
            raise ValueError(f"Item {item_id} is missing analysis.")
        if "translations" not in item or "eng" not in item["translations"]:
            raise ValueError(f"Item {item_id} is missing translations.eng.")

        shared_items.append({
            "id": item_id,
            "character": character,
            "source": original["rus"],
            "reference": reference["eng"],
            "analysis": analysis,
            "translations_eng": item["translations"]["eng"],
        })
    return shared_items


def parse_csv_filter(raw_value):
    if raw_value is None:
        return None
    values = [x.strip() for x in raw_value.split(",") if x.strip()]
    return set(values) if values else None


def discover_method_runs(shared_items):
    methods = {}
    for item in shared_items:
        eng = item["translations_eng"]
        if not isinstance(eng, dict):
            raise ValueError(f"Item {item['id']}: translations.eng must be an object.")
        for method_name, runs in eng.items():
            if not isinstance(runs, dict):
                raise ValueError(f"Item {item['id']}: translations.eng.{method_name} must be an object.")
            methods.setdefault(method_name, set())
            for run_id in runs.keys():
                methods[method_name].add(str(run_id))
    return methods


def build_translation_tasks(shared_items, method_filter=None, run_filter=None):
    discovered = discover_method_runs(shared_items)
    tasks = []

    for method_name in sorted(discovered.keys()):
        if method_filter is not None and method_name not in method_filter:
            continue

        for run_id in sorted(discovered[method_name], key=lambda x: (len(x), x)):
            if run_filter is not None and str(run_id) not in run_filter:
                continue

            payload_items = []
            missing_for_this_task = []

            for item in shared_items:
                runs = item["translations_eng"].get(method_name, {})
                if str(run_id) not in runs:
                    missing_for_this_task.append(item["id"])
                    continue

                candidate = runs[str(run_id)]
                if not isinstance(candidate, str):
                    raise ValueError(
                        f"Item {item['id']}: translations.eng.{method_name}.{run_id} must be a string."
                    )

                payload_items.append({
                    "id": item["id"],
                    "character": item["character"],
                    "source": item["source"],
                    "reference": item["reference"],
                    "analysis": item["analysis"],
                    "candidate": candidate,
                })

            if missing_for_this_task:
                raise ValueError(
                    f"Method '{method_name}' run '{run_id}' is missing candidate translations "
                    f"for item IDs: {missing_for_this_task}"
                )

            tasks.append({
                "method": method_name,
                "run": str(run_id),
                "items": payload_items,
            })

    return tasks


# =========================
# OUTPUT PATHS
# =========================

def method_dir(out_dir: Path, method_name: str) -> Path:
    d = out_dir / method_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def eval_output_path(out_dir: Path, method_name: str, run_id: str, eval_index: int) -> Path:
    return method_dir(out_dir, method_name) / f"run_{run_id}_eval_{eval_index}.json"


def next_eval_index(out_dir: Path, method_name: str, run_id: str) -> int:
    d = method_dir(out_dir, method_name)
    pattern = re.compile(rf"^run_{re.escape(str(run_id))}_eval_(\d+)\.json$")
    existing = []
    for fname in os.listdir(d):
        match = pattern.match(fname)
        if match:
            existing.append(int(match.group(1)))
    return max(existing, default=0) + 1


# =========================
# OUTPUT NORMALIZATION
# =========================

def normalize_issue(issue):
    if not isinstance(issue, dict):
        return {
            "severity": "",
            "category": "",
            "span": "",
            "justification": str(issue),
        }
    return {
        "severity": issue.get("severity", ""),
        "category": issue.get("category", ""),
        "span": issue.get("span", ""),
        "justification": issue.get("justification", ""),
    }


def normalize_mqm_items(model_json, payload_items):
    returned_items = model_json.get("items", [])
    if not isinstance(returned_items, list):
        returned_items = []

    returned_by_id = {}
    for returned_item in returned_items:
        if isinstance(returned_item, dict) and "id" in returned_item:
            returned_by_id[returned_item["id"]] = returned_item

    normalized_items = []
    matched_by_id = len(returned_by_id) > 0

    for idx, payload_item in enumerate(payload_items):
        returned_item = None

        if matched_by_id:
            returned_item = returned_by_id.get(payload_item["id"])
        elif idx < len(returned_items) and isinstance(returned_items[idx], dict):
            returned_item = returned_items[idx]

        issues = []
        if returned_item is not None and isinstance(returned_item.get("issues", []), list):
            issues = [normalize_issue(issue) for issue in returned_item.get("issues", [])]

        normalized_items.append({
            "id": payload_item["id"],
            "character": payload_item["character"],
            "source": payload_item["source"],
            "reference": payload_item["reference"],
            "analysis": payload_item["analysis"],
            "candidate": payload_item["candidate"],
            "issues": issues,
        })

    return normalized_items


# =========================
# MQM ORCHESTRATOR
# =========================

def mqm_evaluation(translation_task, client, fixer_client, eval_model, translation_model,
                   prompt, output_filename=None, eval_index=None):
    success = False

    full_content = prompt
    full_content += "\n\nDATA TO EVALUATE:\n"
    full_content += json.dumps(
        {
            "method": translation_task["method"],
            "run": translation_task["run"],
            "items": translation_task["items"],
        },
        ensure_ascii=False,
        indent=2,
    )

    if eval_model.startswith("gpt"):
        raw_output, in_t, out_t, cost = call_gpt_mqm(full_content, client, eval_model)
    else:
        raw_output, in_t, out_t, cost = call_gemini_mqm(full_content, client, eval_model)

    clean_json_str = raw_output.strip()

    try:
        mqm_json = json.loads(clean_json_str)
        success = True
    except json.JSONDecodeError:
        print(f"Model {eval_model} returned output that could not be parsed as JSON. Attempting to fix it.\n")
        try:
            fixed_string, fix_in_t, fix_out_t, fix_cost = fix_invalid_json(clean_json_str, fixer_client)
            print(f"Fixed JSON string from {eval_model}:\n{fixed_string}\n\nAttempting to parse fixed string.")
            mqm_json = json.loads(fixed_string)
            success = True
            print(f"Successfully repaired JSON output from {eval_model} after initial parsing failure.")
            in_t += fix_in_t
            out_t += fix_out_t
            cost += fix_cost
        except Exception:
            print(
                f"Model {eval_model} failed to return valid JSON, "
                f"and repair was unsuccessful.\n"
            )

    if success:
        normalized_items = normalize_mqm_items(mqm_json, translation_task["items"])

        final_json = {
            "method": translation_task["method"],
            "run": translation_task["run"],
            "eval_run": eval_index,
            "items": normalized_items,
        }

        score = compute_mqm_score(final_json, final_json)
        final_json["summary"] = score
        final_json["evaluator"] = eval_model
        final_json["translator"] = translation_model

        for k, v in mqm_json.items():
            if k not in {"items", "summary", "evaluator", "translator", "method", "run", "eval_run"}:
                final_json[k] = v

        if output_filename:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(final_json, f, ensure_ascii=False, indent=2)

        return {
            "score": score,
            "input_tokens": in_t,
            "output_tokens": out_t,
            "cost_total": cost,
            "method": translation_task["method"],
            "run": translation_task["run"],
            "eval_run": eval_index,
            "output_file": str(output_filename) if output_filename else None,
        }

    print("Failed to obtain valid MQM JSON output from the evaluator for this evaluation run. Returning None for score.")
    return {
        "score": None,
        "input_tokens": in_t,
        "output_tokens": out_t,
        "cost_total": cost,
        "method": translation_task["method"],
        "run": translation_task["run"],
        "eval_run": eval_index,
        "output_file": str(output_filename) if output_filename else None,
    }


# =========================
# STATS HELPERS
# =========================

def mean_or_zero(values):
    return statistics.mean(values) if values else 0.0


def stdev_or_zero(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def se_from_sd(sd, n):
    return sd / math.sqrt(n) if n > 1 else 0.0


def ci_half_width_from_se(se):
    return Z_95 * se


def pooled_sd_from_groups(groups):
    """Pooled within-group SD from lists of repeated scores."""
    numerator = 0.0
    denominator = 0
    for values in groups:
        if len(values) < 2:
            continue
        sd = statistics.stdev(values)
        numerator += (len(values) - 1) * (sd ** 2)
        denominator += (len(values) - 1)
    return math.sqrt(numerator / denominator) if denominator > 0 else 0.0


def summarize_translation(eval_scores):
    translation_mean = mean_or_zero(eval_scores)
    eval_run_sd = stdev_or_zero(eval_scores)
    eval_noise_se = se_from_sd(eval_run_sd, len(eval_scores))
    ci_95_half_width = ci_half_width_from_se(eval_noise_se)

    return {
        "mean_major_equiv_per_unit": translation_mean,
        "eval_run_sd": eval_run_sd,
        "eval_noise_se": eval_noise_se,
        "ci_95_half_width": ci_95_half_width,
        "n_eval_runs": len(eval_scores),
        "eval_scores": eval_scores,
    }


def summarize_method(per_translation_summary):
    translation_means = [
        stats["mean_major_equiv_per_unit"]
        for _, stats in sorted(per_translation_summary.items(), key=lambda kv: (len(str(kv[0])), str(kv[0])))
    ]

    eval_score_groups = [
        stats["eval_scores"]
        for _, stats in sorted(per_translation_summary.items(), key=lambda kv: (len(str(kv[0])), str(kv[0])))
    ]

    T = len(translation_means)
    mean_major_equiv_per_unit = mean_or_zero(translation_means)
    translation_sd = stdev_or_zero(translation_means)
    se_method = se_from_sd(translation_sd, T)
    ci_95_half_width = ci_half_width_from_se(se_method)
    pooled_eval_run_sd = pooled_sd_from_groups(eval_score_groups)

    actual_eval_runs = sorted({stats["n_eval_runs"] for stats in per_translation_summary.values()})
    if len(actual_eval_runs) == 1:
        common_E = actual_eval_runs[0]
        avg_eval_noise = pooled_eval_run_sd / math.sqrt(common_E) if common_E > 1 else 0.0
    else:
        common_E = None
        avg_eval_noise = mean_or_zero([
            stats["eval_noise_se"] for stats in per_translation_summary.values()
        ])

    return {
        "num_translations": T,
        "eval_runs_per_translation": common_E,
        "observed_eval_runs_per_translation": actual_eval_runs,
        "mean_major_equiv_per_unit": mean_major_equiv_per_unit,
        "translation_sd": translation_sd,
        "se_method": se_method,
        "ci_95_half_width": ci_95_half_width,
        "ci_95_lower": mean_major_equiv_per_unit - ci_95_half_width,
        "ci_95_upper": mean_major_equiv_per_unit + ci_95_half_width,
        "pooled_eval_run_sd": pooled_eval_run_sd,
        "avg_eval_noise": avg_eval_noise,
        "translation_means": translation_means,
    }


def summarize_overall(per_method_summary):
    method_means = [stats["mean_major_equiv_per_unit"] for stats in per_method_summary.values()]
    M = len(method_means)
    overall_mean = mean_or_zero(method_means)
    method_sd = stdev_or_zero(method_means)
    se_over_methods = se_from_sd(method_sd, M)
    ci_95_half_width = ci_half_width_from_se(se_over_methods)

    pooled_eval_groups = []
    for method_stats in per_method_summary.values():
        pooled_eval_groups.append(method_stats["pooled_eval_run_sd"])

    return {
        "num_methods": M,
        "mean_major_equiv_per_unit": overall_mean,
        "method_sd": method_sd,
        "se_method_across_methods": se_over_methods,
        "ci_95_half_width": ci_95_half_width,
        "ci_95_lower": overall_mean - ci_95_half_width,
        "ci_95_upper": overall_mean + ci_95_half_width,
        "method_means": method_means,
    }


def accumulate_usage(results):
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    for res in results:
        total_input_tokens += res["input_tokens"]
        total_output_tokens += res["output_tokens"]
        total_cost += res["cost_total"]

    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost": total_cost,
    }


# =========================
# RUNNER
# =========================

def run_single_evaluation(translation_task, client, fixer_client, eval_model, translation_model,
                          out_file, prompt, task_idx, total_tasks, eval_pos, total_eval_runs_for_task,
                          eval_index):
    print(
        f"  Task {task_idx}/{total_tasks} | eval {eval_pos}/{total_eval_runs_for_task}: "
        f"method={translation_task['method']} run={translation_task['run']} eval_run={eval_index} "
        f"-> {os.path.basename(out_file)}"
    )
    return mqm_evaluation(
        translation_task,
        client,
        fixer_client,
        eval_model,
        translation_model,
        prompt,
        out_file,
        eval_index,
    )


# =========================
# MAIN ENTRY POINT
# =========================

def update_existing_fname(f):
    base = f.stem
    suffix = f.suffix
    parent = f.parent
    counter = 1
    while True:
        new_name = f"{base}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            final_path = new_path
            break
        counter += 1
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate translations stored inside one enriched JSON using MQM with T translations and E evaluation runs."
    )

    parser.add_argument("input_json", type=Path, help="Path to the full JSON file containing items, translations, references, and analysis.")
    parser.add_argument("out_dir", type=Path, help="Directory to save evaluation results")
    parser.add_argument("prompt_file", type=Path, help="Path to the evaluation prompt file")
    parser.add_argument("eval_model", type=str, help="Evaluation model to use (e.g., 'gpt-5.2')")
    parser.add_argument("eval_runs", type=int, help="Number of independent evaluation runs per translation (E)")
    parser.add_argument("--methods", type=str, help="Comma-separated list of methods to evaluate")
    parser.add_argument("--runs", type=str, help="Comma-separated list of translation run IDs to evaluate across methods")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers")

    args = parser.parse_args()

    if args.eval_runs < 1:
        raise ValueError("eval-runs must be at least 1.")

    eval_model = args.eval_model.lower()
    if eval_model not in RATES:
        raise ValueError(f"Unsupported evaluation model for pricing table: {eval_model}")

    method_filter = parse_csv_filter(args.methods)
    run_filter = parse_csv_filter(args.runs)

    if eval_model.startswith("gpt"):
        client = openai.OpenAI(api_key=load_openai_key())
    elif eval_model.startswith("gemini"):
        client = genai.Client(api_key=load_gemini_key())
    else:
        raise ValueError("Evaluation model must start with 'gpt' or 'gemini'")

    fixer_client = openai.OpenAI(api_key=load_openai_key())

    data = load_enriched_json(args.input_json)
    prompt_text = args.prompt_file.read_text(encoding="utf-8")

    translation_model = data.get("model")
    if not isinstance(translation_model, str) or not translation_model.strip():
        raise ValueError("Input JSON must contain a non-empty top-level 'model' field.")
    translation_model = translation_model.strip().lower()

    shared_items = collect_shared_items(data)
    translation_tasks = build_translation_tasks(shared_items, method_filter=method_filter, run_filter=run_filter)

    if not translation_tasks:
        raise RuntimeError("No translation tasks were discovered for the requested methods/runs.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results_by_method_run = {}
    all_results = []

    total_translation_tasks = len(translation_tasks)
    total_evaluation_jobs = total_translation_tasks * args.eval_runs
    max_workers = min(max(1, args.max_workers), total_evaluation_jobs)

    print(f"Translation by {translation_model.upper()} evaluated by {eval_model.upper()}.")
    print("\n=== EVALUATION PLAN ===")
    print(f"Meaning units (items): {len(shared_items)}")
    print(f"Translation tasks discovered (T candidates across methods): {total_translation_tasks}")
    print(f"Requested evaluation runs per translation (E): {args.eval_runs}")
    print(f"Total evaluation jobs to launch: {total_evaluation_jobs}")
    print(f"Methods selected: {', '.join(sorted({t['method'] for t in translation_tasks}))}")
    print(f"Max workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for task_idx, translation_task in enumerate(translation_tasks, start=1):
            first_eval_index = next_eval_index(args.out_dir, translation_task["method"], translation_task["run"])

            for eval_pos in range(1, args.eval_runs + 1):
                eval_index = first_eval_index + eval_pos - 1
                out_file = eval_output_path(args.out_dir, translation_task["method"], translation_task["run"], eval_index)

                futures.append(
                    executor.submit(
                        run_single_evaluation,
                        translation_task,
                        client,
                        fixer_client,
                        eval_model,
                        translation_model,
                        out_file,
                        prompt_text,
                        task_idx,
                        total_translation_tasks,
                        eval_pos,
                        args.eval_runs,
                        eval_index,
                    )
                )

        for future in as_completed(futures):
            res = future.result()
            all_results.append(res)

            method_name = res["method"]
            run_id = str(res["run"])
            results_by_method_run.setdefault(method_name, {})
            results_by_method_run[method_name].setdefault(run_id, []).append(res)

    usage_totals = accumulate_usage(all_results)

    successful_scores = [res["score"]["major_equiv_per_unit"] for res in all_results if res["score"] is not None]
    if not successful_scores:
        raise RuntimeError("No successful evaluation results were collected.")

    per_method = {}
    total_successful_eval_runs = 0

    for method_name in sorted(results_by_method_run.keys()):
        per_translation = {}

        for run_id in sorted(results_by_method_run[method_name].keys(), key=lambda x: (len(str(x)), str(x))):
            sorted_results = sorted(results_by_method_run[method_name][run_id], key=lambda r: r["eval_run"])
            eval_scores = [
                r["score"]["major_equiv_per_unit"]
                for r in sorted_results
                if r["score"] is not None
            ]

            if not eval_scores:
                continue

            total_successful_eval_runs += len(eval_scores)
            per_translation[run_id] = summarize_translation(eval_scores)

        if not per_translation:
            continue

        method_stats = summarize_method(per_translation)
        method_stats["per_translation"] = per_translation
        per_method[method_name] = method_stats

    if not per_method:
        raise RuntimeError("No method-level summaries could be computed.")

    overall = summarize_overall(per_method)

    print("\n=== FINAL MQM RESULTS (MAJOR-EQUIV PER UNIT) ===")
    print(f"Methods (M): {overall['num_methods']}")
    print(f"Successful evaluation runs collected: {total_successful_eval_runs}")

    print("\n--- Per-Method Results ---")
    for method_name, method_stats in per_method.items():
        e_display = (
            str(method_stats["eval_runs_per_translation"])
            if method_stats["eval_runs_per_translation"] is not None
            else ",".join(str(x) for x in method_stats["observed_eval_runs_per_translation"])
        )
        print(
            f"{method_name}: T={method_stats['num_translations']}, "
            f"E={e_display}, "
            f"mean={method_stats['mean_major_equiv_per_unit']:.4f}, "
            f"translation SD={method_stats['translation_sd']:.4f}, "
            f"95% CI ±{method_stats['ci_95_half_width']:.4f}, "
            f"pooled eval-run SD={method_stats['pooled_eval_run_sd']:.4f}, "
            f"avg eval noise={method_stats['avg_eval_noise']:.4f}"
        )

    print("\n--- Per-Translation Results ---")
    for method_name, method_stats in per_method.items():
        for run_id, tr_stats in method_stats["per_translation"].items():
            print(
                f"{method_name}/run_{run_id}: "
                f"mean={tr_stats['mean_major_equiv_per_unit']:.4f}, "
                f"eval-run SD={tr_stats['eval_run_sd']:.4f}, "
                f"95% CI ±{tr_stats['ci_95_half_width']:.4f}, "
                f"E={tr_stats['n_eval_runs']}"
            )

    print("\n--- Overall Across Methods ---")
    print(f"Mean major-equiv per unit: {overall['mean_major_equiv_per_unit']:.4f}")
    print(
        f"95% CI: ±{overall['ci_95_half_width']:.4f} "
        f"[{overall['ci_95_lower']:.4f}, {overall['ci_95_upper']:.4f}]"
    )
    print(f"Method-level SD: {overall['method_sd']:.4f}")

    print("\n--- Usage ---")
    print(f"Total input tokens: {usage_totals['total_input_tokens']}")
    print(f"Total output tokens: {usage_totals['total_output_tokens']}")
    print(f"Total estimated cost (USD): ${usage_totals['total_cost']:.6f}")

    ranking = sorted(
        [
            {
                "method": method_name,
                "num_translations": stats["num_translations"],
                "eval_runs_per_translation": stats["eval_runs_per_translation"],
                "mean_major_equiv_per_unit": stats["mean_major_equiv_per_unit"],
                "ci_95_half_width": stats["ci_95_half_width"],
                "translation_sd": stats["translation_sd"],
                "pooled_eval_run_sd": stats["pooled_eval_run_sd"],
                "avg_eval_noise": stats["avg_eval_noise"],
            }
            for method_name, stats in per_method.items()
        ],
        key=lambda x: x["mean_major_equiv_per_unit"]
    )

    final_report = {
        "backend": {
            "evaluation_model": eval_model,
            "evaluated_translation_model": translation_model,
        },
        "requested_eval_runs_per_translation": args.eval_runs,
        "num_methods": overall["num_methods"],
        "total_successful_eval_runs": total_successful_eval_runs,
        "overall_across_methods": {
            "mean_major_equiv_per_unit": overall["mean_major_equiv_per_unit"],
            "ci_95_lower": overall["ci_95_lower"],
            "ci_95_upper": overall["ci_95_upper"],
            "ci_95_half_width": overall["ci_95_half_width"],
            "method_sd": overall["method_sd"],
            "se_method_across_methods": overall["se_method_across_methods"],
        },
        "per_method": per_method,
        "ranking": ranking,
        "usage": {
            "total_input_tokens": usage_totals["total_input_tokens"],
            "total_output_tokens": usage_totals["total_output_tokens"],
            "total_cost_usd": round(usage_totals["total_cost"], 6),
        },
    }

    final_json_path = args.out_dir / "aggregated_summary.json"
    if final_json_path.exists():
        print(f"Output file {final_json_path} already exists. Generating a new filename to avoid overwriting.")
        final_json_path = update_existing_fname(final_json_path)
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    method_comparison = {
        "methods": ranking
    }
    method_comparison_path = args.out_dir / "method_comparison.json"
    if method_comparison_path.exists():
        print(f"Output file {method_comparison_path} already exists. Generating a new filename to avoid overwriting.")
        method_comparison_path = update_existing_fname(method_comparison_path)
    with open(method_comparison_path, "w", encoding="utf-8") as f:
        json.dump(method_comparison, f, ensure_ascii=False, indent=2)

    print(f"\nAggregated summary saved to: {final_json_path}")
    print(f"Method comparison saved to: {method_comparison_path}")
