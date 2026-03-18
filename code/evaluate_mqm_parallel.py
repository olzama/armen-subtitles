#!/usr/bin/env python3

import sys
import argparse
import json
import os
import math
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


# =========================
# MQM SCORING LOGIC
# =========================

def compute_mqm_score(mqm_json, approved_json=None):
    """Calculates MQM penalty points and normalized statistics.

    If approved_json is provided and has an ``items`` field, that is used for the
    number of meaning units. Otherwise, the MQM file itself is used.
    """
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
    """Handles OpenAI specific API calls and usage metadata."""
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
    """Handles Google GenAI specific API calls and usage metadata."""
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
# JSON REPAIR
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
    """Attempts to fix common JSON formatting issues using GPT."""
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
    """Return {method: set(run_ids)} discovered across all items."""
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


def build_tasks(shared_items, method_filter=None, run_filter=None):
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


def make_task_output_path(out_dir: Path, method_name: str, run_id: str) -> Path:
    method_dir = out_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    return method_dir / f"run_{run_id}.json"


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
    """Preserve the payload data locally and attach model-returned issues to it.

    Expected model JSON: top-level object with items. Each item should ideally have id and issues.
    """
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

def mqm_evaluation(task_payload, client, fixer_client, eval_model, translation_model,
                   prompt, output_filename=None):
    """Prepare prompt, call API, process JSON results, and attach summary."""
    success = False

    full_content = prompt
    full_content += "\n\nDATA TO EVALUATE:\n"
    full_content += json.dumps(
        {
            "method": task_payload["method"],
            "run": task_payload["run"],
            "items": task_payload["items"],
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
        normalized_items = normalize_mqm_items(mqm_json, task_payload["items"])

        final_json = {
            "method": task_payload["method"],
            "run": task_payload["run"],
            "items": normalized_items,
        }

        score = compute_mqm_score(final_json, final_json)
        final_json["summary"] = score
        final_json["evaluator"] = eval_model
        final_json["translator"] = translation_model

        # Preserve any extra top-level fields the model may have returned
        for k, v in mqm_json.items():
            if k not in {"items", "summary", "evaluator", "translator", "method", "run"}:
                final_json[k] = v

        if output_filename:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(final_json, f, ensure_ascii=False, indent=2)

        return {
            "score": score,
            "input_tokens": in_t,
            "output_tokens": out_t,
            "cost_total": cost,
            "method": task_payload["method"],
            "run": task_payload["run"],
            "output_file": str(output_filename) if output_filename else None,
        }

    print(f"Failed to obtain valid MQM JSON output from {eval_model} for this evaluation run. Returning None for score.")
    return {
        "score": None,
        "input_tokens": in_t,
        "output_tokens": out_t,
        "cost_total": cost,
        "method": task_payload["method"],
        "run": task_payload["run"],
        "output_file": str(output_filename) if output_filename else None,
    }


# =========================
# RUNNER UTILITIES
# =========================
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

def run_single_evaluation(task_payload, client, fixer_client, eval_model, translation_model,
                          out_file, prompt, idx, total):
    print(
        f"  Evaluation run {idx}/{total}: "
        f"method={task_payload['method']} run={task_payload['run']} "
        f"-> {os.path.basename(out_file)}"
    )
    result = mqm_evaluation(
        task_payload,
        client,
        fixer_client,
        eval_model,
        translation_model,
        prompt,
        out_file,
    )
    return result


def compute_method_stats(run_scores):
    if not run_scores:
        return None

    method_mean = statistics.mean(run_scores)
    run_sd = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
    se_mean = run_sd / math.sqrt(len(run_scores)) if len(run_scores) > 1 else 0.0
    ci_95_half_width = 1.96 * se_mean

    return {
        "mean_major_equiv_per_unit": method_mean,
        "run_sd": run_sd,
        "se_mean": se_mean,
        "ci_95_half_width": ci_95_half_width,
        "n_runs": len(run_scores),
        "run_scores": run_scores,
    }

def ranking_to_stats_ordered(per_method_summary, ranking):
    ordered = []
    for row in ranking:
        method_name = row["method"]
        ordered.append((method_name, per_method_summary[method_name]))
    return ordered

# =========================
# MAIN ENTRY POINT
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate translations stored inside one enriched JSON using MQM.")

    parser.add_argument("input_json", type=Path, help="Path to the full JSON file containing items "
                                                      "to be evaluated and the corresponding reference, "
                                                      "analysis, etc.")
    parser.add_argument("out_dir", type=Path, help="Directory to save evaluation results")
    parser.add_argument("prompt_file", type=Path, help="Path to the evaluation prompt file")
    parser.add_argument("eval_model", type=str, help="Evaluation model to use (e.g., 'gpt-5-mini')")

    parser.add_argument("--methods", type=str, help="Comma-separated list of methods to evaluate")
    parser.add_argument("--runs", type=str, help="Comma-separated list of run IDs to evaluate across methods")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers (default: 3)")

    args = parser.parse_args()

    eval_model = args.eval_model.lower()

    if eval_model not in RATES:
        raise ValueError(f"Unsupported evaluation model for pricing table: {eval_model}")

    method_filter = parse_csv_filter(args.methods)
    run_filter = parse_csv_filter(args.runs)

    # Initialize evaluation backend
    if eval_model.startswith("gpt"):
        client = openai.OpenAI(api_key=load_openai_key())
    elif eval_model.startswith("gemini"):
        client = genai.Client(api_key=load_gemini_key())
    else:
        raise ValueError("Evaluation model must start with 'gpt' or 'gemini'")

    # JSON fixer always uses GPT so repair is available regardless of evaluation backend
    fixer_client = openai.OpenAI(api_key=load_openai_key())

    # Load required context
    data = load_enriched_json(args.input_json)
    prompt_text = args.prompt_file.read_text(encoding="utf-8")

    translation_model = data.get("model")
    if not isinstance(translation_model, str) or not translation_model.strip():
        raise ValueError("Input JSON must contain a non-empty top-level 'model' field.")
    translation_model = translation_model.strip().lower()

    shared_items = collect_shared_items(data)
    tasks = build_tasks(shared_items, method_filter=method_filter, run_filter=run_filter)

    if not tasks:
        raise RuntimeError("No evaluation tasks were discovered for the requested methods/runs.")

    # Create output directory if needed
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Stats tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    results_by_method = {}
    all_run_major_equiv = []
    all_results = []

    total_tasks = len(tasks)
    max_workers = min(max(1, args.max_workers), total_tasks)

    print(f"Translation by {translation_model.upper()} evaluated by {eval_model.upper()}.")
    print("\n=== EVALUATION PLAN ===")
    print(f"Meaning units (items): {len(shared_items)}")
    print(f"Tasks discovered: {total_tasks}")
    print(f"Methods selected: {', '.join(sorted({t['method'] for t in tasks}))}")
    print(f"Max workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for idx, task in enumerate(tasks, start=1):
            out_file = make_task_output_path(args.out_dir, task["method"], task["run"])
            futures.append(
                executor.submit(
                    run_single_evaluation,
                    task,
                    client,
                    fixer_client,
                    eval_model,
                    translation_model,
                    out_file,
                    prompt_text,
                    idx,
                    total_tasks,
                )
            )

        for future in as_completed(futures):
            res = future.result()
            all_results.append(res)

            method_name = res["method"]
            results_by_method.setdefault(method_name, []).append(res)

            if res["score"] is not None:
                all_run_major_equiv.append(res["score"]["major_equiv_per_unit"])

    usage_totals = accumulate_usage(all_results)
    total_input_tokens = usage_totals["total_input_tokens"]
    total_output_tokens = usage_totals["total_output_tokens"]
    total_cost = usage_totals["total_cost"]

    if not all_run_major_equiv:
        raise RuntimeError("No evaluation results were collected.")

    # Compute per-method summaries from run-level scores
    per_method_summary = {}
    method_major_equiv_means = []

    for method_name in sorted(results_by_method.keys()):
        sorted_method_results = sorted(
            results_by_method[method_name],
            key=lambda r: (len(str(r["run"])), str(r["run"]))
        )
        run_scores = [
            r["score"]["major_equiv_per_unit"]
            for r in sorted_method_results
            if r["score"] is not None
        ]

        if not run_scores:
            continue

        stats = compute_method_stats(run_scores)
        per_method_summary[method_name] = stats
        method_major_equiv_means.append(stats["mean_major_equiv_per_unit"])

    M = len(per_method_summary)

    # Overall method-level mean based on per-method means, preserving the old logic
    overall_major_mean = statistics.mean(method_major_equiv_means) if method_major_equiv_means else 0.0

    # Overall run-level noise across all individual evaluation runs
    overall_run_sd = statistics.stdev(all_run_major_equiv) if len(all_run_major_equiv) > 1 else 0.0

    # Method-level dispersion across method means
    method_sd = statistics.stdev(method_major_equiv_means) if len(method_major_equiv_means) > 1 else 0.0
    se_method = method_sd / math.sqrt(M) if M > 1 else 0.0
    ci_95_half_width = 1.96 * se_method
    ci_lower = overall_major_mean - ci_95_half_width
    ci_upper = overall_major_mean + ci_95_half_width

    print("\n=== FINAL MQM RESULTS (MAJOR-EQUIV PER UNIT) ===")
    print(f"Methods (M): {M}")
    print(f"Total evaluation runs collected: {len(all_run_major_equiv)}")

    print("\n--- Per-Method Results ---")
    for method_name, stats in per_method_summary.items():
        print(
            f"{method_name}: mean={stats['mean_major_equiv_per_unit']:.4f}, "
            f"SD={stats['run_sd']:.4f}, "
            f"SE={stats['se_mean']:.4f}, "
            f"95% CI ±{stats['ci_95_half_width']:.4f}, "
            f"n_runs={stats['n_runs']}"
        )

    print("\n--- Overall Method-Level Result ---")
    print(f"Mean major-equiv per unit: {overall_major_mean:.4f}")
    print(f"95% CI: ±{ci_95_half_width:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Method-level SD: {method_sd:.4f}")
    print(f"Individual run-level SD (noise): {overall_run_sd:.4f}")

    print("\n--- Per-Method Evaluation Noise ---")
    for method_name, stats in per_method_summary.items():
        print(
            f"{method_name}: average evaluation noise "
            f"(SD of the mean, E={stats['n_runs']}): {stats['se_mean']:.4f}"
        )

    print("\n--- Usage ---")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total estimated cost (USD): ${total_cost:.6f}")

    ranking = sorted(
        [
            {
                "method": method_name,
                "mean_major_equiv_per_unit": stats["mean_major_equiv_per_unit"],
                "ci_95_half_width": stats["ci_95_half_width"],
                "n_runs": stats["n_runs"],
            }
            for method_name, stats in per_method_summary.items()
        ],
        key=lambda x: x["mean_major_equiv_per_unit"]
    )

    final_report = {
        "num_methods": M,
        "total_eval_runs": len(all_run_major_equiv),
        "backend": {
            "evaluation_model": eval_model,
            "evaluated_translation_model": translation_model,
        },
        "major_equiv_per_unit": {
            "mean": overall_major_mean,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "ci_95_half_width": ci_95_half_width,
            "se_method": se_method,
            "method_sd": method_sd,
            "run_sd": overall_run_sd,
        },
        "per_method": per_method_summary,
        "ranking": ranking,
        "usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost, 6),
        },
    }

    final_json_path = args.out_dir / "aggregated_summary.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    method_comparison = {
        "methods": [
            {
                "method": method_name,
                "n_runs": stats["n_runs"],
                "mean_major_equiv_per_unit": stats["mean_major_equiv_per_unit"],
                "ci_95_half_width": stats["ci_95_half_width"],
                "run_sd": stats["run_sd"],
                "se_mean": stats["se_mean"],
            }
            for method_name, stats in ranking_to_stats_ordered(per_method_summary, ranking)
        ]
    }

    method_comparison_path = args.out_dir / "method_comparison.json"
    with open(method_comparison_path, "w", encoding="utf-8") as f:
        json.dump(method_comparison, f, ensure_ascii=False, indent=2)

    print(f"\nAggregated summary saved to: {final_json_path}")
    print(f"Method comparison saved to: {method_comparison_path}")


