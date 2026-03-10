import sys
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
# GPT-5.2 rates
RATE_GPT_INPUT = 1.75 / 1_000_000
RATE_GPT_OUTPUT = 14.00 / 1_000_000

# Gemini-3 rates
RATE_GEMINI_INPUT = 0.30 / 1_000_000
RATE_GEMINI_OUTPUT = 2.5 / 1_000_000


# =========================
# MQM SCORING LOGIC
# =========================

def compute_mqm_score(mqm_json, approved_json=None):
    """Calculates MQM penalty points and normalized statistics.

    If approved_json is provided and has an ``items`` field, that is used for the
    number of meaning units. Otherwise, the MQM file itself is used. This keeps
    backward compatibility while also matching the behavior expected by
    compute-eval-stats.py when summaries are recomputed from saved evaluation
    files.
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
            f"(≈ {major_equiv_per_unit:.2f} major errors per unit; "
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
        temperature=0,
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
    cost = (in_tokens * RATE_GPT_INPUT) + (out_tokens * RATE_GPT_OUTPUT)
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
    # Gemini 3 billable output includes thoughts and candidates
    out_tokens = usage.candidates_token_count + (usage.thoughts_token_count or 0)
    cost = (in_tokens * RATE_GEMINI_INPUT) + (out_tokens * RATE_GEMINI_OUTPUT)
    return raw_text, in_tokens, out_tokens, cost


# =========================
# MQM ORCHESTRATOR
# =========================

def mqm_evaluation(source, translation, client, eval_model, translation_model,
                   prompt=None, summary=None, memes=None, schema=None, output_filename=None):
    """Prepare prompt, call API, process JSON results, and attach summary."""
    full_content = (
        f"Source text:\n ```{source}```\n\n"
        f"Translation:\n ```{translation}```\n\n"
        f"SCHEMA:\n {schema}\n\n"
        f"MEMES:\n {json.dumps(memes, ensure_ascii=False)}\n\n"
        f"SUMMARY:\n {summary}\n\n"
    )
    if prompt:
        full_content += prompt

    if eval_model.startswith("gpt"):
        raw_output, in_t, out_t, cost = call_gpt_mqm(full_content, client, eval_model)
    else:
        raw_output, in_t, out_t, cost = call_gemini_mqm(full_content, client, eval_model)

    clean_json_str = raw_output.strip()
    if clean_json_str.startswith("```"):
        lines = clean_json_str.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean_json_str = "\n".join(lines).strip()

    try:
        mqm_json = json.loads(clean_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model {eval_model} failed to return valid JSON:\n{raw_output}") from e

    score = compute_mqm_score(mqm_json, mqm_json)
    mqm_json["summary"] = score
    mqm_json["evaluator"] = eval_model
    mqm_json["translator"] = translation_model

    if output_filename:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(mqm_json, f, ensure_ascii=False, indent=2)

    return {
        "score": score,
        "input_tokens": in_t,
        "output_tokens": out_t,
        "cost_total": cost,
    }


# =========================
# RUNNER UTILITIES
# =========================

def run_single_evaluation(source, translation, client, eval_model, translation_model,
                          out_file, prompt, summary, memes, schema, idx, total, translation_file):
    print(f"  Evaluation run {idx}/{total} for {os.path.basename(out_file)}")
    result = mqm_evaluation(
        source,
        translation,
        client,
        eval_model,
        translation_model,
        prompt,
        summary,
        memes,
        schema,
        out_file,
    )
    result["translation_file"] = translation_file
    result["output_file"] = out_file
    result["run_index"] = idx
    return result


# =========================
# MAIN ENTRY POINT
# =========================

if __name__ == "__main__":
    if len(sys.argv) < 11:
        print(
            "Usage: python evaluate_mqm_parallel.py <source.txt> <trans_folder> <out_dir> "
            "<prompt.txt> <summary.txt> <memes.json> <schema.txt> <n_runs> <translation_model> <eval_model>"
        )
        sys.exit(1)

    source_file = sys.argv[1]
    trans_folder = sys.argv[2]
    out_dir = sys.argv[3]
    prompt_file = sys.argv[4]
    summary_file = sys.argv[5]
    memes_file = sys.argv[6]
    schema_file = sys.argv[7]
    n_eval_runs = int(sys.argv[8])
    translation_model = sys.argv[9].lower()
    eval_model = sys.argv[10].lower()

    # Initialize backend
    if eval_model.startswith("gpt"):
        with open("./GreenAI-API-key.txt", "r", encoding="utf-8") as f:
            key = f.read().strip()
        client = openai.OpenAI(api_key=key)
    elif eval_model.startswith("gemini"):
        with open("./gemini-personal-API-key.txt", "r", encoding="utf-8") as f:
            key = f.read().strip()
        client = genai.Client(api_key=key)
    else:
        raise ValueError("Evaluation model must start with 'gpt' or 'gemini'")

    # Load context files
    with open(source_file, "r", encoding="utf-8") as f:
        source_text = f.read()
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_text = f.read()
    with open(memes_file, "r", encoding="utf-8") as f:
        memes_data = json.load(f)
    with open(schema_file, "r", encoding="utf-8") as f:
        schema_text = f.read()

    trans_files = [
        os.path.join(trans_folder, f)
        for f in sorted(os.listdir(trans_folder))
        if os.path.isfile(os.path.join(trans_folder, f)) and not f.startswith(".")
    ]

    Path(out_dir).mkdir(parents=True, exist_ok=False)

    # Stats tracking
    all_run_major_equiv = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    # Restore per-translation aggregation functionality
    results_by_file = {f: [] for f in trans_files}
    translation_major_equiv = []

    # Parallel execution
    max_workers = min(8, max(1, len(trans_files) * n_eval_runs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for t_path in trans_files:
            with open(t_path, "r", encoding="utf-8") as f:
                translation_text = f.read()
            for i in range(n_eval_runs):
                result_path = os.path.join(out_dir, f"{os.path.basename(t_path)}_eval_{i + 1}.json")
                futures.append(
                    executor.submit(
                        run_single_evaluation,
                        source_text,
                        translation_text,
                        client,
                        eval_model,
                        translation_model,
                        result_path,
                        prompt_text,
                        summary_text,
                        memes_data,
                        schema_text,
                        i + 1,
                        n_eval_runs,
                        t_path,
                    )
                )

        for future in as_completed(futures):
            res = future.result()
            t_path = res["translation_file"]
            results_by_file[t_path].append(res)

            all_run_major_equiv.append(res["score"]["major_equiv_per_unit"])
            total_input_tokens += res["input_tokens"]
            total_output_tokens += res["output_tokens"]
            total_cost += res["cost_total"]

    # Compute per-translation means from run-level scores
    per_translation_summary = {}
    for t_path in trans_files:
        run_scores = [r["score"]["major_equiv_per_unit"] for r in results_by_file[t_path]]
        if not run_scores:
            continue

        translation_mean = statistics.mean(run_scores)
        translation_sd = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
        translation_se = translation_sd / math.sqrt(len(run_scores)) if len(run_scores) > 1 else 0.0
        translation_ci_half_width = 1.96 * translation_se

        translation_major_equiv.append(translation_mean)
        per_translation_summary[os.path.basename(t_path)] = {
            "mean_major_equiv_per_unit": translation_mean,
            "run_sd": translation_sd,
            "se_mean": translation_se,
            "ci_95_half_width": translation_ci_half_width,
            "n_runs": len(run_scores),
            "run_scores": run_scores,
        }

    T = len(translation_major_equiv)
    E = n_eval_runs

    if not all_run_major_equiv:
        raise RuntimeError("No evaluation results were collected.")

    # Method-level mean should be based on per-translation means, not raw runs.
    overall_major_mean = statistics.mean(translation_major_equiv) if translation_major_equiv else 0.0

    # Run-level noise across all individual evaluation runs
    run_sd = statistics.stdev(all_run_major_equiv) if len(all_run_major_equiv) > 1 else 0.0
    avg_eval_noise = run_sd / math.sqrt(E) if E > 1 else 0.0

    # Translation-level dispersion across translation means
    translation_sd = statistics.stdev(translation_major_equiv) if len(translation_major_equiv) > 1 else 0.0
    se_method = translation_sd / math.sqrt(T) if T > 1 else 0.0
    ci_95_half_width = 1.96 * se_method
    ci_lower = overall_major_mean - ci_95_half_width
    ci_upper = overall_major_mean + ci_95_half_width

    print(f"Translation by {translation_model.upper()} evaluated by {eval_model.upper()}.")
    print("\n=== FINAL MQM RESULTS (MAJOR-EQUIV PER UNIT) ===")
    print(f"Translations (T): {T}")
    print(f"Evaluation runs per translation (E): {E}")

    print("\n--- Per-Translation Results ---")
    for filename, stats in per_translation_summary.items():
        print(
            f"{filename}: mean={stats['mean_major_equiv_per_unit']:.4f}, "
            f"SD={stats['run_sd']:.4f}, "
            f"95% CI ±{stats['ci_95_half_width']:.4f}"
        )

    print("\n--- Method-Level Result ---")
    print(f"Mean major-equiv per unit: {overall_major_mean:.4f}")
    print(f"95% CI: ±{ci_95_half_width:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Translation-level SD: {translation_sd:.4f}")
    print(f"Individual run-level SD (noise): {run_sd:.4f}")
    print(f"Average evaluation noise (SD of the mean, E={E}): {avg_eval_noise:.4f}")

    print("\n--- Usage ---")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total estimated cost (USD): ${total_cost:.6f}")

    final_report = {
        "num_translations": T,
        "eval_runs_per_translation": E,
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
            "translation_sd": translation_sd,
            "run_sd": run_sd,
            "avg_eval_noise": avg_eval_noise,
        },
        "per_translation": per_translation_summary,
        "usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost, 6),
        },
    }

    final_json_path = os.path.join(out_dir, "aggregated_summary.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\nAggregated summary saved to: {final_json_path}")