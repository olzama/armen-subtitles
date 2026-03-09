import sys
import json
import os
import openai
from google import genai
from google.genai import types as gtypes
import statistics
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def compute_mqm_score(mqm_json, approved_json):
    """Calculates penalty points based on MQM severity weights."""
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

    # meaning_units derived from the memes/approved JSON
    num_units = len(approved_json.get("items", [])) if isinstance(approved_json, dict) else 0

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
            {"role": "user", "content": content}
        ]
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
            temperature=0.0
        ),
        contents=content
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
    """Main function to prepare prompt, call API, and process JSON results."""
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

    # Clean Markdown JSON blocks if present
    clean_json_str = raw_output.strip("```json\n").strip("```plaintext\n").strip("```")

    try:
        mqm_json = json.loads(clean_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model {eval_model} failed to return valid JSON:\n{raw_output}") from e

    score = compute_mqm_score(mqm_json, memes)
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
        "cost_total": cost
    }


# =========================
# RUNNER UTILITIES
# =========================

def run_single_evaluation(source, translation, client, ai_type, model_name,
                          out_file, prompt, summary, memes, schema, idx, total):
    print(f"  Evaluation run {idx}/{total} for {os.path.basename(out_file)}")
    return mqm_evaluation(source, translation, client, ai_type, model_name,
                          prompt, summary, memes, schema, out_file)


# =========================
# MAIN ENTRY POINT
# =========================

if __name__ == "__main__":
    if len(sys.argv) < 10:
        print("Usage: python evaluate_mqm_parallel.py <source.txt> <trans_folder> <out_dir> "
              "<prompt.txt> <summary.txt> <memes.json> <schema.txt> <n_runs> <ai_type>")
        sys.exit(1)

    # Setup inputs
    source_file = sys.argv[1]
    trans_folder = sys.argv[2]
    out_dir = sys.argv[3]
    prompt_file = sys.argv[4]
    summary_file = sys.argv[5]
    memes_file = sys.argv[6]
    schema_file = sys.argv[7]
    n_eval_runs = int(sys.argv[8])
    translation_model = sys.argv[9].lower()  # "gpt" or "gemini"
    eval_model = sys.argv[10].lower()  # "gpt" or "gemini"


    # Initialize Backend
    if eval_model.startswith("gpt"):
        with open("./GreenAI-API-key.txt", "r") as f:
            key = f.read().strip()
        client = openai.OpenAI(api_key=key)
    elif eval_model.startswith("gemini"):
        with open("./gemini-personal-API-key.txt", "r") as f:
            key = f.read().strip()
        client = genai.Client(api_key=key)
    else:
        raise ValueError("AI type must be 'gpt' or 'gemini'")

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

    trans_files = [os.path.join(trans_folder, f) for f in sorted(os.listdir(trans_folder))
                   if os.path.isfile(os.path.join(trans_folder, f)) and not f.startswith(".")]

    os.makedirs(out_dir, exist_ok=False)

    # Stats tracking
    translation_major_equiv = []
    all_run_major_equiv = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    # Parallel Execution
    max_workers = min(8, len(trans_files) * n_eval_runs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for t_path in trans_files:
            with open(t_path, "r", encoding="utf-8") as f:
                translation_text = f.read()
            for i in range(n_eval_runs):
                result_path = os.path.join(out_dir, f"{os.path.basename(t_path)}_eval_{i + 1}.json")
                futures.append(executor.submit(
                    run_single_evaluation, source_text, translation_text, client, eval_model, translation_model,
                    result_path, prompt_text, summary_text, memes_data, schema_text, i + 1, n_eval_runs
                ))

        # Collect results per translation to average them
        results_by_file = {f: [] for f in trans_files}
        for future in as_completed(futures):
            res = future.result()
            # Extract original filename from score data or path logic
            # Here we just track global stats for the summary report
            all_run_major_equiv.append(res["score"]["major_equiv_per_unit"])
            total_input_tokens += res["input_tokens"]
            total_output_tokens += res["output_tokens"]
            total_cost += res["cost_total"]

    # Calculate statistics
    T = len(trans_files)
    E = n_eval_runs

    overall_major_mean = statistics.mean(all_run_major_equiv)
    run_sd = statistics.stdev(all_run_major_equiv) if len(all_run_major_equiv) > 1 else 0.0

    # CI calculations (assuming method mean is based on individual runs)
    se_method = run_sd / math.sqrt(len(all_run_major_equiv)) if len(all_run_major_equiv) > 1 else 0.0
    ci_95_half_width = 1.96 * se_method
    ci_lower = overall_major_mean - ci_95_half_width
    ci_upper = overall_major_mean + ci_95_half_width

    # Final Report Output
    print("Translation by {} evaluated by {}.".format(translation_model.upper(), eval_model.upper()))
    print("\n=== FINAL MQM RESULTS (MAJOR-EQUIV PER UNIT) ===")
    print(f"Translations (T): {T}")
    print(f"Evaluation runs per translation (E): {E}")

    print("\n--- Method-Level Result ---")
    print(f"Mean major-equiv per unit: {overall_major_mean:.4f}")
    print(f"95% CI: {ci_95_half_width:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Run-level SD (Noise): {run_sd:.4f}")

    print("\n--- Usage ---")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total estimated cost (USD): ${total_cost:.6f}")

    final_report = {
        "num_translations": T,
        "eval_runs_per_translation": E,
        "backend": {"evaluation model": eval_model, "evaluated translation model": translation_model},
        "major_equiv_per_unit": {
            "mean": overall_major_mean,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "ci_95_half_width": ci_95_half_width,
            "se_method": se_method,
            "run_sd": run_sd
        },
        "usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost, 6)
        }
    }

    final_json_path = os.path.join(out_dir, "aggregated_summary.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\nAggregated summary saved to: {final_json_path}")