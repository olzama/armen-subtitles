import sys
import json
import os
import openai
import statistics
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# COST RATES
# =========================

RATE_INPUT_PER_TOKEN = 1.75 / 1_000_000
RATE_OUTPUT_PER_TOKEN = 14.00 / 1_000_000


# =========================
# MQM SCORING LOGIC
# =========================

def compute_mqm_score(mqm_json, approved_json):
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

    total_points = sum(
        counts[s] * severity_weights[s]
        for s in counts
    )

    num_units = len(approved_json.get("items", []))

    penalty_per_unit = None
    major_equiv_per_unit = None
    interpretation = None

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
# MQM API CALL
# =========================

def mqm(source, translation, client, output_filename,
        prompt=None, summary=None, memes=None, schema=None):

    full_prompt = (
        f"Source text:\n ```{source}```\n\n"
        f"Translation:\n ```{translation}```\n\n"
        f"SCHEMA\n: {schema}\n\n"
        f"MEMES\n: {memes}\n\n"
        f"SUMMARY\n: {summary}\n\n"
    )

    if prompt:
        full_prompt += prompt

    print("Evaluating translation quality using MQM...")

    response = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0,
        top_p=1,
        messages=[
            {
                "role": "system",
                "content": "Expert in literary translation quality assessment using MQM framework."
            },
            {
                "role": "user",
                "content": f"You are given the following literary translation evaluation task:\n\n{full_prompt}"
            }
        ]
    )

    raw_output = response.choices[0].message.content.strip()
    clean_output = raw_output.strip("```plaintext\n").strip("```")

    try:
        mqm_json = json.loads(clean_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON:\n{clean_output}"
        ) from e

    score = compute_mqm_score(mqm_json, memes)
    mqm_json["summary"] = score

    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    cost_input = input_tokens * RATE_INPUT_PER_TOKEN
    cost_output = output_tokens * RATE_OUTPUT_PER_TOKEN
    cost_total = cost_input + cost_output

    print("MQM score summary:", score)
    print(
        f"Estimated cost: ${cost_total:.6f}, "
        f"with {input_tokens} input tokens and {output_tokens} output tokens."
    )

    if output_filename:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(mqm_json, f, ensure_ascii=False, indent=2)

    return {
        "score": score,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_total": cost_total
    }


# =========================
# UTILITIES
# =========================

def list_translation_files(folder_path):
    files = []
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and not fname.startswith("."):
            files.append(fpath)
    return files


def run_single_evaluation(source, translation, client, out_file,
                          prompt, summary, memes, schema,
                          run_idx, total_runs):
    print(f"  Evaluation run {run_idx}/{total_runs}")
    return mqm(
        source,
        translation,
        client,
        out_file,
        prompt,
        summary,
        memes,
        schema
    )


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    # =========================
    # LOAD INPUTS
    # =========================

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        source = f.read()

    translations_folder = sys.argv[2]
    output_dir = sys.argv[3]

    with open(sys.argv[4], "r", encoding="utf-8") as f:
        prompt = f.read()

    with open(sys.argv[5], "r", encoding="utf-8") as f:
        summary = f.read()

    with open(sys.argv[6], "r", encoding="utf-8") as f:
        memes = json.load(f)

    with open(sys.argv[7], "r", encoding="utf-8") as f:
        schema = f.read()

    n_eval_runs = int(sys.argv[8])

    with open("./GreenAI-API-key.txt", "r") as myfile:
        openai_key = myfile.read().replace("\n", "")

    client = openai.OpenAI(api_key=openai_key)

    translation_files = list_translation_files(translations_folder)

    if not translation_files:
        raise ValueError("No translation files found in the folder.")

    print(f"Found {len(translation_files)} translations.")

    os.makedirs(output_dir, exist_ok=False)

    # =========================
    # STORAGE
    # =========================

    translation_major_equiv = []
    all_run_major_equiv = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    max_workers = min(8, len(translation_files) * n_eval_runs)

    # =========================
    # RUN EVALUATIONS
    # =========================

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        for t_idx, t_path in enumerate(translation_files, 1):

            print(
                f"\n=== Translation {t_idx}/{len(translation_files)}: "
                f"{os.path.basename(t_path)} ==="
            )

            with open(t_path, "r", encoding="utf-8") as f:
                translation = f.read()

            futures = []

            for i in range(n_eval_runs):
                out_file = os.path.join(
                    output_dir,
                    f"{os.path.basename(t_path)}_eval_{i+1}.json"
                )

                futures.append(
                    executor.submit(
                        run_single_evaluation,
                        source,
                        translation,
                        client,
                        out_file,
                        prompt,
                        summary,
                        memes,
                        schema,
                        i + 1,
                        n_eval_runs
                    )
                )

            run_major_equiv = []

            for future in as_completed(futures):
                result = future.result()
                score = result["score"]

                run_major_equiv.append(score["major_equiv_per_unit"])
                all_run_major_equiv.append(score["major_equiv_per_unit"])

                total_input_tokens += result["input_tokens"]
                total_output_tokens += result["output_tokens"]
                total_cost += result["cost_total"]

            translation_major_equiv.append(
                statistics.mean(run_major_equiv)
            )

    # =========================
    # METHOD-LEVEL STATISTICS
    # =========================

    T = len(translation_major_equiv)
    E = n_eval_runs

    overall_major_mean = statistics.mean(translation_major_equiv)

    between_translation_sd = (
        statistics.stdev(translation_major_equiv)
        if T > 1 else 0.0
    )

    # Empirical SE of method mean
    se_method = between_translation_sd / math.sqrt(T) if T > 1 else 0.0
    ci_95_half_width = 1.96 * se_method

    ci_lower = overall_major_mean - ci_95_half_width
    ci_upper = overall_major_mean + ci_95_half_width

    # =========================
    # RUN-LEVEL DIAGNOSTIC
    # =========================

    run_sd = statistics.stdev(all_run_major_equiv) if len(all_run_major_equiv) > 1 else 0.0

    # =========================
    # OUTPUT
    # =========================

    print("\n=== FINAL MQM RESULTS (MAJOR-EQUIV PER UNIT) ===")
    print(f"Translations (T): {T}")
    print(f"Evaluation runs per translation (E): {E}")

    print("\n--- Method-Level Result ---")
    print(f"Mean major-equiv per unit: {overall_major_mean:.4f}")
    print(f"95% CI: {ci_95_half_width:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Between-translation SD: {between_translation_sd:.4f}")

    print("\n--- Evaluation Noise (Diagnostic Only) ---")
    print(f"Run-level SD: {run_sd:.4f}")

    print("\n--- Usage ---")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total estimated cost (USD): ${total_cost:.6f}")

    final_report = {
        "num_translations": T,
        "eval_runs_per_translation": E,
        "major_equiv_per_unit": {
            "mean": overall_major_mean,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "ci_95_half_width": ci_95_half_width,
            "between_translation_sd": between_translation_sd,
            "se_method": se_method
        },
        "evaluation_noise_diagnostic": {
            "run_level_sd": run_sd
        },
        "usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost, 6)
        }
    }

    final_json_path = os.path.join(output_dir, "aggregated_summary.json")

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print("\nAggregated summary saved to:")
    print(f"  {final_json_path}")