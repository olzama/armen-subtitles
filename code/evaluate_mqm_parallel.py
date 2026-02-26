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

    os.makedirs(output_dir, exist_ok=True)

    translation_penalties = []
    translation_major_equiv = []

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    max_workers = min(8, len(translation_files) * n_eval_runs)

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

            run_penalties = []
            run_major_equiv = []

            for future in as_completed(futures):
                result = future.result()

                score = result["score"]

                run_penalties.append(score["penalty_per_unit"])
                run_major_equiv.append(score["major_equiv_per_unit"])

                total_input_tokens += result["input_tokens"]
                total_output_tokens += result["output_tokens"]
                total_cost += result["cost_total"]

            translation_penalties.append(statistics.mean(run_penalties))
            translation_major_equiv.append(statistics.mean(run_major_equiv))

    # =========================
    # FINAL AGGREGATION
    # =========================

    mean = statistics.mean(translation_penalties)
    median = statistics.median(translation_penalties)
    sd = statistics.stdev(translation_penalties) if len(translation_penalties) > 1 else 0.0
    ci_95 = 1.96 * sd / math.sqrt(len(translation_penalties)) if len(translation_penalties) > 1 else 0.0

    m_mean = statistics.mean(translation_major_equiv)
    m_median = statistics.median(translation_major_equiv)
    m_sd = statistics.stdev(translation_major_equiv) if len(translation_major_equiv) > 1 else 0.0
    m_ci_95 = 1.96 * m_sd / math.sqrt(len(translation_major_equiv)) if len(translation_major_equiv) > 1 else 0.0

    final_report = {
        "num_translations": len(translation_files),
        "eval_runs_per_translation": n_eval_runs,
        "penalty_per_unit": {
            "mean": mean,
            "median": median,
            "std": sd,
            "ci_95": ci_95
        },
        "major_errors_per_unit": {
            "mean": m_mean,
            "median": m_median,
            "std": m_sd,
            "ci_95": m_ci_95
        },
        "usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost, 6)
        }
    }

    # =========================
    # FINAL AGGREGATION (ACROSS TRANSLATIONS)
    # =========================

    mean = statistics.mean(translation_penalties)
    median = statistics.median(translation_penalties)
    sd = statistics.stdev(translation_penalties) if len(translation_penalties) > 1 else 0.0
    ci_95 = 1.96 * sd / math.sqrt(len(translation_penalties)) if len(translation_penalties) > 1 else 0.0

    m_mean = statistics.mean(translation_major_equiv)
    m_median = statistics.median(translation_major_equiv)
    m_sd = statistics.stdev(translation_major_equiv) if len(translation_major_equiv) > 1 else 0.0
    m_ci_95 = 1.96 * m_sd / math.sqrt(len(translation_major_equiv)) if len(translation_major_equiv) > 1 else 0.0

    final_report = {
        "num_translations": len(translation_files),
        "eval_runs_per_translation": n_eval_runs,
        "per_translation_mean_penalty": translation_penalties,
        "per_translation_mean_major_equiv": translation_major_equiv,
        "penalty_per_unit": {
            "mean": mean,
            "median": median,
            "std": sd,
            "ci_95": ci_95
        },
        "major_errors_per_unit": {
            "mean": m_mean,
            "median": m_median,
            "std": m_sd,
            "ci_95": m_ci_95
        },
        "usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": round(total_cost, 6)
        },
        "interpretation": (
            f"Average MQM penalty per meaning unit: {mean:.2f} "
            f"(STD: {sd:.2f}, 95% CI: ±{ci_95:.2f}). "
            f"Major-equivalent errors per unit: {m_mean:.2f} "
            f"(STD: {m_sd:.2f}, 95% CI: ±{m_ci_95:.2f}). "
            f"(Weights: minor=1, major=5, critical=10)"
        )
    }

    print("\n=== FINAL AGGREGATED MQM RESULTS ===")
    print(f"Translations evaluated: {len(translation_files)}")
    print(f"Evaluation runs per translation: {n_eval_runs}")

    print("\n--- Penalty Per Meaning Unit ---")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Standard deviation: {sd:.2f}")
    print(f"95% CI: ±{ci_95:.2f}")

    print("\n--- Major-Equivalent Errors Per Meaning Unit ---")
    print(f"Mean: {m_mean:.2f}")
    print(f"Median: {m_median:.2f}")
    print(f"Standard deviation: {m_sd:.2f}")
    print(f"95% CI: ±{m_ci_95:.2f}")

    print("\n--- Usage ---")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total estimated cost (USD): ${total_cost:.6f}")

    # =========================
    # SAVE FINAL SUMMARY (IN SAME FOLDER AS EVAL RUNS)
    # =========================

    final_json_path = os.path.join(output_dir, "final_aggregated_summary.json")
    final_txt_path = os.path.join(output_dir, "final_aggregated_summary.txt")

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    with open(final_txt_path, "w", encoding="utf-8") as f:
        f.write(
            "FINAL MQM AGGREGATED RESULTS\n\n"
            f"Translations evaluated: {len(translation_files)}\n"
            f"Evaluation runs per translation: {n_eval_runs}\n\n"

            "Penalty per meaning unit\n"
            f"Mean: {mean:.2f}\n"
            f"Median: {median:.2f}\n"
            f"Standard deviation: {sd:.2f}\n"
            f"95% CI: ±{ci_95:.2f}\n\n"

            "Major-equivalent errors per meaning unit\n"
            f"Mean: {m_mean:.2f}\n"
            f"Median: {m_median:.2f}\n"
            f"Standard deviation: {m_sd:.2f}\n"
            f"95% CI: ±{m_ci_95:.2f}\n\n"

            "Usage\n"
            f"Total input tokens: {total_input_tokens}\n"
            f"Total output tokens: {total_output_tokens}\n"
            f"Total estimated cost (USD): ${total_cost:.6f}\n\n"

            f"{final_report['interpretation']}\n"
        )

    print("\nFinal aggregated summary saved to:")
    print(f"  {final_json_path}")
    print(f"  {final_txt_path}")