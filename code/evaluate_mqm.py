import sys, json
import openai
from chunk import count_tokens_in_text

import re

def compute_mqm_score_classic(mqm_text, translation_text=None):
    severity_weights = {
        "critical": 10,
        "major": 5,
        "minor": 1,
    }

    counts = {s: 0 for s in severity_weights}
    severity_regex = r"\*\*(Critical|Major|Minor)\s+([a-z\-]+):\*\*\s*(\d+)"
    for match in re.finditer(severity_regex, mqm_text, flags=re.IGNORECASE):
        level = match.group(1).lower()
        if level in counts:
            counts[level] += 1

    total_points = sum(counts[s] * severity_weights[s] for s in counts)

    words = None
    normalized = None
    final_score = None
    if translation_text:
        words = len(re.findall(r"\w+", translation_text))
        if words > 0:
            normalized = total_points / words * 1000
            final_score = 100 - normalized

    return {
        "counts": counts,
        "total_points": total_points,
        "words_in_translation": words,
        "normalized_points_per_1k": normalized,
        "final_quality_score": final_score,
    }

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
            if severity in counts:
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

def extract_approved_en_from_json(approved_json):
    return [
        item["en"]
        for item in approved_json.get("items", [])
        if "en" in item
    ]


def mqm(source, translation, client, output_filename, prompt=None, summary=None, memes=None, schema=None):
    full_prompt = (f"Source text:\n ```{source}```\n\n"
                   f"Translation:\n ```{translation}```\n\n "
                   f"SCHEMA\n: {schema}\n\n"
                   f"MEMES\n: {memes}\n\n"
                   f"SUMMARY\n: {summary}\n\n")
    full_prompt += prompt
    print("Evaluating translation quality using MQM...")
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "Expert in literary translation quality assessment using MQM framework."},
            {"role": "user", "content": f"You are given the following literary translation evaluation task: \n\n {full_prompt}"}
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
    mqm_json['summary'] = score
    print("MQM score summary:", score)
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    # Rates for GPT-5 (USD per token)
    rate_input_per_token = 1.25 / 1_000_000
    rate_output_per_token = 10.00 / 1_000_000
    cost_input = input_tokens * rate_input_per_token
    cost_output = output_tokens * rate_output_per_token
    print("Estimated cost: ${:.6f}, with {} input tokens and {} output tokens.".format(cost_input + cost_output,
                                                                                       input_tokens,
                                                                                       output_tokens))
    if output_filename:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(mqm_json, f, ensure_ascii=False, indent=2)
    return mqm_json

if __name__ == "__main__":
    with open(sys.argv[1], "r", encoding='utf-8') as f:
        source = f.read()
    with open(sys.argv[2], "r", encoding='utf-8') as f:
        translation = f.read()
    output_filename = sys.argv[3]
    with open(sys.argv[4], "r", encoding='utf-8') as f:
        prompt = f.read()
    with open(sys.argv[5], "r", encoding='utf-8') as f:
        summary = f.read()
    with open(sys.argv[6], "r", encoding='utf-8') as f:
        memes = json.load(f)
    with open(sys.argv[7], "r", encoding='utf-8') as f:
        schema = f.read()
    with open ("./LYS-API-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    eval = mqm(source, translation, client, output_filename, prompt, summary, memes, schema)