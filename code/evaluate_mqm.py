import sys
import openai
from chunk import count_tokens_in_text

import re

def compute_mqm_score(mqm_text, translation_text=None):
    severity_weights = {
        "critical": 10,
        "major": 5,
        "minor": 1,
    }

    counts = {s: 0 for s in severity_weights}
    for match in re.finditer(r"Severity:\s*(\w+)", mqm_text, flags=re.IGNORECASE):
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


def mqm(source, translation, prompt, summary, narratives, client, output_filename):
    full_prompt = (f"Source text: ```{source}```"
              f"\n\nTranslation: ```{translation}```\n\n ")
    full_prompt += prompt
    print("Evaluating translation quality using MQM...")
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Expert in literary translation quality assessment using MQM framework."},
            {"role": "user", "content": f"You are given the following literary translation evaluation task: \n\n {full_prompt}"}
        ]
    )
    raw_output = response.choices[0].message.content.strip()
    clean_output = raw_output.strip("```plaintext\n").strip("```")
    score = compute_mqm_score(clean_output, translation)
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
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(clean_output + "\n\nMQM Score Summary:\n" + str(score))
    return clean_output

if __name__ == "__main__":
    with open(sys.argv[1], "r", encoding='utf-8') as f:
        source = f.read()
    with open(sys.argv[2], "r", encoding='utf-8') as f:
        translation = f.read()
    with open(sys.argv[6], "r", encoding='utf-8') as f:
        narratives = f.read()
    with open(sys.argv[4], "r", encoding='utf-8') as f:
        prompt = f.read()
    with open(sys.argv[5], "r", encoding='utf-8') as f:
        summary = f.read()
    output_filename = sys.argv[3]
    with open ("./LYS-API-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    n_toks = count_tokens_in_text(source + translation + summary + narratives + prompt)
    print("Input token count: ", n_toks)
    eval = mqm(source, translation, prompt, summary, narratives, client, output_filename)