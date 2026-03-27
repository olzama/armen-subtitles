import argparse
import random
import openai
import pysrt
from google import genai
from google.genai import types as gtypes
from pathlib import Path
from evaluate_mqm_parallel import RATES, load_openai_key, load_gemini_key


# =========================
# API WRAPPERS
# =========================

def call_gpt_translate(content, client, model_name, temp):
    response = client.chat.completions.create(
        model=model_name,
        temperature=temp,
        seed=random.choice([x for x in range(1, 10000) if x != 42]),
        **({'reasoning_effort': RATES[model_name]['reasoning_effort']} if 'reasoning_effort' in RATES[model_name] else {}),
        max_completion_tokens=RATES[model_name].get("max_completion_tokens"),
        messages=[
            {"role": "system", "content": "Expert in subtitles translation."},
            {"role": "user", "content": content}
        ]
    )
    raw_text = response.choices[0].message.content.strip()
    usage = response.usage
    in_tokens = usage.prompt_tokens
    out_tokens = usage.completion_tokens
    reasoning_tokens = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
    cost = (in_tokens * RATES[model_name]["input"]) + (out_tokens * RATES[model_name]["output"])
    return raw_text, None, in_tokens, out_tokens, cost, reasoning_tokens


def call_gemini_translate(content, client, model_name, temp):
    response = client.models.generate_content(
        model=model_name,
        config=gtypes.GenerateContentConfig(
            system_instruction="You are an expert in subtitles translation.",
            temperature=temp,
            seed=random.choice([x for x in range(1, 10000) if x != 42]),
            thinking_config=gtypes.ThinkingConfig(include_thoughts=True)
        ),
        contents=content
    )
    reasoning = None
    raw_text = ""
    for part in response.candidates[0].content.parts:
        if part.thought: reasoning = part.text
        if part.text: raw_text += part.text

    usage = response.usage_metadata
    in_tokens = usage.prompt_token_count
    out_tokens = usage.candidates_token_count + (usage.thoughts_token_count or 0)
    cost = (in_tokens * RATES[model_name]["input"]) + (out_tokens * RATES[model_name]["output"])
    return raw_text.strip(), reasoning, in_tokens, out_tokens, cost


# =========================
# HELPER: CHUNKING
# =========================

def get_chunks(text, max_chars=30000):
    """Uses pysrt to split text into chunks without breaking subtitle blocks."""
    subs = pysrt.from_string(text)
    chunks = []
    current_chunk_subs = []
    current_length = 0

    for sub in subs:
        sub_len = len(str(sub))  # str(sub) returns the formatted SRT block (Index, Time, Text)

        if current_length + sub_len > max_chars and current_chunk_subs:
            # Join the collected SubRipItems with newlines to form valid SRT text
            chunks.append("\n".join(str(s) for s in current_chunk_subs))
            current_chunk_subs = []
            current_length = 0

        current_chunk_subs.append(sub)
        current_length += sub_len

    if current_chunk_subs:
        chunks.append("\n".join(str(s) for s in current_chunk_subs))

    return chunks


# =========================
# CORE LOGIC
# =========================

def translate(text, client, translation_model, temp, output_dir, n_translations, source_lang, target_lang,
              prompt="", summary=None, intermediate_translation=None, unit_list=None,
              given_translation=None, schema=None):
    output_dir = Path(output_dir)
    prompts_dir = output_dir / "prompts"
    translations_dir = output_dir / "translations"
    reasoning_dir = output_dir / "reasoning"

    for d in [prompts_dir, translations_dir, reasoning_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build Prompt Header
    full_prompt_base = (f"You will perform a film subtitles translation task from {source_lang} into {target_lang}.\n"
                        f"{prompt}\n")
    if intermediate_translation: full_prompt_base += f"\nUse this intermediate translation:\n{intermediate_translation}\n"
    if unit_list: full_prompt_base += f"\nAnalyze these memes:\n{unit_list}\n"
    if given_translation: full_prompt_base += f"\nUse approved translations:\n{given_translation}\n"
    if summary: full_prompt_base += f"\nContext/Summary:\n{summary}\n"
    if schema: full_prompt_base += f"\nUse this schema:\n{schema}\n"

    # Chunk the input text
    max_chars = RATES[translation_model].get("max_chunk_chars", 30_000)
    chunks = get_chunks(text, max_chars=max_chars)
    print(f"Total chunks to process: {len(chunks)}")

    existing_trans_nums = [int(f.stem.split("-")[1]) for f in translations_dir.glob("translation-*")]
    next_num = max(existing_trans_nums) + 1 if existing_trans_nums else 1

    for i in range(next_num, next_num + n_translations):
        print(f"--- Starting Run {i}/{next_num + n_translations - 1} ---")
        combined_translation = []
        combined_reasoning = []
        combined_raw = []
        run_total_cost = 0

        for idx, chunk_text in enumerate(chunks):
            print(f"  Processing chunk {idx + 1}/{len(chunks)}...")

            content = (f"{full_prompt_base}\n\n"
                       f"Translate the following subtitles in full, even if it takes time, without stopping midway or asking for clarifications. Preserve time codes EXACTLY. "
                       f"Return the translation ONLY. Text to translate in full:\n\n{chunk_text}")

            reasoning_t = 0
            if translation_model.startswith("gpt"):
                raw_output, reasoning, in_t, out_t, cost, reasoning_t = call_gpt_translate(content, client, translation_model, temp)
            else:
                raw_output, reasoning, in_t, out_t, cost = call_gemini_translate(content, client, translation_model,
                                                                                 temp)

            clean_output = raw_output.replace("```plaintext", "").replace("```", "").strip()
            combined_translation.append(clean_output)
            combined_raw.append(f"CHUNK {idx + 1} RAW:\n{raw_output}")
            if reasoning:
                combined_reasoning.append(f"CHUNK {idx + 1} REASONING:\n{reasoning}")

            run_total_cost += cost
            reasoning_note = f"  ({reasoning_t} reasoning)" if reasoning_t else ""
            print(f"  Chunk tokens: {in_t} in / {out_t} out{reasoning_note}  Cost: ${cost:.4f}")

        # Save Final Glued Artifacts
        final_text = "\n\n".join(combined_translation)
        (translations_dir / f"translation-{i}.txt").write_text(final_text, encoding="utf-8")
        (prompts_dir / f"prompt-{i}.txt").write_text(f"{translation_model}\n{full_prompt_base}", encoding="utf-8")
        (reasoning_dir / f"raw-{i}.txt").write_text("\n\n".join(combined_raw), encoding="utf-8")
        if combined_reasoning:
            (reasoning_dir / f"reasoning-{i}.txt").write_text("\n\n".join(combined_reasoning), encoding="utf-8")

        print(f"Run {i} Finished. Total Cost: ${run_total_cost:.4f}")


# =========================
# MAIN ENTRY
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate subtitles using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Directory convention:\n"
            "  input:  data/films/<film_name>/subs/\n"
            "  output: output/films/<film_name>/translations/<trans_model>/<method>/"
        ),
    )
    parser.add_argument("film_name", type=str, help="Film identifier (e.g. pokrov-gate)")
    parser.add_argument("method", type=str, help="Translation method name (e.g. zero, summary)")
    parser.add_argument("trans_model", type=str, help="Translation model (e.g. gpt-5.2)")
    parser.add_argument("temp", type=float, help="Sampling temperature")
    parser.add_argument("n_runs", type=int, help="Number of translation runs to produce")
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--intermediate_trans", type=Path)
    parser.add_argument("--unit_list", type=Path)
    parser.add_argument("--given_trans", type=Path)
    parser.add_argument("--schema", type=Path)

    args = parser.parse_args()
    translation_model = args.trans_model.lower()

    if translation_model not in RATES:
        raise ValueError(f"Unsupported model '{translation_model}'. Known models: {list(RATES.keys())}")

    input_path = Path("data/films") / args.film_name / "subs"
    output_dir = Path("output/films") / args.film_name / "translations" / translation_model / args.method

    if input_path.is_dir():
        text = "\n".join([f.read_text(encoding="utf-8") for f in sorted(input_path.iterdir()) if f.is_file()])
    else:
        text = input_path.read_text(encoding="utf-8")

    if translation_model.startswith("gpt"):
        client = openai.OpenAI(api_key=load_openai_key())
    else:
        client = genai.Client(api_key=load_gemini_key())

    translate(
        text, client, translation_model, args.temp, output_dir, args.n_runs,
        "Russian", "English",
        args.prompt.read_text(encoding="utf-8") if args.prompt else "",
        args.summary.read_text(encoding="utf-8") if args.summary else None,
        args.intermediate_trans.read_text(encoding="utf-8") if args.intermediate_trans else None,
        args.unit_list.read_text(encoding="utf-8") if args.unit_list else None,
        args.given_trans.read_text(encoding="utf-8") if args.given_trans else None,
        args.schema.read_text(encoding="utf-8") if args.schema else None
    )