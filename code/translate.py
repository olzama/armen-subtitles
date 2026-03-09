import json
import sys
import os
import openai
import httpx
from google import genai
from google.genai import types as gtypes
from pathlib import Path
from summarize import summarize_text

# =========================
# COST RATES (Current 2026)
# =========================
RATE_GPT_INPUT = 1.75 / 1_000_000
RATE_GPT_OUTPUT = 14.00 / 1_000_000

RATE_GEMINI_INPUT = 0.30 / 1_000_000
RATE_GEMINI_OUTPUT = 2.5 / 1_000_000


# =========================
# API WRAPPERS
# =========================

def call_gpt_translate(content, client, model_name, temp):
    """Handles OpenAI specific translation calls."""
    response = client.chat.completions.create(
        model=model_name,
        temperature=temp,
        messages=[
            {"role": "system", "content": "Expert in subtitles translation."},
            {"role": "user", "content": content}
        ]
    )
    raw_text = response.choices[0].message.content.strip()
    usage = response.usage
    in_tokens = usage.prompt_tokens
    out_tokens = usage.completion_tokens
    cost = (in_tokens * RATE_GPT_INPUT) + (out_tokens * RATE_GPT_OUTPUT)
    return raw_text, None, in_tokens, out_tokens, cost


def call_gemini_translate(content, client, model_name, temp):
    """Handles Google GenAI translation calls with thinking support."""
    # Gemini 3 typically uses temp 1.0 for optimal reasoning
    response = client.models.generate_content(
        model=model_name,
        config=gtypes.GenerateContentConfig(
            system_instruction="You are an expert in subtitles translation.",
            temperature=temp,
            thinking_config=gtypes.ThinkingConfig(
                include_thoughts=True  # Changed to True to capture reasoning
            )
        ),
        contents=content
    )

    reasoning = None
    raw_text = ""

    for part in response.candidates[0].content.parts:
        if part.thought:
            reasoning = part.text
        if part.text:
            raw_text += part.text

    usage = response.usage_metadata
    in_tokens = usage.prompt_token_count
    # Billable output includes thoughts and candidates[cite: 2]
    out_tokens = usage.candidates_token_count + (usage.thoughts_token_count or 0)
    cost = (in_tokens * RATE_GEMINI_INPUT) + (out_tokens * RATE_GEMINI_OUTPUT)

    return raw_text.strip(), reasoning, in_tokens, out_tokens, cost


# =========================
# CORE LOGIC
# =========================

def translate(text, client, translation_model, temp, output_dir, n_translations, source_lang, target_lang,
              prompt="", summary=None, intermediate_translation=None, memes_list=None,
              memes_translation=None, schema=None):
    output_dir = Path(output_dir)
    prompts_dir = output_dir / "prompts"
    translations_dir = output_dir / "translations"
    reasoning_dir = output_dir / "reasoning"

    for d in [prompts_dir, translations_dir, reasoning_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build Prompt
    full_prompt = (f"You will perform a film subtitles translation task from {source_lang} into {target_lang}.\n"
                   f"{prompt}\n")

    if intermediate_translation:
        full_prompt += f"\nUse this intermediate translation:\n{intermediate_translation}\n"
    if memes_list:
        full_prompt += f"\nAnalyze these memes in particular; they require special attention/analysis:\n{memes_list}\n"
    if memes_translation:
        full_prompt += f"\nUse the approved meme translations:\n{memes_translation}\n"
    if summary:
        full_prompt += f"\nContext/Summary/Additional information for analysis:\n{summary}\n"
    if schema:
        full_prompt += f"\nUse this schema to analyze the listed memes:\n{schema}\n"

    content = (f"{full_prompt}\n\n"
               f"Translate the following subtitles text. Preserve time codes. "
               f"Return translation ONLY:\n\n{text}")

    all_translations = []

    for i in range(1, n_translations + 1):
        print(f"Translating (run {i}/{n_translations}) with {translation_model}...")

        if translation_model.startswith("gpt"):
            raw_output, reasoning, in_t, out_t, cost = call_gpt_translate(content, client, translation_model, temp)
        else:
            raw_output, reasoning, in_t, out_t, cost = call_gemini_translate(content, client, translation_model, temp)

        # Cleanup
        clean_output = raw_output.replace("```plaintext", "").replace("```", "").strip()

        print(f"Cost: ${cost:.4f} | In: {in_t} | Out: {out_t}")

        # Save Artifacts
        (translations_dir / f"translation-{i}.txt").write_text(clean_output, encoding="utf-8")
        (prompts_dir / f"prompt-{i}.txt").write_text(translation_model+'/n'+content, encoding="utf-8")
        if reasoning:
            (reasoning_dir / f"reasoning-{i}.txt").write_text(reasoning, encoding="utf-8")

        all_translations.append(clean_output)

    return all_translations


# =========================
# MAIN ENTRY
# =========================

if __name__ == "__main__":
    # Simplified loading logic based on evaluate.py style
    if len(sys.argv) < 6:
        print("Usage: python translate.py <input_path> <output_dir> <temp> <n_runs> <ai_type> ...")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = sys.argv[2]
    temp = float(sys.argv[3])
    n_runs = int(sys.argv[4])
    translation_model = sys.argv[5].lower()

    # Load Text
    if input_path.is_dir():
        text = "\n".join([f.read_text(encoding="utf-8") for f in sorted(input_path.iterdir()) if f.is_file()])
    else:
        text = input_path.read_text(encoding="utf-8")

    # API Setup
    if translation_model.startswith("gpt"):
        key = Path("./GreenAI-API-key.txt").read_text().strip()
        client = openai.OpenAI(api_key=key)
    elif translation_model.startswith("gemini"):
        key = Path("./gemini-personal-API-key.txt").read_text().strip()
        client = genai.Client(api_key=key)
    else:
        raise ValueError("Invalid AI type.")


    # Context file helper
    def read_optional(idx):
        if len(sys.argv) > idx:
            p = Path(sys.argv[idx])
            return p.read_text(encoding="utf-8").strip() or None
        return None


    translate(
        text, client, translation_model, temp, output_dir, n_runs, "Russian", "English",
        prompt=read_optional(6),
        summary=read_optional(7),
        intermediate_translation=read_optional(8),
        memes_list=read_optional(9),
        memes_translation=read_optional(10),
        schema=read_optional(11)
    )