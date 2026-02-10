import sys
from budget_estimate import track_usage_and_cost
import openai
from chunk import count_tokens_in_text
import httpx
from summarize import summarize_text
from pathlib import Path


def translate_parts(text_parts, client, output_filename, source_lang, target_lang):
    end_of_p1 = '906'
    summarization_prompt = ("Provide a concise summary of the subtitles to help with coherent creative translation "
                            "in parts. Pay attention to narratives, character traits, intertextual references, "
                            "and other subtle things. You may want to create a coherence bridgee at the breaking point "
                            "between the parts; the parts are identified by the following segment numbers: "
                            "end of part 1: "
                            + end_of_p1 + ".")
    whole_text = "\n".join(text_parts)
    whole_translation = ""
    summary = summarize_text(whole_text, summarization_prompt , client)
    with open("summary_for_translation.txt", "w", encoding='utf-8') as f:
        f.write(summary)
    for text in text_parts:
        print("Translating...")
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "Expert in subtitles translation."},
                {"role": "user", "content": f"Translate the following subtitles text from {source_lang} into {target_lang}: {text}."
                                            f"Consult the summary of the text to perform the task better: {summary}.\n\n"
                                            f"Preserve the time codes. Return the translation only, without any comments.\n"}
            ]
        )
        raw_output = response.choices[0].message.content.strip()
        # Sometimes ChatGPT returns a string prepended with "```plaintext\n" and suffixed with "\```"
        clean_output = raw_output.strip("```plaintext\n").strip("```")
        whole_translation += clean_output + "\n"
        usage = track_usage_and_cost(response.usage, 2.5, 10, "gpt-4o")
        print("Estimated cost: ${}, with {} input tokens and {} output tokens.".format(usage["total_cost"],
                                                                                       usage["input_tokens"],
                                                                                       usage["output_tokens"]))
    if output_filename:
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(whole_translation)
    return whole_translation


def translate(text, summary, prompt, client, output_filename, source_lang, target_lang, english_translation=None,
              memes_translation=None):
    if english_translation:
        prompt += (f"\n\n Use the following English translation to perform the task better:\n{english_translation}\n\n")
    if memes_translation:
        prompt += (f"\n\n Make sure to use the following already approved translations of specific memes, jokes, etc.:\n{memes_translation}\n\n")
    print("Translating...")
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "Expert in subtitles translation."},
            {"role": "user", "content": f"Translate the following subtitles text from {source_lang} into {target_lang}: {text}."
                                        f"\n{prompt}\n"
                                        f"Maske sure to consult the detailed summary to perform the task better: {summary}. "
                                        f"Pay special attention to memes and jokes, etc. For each example of humor etc., mentioned "
                                        f"in the summary, provide an appropriately creative translation that conveys tone, irony, rhyme, etc.\n\n"
                                        f"Preserve the time codes. Return the translation only, without any comments.\n"}
        ]
    )
    raw_output = response.choices[0].message.content.strip()
    # Sometimes ChatGPT returns a string prepended with "```plaintext\n" and suffixed with "\```"
    clean_output = raw_output.strip("```plaintext\n").strip("```")
    usage = track_usage_and_cost(response.usage, 2.5, 10, "gpt-4o")
    print("Estimated cost: ${}, with {} input tokens and {} output tokens.".format(usage["total_cost"],
                                                                                   usage["input_tokens"],
                                                                                   usage["output_tokens"]))
    if output_filename:
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(clean_output)
    return clean_output

if __name__ == "__main__":
    text_parts = []
    # Text may come in parts as files in a folder:
    # Iterate over files in a folder
    if not sys.argv[1].endswith('.srt'):
        input_dir = Path(sys.argv[1])
        for part in input_dir.iterdir():
            if part.is_file():
                with open(part, "r", encoding='utf-8') as f:
                    text = f.read()
            text_parts.append(text)
    else:
        with open(sys.argv[1], "r", encoding='utf-8') as f:
            text = f.read()
    with open(sys.argv[2], "r", encoding='utf-8') as f:
        summary = f.read()
    print(summary)
    with open(sys.argv[3], "r", encoding='utf-8') as f:
        prompt = f.read()
    output_filename = sys.argv[4]
    if len(sys.argv) > 5:
        with open(sys.argv[5], "r", encoding='utf-8') as f:
            english_translation = f.read()
    else:
        english_translation = None
    if len(sys.argv) > 6:
        with open(sys.argv[6], "r", encoding='utf-8') as f:
            memes_translation = f.read()
    else:
        memes_translation = None
    with open ("./open-ai-api-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key, timeout=httpx.Timeout(
                connect=10.0,
                read=600.0,
                write=30.0,
                pool=10.0,
            ))
    #n_toks = count_tokens_in_text('/' + summary)
    #print("Text plus Summary token count: ", n_toks)
    translated_text = translate(text, summary, prompt, client,
                                output_filename, 'Russian', 'English',
                                None, memes_translation)
    #translated_text = translate_parts(text_parts, client, output_filename, 'English', 'Spanish')