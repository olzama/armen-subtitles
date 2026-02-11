import sys
from budget_estimate import track_usage_and_cost
import openai
from chunk import count_tokens_in_text
import pysrt


def translate_simple_text(text, client, source_lang, target_lang, output_filename):
    print("Translating...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "Expert in literary translation, with the focus on postmodern and metamodern literature."},
            {"role": "user",
             "content": f"You are given a text about world literature, with many references to various texts."
                        f"Translate the text from {source_lang} into {target_lang}: {text}."
                        f"Only return the translation, nothing else.\n"}
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
    subs = pysrt.open(sys.argv[1])
    text = ' '.join([sub.text for sub in subs])
    output_filename = sys.argv[2]
    with open ("./LYS-API-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    n_toks = count_tokens_in_text(text)
    print("Text token count: ", n_toks)
    #translated_text = translate_simple_text(text, client, 'en', 'ru', output_filename,)
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(text)