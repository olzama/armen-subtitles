import sys
import openai
from chunk import count_tokens_in_text

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
    usage = response.usage
    print("Estimated cost: ${}, with {} input tokens and {} output tokens.".format(
        usage["total_tokens"] * 0.03 / 1000, usage["input_tokens"], usage["output_tokens"]))
    if output_filename:
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(clean_output)
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