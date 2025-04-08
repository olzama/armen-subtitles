import os
import sys
import openai
from chunk import count_tokens_in_text, combine_srt_chunks
from budget_estimate import track_usage_and_cost


def identify_narratives(text, prompt, client, summary=None, output_filename=None):
    if summary:
        prompt += (f"\n\nUse the following summary of references to perform the task better:\n\n"
                   f"{summary}\n\n")
    print("Identifying narratives...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Expert in identifying narratives from other texts in literature."},
            {"role": "user", "content": f"You are given the following subtitles text: {text}.\n\n {prompt}\n"
                                        f"Return the list of the narratives, very briefly summarized.\n"}
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
    input_dir = sys.argv[1]
    output_filename = sys.argv[2]
    with open(sys.argv[3], "r", encoding='utf-8') as f:
        prompt = f.read()
    #with open ("./open-ai-api-key.txt", "r") as myfile:
    with open("./LYS-API-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    combined_srt_chunks = ''
    for chunk in sorted(os.listdir(input_dir)):
        if chunk.endswith('.srt'):
            with open(os.path.join(input_dir, chunk), "r", encoding='utf-8') as f:
                combined_srt_chunks += f.read()
    combined_text = '\n'.join([ chunk for chunk in combine_srt_chunks(input_dir)])
    n_toks = count_tokens_in_text(combined_text)
    print("Text token count: ", n_toks)
    narratives = identify_narratives(combined_text, prompt, client, output_filename=output_filename)

