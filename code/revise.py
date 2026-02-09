import sys
from budget_estimate import track_usage_and_cost
import openai
from chunk import count_tokens_in_text

def revise(text, summary, narratives, prompt, client, output_filename=None):
    print("Revising...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Expert in literary text editing of postmodernist texts."},
            {"role": "user", "content": f"You are given the following subtitles text: {text} and its summary: {summary}."
                                        #f"You are also given the relevant narratives: {narratives}.\n"
                                        f"\n\n {prompt}"}
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
    with open(sys.argv[1], "r", encoding='utf-8') as f:
        text = f.read()
    with open(sys.argv[2], "r", encoding='utf-8') as f:
        summary = f.read()
    with open(sys.argv[3], "r", encoding='utf-8') as f:
        prompt = f.read()
    with open(sys.argv[5], "r", encoding='utf-8') as f:
        narratives = f.read()
    #with open(sys.argv[6], "r", encoding='utf-8') as f:
    #    specific_feedback = f.read()
    #with open(sys.argv[6], "r", encoding='utf-8') as f:
    #    original_subs = f.read()
    output_filename = sys.argv[4]
    with open ("./LYS-API-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    n_toks = count_tokens_in_text(text + summary)
    print("Text plus Summary token count: ", n_toks)
    revised_text = revise(text, summary, narratives, prompt, client, output_filename)