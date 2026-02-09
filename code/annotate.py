import sys
import openai
from chunk import combine_srt_chunks, count_tokens_in_text
from budget_estimate import track_usage_and_cost



def annotate_text(text, prompt, summary, client):
    print("Annotating...")
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "Expert in translating modern and postmodern literature and film."},
            {"role": "user", "content": f"You are given the following text: {text}.\n\n {prompt}"
                                        f"Consult the summary of the text to perform the task better: {summary}. "
             }
        ]
    )
    raw_output = response.choices[0].message.content.strip()
    # Sometimes ChatGPT returns a string prepended with "```plaintext\n" and suffixed with "\```"
    clean_output = raw_output.strip("```plaintext\n").strip("```")
    usage = track_usage_and_cost(response.usage, 2.5, 10, "gpt-4o")
    print("Estimated cost: ${}, with {} input tokens and {} output tokens.".format(usage["total_cost"],
                                                                                   usage["input_tokens"], usage["output_tokens"]))
    return clean_output


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_filename = sys.argv[2]
    with open(sys.argv[3], "r", encoding='utf-8') as f:
        prompt = f.read()
    with open(sys.argv[4], "r", encoding='utf-8') as f:
        summary = f.read()
    with open ("./open-ai-api-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    with open (input_file, "r", encoding='utf-8') as f:
        text = f.read()
    annotated = annotate_text(text, prompt, summary, client)
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(annotated)
