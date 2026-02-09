import sys
from budget_estimate import track_usage_and_cost
import openai
from chunk import count_tokens_in_text


def adjust(text, client, output_filename):
    print("Adjusting time codes...")
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "Expert in subtitles."},
            {"role": "user", "content": f"Make the shorter subtitle segments (less than 5-6 seconds) a bit longer; keep the start times as is, add about 1,5 seconds to the finish timestamps; allow a bit of overlap.: {text}."
                                        f"No other changes. Return the adjusted text only, without any comments.\n"}
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
    output_filename = sys.argv[2]
    with open ("./open-ai-api-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    n_toks = count_tokens_in_text(text)
    print("Text plus Summary token count: ", n_toks)
    translated_text = adjust(text, client, output_filename)