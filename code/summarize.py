import sys
import openai
from chunk import combine_srt_chunks
from budget_estimate import track_usage_and_cost



def summarize_text(text, prompt, client):
    print("Summarizing...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Expert in intertextuality in postmodern literature."},
            {"role": "user", "content": f"You are given the following text: {text}.\n\n {prompt}"}
        ]
    )
    raw_output = response.choices[0].message.content.strip()
    # Sometimes ChatGPT returns a string prepended with "```plaintext\n" and suffixed with "\```"
    clean_output = raw_output.strip("```plaintext\n").strip("```")
    return clean_output


if __name__ == "__main__":
    input_dir = sys.argv[1]#"../data/demons/original-auto/captions demons 2.srt"
    output_filename = sys.argv[2]
    with open(sys.argv[3], "r", encoding='utf-8') as f:
        prompt = f.read()
    with open ("./open-ai-api-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    combined_text_chunks = combine_srt_chunks(input_dir)
    summaries = []
    for text_chunk in combined_text_chunks:
        summaries.append(summarize_text(text_chunk, prompt, client))
    all_summaries = "\n".join(summaries)
    with open(output_filename, "w", encoding='utf-8') as f:
        f.write(all_summaries)
