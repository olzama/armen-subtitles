import os
import sys
import openai
from chunk import count_tokens_in_text, combine_srt_chunks
from budget_estimate import track_usage_and_cost

def identify_errors(text, prompt, client, summary=None, output_filename=None):
    if summary:
        prompt += (f"\n\nIMPORTANT: Often, the transcription errors are due to complex references in the text, which"
                   f"the transcription model did not expect. Use the following summary to identify such errors better:\n\n"
                   f"{summary}\n\n")
    print("Identifying transcription errors...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Expert in correcting audiotranscription errors in complex texts about literature."},
            {"role": "user", "content": f"You are given the following subtitles text: {text}.\n\n {prompt}"}
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

def correct_errors(text, prompt, client, summary=None, narratives=None, output_filename=None, original_subs=None):
    if summary:
        prompt += (f"\n\nIMPORTANT: Often, the transcription errors are due to complex references in the text, which"
                   f"the transcription model did not expect. Use the following summary to identify such errors better:\n\n"
                   f"{summary}\n\n")
    if narratives:
        prompt += (f"\n\nIMPORTANT: Use the following list of narratives to identify any subtle errors better:\n\n"
                   f"{narratives}\n\n Make sure to check that the text correctly states the facts from the narratives "
                   f"(who did what to whom, who said what). If there are mismatches, look for possible errors due to"
                   f"a misheard foreign word which later caused a misunderstanding.\n\n")
    if original_subs:
        prompt += (f"\n\nIMPORTANT: Use the following original subtitles for reference. "
                   f"They contain all the original errors, but can be helpful to identify missing acoustic cues.\n\n"
                   f"{original_subs}\n\n")
    print("Correcting transcription errors...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Expert in correcting audiotranscription errors."},
            {"role": "user", "content": f"You are given the following subtitles text: {text}.\n\n {prompt}"
                                        f"ONLY return the improved text WITH TIMECODES INTACT, and nothing else.\n"}
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
    with open(sys.argv[2], "r", encoding='utf-8') as f:
        summary = f.read()
    output_filename = sys.argv[3]
    with open(sys.argv[4], "r", encoding='utf-8') as f:
        prompt = f.read()
    with open(sys.argv[5], "r", encoding='utf-8') as f:
        narratives = f.read()
    with open(sys.argv[6], "r", encoding='utf-8') as f:
        original_subs = f.read()
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
    n_toks = count_tokens_in_text(combined_srt_chunks + summary + narratives)
    print("Subtitle plus Summary token count: ", n_toks)
    #errors = identify_errors(combined_srt_chunks, prompt, client, summary, output_filename)
    #narratives = identify_narratives(combined_text, prompt, client, output_filename=output_filename)
    improved_text = correct_errors(combined_srt_chunks, prompt, client, summary=summary, narratives=narratives,
                                   output_filename=output_filename, original_subs=original_subs)

