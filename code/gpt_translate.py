import sys
import openai
import pysrt
from process_srt import create_subtitle_mapping, txt2lines, txt2srt
from chunk import split_srt_file_with_AI, combine_lines

def translate_full_text(text, output_dir, client, prompt, source_lang='ru', target_lang='en'):
    print("Translating full text...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": f"Translate the following text from {source_lang} to {target_lang}: '{text}'"
                                        f"{prompt}"
             }
        ]
    )
    translated_text = response.choices[0].message.content.strip()
    #with open(f"{output_dir}/translated_full_text.txt", "w", encoding='utf-8') as f:
    #    f.write(translated_text)
    return translated_text


def revise_text(original_text, translated_text, client, prompt):
    print("Revising the chunk: {}".format(translated_text))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a linguistic processor for copy editing."},
            {"role": "user", "content": f"Revise the following translated text given the possibly imperfect "
                                        f"original text and the additional instructions:\n"
                                        f"Original: {original_text}\n"
                                        f"Translated: {translated_text}\nInstructions: {prompt}\n"
                                        f"Revise the translation and return only the final revised version as output.\n"}
        ]
    )
    revised_text = response.choices[0].message.content.strip()
    print("Revised text: {}".format(revised_text))
    return revised_text




def translate_with_time_codes_and_ref(subs, translation, source_lang, target_lang, client, reference_subs):
    print("Translating the chunk with time codes...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in translating text with time codes."},
            {"role": "user", "content": f"You are given: 1) original subtitles in {source_lang} in SRT format; "
             f"2) a high-quality translation of the text in {target_lang}; "
             f"3) a translation with time codes in SRT format in {target_lang} which may be of poorer quality.\n"
             f"The REQUIRED OUTPUT is a high-quality translation in SRT format with the exact same time codes as the original.\n\n"
             f": 1) Original subs: \n{subs}.\n\n"
             f"  2) Good translation for reference:\n{translation}.\n\n"
             f"  3) A (possibly not so good) translation with time codes for reference:\n{reference_subs}.\n"
             f"Important: Sometimes, the translated sentence sums up more than one line of the original text. In such cases, "
             f"you must either break the translated line into two or add an empty line corresponding the extra "
             f"line in the original text. Map the high quality translation to the SRT translation.\n"
             f"It is of ultimate importance that the translated lines do not go off track with respect to time codes, "
             f"because that kills the whole purpose of the task.\n"
             f"Return ONLY the translated text, without any additional comments or explanations."}
        ]
    )
    raw_output = response.choices[0].message.content.strip()
    # Sometimes ChatGPT returns a string prepended with "```plaintext\n" and suffixed with "\```"
    clean_output = raw_output.strip("```plaintext\n").strip("```")
    clean_output = clean_output.strip("```srt\n").strip("```")

    return clean_output

def translate_with_time_codes(subs, translation, source_lang, target_lang, N, client):
    print("Translating the chunk with time codes...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in translating text with time codes."},
            {"role": "user", "content": f"You are given: 1) original subtitles in {source_lang} in SRT format; "
             f"2) a high-quality translation of the text in {target_lang}; "
             f"The REQUIRED OUTPUT is a high-quality translation in SRT format with EXACTLY {N} lines, as in the original.\n\n"
             f": 1) Original subs with {N} lines: \n{subs}.\n\n"
             f"  2) Good translation for reference:\n{translation}.\n\n"
             f"Important: Sometimes, the translated sentence sums up more than one line of the original text. In such cases, "
             f"you must either break the translated line into two or add an empty line corresponding the extra "
             f"line in the original text. Map the high quality translation to the SRT structure as close as possible.\n"
             f"It is of ULTIMATE importance that the translated lines do not go off track with respect to time codes, "
             f"because that kills the whole purpose of the task. This means, it is BETTER to insert a blank line than "
                                        f"to use a line from the translation which is ahead of the original.\n"
             f"You must carefully compare line to line, prioritizing that the subtitles stay on track. "
             f"If it seems like you are getting off track (the subtitles get ahead or behind the original), CORRECT"
             f" by either appending the translation to the previous line, "
             f"splitting it between the current and the next line, or even inserting a blank line.\n\n"
             f"Return ONLY the translated text, without any additional comments or explanations."}
        ]
    )
    raw_output = response.choices[0].message.content.strip()
    # Sometimes ChatGPT returns a string prepended with "```plaintext\n" and suffixed with "\```"
    clean_output = raw_output.strip("```plaintext\n").strip("```")
    clean_output = clean_output.strip("```srt\n").strip("```")

    return clean_output




def translate_srt_file(input_file, output_dir, output_file, client, ref_trans=None, full_text=None,
                       full_translation=None, prompt=''):
    subs = pysrt.open(input_file)
    mapping = create_subtitle_mapping(subs)
    chunks = split_srt_file_with_AI(mapping, 1000, client, 2, 100, 5, prompt='')
    print("Split into {} chunks.".format(len(chunks)))
    #chunks = [{'mapping':mapping, 'combined_text':combined_full_text}]
    translated_lines = []
    for chunk in chunks:
        if not full_translation:
            translated_text = translate_full_text(chunk['combined_text'], output_dir, client, '', 'ru', 'en')
        else:
            translated_text = full_translation
        #revised_translation = revise_text(chunk['combined_text'], translated_text, client, prompt)
        translated_subs = translate_with_time_codes(chunk['raw_text'], translated_text, 'ru',
                                                          'eng',len(chunk['mapping']),client)
        lines = txt2lines(translated_subs, chunk['mapping'])
        combine_lines(translated_lines,lines)
    translated_srt = pysrt.SubRipFile()
    translated_srt.extend(txt2srt(translated_lines))
    translated_srt.save(output_dir+output_file, encoding='utf-8')


if __name__ == "__main__":
    input_srt = sys.argv[1]#"../data/demons/original-auto/captions demons 2.srt"
    output_dir = sys.argv[2]
    output_srt = sys.argv[3]#"../output/demons/demons-translated.srt"
    #model_name = sys.argv[4]
    full_text = None
    full_translation = None
    ref_trans = None # A reference translation with time codes (e.g. from youtube)
    prompt = ''
    if len(sys.argv) > 4:
        with open(sys.argv[4], "r", encoding='utf-8') as f:
            prompt = f.read()
    if len(sys.argv) > 5:
        with open(sys.argv[5], "r", encoding='utf-8') as f:
            ref_trans = f.read()
    if len(sys.argv) > 5:
        with open(sys.argv[6], "r", encoding='utf-8') as f:
            full_text = f.read()
        with open(sys.argv[7], "r", encoding='utf-8') as f:
            full_translation = f.read()
    with open ("./open-ai-api-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    #response = client.models.list()
    #for model_info in response.data:
    #    print(model_info.id)
    translate_srt_file(input_srt, output_dir, output_srt, client, ref_trans, full_text, full_translation, prompt)
