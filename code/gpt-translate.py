import string
import sys
import openai
import pysrt
import difflib
import re

def create_subtitle_mapping(subs):
    """
    Creates a mapping of line numbers to time codes and subtitle text.
    Returns a list of dictionaries containing index, start time, end time, and text.
    """
    mapping = []
    for sub in subs:
        mapping.append({
            "index": sub.index,
            "start": sub.start,
            "end": sub.end,
            "text": sub.text.strip()
        })
    return mapping

def combine_text(mapping):
    """
    Combines all subtitle text into one large string for AI processing.
    """
    return ' '.join([entry["text"] for entry in mapping])

def split_translation_into_lines_AI(original_mapping, translated_text, client):
    original_lines = [entry["text"] for entry in original_mapping]
    translation_lines = []
    print("Calling ChatGPT API to split the translation into lines...")
    for ln in original_lines:
        input_prompt = (f"You are an expert in subtitle formatting and translation mapping. "
                        f"Your task is to take a fully translated subtitle text {translated_text}"
                        f"and find a substring in it "
                        f"that corresponds to the original subtitle line ({ln}),"
                        f"ensuring alignment while preserving natural readability."
                        f"The original subtitles do not follow standard sentence structures, "
                        f"and subtitle breaks often occur within phrases, "
                        f"so your output should prioritize that the output is not much longer than the input."
                        f"Pay special attention to not output very long lines."
                        f"The text may contain labels such as [Music], which is always its own line."
                        f"Do not include any additional comments or anything else in the output, just the substring."
                        f"Here is an example. Suppose the full translation is: '[Music] At around midnight, "
                        f"in a small village, a group of people enter an inn.' Suppose the original subtitle lines are:"
                        f" '[музыка]'; 'В полночь'; 'в небольшой деревне'; 'группа людей входит'; 'в таверну.' The"
                        f"output must be: '[Music]'; 'At around midnight,'; 'in a small village,'; "
                        f"'a group of people enter'; 'an inn.'")
        response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in subtitle formatting and translation mapping."},
                    {"role": "user", "content": input_prompt}
                ])
        translation_lines.append(response.choices[0].message.content.strip())
    #clean_lines = remove_overlap_with_ai(translation_lines, client)
    clean_lines = reduce_overlap(translation_lines, 20)
    return clean_lines

def remove_overlap_with_ai(lines, client):
    """
    Iterates through a list of strings, comparing each with the next one and the one after.
    Uses AI to detect and remove substantial overlap while preserving readability.
    Ensures that the resulting list is of the exact same length as the input list.

    :param lines: List of strings.
    :return: List of processed strings with overlaps removed.
    """
    cleaned_lines = lines[:]
    n = len(lines)

    for i in range(n - 1):
        line_i = cleaned_lines[i]
        line_next = cleaned_lines[i + 1] if i + 1 < n else ""
        line_next_next = cleaned_lines[i + 2] if i + 2 < n else ""

        input_prompt = (
                f"Here are three consecutive lines:\n"
                f"1: {line_i}\n"
                f"2: {line_next}\n"
                f"3: {line_next_next}\n\n"
                f"Analyze the overlap between them. Remove redundant content where it least harms readability," 
                f"or remove from the longest string."
        f"Return the modified three lines, each on a separate line, keeping the list the same length."
                f"There must not be any additional comments, numbers, or anything else in the output, just the three lines."
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a linguistic processor for removing redundant text."},
                {"role": "user", "content": input_prompt}
            ]
        )
        modified_lines = response.choices[0].message.content.strip().split('\n')

        # Ensure we replace only the correct number of lines
        if len(modified_lines) == 3:
            cleaned_lines[i] = modified_lines[0]
            cleaned_lines[i + 1] = modified_lines[1]
            if i + 2 < n:
                cleaned_lines[i + 2] = modified_lines[2]
        else:
            print("Invalid response length. Skipping this set of lines.")
    return cleaned_lines

def remove_overlap_simple(lines, threshold):
    clean_lines = []
    for i in range(len(lines)-1):
        cur_ln = lines[i]
        next_ln = lines[i+1]
        overlap_indices = find_overlap_indices(cur_ln, next_ln)
        if overlap_indices and overlap_indices[1] - overlap_indices[0] > threshold:
            clean_lines.append(cur_ln[:overlap_indices[0]])
        else:
            clean_lines.append(cur_ln)
    clean_lines.append(lines[-1])
    return clean_lines

def remove_full_overlap(source, target, min_overlap=5):
    """Finds and removes a fully overlapping portion from the target text, ensuring clean splits."""
    matcher = difflib.SequenceMatcher(ignore_in_comparison, source, target)
    match = max(matcher.get_matching_blocks(), key=lambda m: m.size)
    if match.size >= min_overlap:
        return ensure_word_boundary(clean_punctuation(target[match.size:]))
    return target


def clean_punctuation(text):
    """Removes leading/trailing punctuation and ensures proper spacing."""
    return re.sub(r'^[^\w]+|[^\w]+$', '', text).strip()


def ensure_word_boundary(text):
    """Ensures that text starts at a word boundary by removing leading non-word characters."""
    return re.sub(r'^\W+', '', text).strip()


def ignore_in_comparison(c):
    """Defines characters to be ignored during sequence matching, such as whitespace and punctuation."""
    return c in string.whitespace + string.punctuation


def is_standalone_label(text):
    """Checks if a line is a standalone label, such as '[Music]' or '[Laughter]'."""
    return bool(re.match(r'^\[.*\]$', text.strip()))


def find_overlap(source, target, min_overlap=5):
    """Finds the longest overlapping substring between two texts, ignoring punctuation and spaces."""
    matcher = difflib.SequenceMatcher(ignore_in_comparison, source, target)
    match = max(matcher.get_matching_blocks(), key=lambda m: m.size)
    return match if match.size >= min_overlap else None


def remove_overlap(source, target, min_overlap=5):
    """Removes overlap from the target string, ensuring full word removal if necessary."""
    match = find_overlap(source, target, min_overlap)
    if match:
        target_cut = target[match.b + match.size:].strip()
        return ensure_word_boundary(clean_punctuation(target_cut))
    return target


def reduce_overlap(lines, min_overlap=5):
    """
    Detects and removes substantial overlap between subtitle lines while ensuring:
    - Proper word boundary preservation.
    - No punctuation artifacts.
    - No trailing single-letter artifacts from overlap removal.
    - Detects when a line overlaps with an earlier, non-adjacent line.
    - Prefers empty lines over lines with duplicated content or nonsensical word fragments.
    - Prevents removal of standalone labels like '[Music]' or '[Laughter]'.

    :param lines: List of subtitle lines.
    :param min_overlap: Minimum length of an overlap to be removed.
    :param combination_threshold: Ratio of similarity for detecting combination lines.
    :return: A new list of cleaned subtitle lines.
    """
    cleaned_lines = lines[:]
    for i in range(len(cleaned_lines) - 1):
        line_i = cleaned_lines[i]
        line_next = cleaned_lines[i + 1]
        # Skip standalone labels to ensure they are never removed
        if is_standalone_label(line_next):
            continue
        # Remove overlap with the previous line
        cleaned_lines[i + 1] = remove_overlap(line_i, line_next, min_overlap)
        # Detect and remove overlap with earlier lines (not just the immediately previous one)
        for j in range(max(0, i - 2), i):  # Check up to 2 lines before
            earlier_line = cleaned_lines[j]
            cleaned_lines[i + 1] = remove_overlap(earlier_line, cleaned_lines[i + 1], min_overlap)
    cleaned_lines = redistribute_subtitles(cleaned_lines, 40)  # Redistribute long lines
    return cleaned_lines

def redistribute_subtitles(subs, threshold):
    """
    If a line is empty and the previous line is longer than `threshold`,
    split the previous line into two parts while preserving word boundaries,
    and distribute the second part into the empty line.

    Args:
        subs (list of str): List of subtitle lines.
        threshold (int): The character length threshold for splitting.

    Returns:
        list of str: The modified list of subtitles.
    """
    result = []

    for i in range(len(subs)):
        if i > 0 and not subs[i].strip() and len(subs[i - 1]) > threshold:
            words = subs[i - 1].split()
            split_idx = len(words) // 2  # Find middle word index

            first_part = ' '.join(words[:split_idx]).strip()
            second_part = ' '.join(words[split_idx:]).strip()

            result[-1] = first_part  # Modify previous line
            result.append(second_part)  # Add to current empty line
        else:
            result.append(subs[i])

    return result

def find_overlap_indices(s1, s2):
    """
    Finds the longest overlap between two strings and returns the
    start and end indices of the overlap in both strings.

    :param s1: First string.
    :param s2: Second string.
    :return: Tuple (start_index_s1, end_index_s1, start_index_s2, end_index_s2),
             or None if there is no overlap.
    """
    max_overlap = 0
    best_indices = None
    # Try all suffixes of s1 that could match prefixes of s2
    for i in range(len(s1)):
        if s2.startswith(s1[i:]):
            overlap_length = len(s1) - i
            if overlap_length > max_overlap:
                max_overlap = overlap_length
                best_indices = (i, len(s1), 0, overlap_length)
    # Try all suffixes of s2 that could match prefixes of s1
    return best_indices if best_indices else None

def translate_line_by_line(lines, full_translation_reference, client, source_lang='ru', target_lang='en'):
    print("Translating line by line...")
    translated_lines = []
    for ln in lines:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a translator of podcast subtitles."},
                {"role": "user", "content": f"Translate the following text from {source_lang} to {target_lang}: '{ln}'."
                                            f"The line to translate may be a sentence fragment or fragments of different sentences; constituents"
                                            f"may not be respected. The translation can have the same properties,"
                                            f"but for better accuracy, you should use the following translation of the"
                                            f"full text as a reference: '{full_translation_reference}'."
                 }
            ]
        )
        translated_lines.append(response.choices[0].message.content.strip())
    return translated_lines

def translate_full_text(text, client, source_lang='ru', target_lang='en'):
    print("Translating full text...")
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": f"Translate the following text from {source_lang} to {target_lang}: '{text}'"
             }
        ]
    )
    return response.choices[0].message.content.strip()

def map_translated_text_to_timecodes(mapping, translated_lines):
    """
    Maps the translated lines back to their original time codes.
    This preserves the timing while updating the text.
    """
    mapped_subs = []
    index = 1
    for entry, translated_text in zip(mapping, translated_lines):
        mapped_subs.append(pysrt.SubRipItem(
            index=index,
            start=entry["start"],
            end=entry["end"],
            text=translated_text
        ))
        index += 1
    return mapped_subs

def insert_punctuation(text, client, source_lang='ru'):
    """
    Uses AI to insert appropriate punctuation into the combined text before translation.
    This considers only the source language structure.
    """
    print("Inserting punctuation...")
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a linguistic processor for punctuation correction."},
            {"role": "user", "content": f"Insert appropriate punctuation and capitalization into the following text, "
                                        f"considering only the structure of {source_lang}:\n{text}. Consider also"
                                        f"that this is subtitles text, so there will be labels like [Music]. "
                                        f"Also keep in mind that the name of the podcast is 'Армен и Федор', so,"
                                        f"anything that looks like ''Армена Федор' is likely a mistake "
                                        f"in the autogenerated subtitles. "
                                        f"The author's name should be spelled Armen Zakharyan and not Zakharov."}
        ]
    )
    return response.choices[0].message.content.strip()

def translate_srt_file(input_file, output_file, client):
    subs = pysrt.open(input_file)
    mapping = create_subtitle_mapping(subs)
    original_lines = [entry["text"] for entry in mapping]
    combined_text = combine_text(mapping)
    combined_text = insert_punctuation(combined_text, client)
    translated_text = translate_full_text(combined_text, client)
    #translated_lines = translate_line_by_line(original_lines, translated_text, client, source_lang='ru', target_lang='en')
    translated_lines = split_translation_into_lines_AI(mapping, translated_text, client)
    translated_subs = map_translated_text_to_timecodes(mapping, translated_lines)
    translated_srt = pysrt.SubRipFile()
    translated_srt.extend(translated_subs)
    translated_srt.save(output_file, encoding='utf-8')


if __name__ == "__main__":
    input_srt = sys.argv[1]#"../data/demons/original-auto/captions demons 2.srt"
    output_srt = sys.argv[2]#"../output/demons/demons-translated.srt"
    with open ("./open-ai-api-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    translate_srt_file(input_srt, output_srt, client)
