import string
import sys
import openai
import pysrt
import difflib
import re
import tiktoken
from openai import api_key


def count_tokens_in_text(text, model="gpt-4-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def split_srt_file(mapping, input_text, max_tokens):
    chunks = []
    original_lines = [entry["text"] for entry in mapping]
    tokens = count_tokens_in_text(input_text, model='gpt-4-turbo')
    if tokens <= max_tokens:
        return [{"mapping": mapping, "combined_text": input_text}]
    else:
        chunk = {"mapping": [], "combined_text": ''}
        chunk_tokens = 0
        for i, line in enumerate(original_lines):
            chunk_tokens += count_tokens_in_text(line)
            if chunk_tokens <= max_tokens:
                chunk['mapping'].append(mapping[i])
            else:
                chunks.append(chunk)
                chunk = [mapping[i]]
                chunk_tokens = count_tokens_in_text(line)
        chunks.append(chunk)
        return chunks

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



def discard_model_explanation(
            substring: str,
            snippet: str,
            max_length: int = 100,
            sim_threshold: float = 0.2
    ) -> str:
        """
        Returns an empty string if:
          1) substring is longer than max_length, AND
          2) substring's similarity ratio to snippet is less than sim_threshold.
        Otherwise returns substring as-is.

        :param substring: The candidate substring returned by GPT.
        :param snippet: The text from which substring was supposedly taken.
        :param max_length: Maximum allowable length before we consider it 'too long'.
        :param sim_threshold: Minimum similarity ratio to snippet for acceptance.
        :return: substring or "" if conditions fail.
        """
        if len(substring.strip()) > max_length:
            ratio = difflib.SequenceMatcher(None, substring.strip(), snippet).ratio()
            if ratio < sim_threshold:
                print("Line too long and dissilimar to input:")
                print(substring)
                return ""
        return substring


def find_substring_for_line(line, snippet, client, line_threshold=0.9):
    """
    Same logic as your function:
    - GPT tries to pick a substring from 'snippet' that matches 'line'.
    - We do a fuzzy check (SequenceMatcher) to confirm it's found with ratio >= sim_threshold.
    - Returns (found_text, ratio).
    """
    prompt = (
        f"Below is a chunk of text:\n{snippet}\n\n"
        f"Find a substring in it that best corresponds to the original line:\n{line}\n\n"
        f"Constraints:\n"
        f"1. Return only the substring (no quotes, disclaimers, etc.).\n"
        f"2. The substring must come verbatim from the chunk.\n"
        f"3. NEVER return any comments, reasoning, or anything at all except the verbatim substring.\n"
        f"4. If no match is found, return an empty line.\n"

    )

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_output = response.choices[0].message.content.strip()
    if not raw_output:
        return "", 0.0

    # fuzzy check
    snippet_lower = snippet.lower()
    candidate_lower = raw_output.lower()
    best_ratio = 0.0

    if len(candidate_lower) <= len(snippet_lower):
        for start_idx in range(len(snippet_lower) - len(candidate_lower) + 1):
            window = snippet_lower[start_idx:start_idx + len(candidate_lower)]
            ratio = difflib.SequenceMatcher(ignore_in_comparison, candidate_lower, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                if best_ratio >= line_threshold:
                    break

    if best_ratio >= line_threshold:
        return raw_output, best_ratio
    else:
        return "", best_ratio


def process_batch_of_lines(
        lines_batch,
        words,
        pointer,
        chunk_size,
        client,
        line_threshold=0.9,
        step_back=5,
        step_forward=10,
        max_line_retries=3
):
    """
    Processes a batch of lines with the initial snippet from [pointer : pointer+chunk_size].

    For each line in the batch:
      - We do up to max_line_retries attempts to find a substring:
        1) Build snippet
        2) find_substring_for_line
        3) If found => pointer += small_step_forward, store matched substring
        4) If not found => expand snippet by 'step_forward' words, increment retry
           (We also do pointer -= step_back if you want partial fallback.)
      - If after max_line_retries, still no match => store "" and do pointer fallback
        pointer -= step_back, pointer += step_forward (or keep the snippet expanded).

    Returns:
     - matched_list: the matched substring for each line
     - pointer: final pointer after entire batch
     - snippet_text: the text used for combined ratio
    """
    n = len(words)
    # We'll keep track of the snippet end separately, so we can expand it
    snippet_end = min(pointer + chunk_size, n)
    snippet_words = words[pointer:snippet_end]
    snippet_text = " ".join(snippet_words)

    matched_list = []

    for line in lines_batch:
        # We'll do up to max_line_retries attempts if no match
        line_matched = False
        line_substring = ""
        line_retries = 0

        while line_retries < max_line_retries and not line_matched:
            # Rebuild snippet based on current snippet_end
            snippet_words = words[pointer:snippet_end]
            snippet_text = " ".join(snippet_words)

            found_substring, ratio = find_substring_for_line(line, snippet_text, client, line_threshold)
            if ratio >= line_threshold:
                line_matched = True
                line_substring = found_substring
            else:
                # No match: expand snippet and optionally step pointer
                # pointer fallback:
                pointer = max(0, pointer - step_back)
                # expand snippet by step_forward words
                snippet_end = min(snippet_end + step_forward, n)
                line_retries += 1

        if line_matched:
            matched_list.append(line_substring)
        else:
            # after max_line_retries, no match
            matched_list.append("")
            # do final pointer fallback if we want
            pointer = max(0, pointer - step_back)
            pointer = min(pointer + step_forward, n)

    # snippet_text returned is from the last built snippet
    return matched_list, pointer, snippet_text


def split_translation_into_lines_AI(
    original_lines,
    translated_text,
    client,
    batch_size=3,
    chunk_size=60,
    line_threshold=0.9,
):
    """
    We handle lines in batches.
    For each batch:
      - process lines individually => pointer moves a bit for each line success/fail
      - compute combined ratio => if >= combined_threshold => pointer += 5
    """
    words = translated_text.split()
    n = len(words)
    pointer = 0
    results = []

    idx = 0
    while idx < len(original_lines):
        lines_batch = original_lines[idx : idx + batch_size]

        # 1) process the batch
        matched_list, pointer, snippet_text = process_batch_of_lines(
            lines_batch, words, pointer, chunk_size, client, line_threshold
        )

        # 2) measure combined ratio
        combined_str = " ".join(x for x in matched_list if x)
        combined_str = remove_largest_common_substring(combined_str, 2)
        snippet_lower = snippet_text.lower()
        combined_lower = combined_str.lower()
        combined_ratio = difflib.SequenceMatcher(ignore_in_comparison, combined_lower, snippet_lower).ratio()

        # if combined_ratio >= combined_threshold => pointer += e.g. 5
        matched_words_count = len(combined_str.split(' '))
        dynamic_step = int((matched_words_count * combined_ratio) / 2) + 1
        pointer = min(pointer + dynamic_step, n)

        # store matched_list in results
        results.extend(matched_list)
        idx += batch_size

    # fill leftover if needed
    while len(results) < len(original_lines):
        results.append("")

    return results


def remove_largest_common_substring(text: str, n: int) -> str:
    """
    Finds the largest repeated substring (in terms of word count) in 'text'.
    If its length > n words, removes the second occurrence of that substring.
    Otherwise, leaves the text unchanged.

    :param text: The input text from which we remove a repeated substring if it's too long.
    :param n: The minimum number of words for a substring to be considered too long.
    :return: The text with the second occurrence of the largest repeated substring (longer than n words) removed.
    """
    words = text.split()
    length = len(words)

    # Track the best repeated substring info
    best_len = 0
    best_i = 0   # start index of first occurrence
    best_j = 0   # start index of second occurrence

    # Naive approach: For each pair (i, j), find how many consecutive words match
    # i < j to ensure the substring is repeated in a separate place
    for i in range(length):
        for j in range(i+1, length):
            # Match consecutive words starting at i and j
            match_len = 0
            while (i + match_len < length and
                   j + match_len < length and
                   words[i + match_len] == words[j + match_len]):
                match_len += 1

            # If we found a new best, store it
            if match_len > best_len:
                best_len = match_len
                best_i = i
                best_j = j

    # If the longest repeated substring is longer than n
    # remove the second occurrence from the text
    if best_len > n:
        # Remove words[best_j : best_j + best_len]
        new_words = words[:best_j] + words[best_j + best_len:]
        return " ".join(new_words)
    else:
        # Nothing to remove
        return text


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
    print("Reducing overlap...")
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
    cleaned_lines = remove_combined_lines(cleaned_lines, 80, 5, 0.8)  # Remove combined lines
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
    print("Redistributing subtitles if there's empty lines...")
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


def remove_combined_lines(lines, length_threshold=80, N=5, similarity_threshold=0.8):
    """
    Removes any line longer than 'length_threshold' if it is highly similar
    (>= similarity_threshold) to the combination of up to N neighboring lines
    (previous or subsequent). The combination can involve fewer than N lines,
    e.g., 1..N lines if needed.

    :param lines: List of strings (subtitle lines, for example).
    :param length_threshold: Only consider removing lines whose length exceeds this threshold.
    :param N: Max number of neighboring lines to consider for combination (previous or subsequent).
    :param similarity_threshold: The ratio above which we consider two texts 'similar'.
    :return: A new list of lines with certain lines removed.
    """
    cleaned = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # If the line is not long enough, just keep it
        if len(line) <= length_threshold:
            cleaned.append(line)
            i += 1
            continue
        # Build combos of up to N previous lines
        remove_line = False
        for k in range(1, N + 1):
            # Start index for previous lines
            start_idx = max(0, i - k)
            # The combination of k previous lines
            prev_combination = " ".join(lines[start_idx:i])
            if prev_combination:
                ratio_prev = difflib.SequenceMatcher(None, line, prev_combination).ratio()
                if ratio_prev >= similarity_threshold:
                    remove_line = True
                    break
            # Also build combos of k subsequent lines if we haven't decided to remove yet
            if not remove_line:
                end_idx = min(len(lines), i + k + 1)
                next_combination = " ".join(lines[i+1:end_idx])
                if next_combination:
                    ratio_next = difflib.SequenceMatcher(None, line, next_combination).ratio()
                    if ratio_next >= similarity_threshold:
                        remove_line = True
                        break
        if remove_line:
            # Skip adding this line to 'cleaned'
            i += 1
        else:
            cleaned.append(line)
            i += 1
    return cleaned

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

def translate_full_text(text, output_dir, client, source_lang='ru', target_lang='en'):
    print("Translating full text...")
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": f"Translate the following text from {source_lang} to {target_lang}: '{text}'"
             }
        ]
    )
    translated_text = response.choices[0].message.content.strip()
    with open(f"{output_dir}/translated_full_text.txt", "w", encoding='utf-8') as f:
        f.write(translated_text)
    return translated_text

def map_translated_text_to_timecodes(mapping, translated_lines):
    """
    Maps the translated lines back to their original time codes.
    This preserves the timing while updating the text.
    """
    mapped_subs = []
    print("Mapping translated text to time codes...")
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

def insert_punctuation(text, output_dir, client, source_lang='ru'):
    """
    Uses AI to insert appropriate punctuation into the combined text before translation.
    This considers only the source language structure.
    """
    print("Inserting punctuation...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a linguistic processor for copy editing."},
            {"role": "user", "content": f"Insert appropriate punctuation and capitalization into the following text, "
                                        f"considering only the structure of {source_lang}:\n{text}. Consider also"
                                        f"that this is subtitles text, so there will be labels like [Музыка]. "
                                        f"Also keep in mind that the name of the podcast is 'Армен и Федор', so,"
                                        f"anything that looks like ''Армена Федор' is likely a mistake "
                                        f"in the autogenerated subtitles."}
        ]
    )
    text = response.choices[0].message.content.strip()
    with open(f"{output_dir}/text_with_punctuation.txt", "w", encoding='utf-8') as f:
        f.write(text)
    return text

def translate_srt_file(input_file, output_dir, output_file, client, full_text=None, full_translation=None):
    subs = pysrt.open(input_file)
    mapping = create_subtitle_mapping(subs)
    combined_full_text = combine_text(mapping)
    #chunks = split_srt_file(mapping, combined_full_text,3000)
    chunks = [{'mapping':mapping, 'combined_text':combined_full_text}]
    translated_lines = []
    translated_subs = []
    for chunk in chunks:
        if not full_text:
            chunk['combined_text'] = insert_punctuation(chunk['combined_text'], output_dir, client)
        else:
            chunk['combined_text'] = full_text
        if not full_translation:
            translated_text = translate_full_text(chunk['combined_text'], output_dir, client)
        else:
            translated_text = full_translation
        translated_lines.extend(split_translation_into_lines_AI(chunk['mapping'], translated_text, client))
        translated_subs.extend(map_translated_text_to_timecodes(chunk['mapping'], translated_lines))
    translated_srt = pysrt.SubRipFile()
    translated_srt.extend(translated_subs)
    translated_srt.save(output_dir+output_file, encoding='utf-8')


if __name__ == "__main__":
    input_srt = sys.argv[1]#"../data/demons/original-auto/captions demons 2.srt"
    output_dir = sys.argv[2]
    output_srt = sys.argv[3]#"../output/demons/demons-translated.srt"
    #model_name = sys.argv[4]
    full_text = None
    full_translation = None
    if len(sys.argv) > 4:
        with open(sys.argv[4], "r", encoding='utf-8') as f:
            full_text = f.read()
        with open(sys.argv[5], "r", encoding='utf-8') as f:
            full_translation = f.read()
    with open ("./open-ai-api-key.txt", "r") as myfile:
        openai_key = myfile.read().replace('\n', '')
    client = openai.OpenAI(api_key=openai_key)
    translate_srt_file(input_srt, output_dir, output_srt, client, full_text, full_translation)
