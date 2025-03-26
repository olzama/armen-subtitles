import string
import sys
import openai
import pysrt
import difflib
import re
import tiktoken
import json
import stanza



def count_tokens_in_text(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def find_breakpoint_with_api(prev_context, next_context, client):
    """
    Ask ChatGPT to determine the optimal breakpoint between previous and next context lines.

    :param prev_context: List of subtitle lines at the end of the current chunk.
    :param next_context: List of subtitle lines at the start of the next chunk.
    :param client: OpenAI API client.
    :return: Optimal index in prev_context to split at (inclusive).
    """
    prompt = (
        "You're given two sets of subtitle lines: a previous context and a following context.\n"
        "Choose the best line index (0-based, counting in the previous context) after which to split, "
        "so as not to break sentences or ideas abruptly. The break happens immediately after this line.\n"
        "Keep in mind that the text is subtitles, so it misses almost all punctuation and much of capitalization.\n\n"
        "Previous context:\n"
    )

    for idx, line in enumerate(prev_context):
        prompt += f"{idx}: {line}\n"

    prompt += "\nFollowing context:\n"
    for idx, line in enumerate(next_context):
        prompt += f"{idx}: {line}\n"

    prompt += "\nReturn only the chosen index from the previous context (an integer)."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": "You determine optimal breakpoints in subtitles, avoiding mid-sentence splits."},
                {"role": "user", "content": prompt}
            ]
        )
        index_str = response.choices[0].message.content.strip()
        index = int(index_str)

        if 0 <= index < len(prev_context):
            return index
        else:
            return len(prev_context) - 1

    except Exception as e:
        print(f"API call failed: {e}")
        return len(prev_context) - 1

def normalize(text):
    """
    Normalize text by removing punctuation, converting to lowercase, and reducing whitespace.
    """
    return re.sub(r'\W+', ' ', text).lower().strip()

def normalize_word(word):
    return word.lower().strip(string.punctuation)

def tokenize_with_sentences(text, nlp):
    """
    Tokenizes text into sentences and words using Stanza.
    Returns:
        tokens: list of (normalized_word, start_char, end_char, sentence_id)
        sentences: list of (start_char, end_char)
    """
    doc = nlp(text)
    tokens = []
    sentences = []
    for sid, sentence in enumerate(doc.sentences):
        sentence_start = sentence.tokens[0].start_char
        sentence_end = sentence.tokens[-1].end_char
        sentences.append((sentence_start, sentence_end))
        for token in sentence.tokens:
            word = normalize_word(token.text)
            tokens.append((word, token.start_char, token.end_char, sid))
    return tokens, sentences


def tokenwise_fuzzy_ratio(tokens1, tokens2):
    """
    Computes the average token-level fuzzy similarity ratio between two token lists.
    Assumes tokens1 and tokens2 are of equal length.
    """
    if len(tokens1) != len(tokens2) or len(tokens1) == 0:
        return 0.0

    scores = [
        difflib.SequenceMatcher(None, t1, t2).ratio()
        for t1, t2 in zip(tokens1, tokens2)
    ]
    return sum(scores) / len(scores)


def find_approximate_substring(hay_tokens, needle, nlp, threshold=0.9):
    """
    Finds approximate match of `needle` inside `haystack` using per-token fuzzy similarity.
    Returns: (start_char, end_char, sentence_id) or (-1, -1, -1) if no match.
    """
    needle_tokens, _ = tokenize_with_sentences(needle, nlp)

    hay_words = [w for w, _, _, _ in hay_tokens if w]
    needle_words = [w for w, _, _, _ in needle_tokens if w]
    needle_len = len(needle_words)

    if needle_len == 0 or len(hay_words) < needle_len:
        return -1, -1, -1

    best_ratio = 0.0
    best_span = (-1, -1)

    for i in range(len(hay_words) - needle_len + 1):
        window = hay_words[i:i + needle_len]
        ratio = tokenwise_fuzzy_ratio(needle_words, window)
        if ratio > best_ratio:
            best_ratio = ratio
            best_span = (i, i + needle_len)
            if ratio >= threshold:
                break  # early exit if good enough match found

    if best_ratio >= threshold:
        start_idx, end_idx = best_span
        start_char = hay_tokens[start_idx][1]
        end_char = hay_tokens[end_idx - 1][2]
        sentence_id = hay_tokens[start_idx][3]
        return start_char, end_char, sentence_id

    return -1, -1, -1

def split_srt_file_with_AI(mapping, max_tokens, client, n_overlap=2, flex_tokens=50, context_size=5):
    """
    Splits subtitles intelligently using the ChatGPT API with context on both sides of the breakpoint.

    :param mapping: List of subtitle entries.
    :param max_tokens: Approximate token limit per chunk.
    :param client: OpenAI API client.
    :param n_overlap: Lines to overlap between consecutive chunks.
    :param flex_tokens: Allowed margin around token limit.
    :param context_size: Number of context lines before and after breakpoint.
    """
    chunks = []
    current_chunk = {'mapping': [], 'combined_text': ''}
    chunk_tokens = 0
    i = 0
    total_entries = len(mapping)
    nlp = stanza.Pipeline(lang='ru', processors='tokenize')

    while i < total_entries:
        entry = mapping[i]
        line_tokens = count_tokens_in_text(str(entry), model='gpt-4o')
        tentative_total = chunk_tokens + line_tokens

        if tentative_total > (max_tokens + flex_tokens) and current_chunk['mapping']:
            next_context = [mapping[j] for j in range(i, min(i + context_size, total_entries))]
            improved_chunk = improve_original_chunk(current_chunk, next_context, nlp, client)
            chunks.append(improved_chunk)

            if i >= total_entries - n_overlap:
                # No more entries left for another chunk (only overlap remains) → break out
                return chunks
            overlap_entries = improved_chunk['mapping'][-n_overlap:] \
                if n_overlap <= len(improved_chunk['mapping']) else improved_chunk['mapping']
            current_chunk = {'mapping': overlap_entries.copy(), 'combined_text': ''}
            chunk_tokens = sum(count_tokens_in_text(str(e), model='gpt-4o') for e in current_chunk['mapping'])
        else:
            current_chunk['mapping'].append(entry)
            chunk_tokens += line_tokens
            i += 1

    # Add final chunk if any content remains
    if current_chunk['mapping']:
        current_chunk = improve_original_chunk(current_chunk, [], nlp, client)
        chunks.append(current_chunk)

    return chunks

def add_overlap_to_chunks(chunks, n_overlap):
    """
    Adds overlap to each chunk in a list of chunks.

    :param chunks: List of chunk dictionaries.
    :param n_overlap: Number of lines to overlap between consecutive chunks.
    :return: List of chunk dictionaries with overlap added.
    """
    chunks_with_overlap = []
    for i, chunk in enumerate(chunks):
        overlap_entries = chunk['mapping'][-n_overlap:] if n_overlap <= len(chunk['mapping']) else chunk['mapping']
        chunk_with_overlap = {'mapping': overlap_entries.copy(), 'combined_text': ''}
        chunk_with_overlap['combined_text'] = ' '.join(e["text"] for e in chunk_with_overlap['mapping'])
        chunks_with_overlap.append(chunk_with_overlap)
    return chunks_with_overlap

def split_srt_file(mapping, max_tokens, n_overlap=2):
    chunks = []
    current_chunk = {'mapping': [], 'combined_text': ''}
    chunk_tokens = 0
    i = 0
    total_entries = len(mapping)

    while i < total_entries:
        entry = mapping[i]
        line_tokens = count_tokens_in_text(str(entry), model='gpt-4o')

        if chunk_tokens + line_tokens > max_tokens and current_chunk['mapping']:
            # Finish current chunk
            current_chunk['combined_text'] = ' '.join(e["text"] for e in current_chunk['mapping'])
            chunks.append(current_chunk)

            # Start new chunk with overlap
            overlap_entries = current_chunk['mapping'][-n_overlap:] if n_overlap <= len(current_chunk['mapping']) else current_chunk['mapping']
            current_chunk = {'mapping': overlap_entries.copy(), 'combined_text': ''}
            chunk_tokens = sum(count_tokens_in_text(e["text"], model='gpt-4o') for e in overlap_entries)

        else:
            # Add current entry to chunk
            current_chunk['mapping'].append(entry)
            chunk_tokens += line_tokens
            i += 1

    # Add the final chunk
    if current_chunk['mapping']:
        current_chunk['combined_text'] = ' '.join(e["text"] for e in current_chunk['mapping'])
        chunks.append(current_chunk)

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

def combine_lines_with_mapping(lines, separator=' '):
    """
    Combines a list of lines into a single text,
    keeping track of the start and end indices of each original line.

    :param lines: list of text lines
    :param separator: character used to join lines (default: space)
    :return: tuple of (combined_text, mapping)
             mapping: list of dicts with 'line', 'start_idx', 'end_idx'
    """
    combined_text = ''
    mapping = []
    current_idx = 0

    for i,line in enumerate(lines):
        start_idx = current_idx
        combined_text += line['text']
        current_idx += len(line['text'])
        end_idx = current_idx

        mapping.append({
            'line idx': i,
            'start idx': start_idx,
            'end idx': end_idx
        })

        combined_text += separator
        current_idx += len(separator)

    combined_text = combined_text.rstrip(separator)  # Remove trailing separator
    return combined_text, mapping

def improve_substring_for_line(line, snippet, client):
    prompt = (
        f"Below is a chunk of text:\n{snippet}\n\n"
        f"Revise the following line:\n{line}\n\n"
        f"If the line is seems to be missing a main verb IN THE MIDDLE"
        f" or some other important word, insert a word that makes sense.\n"
        f"Use the snippet for reference.\n"
        f"Constraints:\n"
        f"1. Return only the improved or unchanged line (no quotes, disclaimers, etc.).\n"
        f"2. You CANNOT add more than 2-3 words.\n"
        f"3. DO NOT CHANGE lines which are grammatical and make sense, even if they seem like fragments.\n"
        f"3. NEVER return any comments, reasoning, or anything at all except the revised line.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_output = response.choices[0].message.content.strip()
    if raw_output != line:
        print("Changing '{}' to '{}'".format(line, raw_output))
    else:
        print("No change to line '{}'.".format(line))
    return raw_output

def find_matching_translated_lines(chunk, snippet, client, line_threshold=0.9, prompt_refinement=""):
    lines = chunk['mapping']
    combined_lines = chunk['combined_text']
    prompt = (
        f"You are given a batch of subtitle lines with time codes etc.:\n{lines}\n\n"
        f"Also, here is an improved text with punctuation and capitalization, corresponding to the lines:\n{combined_lines}\n"
        f"This improved text can help you with the task.\n\n"
        f"Below is the translation of the above text:\n{snippet}\n\n"
        f"1. I need to split the translation into the same time codes that are found in the original lines.\n"
        f"2. The translated lines must come verbatim from the already provided translation.\n"
        f"3. NEVER return any comments, reasoning, or anything at all except the verbatim chunk.\n"
        f"4. Keep the time codes as in the original.\n"
        f"5. Finally, return a string that can be directly parsed into a JSON list by calling json.loads().\n"
        f"{prompt_refinement}\n"

    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    raw_output = response.choices[0].message.content.strip()
    json_output = raw_output.strip("```json\n").strip("```")
    output_list = json.loads(json_output)
    return output_list

def find_substring_for_line(line, snippet, client, line_threshold=0.9, prompt_refinement=""):
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
        f"{prompt_refinement}\n"

    )

    response = client.chat.completions.create(
        model="gpt-4o",
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


def process_batch_by_line(
        lines_batch,
        words,
        pointer,
        chunk_size,
        client,
        line_threshold=0.9,
        step_back=5,
        step_forward=10,
        max_line_retries=3,
        allowed_length_diff = 10
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
    original_lines = [line['text'] for line in lines_batch]
    for line in original_lines:
        print("Processing line '{}'".format(line))
        # We'll do up to max_line_retries attempts if no match
        line_matched = False
        line_substring = ""
        line_retries = 0
        prompt_refinement = ''

        while line_retries < max_line_retries and not line_matched:
            # Rebuild snippet based on current snippet_end
            snippet_words = words[pointer:snippet_end]
            snippet_text = " ".join(snippet_words)

            found_substring, ratio = find_substring_for_line(line, snippet_text, client, line_threshold, prompt_refinement)
            print("Got '{}' with similarity ratio {}".format(found_substring, ratio))
            if found_substring and len(found_substring.split(' ')) > 3:
                found_substring = improve_substring_for_line(found_substring, snippet_text, client)
            len_diff = len(found_substring.split(' ')) - len(line.split(' '))
            if ratio >= line_threshold and len_diff <= allowed_length_diff:
                line_matched = True
                line_substring = found_substring
            else:
                print("Trying again, attempt {}...".format(line_retries))
                # Expand snippet if it is not too big already
                if snippet_end - pointer < 200:
                    pointer = max(0, pointer - step_back)
                    snippet_end = min(snippet_end + step_forward, n)
                line_threshold -= 0.02  # relax threshold a bit
                line_retries += 1
                if len_diff > allowed_length_diff:
                    prompt_refinement = ("\nLast time, you retrieved too long of a substring. "
                                         "Make sure to allow fragments and do not output long lines.")

        if line_matched:
            matched_list.append(line_substring)
        else:
            # after max_line_retries, no match
            matched_list.append("")
    # snippet_text returned is from the last built snippet
    return matched_list, pointer, snippet_text

def process_batch_of_lines(
        chunk,
        words,
        pointer,
        chunk_size,
        client,
        line_threshold=0.9
):
    n = len(words)
    # We'll keep track of the snippet end separately, so we can expand it
    snippet_end = min(pointer + chunk_size, n)
    snippet_words = words[pointer:snippet_end]
    snippet_text = " ".join(snippet_words)
    print("Processing text: '{}'".format(snippet_text))
    matched_list = find_matching_translated_lines(chunk, client, line_threshold)
    return matched_list, pointer, snippet_text

def process_whole_chunk(chunk, translated_text, chunk_size, client):
    words = translated_text.split()
    pointer = 0
    results = []
    # split the chunk into parts:
    #chunk_parts = [original_lines[i:i + chunk_size] for i in range(0, len(original_lines), chunk_size)]
    matched_list, pointer, snippet_text = process_batch_of_lines(chunk, words, pointer, chunk_size, client)
    results.extend(matched_list)
    return results

def split_translation_into_lines_AI(
    original_lines,
    translated_text,
    client,
    batch_size=10,
    chunk_size=100,
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
        snippet_lower = remove_punctuation(snippet_text.lower())
        combined_lower = remove_punctuation(combined_str.lower())
        #ms = difflib.SequenceMatcher(ignore_in_comparison, combined_lower, snippet_lower)

        combined_ratio = lcs_ratio(combined_lower, snippet_lower)
        # if combined_ratio >= combined_threshold => pointer += e.g. 5
        matched_words_count = len(combined_str.split(' '))
        dynamic_step = int(matched_words_count * combined_ratio)
        pointer = min(pointer + dynamic_step, n)

        # store matched_list in results
        results.extend(matched_list)
        idx += batch_size

    # fill leftover if needed
    while len(results) < len(original_lines):
        results.append("")

    return results

def remove_punctuation(text):
    """
    Removes all punctuation characters listed in string.punctuation
    from the input text.
    """
    # Create a translation table that maps punctuation to None
    translator = str.maketrans('', '', string.punctuation)
    # Translate the text
    return text.translate(translator)


def lcs_words(seq1, seq2):
    """
    Computes the Longest Common Subsequence (LCS) between two lists of words (seq1, seq2).

    Returns:
      (lcs_length, lcs_sequence)
        - lcs_length is an integer length of the LCS.
        - lcs_sequence is the list of words in the LCS itself.

    This implementation uses dynamic programming and reconstructs the LCS.

    Example:
        seq1 = ["a", "young", "man", "about", "27", "years", "old"]
        seq2 = ["at", "night,", "around", "one", "o'clock,",
                "a", "young", "man,", "about", "27", "years", "old", "visits", ...]
        length, lcs_seq = lcs_words(seq1, seq2)
        print(length, lcs_seq)
    """
    len1, len2 = len(seq1), len(seq2)

    # dp[i][j] will store length of LCS of seq1[:i] and seq2[:j]
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Fill dp bottom-up
    for i in range(len1):
        for j in range(len2):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    # The length of the LCS is now dp[len1][len2]
    lcs_length = dp[len1][len2]

    # Reconstruct the LCS sequence by tracing back through dp
    lcs_sequence = []
    i, j = len1, len2
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            # This word is part of the LCS, add it
            lcs_sequence.append(seq1[i - 1])
            i -= 1
            j -= 1
        else:
            # Move in the direction of the larger subproblem
            if dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

    # lcs_sequence was built in reverse
    lcs_sequence.reverse()

    return lcs_length, lcs_sequence


def lcs_ratio(seq1, seq2):
    """
    A convenient helper that returns a ratio for the LCS length
    normalized by the sum of lengths of seq1 and seq2.

    ratio = 2 * LCS_length / (len(seq1) + len(seq2))

    For example, if seq1 has 7 words, seq2 has 14 words, and the LCS is length 6,
    ratio = (2 * 6) / (7 + 14) = 12/21 = 0.57
    """
    length, _ = lcs_words(seq1, seq2)
    total = len(seq1) + len(seq2)
    if total == 0:
        return 0.0
    return (2.0 * length) / total

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


def translate_full_text(text, output_dir, client, source_lang='ru', target_lang='en'):
    print("Translating full text...")
    response = client.chat.completions.create(
        model="gpt-4o",
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
            text=translated_text['text']
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

def improve_original_chunk(chunk, context, nlp, client, source_lang='ru'):
    """
    Uses AI to insert appropriate punctuation into the combined text before translation.
    This considers only the source language structure.
    """
    text_lines = [entry for entry in chunk['mapping']]
    text_lines.extend(context)
    combined_text, text2lines = combine_lines_with_mapping(text_lines)

    print("Inserting punctuation into the chunk...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a linguistic processor for copy editing."},
            {"role": "user", "content": f"Insert appropriate punctuation and capitalization into the following text, "
                                        f"considering only the structure of {source_lang}:\n{combined_text}. Consider also "
                                        f"that this is subtitles text, so there will be labels like [Музыка]. "
                                        f"Also keep in mind that the name of the podcast is 'Армен и Федор', so "
                                        f"anything that looks like 'Армена Федор' is likely a mistake "
                                        f"in the autogenerated subtitles. "
                                        f"Return ONLY the improved text, without any additional comments or explanations."}
        ]
    )
    raw_output = response.choices[0].message.content.strip()

    # Use find_approximate_substring to map each original line to a span + sentence id
    positions_in_improved_text = []
    hay_tokens, _ = tokenize_with_sentences(raw_output, nlp)
    hay_tokens = [ t for t in hay_tokens if t[0].strip() ]
    for i, ln in enumerate(text_lines):
        ln_text = ln['text']
        start, end, sid = find_approximate_substring(hay_tokens, ln_text, nlp, 0.92)
        positions_in_improved_text.append((start, end, sid))

    improved_chunk = {'mapping': [], 'combined_text': ''}

    # Update chunk with improved lines and sid
    for i, ln in enumerate(chunk['mapping']):
        start, end, sid = positions_in_improved_text[i]
        improved_line = raw_output[start:end]
        improved_chunk['mapping'].append({**ln, 'text': improved_line, 'sid': sid})

    improved_chunk['combined_text'] = ' '.join(e["text"] for e in improved_chunk['mapping'])

    # Add any lines from the extended context that belong to the same sentence as the last chunk line
    if positions_in_improved_text:
        last_sid = positions_in_improved_text[len(chunk['mapping']) - 1][2]

        for j in range(len(chunk['mapping']), len(text2lines)):
            start, end, sid = positions_in_improved_text[j]
            if sid == last_sid:
                improved_chunk['mapping'].append({**text_lines[j], 'text': raw_output[start:end], 'sid': sid})
                relevant_spans = [(start, end) for _, start, end, sid in hay_tokens if sid == last_sid]
                if relevant_spans:
                    min_start = min(start for start, _ in relevant_spans)
                    max_end = max(end for _, end in relevant_spans)
                    improved_chunk['combined_text'] += raw_output[min_start:max_end].strip()
            else:
                break  # Stop as soon as sentence ID changes


    return improved_chunk

def translate_srt_file(input_file, output_dir, output_file, client, full_text=None, full_translation=None):
    subs = pysrt.open(input_file)
    mapping = create_subtitle_mapping(subs)
    chunks = split_srt_file_with_AI(mapping, 1000, client, 2, 100, 5)
    print("Split into {} chunks.".format(len(chunks)))
    #chunks = [{'mapping':mapping, 'combined_text':combined_full_text}]
    translated_lines = []
    translated_subs = []
    for chunk in chunks:
        if not full_translation:
            translated_text = translate_full_text(chunk['combined_text'], output_dir, client)
        else:
            translated_text = full_translation
        translated_lines.extend(process_whole_chunk(chunk, translated_text, 50, client))
        #translated_lines.extend(split_translation_into_lines_AI(chunk['mapping'], translated_text, client))
        #translated_lines = reduce_overlap(translated_lines)
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
    #response = client.models.list()
    #for model_info in response.data:
    #    print(model_info.id)
    translate_srt_file(input_srt, output_dir, output_srt, client, full_text, full_translation)
