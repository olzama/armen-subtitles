import string
import sys
import openai
import pysrt
import difflib
import re
import tiktoken

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

            found_substring, ratio = find_substring_for_line(line['text'], snippet_text, client, line_threshold)
            if ratio >= line_threshold:
                line_matched = True
                line_substring = found_substring
            else:
                # No match: expand snippet and optionally step pointer
                # pointer fallback:
                pointer = max(0, pointer - step_back)
                # expand snippet by step_forward words
                snippet_end = min(snippet_end + step_forward, n)
                line_threshold -= 0.05  # relax threshold a bit
                line_retries += 1

        if line_matched:
            matched_list.append(line_substring)
        else:
            # after max_line_retries, no match
            matched_list.append("")

    # snippet_text returned is from the last built snippet
    return matched_list, pointer, snippet_text


def split_translation_into_lines_AI(
    original_lines,
    translated_text,
    client,
    batch_size=3,
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
            lines_batch, words, pointer, chunk_size+pointer, client, line_threshold
        )

        # 2) measure combined ratio
        combined_str = " ".join(x for x in matched_list if x)
        combined_str = remove_largest_common_substring(combined_str, 2)
        snippet_lower = snippet_text.lower()
        combined_lower = combined_str.lower()
        combined_ratio = difflib.SequenceMatcher(ignore_in_comparison, combined_lower, snippet_lower).ratio()

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