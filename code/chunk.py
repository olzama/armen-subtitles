import copy
import sys, os
import difflib
from copy import deepcopy

import tiktoken
import string
import openai
from sympy.physics.units import milliseconds

from process_srt import srt2text, combine_lines_with_mapping
import pysrt
from process_srt import create_subtitle_mapping, redistribute_subs
from budget_estimate import track_usage_and_cost

def combine_srt_chunks(input_dir, max_tokens=4000):
    combined_text_chunks = []
    combined_text = ''
    tokens_so_far = 0
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".srt"):
            with open(os.path.join(input_dir, filename), "r", encoding='utf-8') as f:
                srt_items = pysrt.from_string(f.read())
                for item in srt_items:
                    new_tokens = count_tokens_in_text(item.text)
                    if tokens_so_far + new_tokens > max_tokens:
                        combined_text_chunks.append(combined_text)
                        combined_text = ''
                        tokens_so_far = 0
                    combined_text += item.text + ' '
    if combined_text:
        combined_text_chunks.append(combined_text)
    return combined_text_chunks

def count_tokens_in_text(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def normalize_word(word):
    return word.lower().strip(string.punctuation)

def tokenize_with_sentences(text, nlp):
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

def ignore_in_comparison(c):
    return c in string.whitespace + string.punctuation

def tokenwise_fuzzy_ratio(tokens1, tokens2, alpha=0.9):
    if not tokens1 or not tokens2:
        return 0.0
    matched_scores = []
    for t1 in tokens1:
        best_score = 0.0
        for t2 in tokens2:
            score = difflib.SequenceMatcher(ignore_in_comparison, t1.lower(), t2.lower()).ratio()
            best_score = max(best_score, score)
            if best_score == 1.0:
                break
        matched_scores.append(best_score)
    bag_score = sum(matched_scores) / len(tokens1)
    s1 = ' '.join(tokens1).lower()
    s2 = ' '.join(tokens2).lower()
    order_score = difflib.SequenceMatcher(ignore_in_comparison, s1, s2).ratio()
    return alpha * bag_score + (1 - alpha) * order_score

def find_approximate_substring(hay_tokens, needle, nlp, threshold=0.91, min_threshold=0.75, max_window_expansion=2):
    needle_tokens, _ = tokenize_with_sentences(needle, nlp)
    hay_words = [w for w, _, _, _ in hay_tokens if w]
    needle_words = [w for w, _, _, _ in needle_tokens if w]
    needle_len = len(needle_words)
    if needle_len == 0 or len(hay_words) == 0:
        return -1, -1, -1
    best_ratio = 0.0
    best_span = (-1, -1)
    for expansion in range(max_window_expansion + 1):
        window_len = needle_len + expansion
        if window_len < 1 or window_len > len(hay_words):
            continue
        for i in range(len(hay_words) - window_len + 1):
            window = hay_words[i:i + window_len]
            ratio = tokenwise_fuzzy_ratio(needle_words, window)
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (i, i + window_len)
                if best_ratio >= threshold:
                    break
        if best_ratio >= threshold:
            break
        else:
            threshold = max(threshold - 0.05, min_threshold)
    if best_ratio >= min_threshold and best_span != (-1, -1):
        start_idx, end_idx = best_span
        start_char = hay_tokens[start_idx][1]
        end_char = hay_tokens[end_idx - 1][2]
        sentence_id = hay_tokens[start_idx][3]
        return start_char, end_char, sentence_id
    return -1, -1, -1

def extract_dependency_phrases(sentence):
    """
    Extract all subtree spans (start_char, end_char) for every word in the sentence.
    Overlapping spans are allowed. These will later be used to decide where to break text.
    """
    spans = []

    # Build a map from head ID to list of dependents
    dep_map = {}
    for word in sentence.words:
        dep_map.setdefault(word.head, []).append(word)

    def collect_subtree(word):
        stack = [word]
        result = [word]
        while stack:
            current = stack.pop()
            for child in dep_map.get(current.id, []):
                if child.id not in {w.id for w in result}:
                    result.append(child)
                    stack.append(child)
        return result

    for word in sentence.words:
        subtree = collect_subtree(word)
        sorted_subtree = sorted(subtree, key=lambda x: x.start_char)
        start = sorted_subtree[0].start_char
        end = sorted_subtree[-1].end_char
        spans.append((start, end))

    return spans

def group_by_dependency_phrases(text, mapping, nlp, max_chars=120, starting_index=0):
    """
    Group subtitle segments into blocks of up to max_chars, breaking only at constituent (dependency) span boundaries.
    Original time codes are preserved.
    """
    doc = nlp(text)
    phrase_spans = []
    for sentence in doc.sentences:
        phrase_spans.extend(extract_dependency_phrases(sentence))

    # Build entry spans by simulating text assembly
    entry_spans = []
    cursor = 0
    for e in mapping:
        e_text = e['text'].replace('\n', ' ').strip()
        if not e_text:
            continue
        start = cursor
        end = start + len(e_text)
        entry_spans.append((e, start, end))
        cursor = end + 1  # space

    entries = []
    index = starting_index + 1
    prev_end_time = None

    i = 0
    while i < len(entry_spans):
        current_len = 0
        group = []
        start_char = entry_spans[i][1]

        # Try to add entries until near limit
        j = i
        while j < len(entry_spans):
            e, _, _ = entry_spans[j]
            t = e['text'].replace('\n', ' ').strip()
            projected_len = current_len + len(t) + (1 if current_len > 0 else 0)
            if projected_len > max_chars:
                break
            group.append(e)
            current_len = projected_len
            j += 1

        # Find best constituent break within range
        end_char = entry_spans[j - 1][2] if group else entry_spans[i][2]
        valid_cuts = [end for (start, end) in phrase_spans if start >= start_char and end <= end_char]
        if valid_cuts:
            boundary = max(valid_cuts)
            # Trim group to only entries within span
            group = [e for e, s, e_ in entry_spans[i:j] if e_ <= boundary]
            j = i + len(group)

        if not group:
            group = [entry_spans[i][0]]
            j = i + 1

        text_chunk = ' '.join(e['text'].replace('\n', ' ').strip() for e in group)
        start_time = group[0]['start']
        end_time = group[-1]['end']

        if prev_end_time:
            import copy
            start_time = copy.deepcopy(prev_end_time)
            start_time.shift(milliseconds=-500)

        entries.append({
            'index': index,
            'start': start_time,
            'end': end_time,
            'text': text_chunk.strip()
        })
        index += 1
        prev_end_time = end_time
        i = j

    return entries

def improve_original_chunk_dep(chunk, context, nlp, client, usage, summary, narratives,
                                source_lang='ru', prompt='', return_carryover=False, start_index=0):
    text_lines = [entry for entry in chunk['mapping']] + context
    combined_text, text2lines = combine_lines_with_mapping(text_lines)
    print("Improving the original chunk...")
    print("{}".format(combined_text))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a linguistic processor for copy editing."},
            {"role": "user", "content": f"Insert appropriate punctuation and capitalization into the following text, "
                                            f"considering only the structure of {source_lang}:\n{combined_text}."
                                            f"Considering the following summary and the narratives referenced in the text,"
                                            f"correct any autotranscription errors in the original "
                                            f"(a typical error would be acoustic confusion, e.g. the  unusual word "
                                            f"идальго mistaken for a common phrase и далеко. The given context should often "
                                            f"be helpful for catching such mistakes. Only look for mistakes that are likely"
                                            f" to be due to autotranscription error. Your cue should be text which does not make sense,"
                                            f"syntactically or semantically, but keep in mind that the text is very complex and can be unusual."
                                            f" Note that something that looks like garbage (foreign letters, numbers...) is almost always "
                                            f"an autotranscription error: do not discard it! "
                                            f"Instead, try to guess what it could have been, given the context "
                                            f"and what you know about the phonetics and phonology of the source language. "
                                            f"If something looks like garbage, what does this garbage sound like and "
                                            f"what else sounds similar that would make sense in the given context?\n"
                                            f"Summary: {summary}\n"
                                            f"List of narratives: {narratives}\n"
                                            f"Return ONLY the improved text, without any additional comments or explanations.\n\n"
                                            f"{prompt}"}
        ]
    )
    raw_output = response.choices[0].message.content.strip()
    track_usage_and_cost(response.usage, 2.5, 10, "gpt-4o", usage=usage)
    hay_tokens, _ = tokenize_with_sentences(raw_output, nlp)
    hay_tokens = [t for t in hay_tokens if t[0].strip()]
    positions_in_improved_text = []
    for ln in text_lines:
        start, end, sid = find_approximate_substring(hay_tokens, ln['text'], nlp, 0.92)
        positions_in_improved_text.append((start, end, sid))
    improved_chunk = {'mapping': [], 'combined_text': '', 'raw_text': ''}
    carryover = []
    if not positions_in_improved_text:
        grouped = group_by_dependency_phrases(improved_chunk['combined_text'], improved_chunk['mapping'], nlp, 120, start_index)
        improved_chunk['mapping'] = grouped
        improved_chunk['raw_text'] = srt2text(grouped)
        improved_chunk['combined_text'] = '\n'.join(e['text'] for e in grouped)
        last_entry_index = grouped[-1]['index']
        return improved_chunk, carryover, last_entry_index
    last_sid = positions_in_improved_text[len(chunk['mapping']) - 1][2]
    filtered_entries = []
    for i, ln in enumerate(chunk['mapping']):
        start, end, sid = positions_in_improved_text[i]
        improved_line = raw_output[start:end]
        updated_entry = {**ln, 'text': improved_line, 'sid': sid}
        improved_chunk['mapping'].append(updated_entry)
        if sid <= last_sid:
            filtered_entries.append(updated_entry)
    context_start = len(chunk['mapping'])
    appended_up_to = -1
    for j in range(context_start, len(text2lines)):
        start, end, sid = positions_in_improved_text[j]
        if sid == last_sid:
            improved_line = raw_output[start:end]
            entry = {**text_lines[j], 'text': improved_line, 'sid': sid}
            improved_chunk['mapping'].append(entry)
            filtered_entries.append(entry)
            appended_up_to = j
        else:
            break
    if appended_up_to >= 0 and appended_up_to < len(text2lines):
        full_entry = text_lines[appended_up_to]
        start, end, sid = positions_in_improved_text[appended_up_to]
        tokens = [t for t in hay_tokens if start <= t[1] < end]
        carry_tokens = [t for t in tokens if t[3] != last_sid]
        if carry_tokens:
            t_start = carry_tokens[0][1]
            t_end = carry_tokens[-1][2]
            carry_text = raw_output[t_start:t_end].strip()
            carry_sid = carry_tokens[0][3]
            carryover.append({**full_entry, 'text': carry_text, 'sid': carry_sid})
    valid_tokens = [t for t in hay_tokens if t[3] <= last_sid]
    if valid_tokens:
        t_start = valid_tokens[0][1]
        t_end = valid_tokens[-1][2]
        improved_chunk['combined_text'] = raw_output[t_start:t_end].strip()

    grouped = group_by_dependency_phrases(improved_chunk['combined_text'], improved_chunk['mapping'], nlp, 120, start_index)
    improved_chunk['mapping'] = grouped
    improved_chunk['raw_text'] = srt2text(grouped)
    improved_chunk['combined_text'] = '\n'.join(e['text'] for e in grouped)
    last_entry_index = grouped[-1]['index']
    return improved_chunk, carryover, last_entry_index

def split_srt_file_with_AI(mapping, usage, max_tokens, client, summary, narratives, n_overlap=2, flex_tokens=50, context_size=5, prompt=''):
    import stanza
    nlp = stanza.Pipeline(lang='ru', processors='tokenize, pos, lemma, depparse')
    #nlp = stanza.Pipeline(lang='ru', processors='tokenize')
    chunks = []
    current_chunk = {'mapping': [], 'combined_text': '', 'raw_text': ''}
    chunk_tokens = 0
    carryover = []
    i = 0
    total_entries = len(mapping)
    expanded_index = 0
    while i < total_entries:
        entry = mapping[i]
        line_tokens = count_tokens_in_text(str(entry), model='gpt-4o')
        tentative_total = chunk_tokens + line_tokens
        if tentative_total > (max_tokens + flex_tokens) and current_chunk['mapping']:
            next_context = [mapping[j] for j in range(i, min(i + context_size, total_entries))]
            improved_chunk, new_carryover, expanded_index = improve_original_chunk_dep(
                current_chunk, next_context, nlp, client, usage, summary, narratives,
                'ru', prompt=prompt, return_carryover=True, start_index=expanded_index
            )
            improved_chunk["raw_text"] = srt2text(improved_chunk["mapping"])
            chunks.append(improved_chunk)
            carryover = new_carryover
            if i >= total_entries - n_overlap:
                return chunks
            overlap_start = max(0, i - n_overlap)
            current_chunk = {'mapping': mapping[overlap_start:i], 'combined_text': '', 'raw_text': ''}
            chunk_tokens = sum(count_tokens_in_text(str(e), model='gpt-4o') for e in current_chunk['mapping'])
        else:
            current_chunk['mapping'].append(entry)
            chunk_tokens += line_tokens
            i += 1
    if current_chunk['mapping']:
        if carryover:
            current_chunk['mapping'].extend(carryover)
        current_chunk, _, expanded_index = improve_original_chunk_dep(
            current_chunk, [], nlp, client, usage, summary, narratives,'ru',
            prompt=prompt, return_carryover=False, start_index=expanded_index
        )
        chunks.append(current_chunk)
    return chunks

def timestamp_to_seconds(timestamp):
    return timestamp.ordinal / 1000.0  # ordinal is in milliseconds

def expand_timecodes(chunk, max_chars=120, starting_index=0):
    entries = chunk['mapping']
    new_mapping = []
    index = starting_index + 1
    i = 0
    total = len(entries)

    while i < total:
        group = []
        char_count = 0

        # Collect lines into a group while keeping under max_chars
        while i < total:
            entry = entries[i]
            entry_text = entry['text'].replace('\n', ' ').strip()
            entry_len = len(entry_text)

            if group and (char_count + entry_len + 1 > max_chars):
                break  # stop appending to group

            group.append(entry)
            char_count += entry_len + 1  # +1 for space
            i += 1

        # Construct combined block
        combined_text = ' '.join(e['text'].replace('\n', ' ').strip() for e in group)
        start_time = group[0]['start']
        end_time = group[-1]['end']

        if new_mapping:
            prev_end = new_mapping[-1]['end']
            start_time = copy.deepcopy(prev_end)
            start_time.shift(milliseconds=-500)

        new_mapping.append({
            'index': index,
            'start': start_time,
            'end': end_time,
            'text': combined_text.strip()
        })
        index += 1

    return {
        'mapping': new_mapping,
        'combined_text': '\n'.join(e['text'] for e in new_mapping),
        'raw_text': '\n'.join([
            f"{e['index']}\n{e['start']} --> {e['end']}\n{e['text']}\n" for e in new_mapping
        ])
    }



def improve_original_chunk(chunk, context, nlp, client, usage, summary, narratives,
                           source_lang='ru', prompt='', return_carryover=False, start_index=0):
    text_lines = [entry for entry in chunk['mapping']] + context
    combined_text, text2lines = combine_lines_with_mapping(text_lines)
    print("Improving the original chunk...")
    print("{}".format(combined_text))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a linguistic processor for copy editing."},
            {"role": "user", "content": f"Insert appropriate punctuation and capitalization into the following text, "
                                        f"considering only the structure of {source_lang}:\n{combined_text}."
                                        f"Considering the following summary and the narratives referenced in the text,"
                                        f"correct any autotranscription errors in the original "
                                        f"(a typical error would be acoustic confusion, e.g. the  unusual word "
                                        f"идальго mistaken for a common phrase и далеко. The given context should often "
                                        f"be helpful for catching such mistakes. Only look for mistakes that are likely"
                                        f" to be due to autotranscription error. Your cue should be text which does not make sense,"
                                        f"syntactically or semantically, but keep in mind that the text is very complex and can be unusual."
                                        f" Note that something that looks like garbage (foreign letters, numbers...) is almost always "
                                        f"an autotranscription error: do not discard it! "
                                        f"Instead, try to guess what it could have been, given the context "
                                        f"and what you know about the phonetics and phonology of the source language. "
                                        f"If something looks like garbage, what does this garbage sound like and "
                                        f"what else sounds similar that would make sense in the given context?\n"
                                        f"Summary: {summary}\n"
                                        f"List of narratives: {narratives}\n"
                                        f"Return ONLY the improved text, without any additional comments or explanations.\n\n"
                                        f"{prompt}"}
        ]
    )
    raw_output = response.choices[0].message.content.strip()
    track_usage_and_cost(response.usage, 2.5, 10, "gpt-4o", usage=usage)
    hay_tokens, _ = tokenize_with_sentences(raw_output, nlp)
    hay_tokens = [t for t in hay_tokens if t[0].strip()]
    positions_in_improved_text = []
    for ln in text_lines:
        start, end, sid = find_approximate_substring(hay_tokens, ln['text'], nlp, 0.92)
        positions_in_improved_text.append((start, end, sid))
    improved_chunk = {'mapping': [], 'combined_text': '', 'raw_text': ''}
    carryover = []
    if not positions_in_improved_text:
        even_better = expand_timecodes(improved_chunk, 120, start_index)
        last_entry_index = even_better['mapping'][-1]['index']
        return even_better, carryover, last_entry_index
    last_sid = positions_in_improved_text[len(chunk['mapping']) - 1][2]
    filtered_entries = []
    for i, ln in enumerate(chunk['mapping']):
        start, end, sid = positions_in_improved_text[i]
        improved_line = raw_output[start:end]
        updated_entry = {**ln, 'text': improved_line, 'sid': sid}
        improved_chunk['mapping'].append(updated_entry)
        if sid <= last_sid:
            filtered_entries.append(updated_entry)
    context_start = len(chunk['mapping'])
    appended_up_to = -1
    for j in range(context_start, len(text2lines)):
        start, end, sid = positions_in_improved_text[j]
        if sid == last_sid:
            improved_line = raw_output[start:end]
            entry = {**text_lines[j], 'text': improved_line, 'sid': sid}
            improved_chunk['mapping'].append(entry)
            filtered_entries.append(entry)
            appended_up_to = j
        else:
            break
    if appended_up_to >= 0 and appended_up_to < len(text2lines):
        full_entry = text_lines[appended_up_to]
        start, end, sid = positions_in_improved_text[appended_up_to]
        tokens = [t for t in hay_tokens if start <= t[1] < end]
        carry_tokens = [t for t in tokens if t[3] != last_sid]
        if carry_tokens:
            t_start = carry_tokens[0][1]
            t_end = carry_tokens[-1][2]
            carry_text = raw_output[t_start:t_end].strip()
            carry_sid = carry_tokens[0][3]
            carryover.append({**full_entry, 'text': carry_text, 'sid': carry_sid})
    valid_tokens = [t for t in hay_tokens if t[3] <= last_sid]
    if valid_tokens:
        t_start = valid_tokens[0][1]
        t_end = valid_tokens[-1][2]
        improved_chunk['combined_text'] = raw_output[t_start:t_end].strip()
    print("Final text:\n{}".format(improved_chunk['combined_text']))
    even_better = expand_timecodes(improved_chunk, 120, start_index)
    last_entry_index = even_better['mapping'][-1]['index']
    return even_better, carryover, last_entry_index

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    with open(sys.argv[3], "r", encoding='utf-8') as f:
        summary = f.read()
    with open(sys.argv[4], "r", encoding='utf-8') as f:
        narratives = f.read()
    with open("./LYS-API-key.txt", "r") as myfile:
        openai_key = myfile.read().strip()
    client = openai.OpenAI(api_key=openai_key)
    subs = pysrt.open(input_file)
    mapping = create_subtitle_mapping(subs)
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "total_cost": 0.0
    }
    chunks = split_srt_file_with_AI(mapping, usage, 20000, client, summary, narratives,2, 100, 5, prompt='')
    pad_length = len(str(len(chunks)))
    for idx, chunk in enumerate(chunks):
        filename = f"chunk_{idx:0{pad_length}}.srt"
        with open(os.path.join(output_dir, filename), "w", encoding='utf-8') as f:
            f.write(chunk["raw_text"])
    print("Split into {} chunks.".format(len(chunks)))
    print("Estimated cost: ${}, with {} input tokens and {} output tokens.".format(
        usage["total_cost"], usage["input_tokens"], usage["output_tokens"]))
