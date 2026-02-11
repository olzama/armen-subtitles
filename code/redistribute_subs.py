import pysrt
import stanza
from datetime import datetime, timedelta
import sys


def estimate_reading_time(text):
    """Estimate reading time assuming 2.5 words per second."""
    word_count = len(text.split())
    return timedelta(seconds=word_count / 2.5)


def combine_text_with_mapping(subs):
    """Combine subtitle text while mapping character spans to subtitle entries."""
    full_text = ""
    mapping = []
    for i, sub in enumerate(subs):
        start_char = len(full_text)
        full_text += sub.text.strip() + " "
        end_char = len(full_text)
        mapping.append({
            'index': i,
            'start_time': sub.start.to_time(),
            'end_time': sub.end.to_time(),
            'start_char': start_char,
            'end_char': end_char
        })
    return full_text.strip(), mapping


def format_time(t):
    """Convert a `datetime.time` object into SRT-compatible string."""
    return t.strftime('%H:%M:%S,%f')[:-3]


def find_segment_times(start_char, end_char, text, mapping):
    """Estimate start and end time for a text span."""
    start_map = next(m for m in mapping if m['start_char'] <= start_char < m['end_char'])
    end_map = next(m for m in mapping if m['start_char'] < end_char <= m['end_char'])

    offset_start = estimate_reading_time(text[start_map['start_char']:start_char])
    offset_end = estimate_reading_time(text[end_map['start_char']:end_char])

    start_time = (datetime.combine(datetime.min, start_map['start_time']) + offset_start).time()
    end_time = (datetime.combine(datetime.min, end_map['start_time']) + offset_end).time()
    return start_time, end_time


def redistribute_by_phrases(text, doc, mapping, min_len=40, max_len=120):
    """Split long sentences and merge short ones, ensuring full data coverage."""
    spans = []
    for sent in doc.sentences:
        tokens = sent.tokens
        current_start = tokens[0].start_char
        for i in range(1, len(tokens) + 1):
            if i == len(tokens):
                current_end = tokens[-1].end_char
            else:
                current_end = tokens[i].start_char
            segment_len = current_end - current_start
            if segment_len >= max_len:
                spans.append((current_start, current_end))
                if i < len(tokens):
                    current_start = tokens[i].start_char
        if current_start < tokens[-1].end_char:
            spans.append((current_start, tokens[-1].end_char))

    # Merge small segments with neighbors if < min_len
    merged = []
    i = 0
    while i < len(spans):
        start, end = spans[i]
        if end - start < min_len and i + 1 < len(spans) and text[start:end].strip() != "[музыка]":
            next_start, next_end = spans[i + 1]
            merged.append((start, next_end))
            i += 2
        else:
            merged.append((start, end))
            i += 1

    return [(find_segment_times(s, e, text, mapping), text[s:e].strip()) for s, e in merged]


def generate_new_srt(segments, output_path):
    """Write the new segments into an SRT file."""
    subs = pysrt.SubRipFile()
    for i, ((start, end), text) in enumerate(segments, 1):
        sub = pysrt.SubRipItem()
        sub.index = i
        sub.start = pysrt.SubRipTime.from_string(format_time(start))
        sub.end = pysrt.SubRipTime.from_string(format_time(end))
        sub.text = text
        subs.append(sub)
    subs.save(output_path, encoding='utf-8')


if __name__ == '__main__':
    stanza.download('ru')  # Only needed once
    nlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma,depparse')

    input_srt_path = sys.argv[1]
    output_srt_path = sys.argv[2]

    # Step 1: Load subtitles
    subs = pysrt.open(input_srt_path, encoding='utf-8')

    # Step 2: Combine text and map spans
    combined_text, mapping = combine_text_with_mapping(subs)

    # Step 3: Parse with Stanza
    doc = nlp(combined_text)

    # Step 4: Redistribute syntactic chunks
    segments = redistribute_by_phrases(combined_text, doc, mapping)

    # Step 5: Save as new SRT
    generate_new_srt(segments, output_srt_path)
