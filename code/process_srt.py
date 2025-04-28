import pysrt
import sys
from datetime import timedelta

def srt2text(entries):
    """Takes a list of subtitle mapping entries and returns raw .srt-format string."""
    srt_lines = []
    for i, entry in enumerate(entries):
        index = entry.get("index", i + 1)
        start = entry.get("start", "")
        end = entry.get("end", "")
        text = entry.get("text", "").strip()

        srt_lines.append(str(index))
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")  # empty line between entries

    return "\n".join(srt_lines).strip()

def txt2srt(lines):
    srt_items = []
    for ln in lines:
        start = ln['start']
        end = ln['end']
        text = ln['text']
        index = ln['index']
        srt_items.append(pysrt.SubRipItem(
            index=index,
            start=start,
            end=end,
            text=text
        ))
    return srt_items

def txt2lines(subs, mapping):
    srt_items = pysrt.from_string(subs)
    lines = []
    for i, entry in enumerate(srt_items):
        original_entry = mapping[i]
        start = original_entry['start']
        end = original_entry['end']
        text = entry.text
        index = original_entry['index']
        my_index = entry.index
        if my_index != index:
            print("Warning: index mismatch! {} vs {}".format(my_index, index))
        lines.append({
            'start': start,
            'end': end,
            'text': text,
            'index': my_index
        })
        print("Added line {}: {} -> {}".format(my_index, original_entry['text'], entry.text))
    return lines

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

def combined_text(srt_filename):
    """
    Takes a .srt file and returns a string with all the text combined.
    """
    subs = pysrt.open(srt_filename)
    combined_text = ' '.join(sub.text for sub in subs)
    return combined_text

def to_subrip_time(time_str):
    """Convert SRT timestamp string to pysrt.SubRipTime."""
    return pysrt.SubRipTime.from_string(time_str)

def to_dict(sub):
    """Convert a pysrt.SubRipItem to a dictionary."""
    return {
        'index': sub.index,
        'start': str(sub.start),
        'end': str(sub.end),
        'text': sub.text
    }

def split_text(text, max_len=120):
    words = text.split()
    chunks, current_line = [], []

    for word in words:
        test_line = ' '.join(current_line + [word])
        if len(test_line) <= max_len:
            current_line.append(word)
        else:
            if current_line:
                chunks.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        chunks.append(' '.join(current_line))

    return chunks

def to_timedelta(srt_time):
    return timedelta(
        hours=srt_time.hours,
        minutes=srt_time.minutes,
        seconds=srt_time.seconds,
        milliseconds=srt_time.milliseconds
    )

def redistribute_subs(subs_dicts, max_len=120, max_overlap_ms=5):
    new_subs = []
    index = 1
    max_overlap = timedelta(milliseconds=max_overlap_ms)

    for sub in subs_dicts:
        start = sub['start']
        end = sub['end']
        duration = to_timedelta(end) - to_timedelta(start)

        chunks = split_text(sub['text'], max_len=max_len)
        segment_duration = duration / len(chunks)

        for i, chunk in enumerate(chunks):
            seg_start = to_timedelta(start) + i * segment_duration
            seg_end = seg_start + segment_duration

            # Ensure max 5ms overlap
            if new_subs:
                prev_end = to_timedelta(to_subrip_time(new_subs[-1]['end']))
                if seg_start < prev_end - max_overlap:
                    seg_start = prev_end - max_overlap
                    seg_end = seg_start + segment_duration
                elif seg_start < prev_end:
                    seg_start = prev_end + timedelta(milliseconds=1)
                    seg_end = seg_start + segment_duration

            new_item = pysrt.SubRipItem(
                index=index,
                start=pysrt.SubRipTime.from_ordinal(int(seg_start.total_seconds() * 1000)),
                end=pysrt.SubRipTime.from_ordinal(int(seg_end.total_seconds() * 1000)),
                text=chunk
            )
            new_subs.append(to_dict(new_item))
            index += 1

    return new_subs

if __name__ == "__main__":
    combined_text = combined_text(sys.argv[1])
    with open(sys.argv[2], "w", encoding='utf-8') as f:
        f.write(combined_text)