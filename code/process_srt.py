import pysrt


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
