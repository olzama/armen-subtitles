#!/usr/bin/env python3
"""Print timestamps for each meme in a reference/analysis JSON."""

import json
import sys
from pathlib import Path
import pysrt


def parse_srt(srt_path):
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    return {sub.index: str(sub.start) for sub in subs}


def main():
    if len(sys.argv) != 3:
        print("Usage: python meme_timestamps.py reference.json subtitles.srt")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    srt_path = Path(sys.argv[2])

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    timestamps = parse_srt(srt_path)

    sorted_items = sorted(data["items"], key=lambda x: x.get("segment_number", [float("inf")])[0])

    for item in sorted_items:
        segments = item.get("segment_number", [])
        times = [timestamps.get(s, "?") for s in segments]
        time_str = " / ".join(times)
        russian = item["original"]["rus"] if "original" in item else item.get("rus", "")
        english = item.get("reference", {}).get("eng", "")
        print(f"[{item['id']}] {item['character']}: {russian[:60]}")
        if english:
            print(f"     EN: {english}")
        print(f"     Segments {segments} → {time_str}")
        print()


if __name__ == "__main__":
    main()
