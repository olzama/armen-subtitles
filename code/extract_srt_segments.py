#!/usr/bin/env python3

import json
import sys
from pathlib import Path
import pysrt


def load_segment_ids(json_path, field="segment_number"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []

    def visit(x):
        if isinstance(x, dict):
            if field in x and x[field] is not None:
                value = x[field]

                if not isinstance(value, list):
                    raise ValueError(f"{field} must be a list of integers")

                for item in value:
                    if not isinstance(item, int):
                        raise ValueError(f"All values in {field} must be integers")
                    if item > 0:
                        ids.append(item)

            for v in x.values():
                visit(v)

        elif isinstance(x, list):
            for v in x:
                visit(v)

    visit(data)
    return set(ids)


def expand_with_context(wanted_set, n):
    if n <= 0:
        return wanted_set

    expanded = set(wanted_set)
    for i in wanted_set:
        for k in range(i - n, i + n + 1):
            if k > 0:
                expanded.add(k)

    return expanded


def extract_from_file(srt_path, wanted_set):
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    extracted = pysrt.SubRipFile()

    for it in subs:
        if it.index in wanted_set:
            extracted.append(it)

    return extracted


def main():
    if len(sys.argv) not in (4, 5):
        print("Usage: python script.py input_folder reference.json output_folder [N]")
        sys.exit(2)

    input_folder = Path(sys.argv[1])
    json_path = sys.argv[2]
    output_folder = Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) == 5 else 2

    if not input_folder.is_dir():
        print("Input folder not found")
        sys.exit(2)

    output_folder.mkdir(parents=True, exist_ok=True)

    wanted_set = load_segment_ids(json_path, field="segment_number")
    wanted_set = expand_with_context(wanted_set, n)

    for srt_file in input_folder.glob("*.srt"):
        extracted = extract_from_file(srt_file, wanted_set)
        extracted.save(str(output_folder / srt_file.name), encoding="utf-8")

    print("Done")


if __name__ == "__main__":
    main()