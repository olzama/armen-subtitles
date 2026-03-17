#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import pysrt


SUPPORTED_SRT_EXTENSIONS = {".srt", ".txt"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_srt_as_map(srt_path: Path) -> dict[int, str]:
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    srt_map: dict[int, str] = {}

    for sub in subs:
        text = sub.text.replace("\r\n", "\n").replace("\r", "\n").strip()
        srt_map[sub.index] = text

    return srt_map


def validate_segment_list(item: dict, item_id) -> list[int]:
    if "segment_number" not in item:
        raise ValueError(f'Item {item_id} is missing required field "segment_number".')
    segments = item["segment_number"]
    if not isinstance(segments, list):
        raise ValueError(f'Item {item_id}: "segment_number" must be a list of integers.')
    if not all(isinstance(x, int) for x in segments):
        raise ValueError(f'Item {item_id}: all values in "segment_number" must be integers.')
    return segments


def collect_translation(segments: list[int], srt_map: dict[int, str], item_id) -> str:
    missing = [seg for seg in segments if seg not in srt_map]
    if missing:
        raise ValueError(
            f"Item {item_id}: the following segment numbers were not found in the SRT: {missing}"
        )
    parts = [srt_map[seg] for seg in segments]
    return "\n".join(parts).strip()


def find_srt_files(srt_dir: Path) -> list[Path]:
    if not srt_dir.is_dir():
        raise ValueError(f"SRT folder not found: {srt_dir}")
    files = sorted(
        p for p in srt_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SRT_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No .srt files found in folder: {srt_dir}")
    return files


def find_method_dirs(methods_root: Path) -> list[Path]:
    if not methods_root.is_dir():
        raise ValueError(f"Methods root folder not found: {methods_root}")

    method_dirs = sorted(p for p in methods_root.iterdir() if p.is_dir())

    if not method_dirs:
        raise ValueError(f"No method subdirectories found in folder: {methods_root}")

    return method_dirs


def enrich_json(
    data: dict,
    method_to_srt_files: dict[str, list[Path]],
    model_name: str,
):
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a top-level object.")

    if "items" not in data:
        raise ValueError('Input JSON must contain a top-level "items" field.')

    if not isinstance(data["items"], list):
        raise ValueError('Top-level "items" must be a list.')

    method_to_srt_maps: dict[str, list[dict[int, str]]] = {
        method_name: [load_srt_as_map(path) for path in srt_files]
        for method_name, srt_files in method_to_srt_files.items()
    }

    output = dict(data)
    output["model"] = model_name
    output_items = []

    for item in data["items"]:
        if not isinstance(item, dict):
            raise ValueError("Each item in 'items' must be an object.")

        item_id = item.get("id", "<no id>")
        segments = validate_segment_list(item, item_id)

        new_item = dict(item)
        translations = dict(new_item.get("translations", {}))
        eng_translations = dict(translations.get("eng", {}))

        for method_name, srt_maps in method_to_srt_maps.items():
            method_runs: dict[str, str] = {}

            for run_idx, srt_map in enumerate(srt_maps, start=1):
                print(f"Mapping item {item_id} segments {segments} to method {method_name} run {run_idx}...")
                translation = collect_translation(segments, srt_map, item_id)
                method_runs[str(run_idx)] = translation

            eng_translations[method_name] = method_runs

        translations["eng"] = eng_translations
        new_item["translations"] = translations
        output_items.append(new_item)

    output["items"] = output_items
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an outer folder whose subdirectories are methods and whose method "
            "subdirectories contain translated SRT files. Read a JSON file with items "
            'containing "segment_number": [list of subtitle indices], then write out '
            "the same JSON with translations nested as "
            "translations['eng'][METHOD][RUN_NUMBER]."
        )
    )
    parser.add_argument(
        "methods_root",
        help="Path to the outer folder containing method subdirectories"
    )
    parser.add_argument("json_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Output filename")
    parser.add_argument("model_name", help="Model name to store in the output JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    methods_root = Path(args.methods_root)
    json_path = Path(args.json_file)
    output_filename = Path(args.output_file)

    if not json_path.is_file():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = load_json(json_path)

        method_dirs = find_method_dirs(methods_root)
        method_to_srt_files: dict[str, list[Path]] = {}

        for method_dir in method_dirs:
            print(f"Processing method directory: {method_dir}")
            method_name = method_dir.name
            srt_files = find_srt_files(method_dir / "translations")
            method_to_srt_files[method_name] = srt_files

        enriched = enrich_json(data, method_to_srt_files, args.model_name)
        save_json(enriched, output_filename)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()