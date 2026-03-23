#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import re

import pysrt


SUPPORTED_SRT_EXTENSIONS = {".srt", ".txt"}

def extract_run_number_from_filename(path: Path) -> str:
    """
    Expect filenames like:
      translation-1.txt
      translation-12.srt

    Returns the numeric part as a string, e.g. "1", "12".
    """
    m = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if not m:
        raise ValueError(
            f"Could not extract run number from filename: {path.name}. "
            f"Expected something like translation-12.txt"
        )
    return str(int(m.group(1)))

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
        raise ValueError(f"No .srt or .txt files found in folder: {srt_dir}")
    print(f"Found SRT files: {[f.name for f in files]} in {srt_dir}")
    return files


def find_method_dirs(methods_root: Path) -> list[Path]:
    if not methods_root.is_dir():
        raise ValueError(f"Methods root folder not found: {methods_root}")

    method_dirs = sorted(p for p in methods_root.iterdir() if p.is_dir())

    if not method_dirs:
        raise ValueError(f"No method subdirectories found in folder: {methods_root}")

    return method_dirs


def build_method_to_srt_files(methods_root: Path) -> dict[str, list[Path]]:
    method_dirs = find_method_dirs(methods_root)
    method_to_srt_files: dict[str, list[Path]] = {}

    for method_dir in method_dirs:
        method_name = method_dir.name
        srt_files = find_srt_files(method_dir / "translations")
        method_to_srt_files[method_name] = srt_files

    return method_to_srt_files


def build_new_translations_for_items(
    data: dict,
    method_to_srt_files: dict[str, list[Path]],
) -> dict[Any, dict[str, dict[str, str]]]:
    """
    Returns:
      item_id -> method_name -> run_number(str) -> translation_text

    Run numbers are taken from filenames, not assigned sequentially.
    """
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a top-level object.")

    if "items" not in data:
        raise ValueError('Input JSON must contain a top-level "items" field.')

    if not isinstance(data["items"], list):
        raise ValueError('Top-level "items" must be a list.')

    # Keep both path and parsed map so we can use filename-derived run numbers
    method_to_runs: dict[str, list[tuple[str, Path, dict[int, str]]]] = {}

    for method_name, srt_files in method_to_srt_files.items():
        runs_for_method = []
        seen_run_numbers = set()

        for path in srt_files:
            run_number = extract_run_number_from_filename(path)
            if run_number in seen_run_numbers:
                raise ValueError(
                    f"Duplicate run number {run_number} for method {method_name}. "
                    f"Check filenames in {path.parent}"
                )
            seen_run_numbers.add(run_number)
            runs_for_method.append((run_number, path, load_srt_as_map(path)))

        runs_for_method.sort(key=lambda x: int(x[0]))
        method_to_runs[method_name] = runs_for_method

    result: dict[Any, dict[str, dict[str, str]]] = {}

    for item in data["items"]:
        if not isinstance(item, dict):
            raise ValueError("Each item in 'items' must be an object.")

        item_id = item.get("id", "<no id>")
        segments = validate_segment_list(item, item_id)

        item_methods: dict[str, dict[str, str]] = {}

        for method_name, runs in method_to_runs.items():
            method_runs: dict[str, str] = {}

            for run_number, path, srt_map in runs:
                print(
                    f"Mapping item {item_id} segments {segments} "
                    f"to method {method_name} run {run_number} from {path.name}..."
                )
                translation = collect_translation(segments, srt_map, item_id)
                method_runs[run_number] = translation

            item_methods[method_name] = method_runs

        result[item_id] = item_methods

    return result


def item_key(item: dict) -> Any:
    return item.get("id", "<no id>")


def get_existing_max_run_number(method_runs: dict) -> int:
    numeric_keys = [int(k) for k in method_runs.keys() if str(k).isdigit()]
    return max(numeric_keys) if numeric_keys else 0


def merge_method_runs(
    existing_method_runs: dict[str, str],
    incoming_method_runs: dict[str, str],
) -> dict[str, str]:
    """
    Keep existing runs.
    Add only incoming runs whose run-number key is not already present.
    Do not renumber anything.
    """
    merged = dict(existing_method_runs)

    def sort_key(k: str):
        return (0, int(k)) if str(k).isdigit() else (1, str(k))

    for run_key in sorted(incoming_method_runs.keys(), key=sort_key):
        if run_key in merged:
            print(f"Skipping existing run {run_key}")
            continue
        merged[run_key] = incoming_method_runs[run_key]
        print(f"Added new run {run_key}")

    return merged

def ensure_translations_eng(item: dict) -> dict[str, Any]:
    translations = item.get("translations")
    if translations is None:
        translations = {}
        item["translations"] = translations
    elif not isinstance(translations, dict):
        raise ValueError(
            f'Item {item.get("id", "<no id>")}: "translations" must be an object if present.'
        )

    eng = translations.get("eng")
    if eng is None:
        eng = {}
        translations["eng"] = eng
    elif not isinstance(eng, dict):
        raise ValueError(
            f'Item {item.get("id", "<no id>")}: translations["eng"] must be an object if present.'
        )

    return eng


def merge_into_existing_json(
    base_data: dict,
    incoming_template_data: dict,
    new_translations_by_item: dict[Any, dict[str, dict[str, str]]],
    model_name: str,
) -> dict:
    if not isinstance(base_data, dict):
        raise ValueError("Base JSON must be a top-level object.")
    if "items" not in base_data:
        raise ValueError('Base JSON must contain a top-level "items" field.')
    if not isinstance(base_data["items"], list):
        raise ValueError('Base JSON field "items" must be a list.')

    merged = dict(base_data)
    merged["model"] = model_name

    existing_items = merged["items"]
    existing_index: dict[Any, dict] = {}

    for item in existing_items:
        if not isinstance(item, dict):
            raise ValueError("Each item in base_data['items'] must be an object.")
        existing_index[item_key(item)] = item

    incoming_template_index: dict[Any, dict] = {}
    for item in incoming_template_data["items"]:
        if not isinstance(item, dict):
            raise ValueError("Each item in incoming_template_data['items'] must be an object.")
        incoming_template_index[item_key(item)] = item

    print(f"\n[DEBUG] Base JSON items count: {len(existing_items)}")
    print(f"[DEBUG] Incoming item ids count: {len(new_translations_by_item)}")

    for inc_item_id, methods_dict in new_translations_by_item.items():
        print(f"\n[DEBUG] Merging item_id={inc_item_id}")

        if inc_item_id not in existing_index:
            if inc_item_id not in incoming_template_index:
                raise ValueError(
                    f"Internal error: incoming item {inc_item_id} not found in template JSON."
                )
            new_item = dict(incoming_template_index[inc_item_id])
            merged["items"].append(new_item)
            existing_index[inc_item_id] = new_item
            print(f"[DEBUG]   item was absent in base JSON and was appended")

        target_item = existing_index[inc_item_id]
        eng_translations = ensure_translations_eng(target_item)

        print(f"[DEBUG]   existing methods before merge: {sorted(eng_translations.keys())}")

        for method_name, incoming_method_runs in methods_dict.items():
            existing_method_runs = eng_translations.get(method_name, {})
            if existing_method_runs is None:
                existing_method_runs = {}
            if not isinstance(existing_method_runs, dict):
                raise ValueError(
                    f'Item {inc_item_id}: translations["eng"]["{method_name}"] '
                    f"must be an object if present."
                )

            print(
                f"[DEBUG]   method={method_name} "
                f"existing_run_count={len(existing_method_runs)} "
                f"incoming_run_count={len(incoming_method_runs)}"
            )

            eng_translations[method_name] = merge_method_runs(
                existing_method_runs,
                incoming_method_runs,
            )

            print(
                f"[DEBUG]   method={method_name} "
                f"final_run_count={len(eng_translations[method_name])}"
            )

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an outer folder whose subdirectories are methods and whose method "
            "subdirectories contain translated SRT files. Read a JSON file with items "
            'containing "segment_number": [list of subtitle indices], then write out '
            "the same JSON with translations nested as "
            "translations['eng'][METHOD][RUN_NUMBER]. "
            "If the output JSON already exists, new runs are appended and new methods are added."
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
    output_path = Path(args.output_file)

    if not json_path.is_file():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    try:
        incoming_template_data = load_json(json_path)
        method_to_srt_files = build_method_to_srt_files(methods_root)

        new_translations_by_item = build_new_translations_for_items(
            incoming_template_data,
            method_to_srt_files,
        )

        if output_path.exists():
            print(f"Output file exists, loading as merge base: {output_path}")
            base_data = load_json(output_path)
        else:
            print(f"Output file does not exist, using input JSON as base: {json_path}")
            base_data = incoming_template_data

        merged = merge_into_existing_json(
            base_data=base_data,
            incoming_template_data=incoming_template_data,
            new_translations_by_item=new_translations_by_item,
            model_name=args.model_name,
        )

        # print(f"\n[DEBUG] Output file exists before run: {output_path.exists()}")
        # if output_path.exists():
        #     print(f"[DEBUG] Merging into existing output: {output_path}")
        # else:
        #     print(f"[DEBUG] Creating new output from template: {json_path}")

        save_json(merged, output_path)
        print(f"Saved merged output to: {output_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()