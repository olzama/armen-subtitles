#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import re

import pysrt

SUPPORTED_SRT_EXTENSIONS = {".srt", ".txt"}


def extract_run_number_from_filename(path):
    m = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if not m:
        raise ValueError(
            f"Could not extract run number from filename: {path.name}. "
            f"Expected something like translation-12.txt"
        )
    return str(int(m.group(1)))


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_srt_as_map(srt_path):
    subs = pysrt.open(str(srt_path), encoding="utf-8")
    srt_map = {}
    for sub in subs:
        text = sub.text.replace("\r\n", "\n").replace("\r", "\n").strip()
        srt_map[sub.index] = text
    return srt_map


def validate_segment_list(item, item_id):
    if "segment_number" not in item:
        raise ValueError(f'Item {item_id} is missing required field "segment_number".')
    segments = item["segment_number"]
    if not isinstance(segments, list):
        raise ValueError(f'Item {item_id}: "segment_number" must be a list of integers.')
    if not all(isinstance(x, int) for x in segments):
        raise ValueError(f'Item {item_id}: all values in "segment_number" must be integers.')
    if not segments:
        raise ValueError(f'Item {item_id}: "segment_number" must not be empty.')
    return segments


def normalize_text_for_display(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def join_segments(segment_ids, srt_map):
    parts = [normalize_text_for_display(srt_map[s]) for s in segment_ids if s in srt_map]
    return "\n".join(p for p in parts if p).strip()


def parse_segment_input(s):
    s = s.strip()
    if not s:
        raise ValueError("Empty input.")

    result = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            if a > b:
                raise ValueError(f"Invalid range: {part}")
            result.extend(range(a, b + 1))
        else:
            result.append(int(part))
    return result


def suggest_context_windows(expected_segments, srt_map, window=2):
    expected_segments = sorted(expected_segments)
    start = expected_segments[0]
    end = expected_segments[-1]

    suggestions = []
    seen = set()
    candidate_lists = []

    candidate_lists.append(expected_segments)

    for delta in range(-window, window + 1):
        shifted = [s + delta for s in expected_segments]
        candidate_lists.append(shifted)

    for extra_left in range(0, window + 1):
        for extra_right in range(0, window + 1):
            candidate = list(range(start - extra_left, end + extra_right + 1))
            candidate_lists.append(candidate)

    for delta in range(-window, window + 1):
        for extra_left in range(0, window + 1):
            for extra_right in range(0, window + 1):
                candidate = list(range(start + delta - extra_left, end + delta + extra_right + 1))
                candidate_lists.append(candidate)

    for cand in candidate_lists:
        if not cand:
            continue
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        if all(seg in srt_map for seg in cand):
            suggestions.append((cand, join_segments(cand, srt_map)))

    return suggestions


def get_item_label(item):
    bits = []
    if "id" in item:
        bits.append(f'id={item["id"]}')
    if "character" in item:
        bits.append(f'character={item["character"]}')
    return ", ".join(bits) if bits else "<unknown item>"


def get_reference_translation(item):
    translations = item.get("reference", {})
    reference = translations.get("eng")

    if reference is None:
        return "[NO REFERENCE AVAILABLE]"

    if isinstance(reference, str):
        return reference.strip()
    elif isinstance(reference, dict):
        return json.dumps(reference, ensure_ascii=False, indent=2)
    return str(reference).strip()


def describe_applied_drift(run_number, offset, item_id, expected_segments, proposed_segments):
    print(
        f"Applying learned method-local drift for run {run_number} on item {item_id}: "
        f"{offset:+d} | {expected_segments} -> {proposed_segments}"
    )


def prompt_yes_no(prompt, default=None):
    while True:
        suffix = " [y/n]: "
        if default is True:
            suffix = " [Y/n]: "
        elif default is False:
            suffix = " [y/N]: "

        ans = input(prompt + suffix).strip().lower()

        if not ans and default is not None:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False

        print("Please answer y or n.")


def compute_simple_offset(expected_segments, corrected_segments):
    if len(expected_segments) != len(corrected_segments):
        return None
    offsets = [c - e for e, c in zip(expected_segments, corrected_segments)]
    if len(set(offsets)) == 1:
        return offsets[0]
    return None


def apply_offset_if_possible(expected_segments, offset, srt_map):
    shifted = [s + offset for s in expected_segments]
    if all(seg in srt_map for seg in shifted):
        return shifted
    return expected_segments


def interactive_confirm_item(
        item,
        expected_segments,
        proposed_segments,
        srt_map,
        method_name,
        run_number,
        suggestion_window,
):
    item_label = get_item_label(item)
    reference_translation = get_reference_translation(item)
    current_window = suggestion_window

    while True:
        print("\n" + "=" * 80)
        print(f"METHOD: {method_name} | RUN: {run_number} | ITEM: {item_label}")
        print(f"Reference segments: {expected_segments}")
        print(f"Proposed segments:  {proposed_segments}")
        print("-" * 80)
        print("REFERENCE TRANSLATION:")
        print(reference_translation)
        print("-" * 80)
        print("PROPOSED MAPPED TEXT:")
        print(join_segments(proposed_segments, srt_map) or "[EMPTY]")
        print("-" * 80)

        if prompt_yes_no("Does this mapping look correct?", default=True):
            offset = compute_simple_offset(expected_segments, proposed_segments)
            return proposed_segments, join_segments(proposed_segments, srt_map), offset

        while True:
            print(f"\nNearby suggestions (window={current_window}):")
            suggestions = suggest_context_windows(expected_segments, srt_map, window=current_window)

            shown = 0
            for segs, text in suggestions:
                if shown >= 8:
                    break
                print(f"\nSuggestion {shown + 1}: {segs}")
                print(text or "[EMPTY]")
                shown += 1

            print("\nEnter corrected segment numbers.")
            print("Formats accepted: 15,16,17   or   15-17   or   14,15-17,20")
            print("Enter 'm' to manually type or paste the translated text.")
            print("Enter 's' to show suggestions again.")
            print("Enter 'w' to widen the suggestion window.")
            print("Enter 'e' to keep the expected segments.")
            user_in = input("Corrected segments: ").strip()

            if user_in.lower() == "m":
                print("\nPaste or type the translation text.")
                print("(Press Enter, then Ctrl-D on Linux/Mac or Ctrl-Z on Windows to submit):")
                manual_text = sys.stdin.read().strip()
                if manual_text:
                    # Return None for offset so drift isn't calculated on a manual text entry
                    return expected_segments, manual_text, None
                else:
                    print("Manual entry was empty. Please try again.")
                    continue

            if user_in.lower() == "s":
                continue

            if user_in.lower() == "w":
                current_window += 2
                continue

            if user_in.lower() == "e":
                corrected_segments = expected_segments
            else:
                try:
                    corrected_segments = parse_segment_input(user_in)
                except Exception as exc:
                    print(f"Could not parse input: {exc}")
                    continue

            missing = [seg for seg in corrected_segments if seg not in srt_map]
            if missing:
                print(f"These segments are not present in the SRT: {missing}")
                continue

            proposed_segments = corrected_segments
            break  # break inner loop to re-evaluate the new proposed_segments

def resolve_target_path(target_path):
    """
    Intelligently resolves the target path to accommodate a root folder, a specific method
    folder, a translations folder, or even a specific SRT/TXT file.
    """
    target = Path(target_path)
    method_to_files = defaultdict(list)

    if not target.exists():
        raise ValueError(f"Target path does not exist: {target}")

    # Case 1: Specific File
    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_SRT_EXTENSIONS:
            raise ValueError(f"File must be an SRT or TXT file: {target}")
        # Infer method name from standard structure <method_name>/translations/<file>
        method_name = target.parent.parent.name if target.parent.name == "translations" else target.parent.name
        method_to_files[method_name].append(target)
        return dict(method_to_files)

    # Case 2: Folder
    if target.is_dir():
        # 2a: Directory contains SRT files directly (e.g. user pointed to 'translations' folder)
        direct_files = sorted(
            [p for p in target.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SRT_EXTENSIONS])
        if direct_files:
            method_name = target.parent.name if target.name == "translations" else target.name
            method_to_files[method_name].extend(direct_files)
            return dict(method_to_files)

        # 2b: Directory is a single method folder containing a 'translations' subfolder
        trans_dir = target / "translations"
        if trans_dir.is_dir():
            sub_files = sorted(
                [p for p in trans_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SRT_EXTENSIONS])
            if sub_files:
                method_to_files[target.name].extend(sub_files)
                return dict(method_to_files)

        # 2c: Directory is the root methods folder containing multiple method subdirectories
        for method_dir in sorted(p for p in target.iterdir() if p.is_dir()):
            trans_sub_dir = method_dir / "translations"
            if trans_sub_dir.is_dir():
                files = sorted([p for p in trans_sub_dir.iterdir() if
                                p.is_file() and p.suffix.lower() in SUPPORTED_SRT_EXTENSIONS])
                if files:
                    method_to_files[method_dir.name] = files

    if not method_to_files:
        raise ValueError(f"Could not find any supported SRT/TXT files in {target}")

    return dict(method_to_files)


def merge_method_runs(existing_method_runs, incoming_method_runs):
    merged = dict(existing_method_runs)

    def sort_key(k):
        return (0, int(k)) if str(k).isdigit() else (1, str(k))

    for run_key in sorted(incoming_method_runs.keys(), key=sort_key):
        if run_key in merged:
            print(f"Skipping existing run {run_key} in final merge pass")
            continue
        merged[run_key] = incoming_method_runs[run_key]
        print(f"Added run {run_key}")

    return merged


def ensure_translations_eng(item):
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
        base_data,
        incoming_template_data,
        new_translations_by_item,
        model_name,
):
    if not isinstance(base_data, dict):
        raise ValueError("Base JSON must be a top-level object.")
    if "items" not in base_data:
        raise ValueError('Base JSON must contain a top-level "items" field.')
    if not isinstance(base_data["items"], list):
        raise ValueError('Base JSON field "items" must be a list.')

    merged = dict(base_data)
    merged["model"] = model_name

    existing_items = merged["items"]
    existing_index = {}

    for item in existing_items:
        if not isinstance(item, dict):
            raise ValueError("Each item in base_data['items'] must be an object.")
        existing_index[str(item.get("id", "<no id>"))] = item

    incoming_template_index = {}
    for item in incoming_template_data["items"]:
        if not isinstance(item, dict):
            raise ValueError("Each item in incoming_template_data['items'] must be an object.")
        incoming_template_index[str(item.get("id", "<no id>"))] = item

    for inc_item_id, methods_dict in new_translations_by_item.items():
        if inc_item_id not in existing_index:
            if inc_item_id not in incoming_template_index:
                raise ValueError(
                    f"Internal error: incoming item {inc_item_id} not found in template JSON."
                )
            new_item = dict(incoming_template_index[inc_item_id])
            merged["items"].append(new_item)
            existing_index[inc_item_id] = new_item

        target_item = existing_index[inc_item_id]
        eng_translations = ensure_translations_eng(target_item)

        for method_name, incoming_method_runs in methods_dict.items():
            existing_method_runs = eng_translations.get(method_name, {})
            if existing_method_runs is None:
                existing_method_runs = {}
            if not isinstance(existing_method_runs, dict):
                raise ValueError(
                    f'Item {inc_item_id}: translations["eng"]["{method_name}"] must be an object if present.'
                )

            eng_translations[method_name] = merge_method_runs(
                existing_method_runs,
                incoming_method_runs,
            )

    return merged


def get_existing_translations(base_data):
    existing = defaultdict(lambda: defaultdict(dict))
    if not base_data or "items" not in base_data:
        return existing
    for item in base_data["items"]:
        item_id = str(item.get("id"))
        eng = item.get("translations", {}).get("eng", {})
        if isinstance(eng, dict):
            for method, runs in eng.items():
                if isinstance(runs, dict):
                    for run_num, text in runs.items():
                        existing[item_id][method][str(run_num)] = text
    return existing


def build_interactive_translations_for_items(
        data,
        method_to_srt_files,
        suggestion_window,
        existing_translations,
        progress_data,
        progress_file
):
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a top-level object.")
    if "items" not in data:
        raise ValueError('Input JSON must contain a top-level "items" field.')
    if not isinstance(data["items"], list):
        raise ValueError('Top-level "items" must be a list.')

    items = data["items"]

    method_to_runs = {}
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

    # CAST: Initialize result with string keys
    result = {str(item["id"]): {} for item in items}

    method_names = list(method_to_runs.keys())
    if not method_names:
        return result

    for method_index, method_name in enumerate(method_names):
        print("\n" + "#" * 100)
        print(f"PROCESSING METHOD {method_index + 1}/{len(method_names)}: {method_name}")
        print("#" * 100)

        runs = method_to_runs[method_name]

        # Drift is local to this method only.
        learned_offsets_by_run = {}
        observed_offsets_by_run = defaultdict(list)

        # CAST: Initialize method outputs with string keys
        method_outputs = {str(item["id"]): {} for item in items}

        if method_index == 0:
            # First method: always per-item interactive validation.
            for item in items:
                # CAST: Ensure item_id is a string for all dictionary lookups
                item_id = str(item["id"])
                expected_segments = validate_segment_list(item, item_id)

                for run_number, path, srt_map in runs:
                    if item_id in existing_translations and method_name in existing_translations[item_id] and str(
                            run_number) in existing_translations[item_id][method_name]:
                        print(f"Skipping previously verified item {item_id} | method {method_name} | run {run_number}")
                        method_outputs[item_id][run_number] = existing_translations[item_id][method_name][
                            str(run_number)]
                        continue

                    proposed_segments = expected_segments[:]

                    if run_number in learned_offsets_by_run:
                        offset = learned_offsets_by_run[run_number]
                        shifted_segments = apply_offset_if_possible(
                            expected_segments,
                            offset,
                            srt_map,
                        )
                        if shifted_segments != expected_segments:
                            describe_applied_drift(
                                run_number=run_number,
                                offset=offset,
                                item_id=item_id,
                                expected_segments=expected_segments,
                                proposed_segments=shifted_segments,
                            )
                        proposed_segments = shifted_segments

                    corrected_segments, text, offset = interactive_confirm_item(
                        item=item,
                        expected_segments=expected_segments,
                        proposed_segments=proposed_segments,
                        srt_map=srt_map,
                        method_name=method_name,
                        run_number=run_number,
                        suggestion_window=suggestion_window,
                    )

                    method_outputs[item_id][run_number] = text

                    # Update progress mapping immediately
                    if item_id not in progress_data: progress_data[item_id] = {}
                    if method_name not in progress_data[item_id]: progress_data[item_id][method_name] = {}
                    progress_data[item_id][method_name][run_number] = text
                    save_json(progress_data, progress_file)

                    if offset is not None:
                        observed_offsets_by_run[run_number].append(offset)
                        learned_offsets_by_run[run_number] = Counter(
                            observed_offsets_by_run[run_number]
                        ).most_common(1)[0][0]
                        print(
                            f"Learned method-local drift for run {run_number}: "
                            f"{learned_offsets_by_run[run_number]:+d}"
                        )

            for item in items:
                result[str(item["id"])][method_name] = method_outputs[str(item["id"])]

        else:
            # Later methods: check if any unreviewed runs exist.
            unreviewed_runs = []
            for item in items:
                # CAST: Ensure item_id is a string
                item_id = str(item["id"])
                expected_segments = validate_segment_list(item, item_id)

                for run_number, path, srt_map in runs:
                    if item_id in existing_translations and method_name in existing_translations[item_id] and str(
                            run_number) in existing_translations[item_id][method_name]:
                        method_outputs[item_id][run_number] = existing_translations[item_id][method_name][
                            str(run_number)]
                    else:
                        proposed_segments = expected_segments[:]
                        method_outputs[item_id][run_number] = join_segments(proposed_segments, srt_map)
                        unreviewed_runs.append((item, run_number, path, srt_map))

            if not unreviewed_runs:
                print(f"All items for method '{method_name}' are already reviewed. Skipping.")
                for item in items:
                    result[str(item["id"])][method_name] = method_outputs[str(item["id"])]
                continue

            print("\nMethod-level preview:")
            for item in items:
                item_id = str(item["id"])
                print("\n" + "-" * 80)
                print(f"ITEM {get_item_label(item)}")
                print("\nREFERENCE TRANSLATION:")
                print(get_reference_translation(item))
                for run_number, _, _ in runs:
                    print(f"\nRun {run_number}:")
                    print(method_outputs[item_id][run_number] or "[EMPTY]")

            approved = prompt_yes_no(
                f"\nApprove mapping for the whole method '{method_name}'?",
                default=True,
            )

            if approved:
                for item in items:
                    item_id = str(item["id"])
                    if item_id not in progress_data: progress_data[item_id] = {}
                    if method_name not in progress_data[item_id]: progress_data[item_id][method_name] = {}

                    for run_number, _, _ in runs:
                        result[item_id][method_name] = method_outputs[item_id]
                        progress_data[item_id][method_name][run_number] = method_outputs[item_id][run_number]

                # Save progress mapping immediately on batch approval
                save_json(progress_data, progress_file)
                print(f"Method '{method_name}' approved and progress saved.")
            else:
                print(
                    f"Method '{method_name}' not approved. "
                    f"Falling back to per-item validation."
                )

                # CAST: Reset method_outputs with string keys
                method_outputs = {str(item["id"]): {} for item in items}

                for item in items:
                    item_id = str(item["id"])
                    expected_segments = validate_segment_list(item, item_id)

                    for run_number, path, srt_map in runs:
                        if item_id in existing_translations and method_name in existing_translations[item_id] and str(
                                run_number) in existing_translations[item_id][method_name]:
                            print(
                                f"Skipping previously verified item {item_id} | method {method_name} | run {run_number}")
                            method_outputs[item_id][run_number] = existing_translations[item_id][method_name][
                                str(run_number)]
                            continue

                        proposed_segments = expected_segments[:]

                        if run_number in learned_offsets_by_run:
                            offset = learned_offsets_by_run[run_number]
                            shifted_segments = apply_offset_if_possible(
                                expected_segments,
                                offset,
                                srt_map,
                            )
                            if shifted_segments != expected_segments:
                                describe_applied_drift(
                                    run_number=run_number,
                                    offset=offset,
                                    item_id=item_id,
                                    expected_segments=expected_segments,
                                    proposed_segments=shifted_segments,
                                )
                            proposed_segments = shifted_segments
                        corrected_segments, text, offset = interactive_confirm_item(
                            item=item,
                            expected_segments=expected_segments,
                            proposed_segments=proposed_segments,
                            srt_map=srt_map,
                            method_name=method_name,
                            run_number=run_number,
                            suggestion_window=suggestion_window,
                        )

                        method_outputs[item_id][run_number] = text

                        # Update progress mapping immediately
                        if item_id not in progress_data: progress_data[item_id] = {}
                        if method_name not in progress_data[item_id]: progress_data[item_id][method_name] = {}
                        progress_data[item_id][method_name][run_number] = text
                        save_json(progress_data, progress_file)

                        if offset is not None:
                            observed_offsets_by_run[run_number].append(offset)
                            learned_offsets_by_run[run_number] = Counter(
                                observed_offsets_by_run[run_number]
                            ).most_common(1)[0][0]
                            print(
                                f"Learned method-local drift for run {run_number}: "
                                f"{learned_offsets_by_run[run_number]:+d}"
                            )

                for item in items:
                    result[str(item["id"])][method_name] = method_outputs[str(item["id"])]

    return result

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively map subtitle segments from translated SRT files into the JSON. "
            "For the first method, the user approves item by item; for later methods, "
            "the user can approve the whole method or fall back to item-by-item correction."
        )
    )
    # Require the film name
    parser.add_argument("film_name", help="Identifier for the film (used to create a unique progress file)")
    parser.add_argument(
        "input_target",
        help="Path to a root folder of methods, a specific method folder, a translations folder, or a specific SRT/TXT file."
    )
    parser.add_argument("json_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Output filename")
    parser.add_argument("model_name", help="Model name to store in the output JSON")
    parser.add_argument(
        "--suggestion-window",
        type=int,
        default=2,
        help="How many neighboring segments to include in context suggestions. Default: 2"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_target = Path(args.input_target)
    json_path = Path(args.json_file)
    output_path = Path(args.output_file)

    # Dynamically name the progress file so films never overwrite each other
    progress_file_path = Path(f"{args.film_name}_progress.json")

    if not json_path.is_file():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    try:
        incoming_template_data = load_json(json_path)

        # Flexibly resolve whatever path the user threw at us
        method_to_srt_files = resolve_target_path(input_target)

        # 1. Load progress mapping (if exists) for THIS specific film
        if progress_file_path.exists():
            print(f"Loading progress tracking from: {progress_file_path}")
            progress_data = load_json(progress_file_path)
        else:
            progress_data = {}

        # 2. Load partial output file (if exists) as base
        if output_path.exists():
            print(f"Output file exists, loading as merge base: {output_path}")
            base_data = load_json(output_path)
        else:
            print(f"Output file does not exist, using input JSON as base: {json_path}")
            base_data = incoming_template_data

        # 3. Compile all existing decisions to resume smoothly
        existing_translations = get_existing_translations(base_data)

        # Merge discrete progress mapping into general existing translations buffer
        for item_id, methods in progress_data.items():
            for method, runs in methods.items():
                for run_num, text in runs.items():
                    existing_translations[item_id][method][str(run_num)] = text

        new_translations_by_item = build_interactive_translations_for_items(
            data=incoming_template_data,
            method_to_srt_files=method_to_srt_files,
            suggestion_window=args.suggestion_window,
            existing_translations=existing_translations,
            progress_data=progress_data,
            progress_file=progress_file_path
        )

        merged = merge_into_existing_json(
            base_data=base_data,
            incoming_template_data=incoming_template_data,
            new_translations_by_item=new_translations_by_item,
            model_name=args.model_name,
        )

        save_json(merged, output_path)
        print(f"\nSuccessfully saved updated merged output to: {output_path}")

    except KeyboardInterrupt:
        print(f"\nInterrupted by user. Progress up to the last mapped item was saved to {progress_file_path}.",
              file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()