#!/usr/bin/env python3

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import re

import pysrt
from lang_utils import normalize_lang

SUPPORTED_SRT_EXTENSIONS = {".srt", ".txt"}
DEFAULT_BATCH_SIZE = 6


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Text / segment helpers
# ---------------------------------------------------------------------------

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

    candidate_lists = [expected_segments]

    for delta in range(-window, window + 1):
        candidate_lists.append([s + delta for s in expected_segments])

    for extra_left in range(0, window + 1):
        for extra_right in range(0, window + 1):
            candidate_lists.append(list(range(start - extra_left, end + extra_right + 1)))

    for delta in range(-window, window + 1):
        for extra_left in range(0, window + 1):
            for extra_right in range(0, window + 1):
                candidate_lists.append(
                    list(range(start + delta - extra_left, end + delta + extra_right + 1))
                )

    seen = set()
    suggestions = []
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


# ---------------------------------------------------------------------------
# Item label / reference helpers
# ---------------------------------------------------------------------------

def get_item_label(item):
    bits = []
    if "id" in item:
        bits.append(f'id={item["id"]}')
    if "character" in item:
        bits.append(f'character={item["character"]}')
    return ", ".join(bits) if bits else "<unknown item>"


def get_reference_translation(item, target_lang):
    translations = item.get("reference", {})
    reference = translations.get(target_lang)

    if reference is None:
        return "[NO REFERENCE AVAILABLE]"
    if isinstance(reference, str):
        return reference.strip()
    if isinstance(reference, dict):
        return json.dumps(reference, ensure_ascii=False, indent=2)
    return str(reference).strip()


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

_similarity_model = None


def _get_similarity_model():
    global _similarity_model
    if _similarity_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformers model...")
        _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _similarity_model


def find_best_match_in_window(expected_segs, reference_text, srt_map, window):
    """
    Search all candidate segment groups within the context window and return
    (best_segs, score) where score is cosine similarity to reference_text.
    Falls back to expected_segs if reference_text is empty or no candidates exist.
    After finding the best group, trims leading/trailing segments when doing so
    does not reduce the similarity score, preferring the shortest match.
    """
    if not reference_text or reference_text == "[NO REFERENCE AVAILABLE]":
        return expected_segs[:], 0.0
    candidates = suggest_context_windows(expected_segs, srt_map, window=window)
    if not candidates:
        return expected_segs[:], 0.0

    import numpy as np
    model = _get_similarity_model()
    texts = [text for _, text in candidates]
    all_texts = [reference_text] + texts
    embeddings = model.encode(all_texts, show_progress_bar=False)
    ref_emb = embeddings[0]
    cand_embs = embeddings[1:]

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    scores = [cosine(ref_emb, e) for e in cand_embs]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_segs = list(candidates[best_idx][0])
    best_score = scores[best_idx]

    # Trim trailing then leading segments using batched encoding.
    # For each direction, pre-encode all candidates in one call then simulate the
    # greedy sequential process using the precomputed scores.
    for make_candidates in (
        lambda segs: [segs[:-i] for i in range(1, len(segs))],   # trailing trim
        lambda segs: [segs[i:] for i in range(1, len(segs))],    # leading trim
    ):
        if len(best_segs) <= 1:
            break
        candidates = [c for c in make_candidates(best_segs) if all(s in srt_map for s in c)]
        if not candidates:
            continue
        embs = model.encode([join_segments(c, srt_map) for c in candidates], show_progress_bar=False)
        scores = [cosine(ref_emb, e) for e in embs]
        for trimmed, score in zip(candidates, scores):
            if score >= best_score - 0.01:
                best_segs = trimmed
                best_score = score
            else:
                break

    return best_segs, best_score


def compute_simple_offset(expected_segments, corrected_segments):
    """Return the uniform offset if all segments shifted by the same amount, else None."""
    if len(expected_segments) != len(corrected_segments):
        return None
    offsets = [c - e for e, c in zip(expected_segments, corrected_segments)]
    if len(set(offsets)) == 1:
        return offsets[0]
    return None


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _save_item_progress(progress_data, item_id, method_name, run_number, text, progress_file):
    """Persist a single item/method/run result to the progress file."""
    progress_data.setdefault(item_id, {}).setdefault(method_name, {})[run_number] = text
    save_json(progress_data, progress_file)


def _save_overrides(item_text_hypotheses, overrides_file):
    """Persist text hypotheses to disk."""
    save_json({"hypothesis": item_text_hypotheses}, overrides_file)


def _is_already_reviewed(item_id, method_name, run_number, existing_translations):
    return (
        item_id in existing_translations
        and method_name in existing_translations[item_id]
        and str(run_number) in existing_translations[item_id][method_name]
    )


# ---------------------------------------------------------------------------
# Interactive confirmation UI
# ---------------------------------------------------------------------------


def _print_confirmation_header(item_label, method_name, run_number,
                               expected_segments, proposed_segments,
                               reference_translation, srt_map,
                               prior_hypothesis=None, context_window=2):
    print("\n" + "=" * 80)
    print(f"METHOD: {method_name} | RUN: {run_number} | ITEM: {item_label}")
    print("-" * 80)
    print("REFERENCE TRANSLATION:")
    print(reference_translation)
    if prior_hypothesis is not None:
        print("-" * 80)
        print("PRIOR ACCEPTED TEXT:")
        print(prior_hypothesis)
    print("-" * 80)
    # Show individual nearby segments so the user knows what numbers to pick
    all_segs = sorted(set(expected_segments) | set(proposed_segments))
    lo = min(all_segs) - context_window
    hi = max(all_segs) + context_window
    print("NEARBY SEGMENTS:")
    for seg_id in range(lo, hi + 1):
        if seg_id not in srt_map:
            continue
        marker = ">>>" if seg_id in proposed_segments else "   "
        text = srt_map[seg_id].replace("\n", " / ")
        print(f"  {marker} [{seg_id}]  {text}")
    print("-" * 80)
    print(f"PROPOSED MAPPED TEXT  (segments {proposed_segments}):")
    print(join_segments(proposed_segments, srt_map) or "[EMPTY]")
    print("-" * 80)


def _parse_segment_numbers(user_in, srt_map):
    """Parse user input as segment numbers. Returns list on success, None on failure."""
    try:
        segs = parse_segment_input(user_in)
    except Exception as exc:
        print(f"  Could not parse: {exc}")
        return None
    missing = [s for s in segs if s not in srt_map]
    if missing:
        print(f"  Segments not in SRT: {missing}")
        return None
    return segs


def _accept_or_edit_text(segs, srt_map):
    """Show text from segs. User presses ENTER to accept or types a trimmed/corrected version.
    Returns accepted text string, or None to go back."""
    text = join_segments(segs, srt_map)
    print(f"\nText from segments {segs}:")
    print(text or "[EMPTY]")
    print("ENTER=accept as-is   type to trim or correct   b=back")
    user_in = input("> ").strip()
    if not user_in:
        return text
    if user_in.lower() == "b":
        return None
    return user_in


def interactive_confirm_item(
        item,
        expected_segments,
        proposed_segments,
        srt_map,
        method_name,
        run_number,
        suggestion_window,
        target_lang,
        prior_hypothesis=None,
):
    item_label = get_item_label(item)
    reference_translation = get_reference_translation(item, target_lang)
    query = prior_hypothesis if prior_hypothesis is not None else reference_translation
    current_window = suggestion_window
    current_segs = list(proposed_segments)

    while True:
        _print_confirmation_header(
            item_label, method_name, run_number,
            expected_segments, current_segs,
            reference_translation, srt_map,
            prior_hypothesis=prior_hypothesis,
            context_window=current_window,
        )
        print("ENTER=accept   n=widen search   [number(s) from list above]=pick segments   e=edit proposed text")
        user_in = input("> ").strip()

        if not user_in:
            offset = compute_simple_offset(expected_segments, current_segs)
            return current_segs, join_segments(current_segs, srt_map), offset

        if user_in.lower() == "n":
            current_window += 2
            new_segs, score = find_best_match_in_window(expected_segments, query, srt_map, current_window)
            print(f"  Window ±{current_window} → {new_segs}  [score {score:.2f}]")
            current_segs = new_segs
            continue

        if user_in.lower() == "e":
            result = _accept_or_edit_text(current_segs, srt_map)
            if result is not None:
                return current_segs, result, None
            continue

        segs = _parse_segment_numbers(user_in, srt_map)
        if segs is not None:
            current_segs = segs
            result = _accept_or_edit_text(current_segs, srt_map)
            if result is not None:
                return current_segs, result, None
            continue

        print("  Enter segment numbers, or: ENTER accept   n widen   e edit")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_target_path(target_path):
    """
    Intelligently resolves the target path to accommodate a root folder, a specific method
    folder, a translations folder, or even a specific SRT/TXT file.
    """
    target = Path(target_path)
    method_to_files = defaultdict(list)

    if not target.exists():
        raise ValueError(f"Target path does not exist: {target}")

    # Case 1: Specific file
    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_SRT_EXTENSIONS:
            raise ValueError(f"File must be an SRT or TXT file: {target}")
        # Infer method name from standard structure <method_name>/translations/<file>
        method_name = (
            target.parent.parent.name if target.parent.name == "translations" else target.parent.name
        )
        method_to_files[method_name].append(target)
        return dict(method_to_files)

    if target.is_dir():
        # Case 2a: Directory contains SRT files directly (e.g. user pointed to 'translations' folder)
        direct_files = sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SRT_EXTENSIONS
        )
        if direct_files:
            method_name = target.parent.name if target.name == "translations" else target.name
            method_to_files[method_name].extend(direct_files)
            return dict(method_to_files)

        # Case 2b: Single method folder with a 'translations' subfolder
        trans_dir = target / "translations"
        if trans_dir.is_dir():
            sub_files = sorted(
                p for p in trans_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_SRT_EXTENSIONS
            )
            if sub_files:
                method_to_files[target.name].extend(sub_files)
                return dict(method_to_files)

        # Case 2c: Root methods folder containing multiple method subdirectories
        for method_dir in sorted(p for p in target.iterdir() if p.is_dir()):
            trans_sub_dir = method_dir / "translations"
            if trans_sub_dir.is_dir():
                files = sorted(
                    p for p in trans_sub_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_SRT_EXTENSIONS
                )
                if files:
                    method_to_files[method_dir.name] = files

    if not method_to_files:
        raise ValueError(f"Could not find any supported SRT/TXT files in {target}")

    return dict(method_to_files)


# ---------------------------------------------------------------------------
# JSON merge helpers
# ---------------------------------------------------------------------------

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


def ensure_translations_for_lang(item, target_lang):
    translations = item.get("translations")
    if translations is None:
        translations = {}
        item["translations"] = translations
    elif not isinstance(translations, dict):
        raise ValueError(
            f'Item {item.get("id", "<no id>")}: "translations" must be an object if present.'
        )

    lang_translations = translations.get(target_lang)
    if lang_translations is None:
        lang_translations = {}
        translations[target_lang] = lang_translations
    elif not isinstance(lang_translations, dict):
        raise ValueError(
            f'Item {item.get("id", "<no id>")}: translations["{target_lang}"] must be an object if present.'
        )

    return lang_translations


def _validate_base_data(base_data):
    if not isinstance(base_data, dict):
        raise ValueError("Base JSON must be a top-level object.")
    if "items" not in base_data:
        raise ValueError('Base JSON must contain a top-level "items" field.')
    if not isinstance(base_data["items"], list):
        raise ValueError('Base JSON field "items" must be a list.')


def _build_item_index(items, label):
    index = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Each item in {label} must be an object.")
        index[str(item.get("id", "<no id>"))] = item
    return index


def _merge_item_translations(target_item, inc_item_id, methods_dict, incoming_template_index, target_lang):
    """Merge new method/run translations into a single target item."""
    lang_translations = ensure_translations_for_lang(target_item, target_lang)

    for method_name, incoming_method_runs in methods_dict.items():
        existing_method_runs = lang_translations.get(method_name) or {}
        if not isinstance(existing_method_runs, dict):
            raise ValueError(
                f'Item {inc_item_id}: translations["{target_lang}"]["{method_name}"] '
                f'must be an object if present.'
            )
        lang_translations[method_name] = merge_method_runs(existing_method_runs, incoming_method_runs)


def merge_into_existing_json(
        base_data,
        incoming_template_data,
        new_translations_by_item,
        model_name,
        target_lang,
):
    _validate_base_data(base_data)

    merged = dict(base_data)
    merged["model"] = model_name

    existing_index = _build_item_index(merged["items"], "base_data['items']")
    incoming_template_index = _build_item_index(
        incoming_template_data["items"], "incoming_template_data['items']"
    )

    for inc_item_id, methods_dict in new_translations_by_item.items():
        if inc_item_id not in existing_index:
            if inc_item_id not in incoming_template_index:
                raise ValueError(
                    f"Internal error: incoming item {inc_item_id} not found in template JSON."
                )
            new_item = dict(incoming_template_index[inc_item_id])
            merged["items"].append(new_item)
            existing_index[inc_item_id] = new_item

        _merge_item_translations(
            existing_index[inc_item_id], inc_item_id, methods_dict, incoming_template_index, target_lang
        )

    return merged


def get_existing_translations(base_data, target_lang):
    existing = defaultdict(lambda: defaultdict(dict))
    if not base_data or "items" not in base_data:
        return existing
    for item in base_data["items"]:
        item_id = str(item.get("id"))
        lang_translations = item.get("translations", {}).get(target_lang, {})
        if isinstance(lang_translations, dict):
            for method, runs in lang_translations.items():
                if isinstance(runs, dict):
                    for run_num, text in runs.items():
                        existing[item_id][method][str(run_num)] = text
    return existing


# ---------------------------------------------------------------------------
# Interactive translation building
# ---------------------------------------------------------------------------

def _load_method_runs(method_to_srt_files):
    """Load SRT files into (run_number, path, srt_map) tuples, sorted by run number."""
    method_to_runs = {}
    for method_name, srt_files in method_to_srt_files.items():
        seen_run_numbers = set()
        runs = []
        for path in srt_files:
            run_number = extract_run_number_from_filename(path)
            if run_number in seen_run_numbers:
                raise ValueError(
                    f"Duplicate run number {run_number} for method {method_name}. "
                    f"Check filenames in {path.parent}"
                )
            seen_run_numbers.add(run_number)
            runs.append((run_number, path, load_srt_as_map(path)))
        runs.sort(key=lambda x: int(x[0]))
        method_to_runs[method_name] = runs
    return method_to_runs


def _display_batch(batch, srt_map, method_name, run_number, batch_num, total_batches, target_lang):
    """Display a numbered list of (item, expected_segs, proposed_segs, source_note) for batch review."""
    print(f"\n{'=' * 80}")
    print(f"METHOD: {method_name} | RUN: {run_number} | Batch {batch_num}/{total_batches}")
    print('=' * 80)

    for idx, (item, expected_segs, proposed_segs, source_note) in enumerate(batch, start=1):
        if proposed_segs != expected_segs:
            offset = compute_simple_offset(expected_segs, proposed_segs)
            seg_info = (
                f"{expected_segs} -> {proposed_segs}  [{offset:+d}]"
                if offset is not None
                else f"{expected_segs} -> {proposed_segs}"
            )
        else:
            seg_info = str(proposed_segs)

        if source_note:
            seg_info += f"  {source_note}"

        ref = get_reference_translation(item, target_lang).replace("\n", "\n              ")
        mapped = (join_segments(proposed_segs, srt_map) or "[EMPTY]").replace("\n", "\n              ")

        print(f"\n  [{idx}] {get_item_label(item)}")
        print(f"       Segments: {seg_info}")
        print(f"       REF:      {ref}")
        print(f"       MAP:      {mapped}")

    print()


def _prompt_batch_flags(n_items):
    """Prompt user to flag items for correction. Returns a set of 1-based indices."""
    while True:
        raw = input("Press ENTER to approve all, or enter numbers to flag (e.g. 2 5): ").strip()
        if not raw:
            return set()
        try:
            flagged = set()
            valid = True
            for tok in raw.split():
                n = int(tok)
                if 1 <= n <= n_items:
                    flagged.add(n)
                else:
                    print(f"  {n} is out of range (1-{n_items}). Try again.")
                    valid = False
                    break
            if valid:
                return flagged
        except ValueError:
            print("  Please enter numbers separated by spaces (e.g. 2 5).")


def _build_item_proposal(item_id, item, expected_segs,
                         item_text_hypotheses, srt_map, suggestion_window, target_lang):
    """Return (proposed_segs, source_note) for one unreviewed item.

    If a prior accepted text exists for this item, it is used as the similarity
    query (more reliable than the reference).  Falls back to the reference
    translation when no hypothesis is available.
    """
    hypothesis = item_text_hypotheses.get(item_id)
    query = hypothesis if hypothesis is not None else get_reference_translation(item, target_lang)
    label = "prior" if hypothesis is not None else "sim"
    best_segs, score = find_best_match_in_window(expected_segs, query, srt_map, suggestion_window)
    show_score = best_segs != expected_segs or score < 0.5
    source_note = f"[{label} {score:.2f}]" if show_score else None
    return best_segs, source_note


def _process_batch(batch, srt_map, method_name, run_number, batch_idx, n_batches,
                   method_outputs, progress_data, progress_file,
                   item_text_hypotheses, overrides_file, target_lang):
    """Display one batch, save approved items, return flagged ones."""
    _display_batch(batch, srt_map, method_name, run_number, batch_idx + 1, n_batches, target_lang)
    flagged_1based = _prompt_batch_flags(len(batch))

    hypotheses_updated = False
    flagged_items = []
    for i, (item, expected_segs, proposed_segs, source_note) in enumerate(batch, start=1):
        item_id = str(item["id"])
        if i in flagged_1based:
            flagged_items.append((item, expected_segs, proposed_segs,
                                  item_text_hypotheses.get(item_id)))
        else:
            text = join_segments(proposed_segs, srt_map)
            method_outputs[item_id][run_number] = text
            _save_item_progress(progress_data, item_id, method_name, run_number, text, progress_file)
            # Store approved text as hypothesis for future runs
            if text and text != item_text_hypotheses.get(item_id):
                item_text_hypotheses[item_id] = text
                hypotheses_updated = True

    if hypotheses_updated:
        _save_overrides(item_text_hypotheses, overrides_file)
    return flagged_items


def _handle_flagged_item(item, expected_segs, proposed_segs, prior_hypothesis,
                         srt_map, method_name, run_number, suggestion_window,
                         method_outputs, progress_data, progress_file,
                         item_text_hypotheses, overrides_file, target_lang):
    """Run interactive correction for one flagged item and persist the result."""
    item_id = str(item["id"])
    corrected_segs, text, offset = interactive_confirm_item(
        item=item,
        expected_segments=expected_segs,
        proposed_segments=proposed_segs,
        srt_map=srt_map,
        method_name=method_name,
        run_number=run_number,
        suggestion_window=suggestion_window,
        target_lang=target_lang,
        prior_hypothesis=prior_hypothesis,
    )
    method_outputs[item_id][run_number] = text
    _save_item_progress(progress_data, item_id, method_name, run_number, text, progress_file)
    if text and text != item_text_hypotheses.get(item_id):
        item_text_hypotheses[item_id] = text
        _save_overrides(item_text_hypotheses, overrides_file)


def _process_method(
        items, runs, method_name, suggestion_window, batch_size,
        method_outputs, existing_translations, progress_data, progress_file,
        item_text_hypotheses, overrides_file, target_lang,
):
    """Process all items for all runs of one method using paged batch review."""
    for run_number, _path, srt_map in runs:
        unreviewed = []
        n_skipped = 0

        for item in items:
            item_id = str(item["id"])
            expected_segs = validate_segment_list(item, item_id)
            if _is_already_reviewed(item_id, method_name, run_number, existing_translations):
                method_outputs[item_id][run_number] = (
                    existing_translations[item_id][method_name][str(run_number)]
                )
                n_skipped += 1
            else:
                proposed_segs, source_note = _build_item_proposal(
                    item_id, item, expected_segs,
                    item_text_hypotheses, srt_map, suggestion_window, target_lang,
                )
                unreviewed.append((item, expected_segs, proposed_segs, source_note))

        if n_skipped:
            print(f"  Run {run_number}: {n_skipped} already-reviewed item(s) skipped.")
        if not unreviewed:
            print(f"  Run {run_number}: all items already reviewed.")
            continue
        print(f"  Run {run_number}: {len(unreviewed)} item(s) to review.")

        n_batches = math.ceil(len(unreviewed) / batch_size)
        for batch_idx in range(n_batches):
            batch = unreviewed[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            flagged_items = _process_batch(
                batch, srt_map, method_name, run_number, batch_idx, n_batches,
                method_outputs, progress_data, progress_file,
                item_text_hypotheses, overrides_file, target_lang,
            )
            for item, expected_segs, proposed_segs, prior_hypothesis in flagged_items:
                _handle_flagged_item(
                    item, expected_segs, proposed_segs, prior_hypothesis,
                    srt_map, method_name, run_number, suggestion_window,
                    method_outputs, progress_data, progress_file,
                    item_text_hypotheses, overrides_file, target_lang,
                )


def build_interactive_translations_for_items(
        data,
        method_to_srt_files,
        suggestion_window,
        batch_size,
        existing_translations,
        progress_data,
        progress_file,
        item_text_hypotheses,
        overrides_file,
        target_lang,
):
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a top-level object.")
    if "items" not in data:
        raise ValueError('Input JSON must contain a top-level "items" field.')
    if not isinstance(data["items"], list):
        raise ValueError('Top-level "items" must be a list.')

    items = data["items"]
    method_to_runs = _load_method_runs(method_to_srt_files)
    method_names = list(method_to_runs.keys())

    result = {str(item["id"]): {} for item in items}
    if not method_names:
        return result

    for method_index, method_name in enumerate(method_names):
        print("\n" + "#" * 100)
        print(f"PROCESSING METHOD {method_index + 1}/{len(method_names)}: {method_name}")
        print("#" * 100)

        runs = method_to_runs[method_name]
        method_outputs = {str(item["id"]): {} for item in items}

        _process_method(
            items, runs, method_name, suggestion_window, batch_size,
            method_outputs, existing_translations, progress_data, progress_file,
            item_text_hypotheses, overrides_file, target_lang,
        )

        for item in items:
            result[str(item["id"])][method_name] = method_outputs[str(item["id"])]

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively map subtitle segments from translated SRT files into the JSON. "
            "Items are shown in batches; press ENTER to approve all, or enter numbers to "
            "flag specific items for manual correction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Directory convention:\n"
            "  translations: films/output/translations/<film_name>/<trans_model>/\n"
            "  input JSON:   films/output/translations/<film_name>/<trans_model>.json\n"
            "  output JSON:  films/output/translations/<film_name>/<trans_model>.json"
        ),
    )
    parser.add_argument("film_name", help="Film identifier (e.g. pokrov-gate)")
    parser.add_argument("trans_model", help="Translation model name (e.g. gpt-5.2)")
    parser.add_argument("source_lang", help="Source language (e.g. Russian)")
    parser.add_argument("target_lang", help="Target language (e.g. Galician)")
    parser.add_argument(
        "--suggestion-window",
        type=int,
        default=2,
        help="How many neighboring segments to include in context suggestions. Default: 2",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of items to display per review batch. Default: {DEFAULT_BATCH_SIZE}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_lang = normalize_lang(args.source_lang)
    target_lang = normalize_lang(args.target_lang)

    lang_pair = f"{source_lang}-{target_lang}"
    film_root = Path("films/output/translations") / args.film_name / lang_pair
    input_target = film_root / args.trans_model
    output_path = film_root / f"{args.trans_model}.json"
    reference_path = Path("films/data") / args.film_name / "reference.json"
    progress_file_path = Path(f"{args.film_name}_progress.json")
    overrides_file_path = Path(f"{args.film_name}_overrides.json")
    model_name = args.trans_model

    if output_path.is_file():
        template_path = output_path
    elif reference_path.is_file():
        print(f"Output JSON not found; using reference as template: {reference_path}")
        template_path = reference_path
    else:
        print(f"Error: neither {output_path} nor {reference_path} exists.", file=sys.stderr)
        sys.exit(1)

    try:
        incoming_template_data = load_json(template_path)
        method_to_srt_files = resolve_target_path(input_target)

        # Load or initialise progress
        if progress_file_path.exists():
            print(f"Loading progress from: {progress_file_path}")
            progress_data = load_json(progress_file_path)
        else:
            progress_data = {}

        # Load or initialise overrides
        if overrides_file_path.exists():
            print(f"Loading overrides from: {overrides_file_path}")
            _ov = load_json(overrides_file_path)
            item_text_hypotheses = dict(_ov.get("hypothesis", {}))
            print(f"  {len(item_text_hypotheses)} text hypothesis/hypotheses loaded.")
        else:
            item_text_hypotheses = {}

        # incoming_template_data is already the right base (output file if it exists, else reference)
        base_data = incoming_template_data

        # Compile all existing decisions so we can resume smoothly
        existing_translations = get_existing_translations(base_data, target_lang)
        for item_id, methods in progress_data.items():
            for method, runs in methods.items():
                for run_num, text in runs.items():
                    existing_translations[item_id][method][str(run_num)] = text

        new_translations_by_item = build_interactive_translations_for_items(
            data=incoming_template_data,
            method_to_srt_files=method_to_srt_files,
            suggestion_window=args.suggestion_window,
            batch_size=args.batch_size,
            existing_translations=existing_translations,
            progress_data=progress_data,
            progress_file=progress_file_path,
            item_text_hypotheses=item_text_hypotheses,
            overrides_file=overrides_file_path,
            target_lang=target_lang,
        )

        merged = merge_into_existing_json(
            base_data=base_data,
            incoming_template_data=incoming_template_data,
            new_translations_by_item=new_translations_by_item,
            model_name=model_name,
            target_lang=target_lang,
        )

        save_json(merged, output_path)
        print(f"\nSuccessfully saved updated merged output to: {output_path}")

    except KeyboardInterrupt:
        print(
            f"\nInterrupted by user. Progress up to the last mapped item was saved to {progress_file_path}.",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
