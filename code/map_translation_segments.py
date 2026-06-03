#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import re

import pysrt
from lang_utils import normalize_lang, lang_code

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
    subs = pysrt.open(str(srt_path), encoding="utf-8-sig")
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


def build_text_stream(segment_ids, srt_map):
    """Concatenate segments into a continuous stream with a char→segment index.

    Returns (stream_text, char_to_seg) where char_to_seg[i] is (seg_id, offset)
    for text characters or None for inter-segment separator characters.
    """
    segments = sorted(s for s in segment_ids if s in srt_map)
    parts = []
    char_to_seg = []
    for seg_id in segments:
        if parts:
            parts.append('\n')
            char_to_seg.append(None)
        text = normalize_text_for_display(srt_map[seg_id])
        parts.append(text)
        for i in range(len(text)):
            char_to_seg.append((seg_id, i))
    return ''.join(parts), char_to_seg


def find_span_in_stream(stream_text, char_to_seg, reference_texts, min_ratio=0.55):
    """Find the text span in stream_text that best matches any of reference_texts.

    Uses SequenceMatcher to locate where in the SRT stream the reference content
    appears, then snaps to word boundaries.  Returns (span_text, span_seg_ids)
    or (None, None) if no good match is found.
    """
    from difflib import SequenceMatcher as SM
    if not stream_text:
        return None, None
    refs = [r.strip() for r in reference_texts if r and r.strip() and r != "[NO REFERENCE AVAILABLE]"]
    if not refs:
        return None, None

    # Fast path: any reference is a literal substring of the stream.
    for ref in refs:
        if ref in stream_text:
            idx = stream_text.find(ref)
            span_segs = sorted({
                info[0] for info in char_to_seg[idx:idx + len(ref)] if info is not None
            })
            if span_segs:
                return ref, span_segs

    best_ratio = 0.0
    best_text = None
    best_segs = None

    for ref in refs:
        matcher = SM(None, stream_text, ref, autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size > 2]
        if not blocks:
            continue
        # Require that the reference is reasonably well-covered end-to-end.
        covered = (blocks[-1].b + blocks[-1].size - blocks[0].b) / max(len(ref), 1)
        if covered < 0.5:
            continue

        raw_start = blocks[0].a
        raw_end = blocks[-1].a + blocks[-1].size

        # Snap start backwards to the nearest word boundary.
        span_start = raw_start
        while span_start > 0 and not stream_text[span_start - 1].isspace():
            span_start -= 1

        # Snap end forward only when genuinely mid-word (previous char is
        # alphanumeric).  When the match ends right after punctuation such as
        # an em-dash, do not snap — that would wrongly pull in the next word
        # (e.g. "—he").
        span_end = raw_end
        if (span_end < len(stream_text)
                and not stream_text[span_end].isspace()
                and span_end > 0 and stream_text[span_end - 1].isalnum()):
            while span_end < len(stream_text) and not stream_text[span_end].isspace():
                span_end += 1

        span_text = stream_text[span_start:span_end].strip()
        if not span_text:
            continue

        ratio = SM(None, span_text, ref, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_text = span_text
            span_segs = sorted({
                info[0] for info in char_to_seg[span_start:span_end] if info is not None
            })
            best_segs = span_segs or None

    if best_ratio < min_ratio or not best_text or not best_segs:
        return None, None
    return best_text, best_segs


def _refine_to_span(best_segs, srt_map, hyps, ref):
    """Build a stream around best_segs (±1 neighbor) and find the best-matching span.

    Tries hypotheses first (same-language, highest reliability) then the reference.
    Returns (span_text, span_seg_ids) or (None, None).
    """
    if not best_segs:
        return None, None
    srt_keys = set(srt_map.keys())
    lo, hi = min(best_segs), max(best_segs)
    stream_segs = sorted({s for s in best_segs}
                         | ({lo - 1} if lo - 1 in srt_keys else set())
                         | ({hi + 1} if hi + 1 in srt_keys else set()))
    stream_text, char_to_seg = build_text_stream(stream_segs, srt_map)
    ref_texts = list(hyps) + ([ref] if ref and ref != "[NO REFERENCE AVAILABLE]" else [])
    return find_span_in_stream(stream_text, char_to_seg, ref_texts)


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
    """Generate candidate segment groups to search.

    Shifts the expected group across [-window, +window] and pairs each shift
    with a small expansion (±0..2 extra segments).  Expansion is capped at 2
    regardless of window size — larger positional gaps are handled by the
    segment_memory offset, not by a wider expansion.  This keeps the candidate
    count at O(window) rather than O(window³).
    """
    expected_segments = sorted(expected_segments)
    start = expected_segments[0]
    end = expected_segments[-1]
    max_expand = min(window, 2)

    candidate_lists = []
    for delta in range(-window, window + 1):
        for extra_left in range(0, max_expand + 1):
            for extra_right in range(0, max_expand + 1):
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


def _get_lang_value(d, target_lang):
    """Look up target_lang in dict d, trying both ISO code and full name."""
    val = d.get(target_lang)
    if val is not None:
        return val
    try:
        val = d.get(lang_code(target_lang))
        if val is not None:
            return val
    except ValueError:
        pass
    try:
        val = d.get(normalize_lang(target_lang))
        if val is not None:
            return val
    except ValueError:
        pass
    return None


def get_reference_translation(item, target_lang):
    translations = item.get("reference", {})
    reference = _get_lang_value(translations, target_lang)

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
_embedding_cache: dict = {}


def _get_similarity_model():
    global _similarity_model
    if _similarity_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformers model...")
        _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _similarity_model


def _cached_encode(model, texts):
    """Encode texts, returning cached embeddings for any previously seen text."""
    import numpy as np
    new_texts = [t for t in texts if t not in _embedding_cache]
    if new_texts:
        embs = model.encode(new_texts, show_progress_bar=False)
        for t, e in zip(new_texts, embs):
            _embedding_cache[t] = e
    return np.stack([_embedding_cache[t] for t in texts])


def find_best_match_in_window(expected_segs, reference_text, srt_map, window):
    """
    Search all candidate segment groups within the context window and return
    (best_segs, score) where score is cosine similarity to reference_text.
    Falls back to expected_segs if reference_text is empty or no candidates exist.
    After finding the best group, trims leading/trailing segments when doing so
    does not reduce the similarity score, preferring the shortest match.
    """
    # reference_text may be a str or a list of str (multiple hypotheses)
    ref_texts = reference_text if isinstance(reference_text, list) else [reference_text]
    ref_texts = [t for t in ref_texts if t and t != "[NO REFERENCE AVAILABLE]"]
    if not ref_texts:
        return expected_segs[:], 0.0
    candidates = suggest_context_windows(expected_segs, srt_map, window=window)
    if not candidates:
        return expected_segs[:], 0.0

    import numpy as np
    model = _get_similarity_model()
    cand_texts = [text for _, text in candidates]
    all_texts = ref_texts + cand_texts
    embeddings = _cached_encode(model, all_texts)
    ref_embs = embeddings[:len(ref_texts)]
    cand_embs = embeddings[len(ref_texts):]

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # Score each candidate as max similarity across all reference hypotheses.
    scores = [max(cosine(r, e) for r in ref_embs) for e in cand_embs]

    # Prefer candidates whose length matches expected to avoid over-long matches.
    n_exp = len(expected_segs)
    def length_penalty(cand_segs):
        return n_exp / max(len(cand_segs), n_exp)

    best_idx = max(range(len(scores)), key=lambda i: scores[i] * length_penalty(candidates[i][0]))
    best_segs = list(candidates[best_idx][0])
    best_score = scores[best_idx]

    # Trim trailing then leading segments: drop any segment whose removal does not
    # reduce similarity by more than 0.01, including segments below the expected count.
    for make_candidates in (
        lambda segs: [segs[:-i] for i in range(1, len(segs))],   # trailing trim
        lambda segs: [segs[i:] for i in range(1, len(segs))],    # leading trim
    ):
        if len(best_segs) <= 1:
            break
        candidates = [c for c in make_candidates(best_segs) if len(c) >= 1 and all(s in srt_map for s in c)]
        if not candidates:
            continue
        embs = _cached_encode(model, [join_segments(c, srt_map) for c in candidates])
        scores = [max(cosine(r, e) for r in ref_embs) for e in embs]
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
    """Persist text hypotheses to disk. Values are lists of known-good translations."""
    save_json({"hypotheses": item_text_hypotheses}, overrides_file)


def _save_segment_memory(segment_memory, segment_memory_file):
    save_json(segment_memory, segment_memory_file)


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


def _apply_hypothesis_trim(srt_text, hypothesis, min_ratio=0.6):
    """Return the portion of srt_text that best matches hypothesis.

    Trim positions are derived from the first/last SequenceMatcher matching
    blocks, then snapped to word boundaries so that no word is ever split:
    - trim_start, if it falls mid-word, is advanced forward to the start of
      the next word (the partial preamble word is dropped entirely).
    - trim_end, if it falls mid-alphanumeric-word, is advanced forward to the
      end of that word (the partial tail word is kept whole).  When the
      character before raw_end is punctuation (e.g. an em-dash), no forward
      snap is applied — this prevents spurious trailing fragments like "—he"
      from being included when the match correctly ends at the punctuation.

    The leading trim is additionally gated on blocks[0].b == 0: hypothesis
    must itself begin at the start of its first matching block, which prevents
    false leading clips when srt_text and hypothesis are different translation
    variants that share a common suffix fragment.

    Returns trimmed text if similarity to hypothesis exceeds min_ratio,
    otherwise None (caller falls back to raw srt_text)."""
    if not srt_text or not hypothesis:
        return None
    srt_stripped = srt_text.strip()
    hyp_stripped = hypothesis.strip()
    # Fast path: hypothesis is a literal substring of srt_text.
    if hyp_stripped in srt_stripped:
        return hyp_stripped

    if SequenceMatcher(None, srt_stripped, hyp_stripped, autojunk=False).ratio() > 0.95:
        return srt_stripped

    matcher = SequenceMatcher(None, srt_stripped, hyp_stripped, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 2]
    if not blocks:
        return None

    raw_start = blocks[0].a if blocks[0].b == 0 else 0
    raw_end = blocks[-1].a + blocks[-1].size

    # Snap trim_start forward to the next word start if it falls mid-word.
    trim_start = raw_start
    if trim_start > 0 and not srt_stripped[trim_start - 1].isspace():
        while trim_start < len(srt_stripped) and not srt_stripped[trim_start].isspace():
            trim_start += 1
        while trim_start < len(srt_stripped) and srt_stripped[trim_start].isspace():
            trim_start += 1

    # Snap trim_end forward to the end of the current word if it falls mid-word.
    trim_end = raw_end
    if trim_end < len(srt_stripped) and not srt_stripped[trim_end].isspace():
        while trim_end < len(srt_stripped) and not srt_stripped[trim_end].isspace():
            trim_end += 1

    trimmed = srt_stripped[trim_start:trim_end].strip()
    if not trimmed:
        return None
    ratio = SequenceMatcher(None, trimmed, hyp_stripped, autojunk=False).ratio()
    return trimmed if ratio >= min_ratio else None


def _edit_in_editor(text):
    """Open text in $EDITOR and return the saved result, or None if unchanged/cancelled."""
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = f.name
    try:
        subprocess.run([editor, tmp], check=True)
        result = Path(tmp).read_text(encoding="utf-8").strip()
        return result if result != text.strip() else text
    except Exception as exc:
        print(f"  Editor error: {exc}")
        return text
    finally:
        Path(tmp).unlink(missing_ok=True)


def _accept_or_edit_text(segs, srt_map, initial_text=None):
    """Show text for segs. User presses ENTER to accept, e to open in editor,
    or types a replacement. Returns accepted text string, or None to go back.

    initial_text overrides join_segments(segs) as the starting text — used
    when a refined span is already available.
    """
    text = initial_text if initial_text is not None else join_segments(segs, srt_map)
    lines = text.splitlines() if text else []
    print(f"\nText from segments {segs}:")
    for i, line in enumerate(lines, 1):
        print(f"  {i}: {line}")
    if not lines:
        print("  [EMPTY]")
    print("ENTER=accept   e=open in editor   b=back   or type replacement")
    user_in = input("> ").strip()
    if not user_in:
        return text
    if user_in.lower() == "b":
        return None
    if user_in.lower() == "e":
        return _edit_in_editor(text)
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
        proposed_text=None,
):
    item_label = get_item_label(item)
    reference_translation = get_reference_translation(item, target_lang)
    query = prior_hypothesis if prior_hypothesis is not None else reference_translation
    current_window = suggestion_window
    current_segs = list(proposed_segments)
    # current_text is the span-refined text; None means fall back to join_segments.
    current_text = proposed_text
    n_presses = 0

    while True:
        _print_confirmation_header(
            item_label, method_name, run_number,
            expected_segments, current_segs,
            reference_translation, srt_map,
            prior_hypothesis=prior_hypothesis,
            context_window=current_window,
        )
        print("ENTER=accept   b=back   n=widen search   e=edit/trim in $EDITOR   [segment numbers, e.g. 569 or 569,570]=pick segments")
        user_in = input("> ").strip()

        if not user_in:
            text_out = join_segments(current_segs, srt_map)
            offset = compute_simple_offset(expected_segments, current_segs)
            return current_segs, text_out, offset, n_presses

        if user_in.lower() == "b":
            return None, None, None, 0

        if user_in.lower() == "n":
            n_presses += 1
            current_window += 2
            new_segs, score = find_best_match_in_window(expected_segments, query, srt_map, current_window)
            hyps = [prior_hypothesis] if prior_hypothesis is not None else []
            span_text, span_segs = _refine_to_span(new_segs, srt_map, hyps, reference_translation)
            if span_text is not None:
                current_segs = span_segs
                current_text = span_text
                print(f"  Window ±{current_window} → segments {span_segs} (span)  [score {score:.2f}]")
            else:
                current_segs = new_segs
                current_text = None
                print(f"  Window ±{current_window} → {new_segs}  [score {score:.2f}]")
            continue

        if user_in.lower() == "e":
            result = _accept_or_edit_text(current_segs, srt_map)
            if result is not None:
                return current_segs, result, None, n_presses
            continue

        segs = _parse_segment_numbers(user_in, srt_map)
        if segs is not None:
            current_segs = segs
            current_text = None  # manual seg pick resets span; user can edit from there
            result = _accept_or_edit_text(current_segs, srt_map)
            if result is not None:
                return current_segs, result, None, n_presses
            continue

        print("  Enter segment numbers, or: ENTER accept   b back   n widen   e edit")


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
        incoming_val = incoming_method_runs[run_key]
        if run_key in merged:
            if merged[run_key] != incoming_val:
                print(f"Updating run {run_key} (was: {merged[run_key]!r}, now: {incoming_val!r})")
                merged[run_key] = incoming_val
        else:
            merged[run_key] = incoming_val
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

    # Find existing entry under code or full-name variant, then normalise the key.
    lang_translations = _get_lang_value(translations, target_lang)
    if lang_translations is None:
        lang_translations = {}
    elif not isinstance(lang_translations, dict):
        raise ValueError(
            f'Item {item.get("id", "<no id>")}: translations["{target_lang}"] must be an object if present.'
        )
    else:
        # Remove stale key if it differs from the canonical one (migrate on write).
        old_key = next((k for k, v in translations.items() if v is lang_translations), None)
        if old_key and old_key != target_lang:
            del translations[old_key]
    translations[target_lang] = lang_translations

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
        lang_translations = _get_lang_value(item.get("translations", {}), target_lang) or {}
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


def _display_batch(batch, srt_map, method_name, run_number, batch_num, total_batches, target_lang,
                   item_text_hypotheses=None):
    """Display a numbered list of (item, expected_segs, proposed_segs, proposed_text, source_note)."""
    print(f"\n{'=' * 80}")
    print(f"METHOD: {method_name} | RUN: {run_number} | Batch {batch_num}/{total_batches}")
    print('=' * 80)

    for idx, (item, expected_segs, proposed_segs, proposed_text, source_note) in enumerate(batch, start=1):
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

        item_id = str(item["id"])
        hyps = (item_text_hypotheses or {}).get(item_id) or []
        ref = get_reference_translation(item, target_lang).replace("\n", "\n              ")
        srt_text = proposed_text or join_segments(proposed_segs, srt_map) or ""
        if hyps:
            trimmed = next((t for h in reversed(hyps) for t in [_apply_hypothesis_trim(srt_text, h)] if t), None)
            mapped = (trimmed or srt_text or "[EMPTY]").replace("\n", "\n              ")
        else:
            mapped = (srt_text or "[EMPTY]").replace("\n", "\n              ")

        print(f"\n  [{idx}] {get_item_label(item)}")
        print(f"       Segments: {seg_info}")
        print(f"       REF:      {ref}")
        print(f"       MAP:      {mapped}")

    print()


def _prompt_batch_flags(n_items):
    """Prompt user to flag items for correction. Returns a set of 1-based indices."""
    while True:
        raw = input("Press ENTER to approve all, b=back, or enter numbers to flag (e.g. 2 5): ").strip()
        if not raw:
            return set()
        if raw.lower() == "b":
            return None
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


def _normalize_for_autoapprove(t):
    """Strip punctuation and collapse whitespace for exact-match comparison."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', t)).strip().lower()


def _try_auto_approve(item_id, proposed_segs, item_text_hypotheses, srt_map, proposed_text=None):
    """Return stored hypothesis text if the trimmed proposed text matches it exactly
    (modulo punctuation and whitespace). Hypothesis lookup is flexible (substring/fuzzy
    via _apply_hypothesis_trim), but the final approval requires a near-exact match so
    that truncated or otherwise wrong hypotheses are never silently accepted.

    proposed_text is the span-refined text; if None, falls back to join_segments.
    """
    hyps = item_text_hypotheses.get(item_id) or []
    if not hyps:
        return None
    srt_text = proposed_text if proposed_text is not None else join_segments(proposed_segs, srt_map)
    trimmed = next((t for h in reversed(hyps) for t in [_apply_hypothesis_trim(srt_text, h)] if t), None)
    text = trimmed if trimmed else srt_text
    text_norm = _normalize_for_autoapprove(text)
    for hyp in hyps:
        if _normalize_for_autoapprove(hyp) == text_norm:
            return hyp  # return the stored approved version
    return None


def _build_item_proposal(item_id, item, expected_segs,
                         item_text_hypotheses, srt_map, suggestion_window, target_lang,
                         segment_memory=None):
    """Return (proposed_segs, source_note) for one unreviewed item.

    If a prior accepted text exists for this item, it is used as the similarity
    query (more reliable than the reference).  Falls back to the reference
    translation when no hypothesis is available.

    If segment_memory has an entry for this item, the remembered offset between
    expected and accepted segments is applied to bias the search center, and the
    window is widened by the number of 'n' presses previously required.
    """
    hyps = item_text_hypotheses.get(item_id) or []
    query = hyps if hyps else get_reference_translation(item, target_lang)
    label = "prior" if hyps else "sim"

    search_segs = expected_segs
    search_window = suggestion_window
    mem = (segment_memory or {}).get(item_id)
    if mem:
        mem_expected = mem.get("expected") or []
        mem_accepted = mem.get("accepted") or []
        n_presses = mem.get("n_presses", 0)
        if mem_expected and mem_accepted and len(mem_accepted) == len(mem_expected):
            offset = mem_accepted[0] - mem_expected[0]
            if offset != 0:
                search_segs = [s + offset for s in expected_segs]
                label = "mem"
        search_window = suggestion_window + n_presses * 2

    best_segs, score = find_best_match_in_window(search_segs, query, srt_map, search_window)

    # Refine to a character-level text span, ignoring segment boundaries.
    ref = get_reference_translation(item, target_lang)
    span_text, span_segs = _refine_to_span(best_segs, srt_map, hyps, ref)
    if span_text is not None and span_segs:
        proposed_segs = span_segs
        proposed_text = span_text
    else:
        proposed_segs = best_segs
        proposed_text = join_segments(best_segs, srt_map)

    show_score = proposed_segs != expected_segs or score < 0.5
    source_note = f"[{label} {score:.2f}]" if show_score else None
    return proposed_segs, proposed_text, source_note


def _process_batch(batch, srt_map, method_name, run_number, batch_idx, n_batches,
                   method_outputs, progress_data, progress_file,
                   item_text_hypotheses, overrides_file, target_lang):
    """Display one batch, save approved items, return flagged ones. Returns None if user pressed b."""
    _display_batch(batch, srt_map, method_name, run_number, batch_idx + 1, n_batches, target_lang,
                   item_text_hypotheses=item_text_hypotheses)
    flagged_1based = _prompt_batch_flags(len(batch))

    if flagged_1based is None:
        return None

    hypotheses_updated = False
    flagged_items = []
    for i, (item, expected_segs, proposed_segs, proposed_text, source_note) in enumerate(batch, start=1):
        item_id = str(item["id"])
        if i in flagged_1based:
            flagged_items.append((item, expected_segs, proposed_segs, proposed_text,
                                  (item_text_hypotheses.get(item_id) or [None])[-1]))
        else:
            hyps = item_text_hypotheses.get(item_id) or []
            srt_text = proposed_text or join_segments(proposed_segs, srt_map)
            if hyps:
                trimmed = next((t for h in reversed(hyps) for t in [_apply_hypothesis_trim(srt_text, h)] if t), None)
                text = trimmed if trimmed else srt_text
            else:
                text = srt_text
            method_outputs[item_id][run_number] = text
            _save_item_progress(progress_data, item_id, method_name, run_number, text, progress_file)
            if not hyps and text:
                item_text_hypotheses[item_id] = [text]
                hypotheses_updated = True

    if hypotheses_updated:
        _save_overrides(item_text_hypotheses, overrides_file)
    return flagged_items


def _find_last_progress_item(progress_data, method_name, run_number):
    """Return the item_id of the last saved item for this method/run, or None."""
    candidates = [
        item_id for item_id, methods in progress_data.items()
        if method_name in methods and run_number in methods[method_name]
    ]
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda x: int(x))
    except (ValueError, TypeError):
        return candidates[-1]


def _process_flagged_items(flagged_items, items_by_id, srt_map, method_name, run_number,
                           suggestion_window, method_outputs, existing_translations,
                           progress_data, progress_file,
                           item_text_hypotheses, overrides_file, target_lang,
                           segment_memory=None, segment_memory_file=None):
    """Process flagged items one by one, supporting b=back to re-do the previous item.
    When at the first item, b retrieves the last item saved in progress so it can be redone."""
    flagged_items = list(flagged_items)
    i = 0
    popped_from_progress = False
    while i < len(flagged_items):
        item, expected_segs, proposed_segs, proposed_text, prior_hypothesis = flagged_items[i]
        success = _handle_flagged_item(
            item, expected_segs, proposed_segs, proposed_text, prior_hypothesis,
            srt_map, method_name, run_number, suggestion_window,
            method_outputs, existing_translations, progress_data, progress_file,
            item_text_hypotheses, overrides_file, target_lang,
            segment_memory=segment_memory, segment_memory_file=segment_memory_file,
        )
        if not success:
            if i > 0:
                i -= 1
                prev_id = str(flagged_items[i][0]["id"])
                progress_data.get(prev_id, {}).get(method_name, {}).pop(str(run_number), None)
                save_json(progress_data, progress_file)
                method_outputs.get(prev_id, {}).pop(run_number, None)
            elif not popped_from_progress:
                last_id = _find_last_progress_item(progress_data, method_name, str(run_number))
                if last_id is not None and last_id in items_by_id:
                    popped_from_progress = True
                    last_item = items_by_id[last_id]
                    last_expected_segs = validate_segment_list(last_item, last_id)
                    prior_text = progress_data[last_id][method_name][str(run_number)]
                    progress_data[last_id][method_name].pop(str(run_number))
                    save_json(progress_data, progress_file)
                    method_outputs.get(last_id, {}).pop(run_number, None)
                    flagged_items.insert(0, (last_item, last_expected_segs, last_expected_segs, None, prior_text))
                else:
                    print("  No previous progress to go back to.")
            else:
                print("  Already at the first item.")
        else:
            i += 1


def _handle_flagged_item(item, expected_segs, proposed_segs, proposed_text, prior_hypothesis,
                         srt_map, method_name, run_number, suggestion_window,
                         method_outputs, existing_translations, progress_data, progress_file,
                         item_text_hypotheses, overrides_file, target_lang,
                         segment_memory=None, segment_memory_file=None):
    """Run interactive correction for one flagged item and persist the result.
    Returns True on success, False if the user pressed b to go back."""
    item_id = str(item["id"])
    corrected_segs, text, offset, n_presses = interactive_confirm_item(
        item=item,
        expected_segments=expected_segs,
        proposed_segments=proposed_segs,
        srt_map=srt_map,
        method_name=method_name,
        run_number=run_number,
        suggestion_window=suggestion_window,
        target_lang=target_lang,
        prior_hypothesis=prior_hypothesis,
        proposed_text=proposed_text,
    )
    if text is None:
        return False
    method_outputs[item_id][run_number] = text
    _save_item_progress(progress_data, item_id, method_name, run_number, text, progress_file)
    old_hyps = item_text_hypotheses.get(item_id) or []
    if text and text not in old_hyps:
        # Append new correction; keep list in chronological order.
        item_text_hypotheses[item_id] = old_hyps + [text]
        _save_overrides(item_text_hypotheses, overrides_file)
        # Propagate: update already-loaded entries whose value matches any known
        # old hypothesis — they were accepted automatically and may be stale.
        old_hyps_set = set(old_hyps)
        item_runs = method_outputs.get(item_id, {})
        for rk, rv in list(item_runs.items()):
            if rv in old_hyps_set and rk != run_number:
                method_outputs[item_id][rk] = text
        progress_item = progress_data.get(item_id, {})
        progress_changed = False
        for mname, mruns in progress_item.items():
            for rk, rv in list(mruns.items()):
                if rv in old_hyps_set:
                    progress_data[item_id][mname][rk] = text
                    progress_changed = True
        if progress_changed:
            save_json(progress_data, progress_file)
        for mname, mruns in existing_translations.get(item_id, {}).items():
            for rk, rv in list(mruns.items()):
                if rv in old_hyps_set:
                    existing_translations[item_id][mname][rk] = text
                    # Also push into method_outputs and progress so the correction
                    # reaches the output file even for runs that were skipped earlier.
                    if mname == method_name:
                        method_outputs[item_id][rk] = text
                        _save_item_progress(progress_data, item_id, mname, rk, text, progress_file)
    # Save segment memory when segments differ from expected or window was widened.
    if segment_memory is not None and segment_memory_file is not None:
        if list(corrected_segs) != list(expected_segs) or n_presses > 0:
            old_mem = segment_memory.get(item_id, {})
            new_mem = {
                "expected": list(expected_segs),
                "accepted": list(corrected_segs),
                "n_presses": max(n_presses, old_mem.get("n_presses", 0)),
            }
            if new_mem != old_mem:
                segment_memory[item_id] = new_mem
                _save_segment_memory(segment_memory, segment_memory_file)
    return True


def _process_method(
        items, runs, method_name, suggestion_window, batch_size,
        method_outputs, existing_translations, progress_data, progress_file,
        item_text_hypotheses, overrides_file, target_lang,
        segment_memory=None, segment_memory_file=None, redo_item_ids=None,
):
    """Process all items for all runs of one method using paged batch review.

    Proposals (embedding search) are computed lazily per batch rather than
    upfront for all items, so the first batch appears immediately and there
    is no long pause at method transitions.
    """
    items_by_id = {str(item["id"]): item for item in items}
    for run_number, _path, srt_map in runs:
        # Fast pass: skip already-reviewed, quick-auto-approve via expected segs,
        # collect the rest as (item, expected_segs) needing embedding-based proposals.
        needs_proposal = []
        n_skipped = 0
        n_auto = 0

        for item in items:
            item_id = str(item["id"])
            expected_segs = validate_segment_list(item, item_id)
            if _is_already_reviewed(item_id, method_name, run_number, existing_translations):
                n_skipped += 1
            else:
                # Try auto-approve with expected segs first (no embedding needed).
                # Skip auto-approve for items being redone until new hypotheses accumulate.
                if redo_item_ids is None or item_id not in redo_item_ids:
                    auto_text = _try_auto_approve(item_id, expected_segs, item_text_hypotheses, srt_map)
                else:
                    auto_text = None
                if auto_text is not None:
                    method_outputs[item_id][run_number] = auto_text
                    _save_item_progress(progress_data, item_id, method_name, run_number, auto_text, progress_file)
                    print(f"  AUTO-APPROVED item {get_item_label(item)} (segments {expected_segs}): {auto_text!r}")
                    n_auto += 1
                else:
                    needs_proposal.append((item, expected_segs))

        if n_skipped:
            print(f"  Run {run_number}: {n_skipped} already-reviewed item(s) skipped.")
        if n_auto:
            print(f"  Run {run_number}: {n_auto} item(s) auto-approved (literal hypothesis match).")
        if not needs_proposal:
            print(f"  Run {run_number}: all items already reviewed or auto-approved.")
            continue
        print(f"  Run {run_number}: {len(needs_proposal)} item(s) to review.")

        n_batches = math.ceil(len(needs_proposal) / batch_size)
        batch_idx = 0
        while batch_idx < n_batches:
            # Build proposals for this batch only (embedding happens here, just-in-time).
            raw_batch = needs_proposal[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            unreviewed_in_batch = []
            for item, expected_segs in raw_batch:
                item_id = str(item["id"])
                proposed_segs, proposed_text, source_note = _build_item_proposal(
                    item_id, item, expected_segs,
                    item_text_hypotheses, srt_map, suggestion_window, target_lang,
                    segment_memory=segment_memory,
                )
                auto_text = (
                    None if (redo_item_ids is not None and item_id in redo_item_ids)
                    else _try_auto_approve(item_id, proposed_segs, item_text_hypotheses, srt_map,
                                          proposed_text=proposed_text)
                )
                if auto_text is not None:
                    method_outputs[item_id][run_number] = auto_text
                    _save_item_progress(progress_data, item_id, method_name, run_number, auto_text, progress_file)
                    print(f"  AUTO-APPROVED item {get_item_label(item)} (segments {proposed_segs}): {auto_text!r}")
                    n_auto += 1
                else:
                    unreviewed_in_batch.append((item, expected_segs, proposed_segs, proposed_text, source_note))

            if not unreviewed_in_batch:
                batch_idx += 1
                continue

            batch = unreviewed_in_batch
            result = _process_batch(
                batch, srt_map, method_name, run_number, batch_idx, n_batches,
                method_outputs, progress_data, progress_file,
                item_text_hypotheses, overrides_file, target_lang,
            )
            if result is None:  # user pressed b at the batch prompt
                last_id = _find_last_progress_item(progress_data, method_name, str(run_number))
                if last_id is not None and last_id in items_by_id:
                    last_item = items_by_id[last_id]
                    last_expected_segs = validate_segment_list(last_item, last_id)
                    prior_text = progress_data[last_id][method_name][str(run_number)]
                    progress_data[last_id][method_name].pop(str(run_number))
                    save_json(progress_data, progress_file)
                    method_outputs.get(last_id, {}).pop(run_number, None)
                    _process_flagged_items(
                        [(last_item, last_expected_segs, last_expected_segs, None, prior_text)],
                        items_by_id, srt_map, method_name, run_number, suggestion_window,
                        method_outputs, existing_translations, progress_data, progress_file,
                        item_text_hypotheses, overrides_file, target_lang,
                        segment_memory=segment_memory, segment_memory_file=segment_memory_file,
                    )
                    # re-display the current batch after going back
                else:
                    print("  No previous progress to go back to.")
                    batch_idx += 1
            else:
                _process_flagged_items(
                    result, items_by_id, srt_map, method_name, run_number, suggestion_window,
                    method_outputs, existing_translations, progress_data, progress_file,
                    item_text_hypotheses, overrides_file, target_lang,
                    segment_memory=segment_memory, segment_memory_file=segment_memory_file,
                )
                batch_idx += 1


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
        segment_memory=None,
        segment_memory_file=None,
        redo_item_ids=None,
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
            segment_memory=segment_memory, segment_memory_file=segment_memory_file,
            redo_item_ids=redo_item_ids,
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
            "  translations: experiments/films/output/translations/<film_name>/<trans_model>/\n"
            "  input JSON:   experiments/films/output/translations/<film_name>/<trans_model>.json\n"
            "  output JSON:  experiments/films/output/translations/<film_name>/<trans_model>.json"
        ),
    )
    parser.add_argument("film_name", help="Film identifier (e.g. pokrov-gate)")
    parser.add_argument("trans_model", help="Translation model name (e.g. gpt-5.2)")
    parser.add_argument("source_lang", help="Source language (e.g. Russian)")
    parser.add_argument("target_lang", help="Target language (e.g. Galician)")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods to process (e.g. given,given-lang). Default: all methods.",
    )
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
    parser.add_argument(
        "--redo",
        action="store_true",
        default=False,
        help="Clear saved progress for the selected method(s) and re-review from scratch.",
    )
    parser.add_argument(
        "--redo-items",
        type=str,
        default=None,
        help="Comma-separated item IDs to redo (implies --redo for those items only, e.g. 1,3,7).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_lang = normalize_lang(args.source_lang)   # full name, e.g. "Russian" — used in path
    target_lang = normalize_lang(args.target_lang)   # full name, e.g. "English" — used in path
    target_code = lang_code(args.target_lang)        # ISO code, e.g. "eng" — used as JSON key

    lang_pair = f"{source_lang}-{target_lang}"
    film_root = Path("experiments/films/output/translations") / args.film_name
    input_target = film_root / lang_pair / args.trans_model
    output_path = film_root / f"{args.trans_model}.json"
    reference_path = Path("experiments/films/data") / args.film_name / "reference.json"
    model_mapping_dir = input_target / "mapping"
    lang_mapping_dir = film_root / lang_pair / "mapping"
    model_mapping_dir.mkdir(parents=True, exist_ok=True)
    lang_mapping_dir.mkdir(parents=True, exist_ok=True)
    progress_file_path = model_mapping_dir / "progress.json"
    overrides_file_path = lang_mapping_dir / "overrides.json"
    segment_memory_file_path = lang_mapping_dir / "segment_memory.json"
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
        if args.methods:
            selected = set(args.methods.split(","))
            method_to_srt_files = {k: v for k, v in method_to_srt_files.items() if k in selected}
            if not method_to_srt_files:
                print(f"Error: none of the requested methods {selected} found in {input_target}", file=sys.stderr)
                sys.exit(1)

        # Load or initialise progress
        if progress_file_path.exists():
            print(f"Loading progress from: {progress_file_path}")
            progress_data = load_json(progress_file_path)
        else:
            progress_data = {}

        # Determine which item IDs to redo (None = no redo).
        redo_item_ids = None
        if args.redo_items:
            redo_item_ids = set(args.redo_items.split(","))
        elif args.redo:
            redo_item_ids = {str(item["id"]) for item in incoming_template_data.get("items", [])}

        if redo_item_ids and method_to_srt_files:
            for item_id in list(progress_data.keys()):
                if item_id in redo_item_ids:
                    for method_name in list(method_to_srt_files.keys()):
                        progress_data[item_id].pop(method_name, None)
            desc = f"item(s) {', '.join(sorted(redo_item_ids))} in" if args.redo_items else "all items in"
            print(f"Progress cleared for {desc} method(s): {', '.join(method_to_srt_files)}")

        # Load or initialise segment memory
        if segment_memory_file_path.exists():
            print(f"Loading segment memory from: {segment_memory_file_path}")
            segment_memory = load_json(segment_memory_file_path)
            print(f"  {len(segment_memory)} segment correction(s) loaded.")
        else:
            segment_memory = {}

        # Load or initialise overrides
        if overrides_file_path.exists():
            print(f"Loading overrides from: {overrides_file_path}")
            _ov = load_json(overrides_file_path)
            if "hypotheses" in _ov:
                item_text_hypotheses = {k: v if isinstance(v, list) else [v]
                                        for k, v in _ov["hypotheses"].items()}
            else:
                # migrate old single-value format
                item_text_hypotheses = {k: [v] for k, v in _ov.get("hypothesis", {}).items()}
            print(f"  {len(item_text_hypotheses)} text hypothesis/hypotheses loaded.")
        else:
            item_text_hypotheses = {}

        # incoming_template_data is already the right base (output file if it exists, else reference)
        base_data = incoming_template_data

        # Compile all existing decisions so we can resume smoothly
        existing_translations = get_existing_translations(base_data, target_code)
        for item_id, methods in progress_data.items():
            for method, runs in methods.items():
                for run_num, text in runs.items():
                    existing_translations[item_id][method][str(run_num)] = text

        if redo_item_ids and method_to_srt_files:
            for item_id in list(existing_translations.keys()):
                if item_id in redo_item_ids:
                    for method_name in list(method_to_srt_files.keys()):
                        existing_translations[item_id].pop(method_name, None)

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
            target_lang=target_code,
            segment_memory=segment_memory,
            segment_memory_file=segment_memory_file_path,
            redo_item_ids=redo_item_ids,
        )

        merged = merge_into_existing_json(
            base_data=base_data,
            incoming_template_data=incoming_template_data,
            new_translations_by_item=new_translations_by_item,
            model_name=model_name,
            target_lang=target_code,
        )

        save_json(merged, output_path)
        print(f"\nSuccessfully saved updated merged output to: {output_path}")
        if progress_file_path.exists():
            progress_file_path.unlink()
            print(f"Progress file cleared: {progress_file_path}")

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
