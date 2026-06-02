#!/usr/bin/env python3
"""
Restructure the 'analysis' field in reference.json and *-analysis.json files.

Splits the flat analysis string into:
  general          -- cultural/comedic context
  language_specific -- translation notes: NB, things_to_avoid, etc.

Usage:
  python restructure_analysis.py [--dry-run] [--no-backup] [file ...]

With no file arguments, processes all reference.json and *-analysis.json
files under films/data/.
"""

import json
import re
import sys
import shutil
from pathlib import Path

FILMS_DATA = Path(__file__).parent.parent / "films" / "data"

FILE_GLOBS = ["reference.json", "*-analysis.json", "list-analysis.json"]

# Markers that introduce language-specific notes.
# Each tuple: (key_name, compiled_regex)
MARKERS = [
    ("nb",              re.compile(r"(?<!\w)\s+NB:\s*")),
    ("things_to_avoid", re.compile(r"\s+(?:\*{3})?Things to avoid:(?:\*{3})?\s*", re.IGNORECASE)),
]


def split_analysis(text: str) -> dict:
    """Split a flat analysis string into general + language_specific parts."""
    if not text or not text.strip():
        return {"general": text, "language_specific": None}

    # Strip [cite_start] artifacts left by citation tools
    text = re.sub(r"\[cite_start\]", "", text).strip()

    # Find all marker positions
    hits = []
    for key, pattern in MARKERS:
        m = pattern.search(text)
        if m:
            hits.append((m.start(), m.end(), key))

    if not hits:
        return {"general": text, "language_specific": None}

    hits.sort()  # sort by position in string

    general = text[: hits[0][0]].strip()
    language_specific = {}

    for i, (start, end, key) in enumerate(hits):
        next_pos = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        language_specific[key] = text[end:next_pos].strip()

    return {
        "general": general,
        "language_specific": language_specific or None,
    }


def restructure_item(item: dict) -> tuple[dict, bool]:
    """Return (new_item, changed). Skips items whose analysis is already a dict."""
    if "analysis" not in item:
        return item, False
    raw = item["analysis"]
    if isinstance(raw, dict):
        return item, False
    return {**item, "analysis": split_analysis(raw)}, True


def restructure_data(data):
    """Handle both top-level list and {items: [...]} object formats."""
    if isinstance(data, list):
        changed = False
        new_list = []
        for item in data:
            new_item, item_changed = restructure_item(item)
            new_list.append(new_item)
            changed = changed or item_changed
        return new_list, changed

    if isinstance(data, dict) and "items" in data:
        changed = False
        new_items = []
        for item in data["items"]:
            new_item, item_changed = restructure_item(item)
            new_items.append(new_item)
            changed = changed or item_changed
        return {**data, "items": new_items}, changed

    return data, False


def process_file(path: Path, *, dry_run: bool, backup: bool) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ERROR  {path}: {exc}", file=sys.stderr)
        return False

    new_data, changed = restructure_data(data)

    if not changed:
        print(f"  skip   {path.relative_to(FILMS_DATA.parent.parent)}")
        return False

    if dry_run:
        print(f"  would update  {path.relative_to(FILMS_DATA.parent.parent)}")
        return True

    if backup:
        shutil.copy2(path, path.with_suffix(".json.bak"))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"  updated {path.relative_to(FILMS_DATA.parent.parent)}")
    return True


def find_files() -> list[Path]:
    found = set()
    for glob in FILE_GLOBS:
        found.update(FILMS_DATA.rglob(glob))
    return sorted(found)


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    no_backup = "--no-backup" in args
    paths = [Path(a) for a in args if not a.startswith("--")]

    if not paths:
        paths = find_files()

    if not paths:
        print("No files found.")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Processing {len(paths)} file(s):\n")
    updated = sum(
        process_file(p, dry_run=dry_run, backup=not no_backup) for p in paths
    )
    print(f"\n{updated}/{len(paths)} file(s) {'would be ' if dry_run else ''}updated.")


if __name__ == "__main__":
    main()
