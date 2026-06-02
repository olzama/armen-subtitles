#!/usr/bin/env python3
"""
Migrate language fields in reference.json and *-analysis.json files.

Changes:
  1. In 'language_specific': rename 'nb' and 'things_to_avoid' to 'eng',
     and add 'spa' and 'glg' as null placeholders.
  2. In 'reference': add 'spa' and 'glg' as null placeholders.

Usage:
  python migrate_language_fields.py [--dry-run] [--no-backup] [file ...]
"""

import json
import sys
import shutil
from pathlib import Path

FILMS_DATA = Path(__file__).parent.parent / "films" / "data"
FILE_GLOBS = ["reference.json", "*-analysis.json", "list-analysis.json"]

LANG_KEYS = ["eng", "spa", "glg"]


def migrate_language_specific(ls) -> dict:
    """Convert language_specific to {eng, spa, glg} structure."""
    if ls is None:
        eng_value = None
    elif isinstance(ls, dict):
        # Gather existing English content (from 'nb', 'things_to_avoid', or 'eng')
        parts = []
        for old_key in ("eng", "nb", "things_to_avoid"):
            if ls.get(old_key):
                parts.append(ls[old_key])
        eng_value = " ".join(parts) if parts else None
    else:
        eng_value = None

    return {
        "eng": eng_value,
        "spa": ls.get("spa") if isinstance(ls, dict) else None,
        "glg": ls.get("glg") if isinstance(ls, dict) else None,
    }


def migrate_reference(ref) -> dict:
    """Add spa and glg placeholders to a reference dict."""
    if not isinstance(ref, dict):
        return ref

    # Normalize 'English' key to 'eng'
    if "English" in ref and "eng" not in ref:
        ref = {"eng": ref["English"], **{k: v for k, v in ref.items() if k != "English"}}

    return {
        "eng": ref.get("eng"),
        "spa": ref.get("spa"),
        "glg": ref.get("glg"),
    }


def migrate_analysis(analysis) -> dict:
    """Migrate the analysis dict's language_specific field."""
    if not isinstance(analysis, dict):
        return analysis
    return {
        **analysis,
        "language_specific": migrate_language_specific(analysis.get("language_specific")),
    }


def migrate_item(item: dict) -> tuple[dict, bool]:
    changed = False
    new_item = dict(item)

    if "analysis" in item and isinstance(item["analysis"], dict):
        new_analysis = migrate_analysis(item["analysis"])
        if new_analysis != item["analysis"]:
            new_item["analysis"] = new_analysis
            changed = True

    if "reference" in item and isinstance(item["reference"], dict):
        new_ref = migrate_reference(item["reference"])
        if new_ref != item["reference"]:
            new_item["reference"] = new_ref
            changed = True

    return new_item, changed


def migrate_data(data):
    if isinstance(data, list):
        changed = False
        new_list = []
        for item in data:
            new_item, item_changed = migrate_item(item)
            new_list.append(new_item)
            changed = changed or item_changed
        return new_list, changed

    if isinstance(data, dict) and "items" in data:
        changed = False
        new_items = []
        for item in data["items"]:
            new_item, item_changed = migrate_item(item)
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

    new_data, changed = migrate_data(data)

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
