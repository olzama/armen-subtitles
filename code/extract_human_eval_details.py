#!/usr/bin/env python3
"""
Extract enriched per-judgment details from human evaluation sessions.

Joins each judgment with the source text, translation shown, reference
translation, character info, and item analysis from the translations JSON.

Usage:
    python extract_human_eval_details.py <eval_dir> <translations_json>

    <eval_dir>           directory containing human_*.json files
    <translations_json>  mapped translations file (e.g. films/output/translations/ivan-vas/gpt-5.2.json)

Output:
    <eval_dir>/human_eval_details.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_items(translations_path):
    data = json.loads(Path(translations_path).read_text(encoding="utf-8"))
    return {item["id"]: item for item in data["items"]}


def load_session(path, items):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    evaluator_id = data["evaluator_id"]
    evaluator_meta = data.get("evaluator_meta", {})
    tasks = data["tasks"]
    judgments = data["judgments"]
    auto_filled_set = set(data.get("auto_filled_set", []))

    records = []
    for idx, task in enumerate(tasks):
        judgment = judgments.get(str(idx))
        if judgment is None:
            continue

        issues = judgment.get("issues", [])
        item_id = task["item_id"]
        method = task["method"]
        run = str(task["run"])
        lang = task.get("target_lang_code", "")

        item = items.get(item_id, {})
        source_text = item.get("original", {}).get("rus", "")
        reference = item.get("reference", {}).get(lang, "")
        character = item.get("character", "")
        analysis = item.get("analysis", {})
        segment_number = item.get("segment_number", [])

        translation_shown = (
            item.get("translations", {})
                .get(lang, {})
                .get(method, {})
                .get(run, None)
        )

        records.append({
            "evaluator_id":    evaluator_id,
            "evaluator_meta":  evaluator_meta,
            "film":            task["film"],
            "trans_model":     task["trans_model"],
            "item_id":         item_id,
            "segment_number":  segment_number,
            "character":       character,
            "method":          method,
            "run":             run,
            "is_repeat":       task.get("is_repeat", False),
            "auto_filled":     idx in auto_filled_set,
            "viewed_analysis": judgment.get("viewed_analysis", False),
            "source_text":     source_text,
            "reference":       reference,
            "translation":     translation_shown,
            "issues":          issues,
            "item_analysis":   analysis,
        })
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Extract enriched per-judgment details from human eval sessions."
    )
    parser.add_argument("eval_dir", help="Directory containing human_*.json files")
    parser.add_argument("translations_json", help="Mapped translations JSON file")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"Error: not a directory: {eval_dir}")
        sys.exit(1)

    items = load_items(args.translations_json)

    human_files = sorted(p for p in eval_dir.glob("human_*.json") if not p.name.startswith("human_eval_"))
    if not human_files:
        print(f"No human_*.json files found in {eval_dir}")
        sys.exit(1)

    all_records = []
    for path in human_files:
        records = load_session(path, items)
        all_records.extend(records)
        print(f"  {path.name}: {len(records)} judgments")

    out_path = eval_dir / "human_eval_details.json"
    out_path.write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(all_records)} records to {out_path}")

    issues_count = sum(
        1 for r in all_records
        if not r["auto_filled"] and any(i.get("category") != "no-issue" for i in r["issues"])
    )
    clean_count = sum(
        1 for r in all_records
        if not r["auto_filled"] and not r["is_repeat"]
        and all(i.get("category") == "no-issue" for i in r["issues"])
    )
    print(f"  Judgments with issues: {issues_count}")
    print(f"  Clean (no-issue) judgments: {clean_count}")
    missing = sum(1 for r in all_records if r["translation"] is None)
    if missing:
        print(f"  WARNING: {missing} judgment(s) had no matching translation text (method/run not found in translations JSON)")


if __name__ == "__main__":
    main()
