#!/usr/bin/env python3
"""Summarize human eval errors by method, showing justifications per item."""

import json
from collections import defaultdict
from pathlib import Path

INPUT = Path("films/output/eval/human-eval/ivan-vas/Russian-Galician/human_eval_details.json")
OUTPUT = Path("films/output/eval/human-eval/ivan-vas/Russian-Galician/human_errors_by_method.txt")

data = json.loads(INPUT.read_text(encoding="utf-8"))

# Group records by method
by_method = defaultdict(list)
for rec in data:
    if not rec.get("is_repeat") and not rec.get("auto_filled"):
        by_method[rec["method"]].append(rec)

def error_count(records):
    return sum(1 for r in records if r.get("issues") and not r.get("is_repeat") and not r.get("auto_filled"))

METHOD_ORDER = sorted(by_method.keys(), key=lambda m: error_count(by_method[m]))

lines = []
lines.append("HUMAN EVALUATION — ERROR SUMMARY BY METHOD (ivan-vas, Galician)")
lines.append("Ordered from fewest to most errors")
lines.append("=" * 70)

for method in METHOD_ORDER:
    records = by_method.get(method, [])
    if not records:
        continue

    with_issues = [r for r in records if r.get("issues")]
    total = len(records)
    n_errors = len(with_issues)

    lines.append(f"\n{'─' * 70}")
    lines.append(f"METHOD: {method}  ({n_errors} errors out of {total} evaluations)")
    lines.append(f"{'─' * 70}")

    if not with_issues:
        lines.append("  No errors marked.")
        continue

    # Group by item
    by_item = defaultdict(list)
    for r in with_issues:
        by_item[r["item_id"]].append(r)

    for item_id in sorted(by_item.keys()):
        item_recs = by_item[item_id]
        first = item_recs[0]
        lines.append(f"\n  [Item {item_id}] {first.get('character', '')} — {first.get('source_text', '')}")
        for r in item_recs:
            lines.append(f"    Translation (run {r['run']}): {r.get('translation', '').replace(chr(10), ' ')}")
            for iss in r.get("issues", []):
                sev = iss.get("severity", "?").upper()
                cat = iss.get("category", "?")
                just = iss.get("justification", "").strip()
                lines.append(f"      → {sev} [{cat}]: {just if just else '(sin justificación)'}")

lines.append(f"\n{'=' * 70}\n")

OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {OUTPUT}")
