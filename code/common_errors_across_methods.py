#!/usr/bin/env python3
"""Find errors that appear consistently across multiple methods for the same item."""

import json
from collections import defaultdict
from pathlib import Path

INPUT = Path("experiments/films/output/eval/human-eval/ivan-vas/Russian-Galician/human_eval_details.json")
OUTPUT = Path("experiments/films/output/eval/human-eval/ivan-vas/Russian-Galician/items_with_errors_across_methods.txt")

data = json.loads(INPUT.read_text(encoding="utf-8"))

# For each item, collect which methods had errors
by_item = defaultdict(lambda: defaultdict(list))
item_info = {}

for rec in data:
    if rec.get("is_repeat") or rec.get("auto_filled"):
        continue
    item_id = rec["item_id"]
    method = rec["method"]
    item_info[item_id] = {
        "source": rec.get("source_text", ""),
        "character": rec.get("character", ""),
    }
    if rec.get("issues"):
        by_item[item_id][method].extend(rec["issues"])

lines = []
lines.append("COMMON ERRORS ACROSS METHODS (ivan-vas, Galician)")
lines.append("Items where multiple methods share the same problem")
lines.append("=" * 70)

for item_id in sorted(by_item.keys()):
    methods_with_errors = by_item[item_id]
    if len(methods_with_errors) < 2:
        continue

    info = item_info[item_id]
    lines.append(f"\n{'─' * 70}")
    lines.append(f"[Item {item_id}] {info['character']} — {info['source']}")
    lines.append(f"Error found in {len(methods_with_errors)} methods: {', '.join(sorted(methods_with_errors.keys()))}")

    for method, issues in sorted(methods_with_errors.items()):
        lines.append(f"\n  {method}:")
        seen = set()
        for iss in issues:
            sev = iss.get("severity", "?").upper()
            cat = iss.get("category", "?")
            just = iss.get("justification", "").strip()
            key = (sev, cat, just)
            if key not in seen:
                seen.add(key)
                lines.append(f"    → {sev} [{cat}]: {just if just else '(sin justificación)'}")

lines.append(f"\n{'=' * 70}\n")

OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {OUTPUT}")
