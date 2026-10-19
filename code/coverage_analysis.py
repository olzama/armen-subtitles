#!/usr/bin/env python3
# Analyse how often LLM whole-text MQM targets culturally significant passages
# vs. random dialogue, per film and aggregated.
# Usage: python code/coverage_analysis.py

import json
from pathlib import Path

FILM_DISPLAY = {
    "ivan-vas":      "Ivan Vasilievich",
    "diamond-arm":   "Diamond Arm",
    "pokrov-gate":   "Pokrovskie Vorota",
    "carnival-night":"Carnival Night",
    "gentelmen":     "Gentlemen of Fortune",
}

films = ["ivan-vas", "diamond-arm", "pokrov-gate", "carnival-night", "gentelmen"]

print(f"{'Film':<25} {'Items':>6} {'Methods':>8} {'Flagged':>8} {'Matched':>8} {'Match%':>7} {'Recall':>7}")
print("-" * 75)

totals = dict(n_items=0, n_pairs=0, n_flagged=0, n_matched=0)

for film in films:
    sidecar = Path(f"experiments/films/output/eval/llm-eval/{film}/Russian-English"
                   "/gpt-5.2-by-gpt-5.4/mqm-wholefile/gpt-5.2-wholefile.json")
    if not sidecar.exists():
        print(f"{FILM_DISPLAY[film]:<25}  (no sidecar)")
        continue

    data = json.loads(sidecar.read_text())
    items = data["items"]
    n_items = len(items)

    # count total issues across all method runs
    wf_root = sidecar.parent
    total_issues = 0
    methods_seen = set()
    for method_dir in sorted(wf_root.iterdir()):
        if not method_dir.is_dir():
            continue
        runs = list(method_dir.glob("run_*_eval_*.json"))
        if not runs:
            continue
        methods_seen.add(method_dir.name)
        for p in runs:
            d = json.loads(p.read_text())
            total_issues += len(d.get("issues", []))

    n_methods = len(methods_seen)

    # count matched issues (from llm_wholefile field) and flagged item×method pairs
    n_flagged_pairs = 0
    n_matched_issues = 0
    n_items_flagged = 0
    for item in items:
        wf = item.get("llm_wholefile", {}).get("eng", {})
        item_flagged = False
        for method, issues in wf.items():
            if issues:
                n_flagged_pairs += 1
                n_matched_issues += len(issues)
                item_flagged = True
        if item_flagged:
            n_items_flagged += 1

    match_pct = 100 * n_matched_issues / total_issues if total_issues else 0
    recall = 100 * n_items_flagged / n_items if n_items else 0

    print(f"{FILM_DISPLAY[film]:<25} {n_items:>6} {n_methods:>8} {total_issues:>8} "
          f"{n_matched_issues:>8} {match_pct:>6.1f}% {recall:>6.1f}%")

    totals["n_items"]   += n_items
    totals["n_pairs"]   += n_methods
    totals["n_flagged"] += total_issues
    totals["n_matched"] += n_matched_issues

print("-" * 75)
match_pct = 100 * totals["n_matched"] / totals["n_flagged"] if totals["n_flagged"] else 0
print(f"{'Total':<25} {totals['n_items']:>6} {'':>8} {totals['n_flagged']:>8} "
      f"{totals['n_matched']:>8} {match_pct:>6.1f}%")
print()
print("Flagged  = total issues raised by LLM whole-text judge (all methods, all runs)")
print("Matched  = issues that overlapped (source-quote) with a culturally significant item")
print("Match%   = fraction of all LLM complaints targeting a cultural item")
print("Recall   = fraction of cultural items flagged in at least one method")
