#!/usr/bin/env python3
"""Enrich a mapped translations JSON (gpt-5.2.json) with LLM whole-text MQM judgments.

For each reference item, finds any LLM issues (from mqm-wholefile eval runs) whose
source_quote overlaps substantially with the item's Russian source text, and adds
them under a new 'llm_wholefile' field keyed by method name.

Usage:
    python code/match_wholefile_to_reference.py \\
        experiments/films/output/translations/ivan-vas/gpt-5.2.json \\
        experiments/films/output/eval/llm-eval/ivan-vas/Russian-English/gpt-5.2-by-gpt-5.4/mqm-wholefile/ \\
        --out experiments/films/output/translations/ivan-vas/gpt-5.2-wholefile.json
"""

import argparse
import json
import re
from pathlib import Path


def tokenize(text):
    return set(re.findall(r"[а-яёa-z\w]{2,}", text.lower()))


def overlap_ratio(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / min(len(a_tokens), len(b_tokens))


def load_issues_from_dir(method_dir):
    """Load and deduplicate issues from all run_*_eval_*.json files in a method dir."""
    seen = {}
    for p in sorted(Path(method_dir).glob("run_*_eval_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for issue in data.get("issues", []):
            key = (
                issue.get("source_quote", ""),
                issue.get("severity", ""),
                issue.get("category", ""),
            )
            if key not in seen:
                seen[key] = {**issue, "seen_in_runs": [p.name]}
            else:
                seen[key]["seen_in_runs"].append(p.name)
    return list(seen.values())


def match_issues(item_text, issues, threshold):
    item_tokens = tokenize(item_text)
    matched = []
    for issue in issues:
        quote = issue.get("source_quote", "")
        if not quote:
            continue
        if overlap_ratio(item_tokens, tokenize(quote)) >= threshold:
            matched.append(issue)
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapped_json",
                        help="Path to mapped translations JSON (e.g. gpt-5.2.json)")
    parser.add_argument("mqm_wholefile_dir",
                        help="Path to mqm-wholefile eval directory (contains method subdirs)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--target-lang", required=True,
                        help="ISO 639-2 target language code (e.g. eng, glg, spa)")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Word-overlap threshold for matching (default: 0.4)")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        base = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"Updating existing output: {args.out}")
    else:
        base = json.loads(Path(args.mapped_json).read_text(encoding="utf-8"))
    items = base.get("items", [])

    mqm_root = Path(args.mqm_wholefile_dir)
    method_dirs = sorted(d for d in mqm_root.iterdir() if d.is_dir())
    if not method_dirs:
        print(f"No method subdirectories found in {mqm_root}")

    # Build per-method issue lists
    method_issues = {}
    for d in method_dirs:
        issues = load_issues_from_dir(d)
        if issues:
            method_issues[d.name] = issues
            print(f"  {d.name}: {len(issues)} deduplicated issues")

    tgt = args.target_lang

    # Match issues to each item, nested under target language
    for item in items:
        src_text = item.get("original", {}).get("rus", "")
        if "llm_wholefile" not in item:
            item["llm_wholefile"] = {}
        item["llm_wholefile"].setdefault(tgt, {})
        for method, issues in method_issues.items():
            matched = match_issues(src_text, issues, args.threshold)
            item["llm_wholefile"][tgt][method] = matched

    output = {**base, "items": items}
    Path(args.out).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Summary
    total_matched = sum(
        1 for item in items
        if any(item["llm_wholefile"].get(tgt, {}).get(m) for m in method_issues)
    )
    print(f"\n{total_matched}/{len(items)} items matched in at least one method ({tgt})")
    for item in items:
        hits = {m: len(v) for m, v in item["llm_wholefile"].get(tgt, {}).items() if v}
        if hits:
            src_preview = item.get("original", {}).get("rus", "")[:60]
            print(f"  id={item['id']:2d}  {hits}  [{src_preview}…]")
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
