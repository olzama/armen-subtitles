#!/usr/bin/env python3
"""
Prepare spellcheck data for the web review interface.
Groups words by spelling similarity, selects diverse examples per word,
and outputs data.js into the web directory.
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).parent.parent
INPUT  = ROOT / "films/output/spellcheck_galician.json"
OUTPUT = ROOT / "films/output/eval/Galician-spellcheck/web/data.js"

MAX_EXAMPLES    = 4
JACCARD_THRESH  = 0.5   # context diversity threshold
GROUP_THRESH    = 0.82  # spelling similarity threshold for grouping
MIN_GROUP_LEN   = 5     # only group words this long or longer


# ── Context diversity ──────────────────────────────────────────────────────

def word_set(text):
    return set(re.findall(r"\w+", text.lower()))

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def select_diverse(occurrences, n=MAX_EXAMPLES, threshold=JACCARD_THRESH):
    by_method = {}
    for occ in occurrences:
        by_method.setdefault(occ["method"], []).append(occ)
    candidates, remainder = [], []
    for method in sorted(by_method):
        candidates.append(by_method[method][0])
        remainder.extend(by_method[method][1:])
    candidates.extend(remainder)
    selected, seen = [], []
    for occ in candidates:
        ws = word_set(occ["context"])
        if all(jaccard(ws, s) < threshold for s in seen):
            selected.append(occ)
            seen.append(ws)
            if len(selected) >= n:
                break
    return selected


# ── Spelling similarity grouping ───────────────────────────────────────────

def spelling_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def build_groups(word_list):
    """Union-Find clustering by spelling similarity."""
    n = len(word_list)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    words = [e["word"] for e in word_list]
    for i in range(n):
        if len(words[i]) < MIN_GROUP_LEN:
            continue
        for j in range(i + 1, n):
            if len(words[j]) < MIN_GROUP_LEN:
                continue
            if spelling_sim(words[i], words[j]) >= GROUP_THRESH:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return groups  # root_index -> [member_indices]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    data = json.load(open(INPUT, encoding="utf-8"))

    word_list = []
    for word, info in data["words"].items():
        examples = select_diverse(info["occurrences"])
        word_list.append({
            "word": word,
            "frequency": info["frequency"],
            "examples": [
                {
                    "context":      e["context"],
                    "method":       e["method"],
                    "run":          e["run"],
                    "subtitle_idx": e["subtitle_idx"],
                }
                for e in examples
            ],
        })

    # Assign group_id and group_size
    groups = build_groups(word_list)
    for members in groups.values():
        canonical = max(members, key=lambda i: word_list[i]["frequency"])
        gid = word_list[canonical]["word"]
        size = len(members)
        for i in members:
            word_list[i]["group_id"]   = gid
            word_list[i]["group_size"] = size

    # Sort: groups together, highest-frequency group first;
    # within a group, canonical (highest freq) member first.
    group_max_freq = {}
    for e in word_list:
        gid = e["group_id"]
        group_max_freq[gid] = max(group_max_freq.get(gid, 0), e["frequency"])

    word_list.sort(key=lambda e: (
        -group_max_freq[e["group_id"]],
        e["group_id"],
        -e["frequency"],
    ))

    js = "const WORDS = " + json.dumps(word_list, ensure_ascii=False, indent=2) + ";\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(js, encoding="utf-8")

    n_groups   = len({e["group_id"] for e in word_list if e["group_size"] > 1})
    n_grouped  = sum(1 for e in word_list if e["group_size"] > 1)
    print(f"Wrote {len(word_list)} words to {OUTPUT}")
    print(f"  {n_groups} similarity groups covering {n_grouped} words")


if __name__ == "__main__":
    main()
