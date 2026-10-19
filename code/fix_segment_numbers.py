#!/usr/bin/env python3
"""Check, and optionally correct, the segment numbers of each meme in reference.json.

For each meme, find the smallest run of consecutive source subtitle entries
whose text covers the meme's Russian text. If the recorded segment numbers
cover it clearly worse than that run, propose the run instead.

Usage:
    python code/fix_segment_numbers.py [--apply]   # --apply writes reference.json (backup: reference.pre-segfix.json)
"""

import argparse
import json
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import pysrt

FILMS = ["ivan-vas", "diamond-arm", "gentelmen", "carnival-night", "pokrov-gate"]


def words(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower().replace("ё", "е"))).strip()


def coverage(meme, text):
    m = SequenceMatcher(None, meme, text, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks()) / max(1, len(meme))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    for film in FILMS:
        ref_path = Path(f"experiments/films/data/{film}/reference.json")
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        src = list(pysrt.open(f"experiments/films/data/{film}/subs/{film}-rus.srt", error_handling=pysrt.ERROR_PASS))
        pos = {s.index: k for k, s in enumerate(src)}
        changed = False
        for it in ref["items"]:
            o = it["original"]
            meme = words(o.get("rus") or o.get("Russian") or next(iter(o.values())))
            segs = it["segment_number"]
            cur = coverage(meme, words(" ".join(src[pos[s]].text for s in segs if s in pos)))
            best = None
            for n in range(1, 6):
                for k in range(len(src) - n + 1):
                    c = coverage(meme, words(" ".join(src[j].text for j in range(k, k + n))))
                    if c >= 0.95 and (best is None or c > best[0] + 0.02):
                        best = (c, k, n)
                if best:
                    break
            if best and best[0] > cur + 0.15:
                new = [src[j].index for j in range(best[1], best[1] + best[2])]
                print(f"{film:15s} item {it['id']:>3}: {segs} (cov {cur:.2f}) -> {new} (cov {best[0]:.2f})")
                print("      " + " / ".join(src[j].text.replace("\n", " ") for j in range(best[1], best[1] + best[2]))[:160])
                if a.apply:
                    it["segment_number"] = new
                    changed = True
        if a.apply and changed:
            backup = ref_path.with_name("reference.pre-segfix.json")
            if not backup.exists():
                shutil.copy2(ref_path, backup)
            ref_path.write_text(json.dumps(ref, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
