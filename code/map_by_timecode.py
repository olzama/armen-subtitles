#!/usr/bin/env python3
"""Decide first-pass mapping proposals by time-code alignment.

For translations that preserve every source entry and time code, the meme's
translated text is the translation of exactly its expected source entries.
This script rewrites each proposal in first_pass.json to that text and marks it:
  accept   the meme covers (nearly) all of the source text of its entries
  partial  the entries also contain non-meme source text; needs a manual trim
  missing  an expected entry has no time-code match in the translation
Run map_first_pass.py apply afterwards (only "accept" is applied).

Usage:
    python code/map_by_timecode.py diamond-arm gpt-6-sol Russian English [--min-coverage 0.8]
"""

import argparse
import json
import re
from pathlib import Path

import pysrt

from lang_utils import normalize_lang


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("film"); ap.add_argument("model"); ap.add_argument("src"); ap.add_argument("tgt")
    ap.add_argument("--min-coverage", type=float, default=0.8)
    a = ap.parse_args()

    pair = f"{normalize_lang(a.src)}-{normalize_lang(a.tgt)}"
    root = Path("experiments/films/output/translations") / a.film / pair / a.model
    src = {s.index: s for s in pysrt.open(f"experiments/films/data/{a.film}/subs/{a.film}-rus.srt",
                                          error_handling=pysrt.ERROR_PASS)}
    fp = root / "mapping" / "first_pass.json"
    proposals = json.loads(fp.read_text(encoding="utf-8"))
    cache, counts = {}, {"accept": 0, "partial": 0, "missing": 0}
    for p in proposals:
        key = (p["method"], p["run"])
        if key not in cache:
            subs = pysrt.open(str(root / p["method"] / "translations" / f"translation-{p['run']}.txt"),
                              error_handling=pysrt.ERROR_PASS)
            cache[key] = {(str(s.start), str(s.end)): s.text for s in subs}
        tgt = cache[key]
        segs = p["expected_segs"]
        texts = [tgt.get((str(src[s].start), str(src[s].end))) if s in src else None for s in segs]
        if any(t is None for t in texts):
            p["decision"] = "missing"
        else:
            src_text = norm(" ".join(src[s].text for s in segs))
            coverage = len(norm(p["original"])) / max(1, len(src_text))
            p["source_text"] = " / ".join(src[s].text.replace("\n", " ") for s in segs)
            p["coverage"] = round(coverage, 2)
            p.setdefault("proposed_text_original", p["text"])
            p["text"] = "\n".join(texts)
            p["edited_by"] = "claude (time-code)"
            p["decision"] = "accept" if coverage >= a.min_coverage else "partial"
        counts[p["decision"]] += 1
    fp.write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
