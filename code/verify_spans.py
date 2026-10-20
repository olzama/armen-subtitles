#!/usr/bin/env python3
"""Check that each mapped meme span actually translates its meme.

Time-code mapping can pick the wrong text where a translation's text is shifted
relative to its time codes. For every translation, one call to a small LLM
receives the (Russian meme, mapped span) pairs and says, for each, whether the
span renders that meme (even badly) or is a different line. Spans judged to be a
different line, or empty, are listed as method:run pairs to re-map by content.

Usage:
    python code/verify_spans.py <film> <model> Russian <Target> [--judge gpt-5.4-mini] [--methods meme-search]
Prints a final line "FLAGGED method:run,method:run,..." and writes mapping/span_verification.json.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_mqm_parallel import load_openai_key
from lang_utils import normalize_lang, lang_code
from extract_meme_spans import rus

PROMPT = """Each numbered pair below gives some Russian film subtitle lines and a stretch of {tgt} subtitles that is supposed to be their translation. Translations may be free, creative, adapted, or flawed. For each pair, answer "yes" if the {tgt} text is a translation (however free or flawed) of those Russian lines or of most of them, and "no" if it translates different dialogue or is empty.

Return JSON only: {{"answers": {{"<number>": "yes" | "no", ...}}}}

{pairs}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("film"); ap.add_argument("model"); ap.add_argument("src_lang"); ap.add_argument("tgt_lang")
    ap.add_argument("--judge", default="gpt-5.4")
    ap.add_argument("--methods", default=None, help="Comma-separated methods to check (default: all)")
    a = ap.parse_args()
    only = set(a.methods.split(",")) if a.methods else None
    tgt, code = normalize_lang(a.tgt_lang), lang_code(a.tgt_lang)
    data = json.loads(Path(f"experiments/films/output/translations/{a.film}/{a.model}.json").read_text(encoding="utf-8"))
    import pysrt
    src = {e.index: e.text.replace("\n", " ") for e in pysrt.open(f"experiments/films/data/{a.film}/subs/{a.film}-rus.srt",
                                                                 error_handling=pysrt.ERROR_PASS)}
    ref = {str(i["id"]): i for i in json.loads(Path(f"experiments/films/data/{a.film}/reference.json").read_text(encoding="utf-8"))["items"]}
    memes = {i: " / ".join(src.get(s, "") for s in ref[i]["segment_number"]) for i in ref}
    runs = {}
    for it in data["items"]:
        for m, rr in (it.get("translations", {}).get(code) or {}).items():
            if only and m not in only:
                continue
            for r, text in rr.items():
                runs.setdefault((m, r), {})[str(it["id"])] = text
    client = openai.OpenAI(api_key=load_openai_key())

    def check(key):
        spans = runs[key]
        pairs = "\n\n".join(f"{i}. Russian: {memes[i]}\n{tgt}: {spans[i] or '(empty)'}" for i in spans)
        for attempt in range(3):
            r = client.chat.completions.create(model=a.judge, messages=[{"role": "user", "content": PROMPT.format(tgt=tgt, pairs=pairs)}])
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", r.choices[0].message.content.strip())
            try:
                return key, json.loads(txt)["answers"]
            except (json.JSONDecodeError, KeyError):
                if attempt == 2:
                    return key, {}

    with ThreadPoolExecutor(8) as ex:
        results = dict(ex.map(check, list(runs)))
    flagged, report = set(), []
    for (m, r), ans in results.items():
        for i, v in ans.items():
            if str(v).lower() != "yes":
                flagged.add(f"{m}:{r}")
                report.append({"method": m, "run": r, "item": i, "span": runs[(m, r)].get(i)})
    suffix = f"-{'-'.join(sorted(only))}" if only else ""
    out = Path(f"experiments/films/output/translations/{a.film}/Russian-{tgt}/{a.model}/mapping/span_verification{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{a.film} {a.model} {tgt}: {len(runs)} translations, {len(report)} spans flagged")
    print("FLAGGED " + ",".join(sorted(flagged)))


if __name__ == "__main__":
    main()
