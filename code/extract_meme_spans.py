#!/usr/bin/env python3
"""Re-map every meme to exactly its own span in each translation, with one rule for all models.

The interactive mapping (map_translation_segments.py) left mixed spans: some
runs kept whole subtitle entries, including dialogue around the meme, others
were trimmed to the meme. Since the evaluator penalizes the surrounding text,
conditions mapped more tightly looked better. Here an LLM is shown, for each
meme, the Russian meme text, the numbered Russian subtitle entries around its
expected position, and the time-code-matched translated entries, and returns
the verbatim substring of the translation that renders the meme. Every
returned line must occur in the translated window; anything that does not is
written to a review list and the item keeps its previous mapping.

Usage:
    python code/extract_meme_spans.py diamond-arm gpt-6-sol Russian English [--methods zero,given] [--dry-run]
Writes <model>.json in place (previous version saved once as <model>.pre-span.json)
and a log at <film>/<pair>/<model>/mapping/span_extraction.json.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
import pysrt

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_mqm_parallel import load_openai_key
from lang_utils import normalize_lang, lang_code

MODEL = "gpt-6-sol"
WINDOW = 3

PROMPT = """You are aligning film subtitles. For each meme below you get the Russian meme (a famous quotation), the Russian subtitle entries around where it occurs, and the {tgt} translation of the same entries (same entry numbers). The subtitle wording may differ slightly from the quotation as usually cited.

The quotation as usually cited may differ from the subtitles in wording and word order, and may abbreviate or rearrange them. For each meme, first locate the contiguous stretch of Russian subtitle text that the quotation refers to, from where its first part starts to where its last part ends, including everything in between even if the citation leaves it out or reorders it. Then return the {tgt} translation of exactly that stretch, in full: drop only translated material that lies before its start or after its end (surrounding dialogue, a preceding sentence ending in the same entry, interjections or nonsense words outside the quoted part). Drop leading dialogue dashes. Copy the {tgt} text verbatim, character for character, one line per subtitle line, joined with "\\n". If the translation leaves the meme out entirely, return "".

Return JSON only: {{"spans": {{"<meme id>": "<verbatim {tgt} text>", ...}}}}

{items}"""


QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2014": "-", "\u2013": "-", "\u2026": "..."})


def rus(item):
    o = item["original"]
    return o.get("rus") or o.get("Russian") or next(iter(o.values()))


def norm(t):
    return re.sub(r"\s+", " ", t.translate(QUOTES)).strip()


def words(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower().replace("ё", "е"))).strip()


def align(src, subs):
    """Map each source entry's time code to the translated text shown during it.

    Exact time-code match first; otherwise the translated entries whose time
    interval overlaps the source entry (handles merged, split and renumbered
    entries); otherwise the entry with the same number.
    """
    by_tc = {(str(s.start), str(s.end)): s.text for s in subs}
    by_idx = {s.index: s.text for s in subs}
    spans = sorted(((s.start.ordinal, s.end.ordinal, s.text) for s in subs))
    out = {}
    for i, e in src.items():
        k = (str(e.start), str(e.end))
        if k in by_tc:
            out[k] = by_tc[k]
            continue
        a, b = e.start.ordinal, e.end.ordinal
        hits = [t for s0, s1, t in spans if s0 < b and s1 > a and min(b, s1) - max(a, s0) >= 0.3 * min(b - a, s1 - s0)]
        if hits:
            out[k] = "\n".join(hits)
        elif i in by_idx:
            out[k] = by_idx[i]
    return out


def tc_match_ratio(src, subs):
    stc = {(str(e.start), str(e.end)) for e in src.values()}
    return len(stc & {(str(t.start), str(t.end)) for t in subs}) / max(1, len(stc))


def content_window(item, src, subs, ratio, seconds=240, entries=25):
    """Translated entries around a meme in a translation whose time codes do not follow the source:
    the union of the entries within +-seconds of the meme's source time and the entries within
    +-entries of its source position."""
    segs = [s for s in item["segment_number"] if s in src]
    if not segs:
        return ""
    subs = list(subs)
    a = src[min(segs)].start.ordinal - seconds * 1000
    b = src[max(segs)].end.ordinal + seconds * 1000
    keep = {k for k, t in enumerate(subs) if t.end.ordinal > a and t.start.ordinal < b}
    order = sorted(src)
    scale = len(subs) / max(1, len(order))  # translations that merge entries have fewer of them
    k0, k1 = int(order.index(min(segs)) * scale), int(order.index(max(segs)) * scale)
    keep |= set(range(max(0, k0 - entries), min(len(subs), k1 + entries + 1)))
    return "\n".join(f"{subs[k].index}: {subs[k].text}" for k in sorted(keep))


def dedup_join(texts):
    """Join entry texts, dropping a text repeated from the previous entry (one translated
    entry overlapping two source entries)."""
    out = []
    for t in texts:
        if not out or t != out[-1]:
            out.append(t)
    return "\n".join(out)


def mode_for(item, src):
    """exact: the meme fills its source entries; trim: its entries also hold other text;
    search: its entries do not contain it (wrong or paraphrased segment numbers)."""
    from difflib import SequenceMatcher
    meme = words(rus(item))
    entries = words(" ".join(src[s].text for s in item["segment_number"] if s in src))
    if not entries:
        return "search"
    m = SequenceMatcher(None, meme, entries, autojunk=False)
    matched = sum(b.size for b in m.get_matching_blocks())
    if matched / max(1, len(meme)) < 0.6:
        return "search"
    return "exact" if len(meme) / len(entries) >= 0.95 else "trim"


def windows(item, src, tgt_by_tc, width=WINDOW):
    segs = item["segment_number"]
    lo, hi = min(segs) - width, max(segs) + width
    ru, tr = [], []
    for i in range(lo, hi + 1):
        if i in src:
            ru.append(f"{i}: {src[i].text}")
            t = tgt_by_tc.get((str(src[i].start), str(src[i].end)))
            if t is not None:
                tr.append(f"{i}: {t}")
    return "\n".join(ru), "\n".join(tr)


def extract(client, items, src, tgt_by_tc, tgt_name, widths, tr_override=None):
    blocks, wins = [], {}
    if not items:
        return {}, wins
    for it in items:
        ru, tr = windows(it, src, tgt_by_tc, widths[str(it["id"])])
        if tr_override is not None:
            tr = tr_override[str(it["id"])]
        wins[str(it["id"])] = tr
        blocks.append(f"### MEME {it['id']}\nRussian meme: {rus(it)}\n"
                      f"Russian entries:\n{ru}\n{tgt_name} entries:\n{tr}")
    content = PROMPT.format(tgt=tgt_name, items="\n\n".join(blocks))
    for attempt in range(3):
        r = client.chat.completions.create(model=MODEL, reasoning_effort="low", messages=[{"role": "user", "content": content}])
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", r.choices[0].message.content.strip())
        try:
            return json.loads(txt)["spans"], wins
        except (json.JSONDecodeError, KeyError):
            if attempt == 2:
                raise


FULL_PROMPT = """You are aligning film subtitles. The {tgt} subtitle file below is a translation of a Russian film whose entry numbers and time codes do not match the Russian original. For each Russian meme (a famous quotation) below, given with the Russian subtitle entries around it, find where the {tgt} translation renders it and return the {tgt} text that renders the meme and nothing else, copied verbatim from the {tgt} file, one line per subtitle line, joined with "\\n". Drop translated material that lies before the quotation's start or after its end. If the translation leaves the meme out, return "".

Return JSON only: {{"spans": {{"<meme id>": "<verbatim {tgt} text>", ...}}}}

MEMES:
{memes}

{tgt} SUBTITLE FILE:
{full}"""


def extract_full(client, items, src, subs, tgt_name):
    blocks = []
    for it in items:
        segs = it["segment_number"]
        ru = "\n".join(f"{i}: {src[i].text}" for i in range(min(segs) - WINDOW, max(segs) + WINDOW + 1) if i in src)
        blocks.append(f"### MEME {it['id']}\nRussian meme: {rus(it)}\nRussian entries:\n{ru}")
    full = "\n".join(f"{t.index}: {t.text}" for t in subs)
    content = FULL_PROMPT.format(tgt=tgt_name, memes="\n\n".join(blocks), full=full)
    for attempt in range(3):
        r = client.chat.completions.create(model=MODEL, reasoning_effort="low", messages=[{"role": "user", "content": content}])
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", r.choices[0].message.content.strip())
        try:
            spans = json.loads(txt)["spans"]
            return spans, {str(it["id"]): full for it in items}
        except (json.JSONDecodeError, KeyError):
            if attempt == 2:
                raise


def verbatim(span, window):
    w = norm(re.sub(r"^\d+: ", "", window, flags=re.M))
    return all(norm(line) in w for line in span.split("\n") if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("film"); ap.add_argument("model"); ap.add_argument("src_lang"); ap.add_argument("tgt_lang")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--runs", default=None,
                    help="Comma-separated method:run pairs, or method:run:item triples, to process "
                         "(everything else untouched); prints RESCORE.")
    ap.add_argument("--content", action="store_true",
                    help="Locate memes by content in the whole translated file for the selected runs.")
    ap.add_argument("--redo-failed", action="store_true",
                    help="Only re-extract spans that failed verification in the previous run's log; "
                         "print the affected method/run pairs so they can be re-scored.")
    ap.add_argument("--only-missing", action="store_true",
                    help="Only fill memes that have no mapping yet; never change a span that may already be scored.")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    tgt_name, code = normalize_lang(a.tgt_lang), lang_code(a.tgt_lang)
    film_root = Path("experiments/films/output/translations") / a.film
    trans_root = film_root / f"{normalize_lang(a.src_lang)}-{tgt_name}" / a.model
    out_json = film_root / f"{a.model}.json"
    if out_json.exists():
        data = json.loads(out_json.read_text(encoding="utf-8"))
    else:  # first mapping for this model: start from the reference items, without translations
        data = json.loads(Path(f"experiments/films/data/{a.film}/reference.json").read_text(encoding="utf-8"))
        data["model"] = a.model
        for it in data["items"]:
            it["translations"] = {}
            it["original"] = {**it["original"], "rus": rus(it)}  # the evaluator reads original.rus
    ref_items = json.loads(Path(f"experiments/films/data/{a.film}/reference.json").read_text(encoding="utf-8"))["items"]
    src = {s.index: s for s in pysrt.open(f"experiments/films/data/{a.film}/subs/{a.film}-rus.srt",
                                          error_handling=pysrt.ERROR_PASS)}
    methods = a.methods.split(",") if a.methods else sorted(p.name for p in trans_root.iterdir()
                                                             if (p / "translations").is_dir() and "." not in p.name)
    jobs = [(m, f) for m in methods for f in sorted((trans_root / m / "translations").glob("translation-*.txt"))]
    failed = None
    if a.runs:
        sel = [x.split(":") for x in a.runs.split(",")]
        want = {(x[0], x[1]) for x in sel}
        jobs = [(m, f) for m, f in jobs if (m, str(int(re.search(r"(\d+)", f.stem).group(1)))) in want]
        failed = set()
        for x in sel:
            items_sel = [x[2]] if len(x) > 2 else [str(it["id"]) for it in ref_items]
            failed |= {(i, x[0], x[1]) for i in items_sel}
    if a.redo_failed:
        prev = json.loads((trans_root / "mapping" / "span_extraction.json").read_text(encoding="utf-8"))
        failed = {(e["item"], e["method"], e["run"]) for e in prev if not e["verbatim"]}
        runs_failed = {(m, r) for _, m, r in failed}
        jobs = [(m, f) for m, f in jobs if (m, str(int(re.search(r"(\d+)", f.stem).group(1)))) in runs_failed]
    client = openai.OpenAI(api_key=load_openai_key())
    modes = {str(it["id"]): mode_for(it, src) for it in ref_items}
    print("modes:", {k: sum(v == k for v in modes.values()) for k in ("exact", "trim", "search")},
          "| trim:", [i for i, v in modes.items() if v == "trim"], "| search:", [i for i, v in modes.items() if v == "search"])

    def run(job):
        m, f = job
        run_id = str(int(re.search(r"(\d+)", f.stem).group(1)))
        subs = pysrt.open(str(f), error_handling=pysrt.ERROR_PASS)
        tgt = align(src, subs)
        ratio = tc_match_ratio(src, subs)
        if ratio < 0.95 or a.content:
            # Time codes do not follow the source closely enough for time-code mapping:
            # locate every meme by content in a wide window of the translation.
            got, w = extract_full(client, ref_items, src, subs, tgt_name)
            return m, run_id, got, w
        spans, wins, llm_items, widths = {}, {}, [], {}
        for it in ref_items:
            iid, mode = str(it["id"]), modes[str(it["id"])]
            texts = [tgt.get((str(src[s].start), str(src[s].end))) for s in it["segment_number"] if s in src]
            if mode == "exact" and texts and all(t is not None for t in texts):
                spans[iid] = dedup_join(texts)
                wins[iid] = dedup_join(texts)
            else:  # trim within the meme's own entries, or search around them
                llm_items.append(it)
                widths[iid] = 0 if mode == "trim" and all(t is not None for t in texts) else WINDOW
        got, w = extract(client, llm_items, src, tgt, tgt_name, widths)
        spans.update(got); wins.update(w)
        return m, run_id, spans, wins

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(run, jobs))

    by_id = {str(it["id"]): it for it in data["items"]}
    log, changed, review = [], 0, 0
    for m, run_id, spans, wins in results:
        for iid, span in spans.items():
            item = by_id.get(str(iid))
            if item is None:
                continue
            runs = item.setdefault("translations", {}).setdefault(code, {}).setdefault(m, {})
            old = runs.get(run_id)
            # An empty span or an empty window is never applied: missing time codes must not
            # turn into an "omitted" meme.
            ok = bool(span.strip()) and bool(wins.get(str(iid))) and verbatim(span, wins[str(iid)])
            entry = {"item": str(iid), "method": m, "run": run_id, "old": old, "new": span, "verbatim": ok}
            if a.only_missing and old is not None:
                continue
            if failed is not None and (str(iid), m, run_id) not in failed:
                continue
            if ok and span != old:
                if not a.dry_run:
                    runs[run_id] = span
                changed += 1
            elif not ok:
                review += 1
            log.append(entry)

    if failed is not None:
        fixed = sorted({(e["method"], e["run"]) for e in log if e["verbatim"] and norm(e["new"]) != norm(e["old"] or "")})
        print("RESCORE", json.dumps(fixed))
        (trans_root / "mapping" / "span_extraction.redo.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path = None
    else:
        log_path = trans_root / "mapping" / "span_extraction.json"
    (trans_root / "mapping").mkdir(exist_ok=True)
    if log_path is not None:
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    if not a.dry_run:
        backup = film_root / f"{a.model}.pre-span.json"
        if not backup.exists() and out_json.exists():
            backup.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{a.film} {a.model}: {len(jobs)} translations, {len(log)} spans, {changed} changed, {review} need review")


if __name__ == "__main__":
    main()
