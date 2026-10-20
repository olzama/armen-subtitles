#!/usr/bin/env python3
"""How many of the audience-selected memes does a model's own meme search find?

Compares the Russian passages quoted in the model's meme-search analysis
(meme-search/meme-analysis/meme-analysis.txt) with the curated memes in
reference.json. A curated meme counts as found if a judge LLM (GPT-5.4, or a Claude model) says the analysis
discusses it, in any language (models sometimes quote the memes in English).

Usage:
    python code/meme_search_coverage.py [--models gpt-5.2,gpt-6-sol] [--judge claude-sonnet-5-5]
Writes experiments/films/output/eval/difficulty/meme_search_coverage[-by-<judge>].json
"""

import argparse
import sys
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
FILMS = ["ivan-vas", "diamond-arm", "gentelmen", "carnival-night", "pokrov-gate"]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower().replace("ё", "е"))).strip()


def quoted(text):
    """Russian passages quoted in the analysis: «...», "...", or **...** spans with Cyrillic."""
    spans = re.findall(r"«([^»]+)»|“([^”]+)”|\*\*([^*]+)\*\*|\"([^\"]+)\"", text)
    out = [next(g for g in s if g) for s in spans]
    return [norm(q) for q in out if re.search(r"[а-яА-Я]", q)]


def found(meme, quotes, threshold):
    m = norm(meme)
    for q in quotes:
        if m in q or q in m and len(q) > 0.5 * len(m):
            return True
        if SequenceMatcher(None, m, q, autojunk=False).ratio() >= threshold:
            return True
    return False


JUDGE = """Below is a model's own list of memorable lines from a film, with notes, followed by a numbered list of famous quotations (memes) from the same film. For each numbered quotation, say whether the model's list discusses that quotation (in any language, in any wording; a different line from the same scene does not count).

Return JSON only: {{"found": {{"<number>": true or false, ...}}}}

MODEL'S LIST:
{analysis}

QUOTATIONS:
{memes}"""


def ask(client, model, prompt):
    if model.startswith("claude"):
        from claude_cli import call_claude
        txt = call_claude(prompt, model, "You are a careful annotator. Answer with JSON only.")[0]
    else:
        r = client.chat.completions.create(model=model, temperature=0, messages=[{"role": "user", "content": prompt}])
        txt = r.choices[0].message.content.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip())
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


def judge(client, model, analysis, items):
    memes = "\n".join(f"{it['id']}: {rus(it)}" for it in items)
    return [k for k, v in ask(client, model, JUDGE.format(analysis=analysis, memes=memes))["found"].items() if v]


STRICT = """Below is a model's own list of memorable lines from a film, with its notes on each, followed by famous quotations (memes) from the film, each with an expert's approved analysis of why it is funny and what a translation must preserve. For each quotation, compare the model's notes on it with the expert analysis:
- "correct": the model's notes identify the same essential mechanism or pitfall the expert analysis identifies;
- "partial": they identify some of it but miss or contradict an essential point (such as a warning about a specific mistranslation);
- "wrong": they discuss the line but miss the point;
- "absent": the model's list does not discuss this quotation.

Return JSON only: {{"verdicts": {{"<number>": "correct" | "partial" | "wrong" | "absent", ...}}}}

MODEL'S LIST:
{analysis}

QUOTATIONS WITH EXPERT ANALYSIS:
{memes}"""


def strict(client, model, analysis, items):
    def expert(it):
        a = it.get("analysis") or {}
        g = a.get("general") or {}
        return " ".join(x for x in (g.get("text", ""), g.get("nb", "")) if x)
    memes = "\n\n".join(f"{it['id']}: {rus(it)}\nExpert analysis: {expert(it)}" for it in items)
    return ask(client, model, STRICT.format(analysis=analysis, memes=memes))["verdicts"]


def rus(it):
    o = it["original"]
    return o.get("rus") or o.get("Russian") or next(iter(o.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-5.2,gpt-6-sol")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--judge", default="gpt-5.4")
    a = ap.parse_args()
    client = None
    if not a.judge.startswith("claude"):
        import openai
        from evaluate_mqm_parallel import load_openai_key
        client = openai.OpenAI(api_key=load_openai_key())
    out = {}
    for model in a.models.split(","):
        for film in FILMS:
            f = Path(f"experiments/films/output/translations/{film}/Russian-English/{model}/meme-search/meme-analysis/meme-analysis.txt")
            if not f.exists():
                continue
            text = f.read_text(encoding="utf-8")
            items = json.loads(Path(f"experiments/films/data/{film}/reference.json").read_text(encoding="utf-8"))["items"]
            hits = judge(client, a.judge, text, items)
            v = strict(client, a.judge, text, items)
            correct = [k for k, x in v.items() if x == "correct"]
            out.setdefault(model, {})[film] = {"memes": len(items), "found": len(hits), "found_ids": hits,
                                               "analysis_correct": len(correct), "verdicts": v}
            print(f"{model:10s} {film:15s} found {len(hits)}/{len(items)}  analysis correct {len(correct)}  "
                  f"{ {k: list(v.values()).count(k) for k in ('correct', 'partial', 'wrong', 'absent')} }")
    suffix = "" if a.judge == "gpt-5.4" else f"-by-{a.judge}"
    p = Path(f"experiments/films/output/eval/difficulty/meme_search_coverage{suffix}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
