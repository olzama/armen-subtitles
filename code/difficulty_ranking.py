#!/usr/bin/env python3
"""Do off-the-shelf translation-difficulty signals find the meme passages?

Scores every source subtitle entry of a film with a difficulty signal, then
reports where the meme entries (segment_number of the items in reference.json)
rank among all entries.

Signals (higher = harder after sign flip where needed):
  sentinel   Prosho/sentinel-src-25 (Proietti et al. 2025), source only; model score = ease
  cometkiwi  Unbabel/wmt22-cometkiwi-da on a zero-shot translation, time-code aligned; score = quality
  llm        an LLM rates each entry 1-5 for how hard it is to translate (batched prompts)

Usage:
    python code/difficulty_ranking.py ivan-vas --signal sentinel
    python code/difficulty_ranking.py ivan-vas --signal cometkiwi --trans-model gpt-5.2 --method zero --run 1
    python code/difficulty_ranking.py ivan-vas --signal llm --llm-model gpt-5-mini
Output: experiments/films/output/eval/difficulty/<film>/<signal>[-<detail>].json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pysrt

sys.path.insert(0, str(Path(__file__).parent))

OUT_ROOT = Path("experiments/films/output/eval/difficulty")

LLM_PROMPT = """You are helping a subtitle translator plan their work on the Soviet film "{film}". Below are numbered Russian subtitle entries, in order. For each entry, rate how difficult it is to translate into {target} for a viewer who does not know Russian culture: 1 = routine, 5 = needs special creative or cultural treatment (e.g. a famous quotation, wordplay, a culturally loaded allusion, an idiom whose effect would be lost if translated literally).

Return JSON only: {{"ratings": {{"<entry number>": <1-5>, ...}}}} with one rating for every entry.

ENTRIES:
{entries}"""


def source_entries(film):
    subs = pysrt.open(f"experiments/films/data/{film}/subs/{film}-rus.srt", error_handling=pysrt.ERROR_PASS)
    return {s.index: s.text.replace("\n", " ").strip() for s in subs}


def meme_entries(film):
    ref = json.loads(Path(f"experiments/films/data/{film}/reference.json").read_text(encoding="utf-8"))
    return {s for item in ref["items"] for s in item.get("segment_number", [])}


def score_sentinel(src):
    from sentinel_metric import download_model, load_from_checkpoint
    import torch
    model = load_from_checkpoint(download_model("Prosho/sentinel-src-25"))
    ids = list(src)
    out = model.predict([{"src": src[i]} for i in ids], batch_size=32, gpus=1 if torch.cuda.is_available() else 0)
    return {i: -s for i, s in zip(ids, out.scores)}  # sentinel scores ease


def score_cometkiwi(film, src, trans_model, method, run, target):
    from evaluate_neural_metrics import CometScorer, hf_login
    path = (Path("experiments/films/output/translations") / film / f"Russian-{target}" / trans_model
            / method / "translations" / f"translation-{run}.txt")
    src_tc = {(str(s.start), str(s.end)): s.index
              for s in pysrt.open(f"experiments/films/data/{film}/subs/{film}-rus.srt", error_handling=pysrt.ERROR_PASS)}
    tgt = {(str(s.start), str(s.end)): s.text.replace("\n", " ").strip()
           for s in pysrt.open(str(path), error_handling=pysrt.ERROR_PASS)}
    ids = [src_tc[k] for k in tgt if k in src_tc and src.get(src_tc[k]) and tgt[k]]
    pairs = [(src[i], tgt[k]) for k in tgt if k in src_tc and src.get(i := src_tc[k]) and tgt[k]]
    hf_login()
    scores = CometScorer("Unbabel/wmt22-cometkiwi-da")(pairs)
    return {i: -s for i, s in zip(ids, scores)}  # kiwi scores quality


def score_llm(film, src, model, target, batch=120):
    import openai
    from evaluate_mqm_parallel import load_openai_key
    client = openai.OpenAI(api_key=load_openai_key())
    ids, scores = list(src), {}
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        entries = "\n".join(f"{i}: {src[i]}" for i in chunk)
        for attempt in range(3):
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": LLM_PROMPT.format(film=film, target=target, entries=entries)}])
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.choices[0].message.content.strip())
            try:
                ratings = json.loads(text)["ratings"]
                break
            except (json.JSONDecodeError, KeyError):
                if attempt == 2:
                    raise
        for i in chunk:
            scores[i] = float(ratings.get(str(i), 1))
        print(f"  rated entries {chunk[0]}-{chunk[-1]}")
    return scores


def rank_report(scores, positives):
    """AUROC (ties count half), recall/precision at K = #positives, median percentile of positives."""
    ids = sorted(scores, key=lambda i: -scores[i])
    pos = [i for i in ids if i in positives]
    neg = [i for i in ids if i not in positives]
    wins = sum((scores[p] > scores[n]) + 0.5 * (scores[p] == scores[n]) for p in pos for n in neg)
    auroc = wins / (len(pos) * len(neg))
    k = len(pos)
    hits = sum(1 for i in ids[:k] if i in positives)
    rank = {i: r for r, i in enumerate(ids)}
    pct = sorted(100 * rank[p] / (len(ids) - 1) for p in pos)
    return {"n_entries": len(ids), "n_meme_entries": k, "auroc": round(auroc, 4),
            "recall_at_k": round(hits / k, 4), "median_percentile_of_memes": round(pct[len(pct) // 2], 1),
            "meme_percentiles": [round(p, 1) for p in pct]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("film")
    ap.add_argument("--signal", choices=["sentinel", "cometkiwi", "llm"], required=True)
    ap.add_argument("--target", default="English")
    ap.add_argument("--trans-model", default="gpt-5.2")
    ap.add_argument("--method", default="zero")
    ap.add_argument("--run", default="1")
    ap.add_argument("--llm-model", default="gpt-5-mini")
    args = ap.parse_args()

    src = source_entries(args.film)
    positives = meme_entries(args.film) & set(src)
    if args.signal == "sentinel":
        scores, tag = score_sentinel(src), "sentinel"
    elif args.signal == "cometkiwi":
        scores = score_cometkiwi(args.film, src, args.trans_model, args.method, args.run, args.target)
        tag = f"cometkiwi-{args.trans_model}-{args.method}-{args.run}"
    else:
        scores, tag = score_llm(args.film, src, args.llm_model, args.target), f"llm-{args.llm_model}"

    report = rank_report(scores, positives & set(scores))
    out = OUT_ROOT / args.film / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"film": args.film, "signal": tag, **report,
                               "scores": {str(i): s for i, s in scores.items()}}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "meme_percentiles"}))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
