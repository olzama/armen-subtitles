#!/usr/bin/env python3
# Compute Spearman rank correlations between COMET, LLM whole-text MQM, and meme-MQM
# across methods, per film and pooled across films.
# Usage: python code/metrics_correlation.py

import json
from pathlib import Path
from scipy.stats import spearmanr

METHOD_ALIASES = {"meme-list": "list", "meme_list": "list"}

def canon(name):
    return METHOD_ALIASES.get(name.strip(), name.strip())

def load_comet(film):
    jsons = list(Path(f"experiments/films/output/eval/comet-eval/{film}").glob(
        "*/*/comet_scores_wmt22-cometkiwi-da.json"))
    if not jsons:
        return {}
    d = json.loads(jsons[0].read_text())
    return {canon(m): v["mean_score"] for m, v in d["methods"].items()}

def load_wholefile(film, lang_pair="Russian-English"):
    root = Path(f"experiments/films/output/eval/llm-eval/{film}/{lang_pair}"
                "/gpt-5.2-by-gpt-5.4/mqm-wholefile")
    if not root.exists():
        return {}
    result = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        runs = list(d.glob("run_*_eval_*.json"))
        if not runs:
            continue
        scores = [json.loads(p.read_text())["summary"]["major_equiv_per_unit"] for p in runs]
        result[canon(d.name)] = sum(scores) / len(scores)
    return result

def load_meme_mqm(film, lang_pair="Russian-English"):
    for evaluator in ("gpt-5-mini", "gpt-5.4-mini"):
        p = Path(f"experiments/films/output/eval/llm-eval/{film}/{lang_pair}"
                 f"/gpt-5.2-by-{evaluator}/mqm-memes/merged_summary.json")
        if p.exists():
            d = json.loads(p.read_text())
            return {canon(m): v["mean_major_equiv_per_unit"]
                    for m, v in d["per_method"].items()}
    return {}

def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(xs, ys)
    return round(r, 3), round(p, 3)

films = ["ivan-vas", "diamond-arm", "pokrov-gate", "carnival-night", "gentelmen"]

FILM_DISPLAY = {
    "ivan-vas":      "Ivan Vasilievich",
    "diamond-arm":   "Diamond Arm",
    "pokrov-gate":   "Pokrovskie Vorota",
    "carnival-night":"Carnival Night",
    "gentelmen":     "Gentlemen of Fortune",
}

print(f"{'Film':<25} {'N':>3}  {'r(C,M)':>8} {'p':>6}  {'r(WF,M)':>8} {'p':>6}  {'r(C,WF)':>8} {'p':>6}")
print("-" * 90)

all_c_m, all_wf_m, all_c_wf = [], [], []

for film in films:
    comet = load_comet(film)
    wf    = load_wholefile(film)
    meme  = load_meme_mqm(film)

    # common methods across all three
    common = sorted(set(comet) & set(wf) & set(meme))
    n = len(common)

    c_vals  = [comet[m] for m in common]
    wf_vals = [wf[m]    for m in common]
    m_vals  = [meme[m]  for m in common]

    # negate COMET so higher=better aligns with lower-is-better for others
    # actually keep natural: for correlation we just note the direction
    r_cm,  p_cm  = spearman(c_vals,  m_vals)
    r_wfm, p_wfm = spearman(wf_vals, m_vals)
    r_cwf, p_cwf = spearman(c_vals,  wf_vals)

    print(f"{FILM_DISPLAY[film]:<25} {n:>3}  {r_cm:>8.3f} {p_cm:>6.3f}  {r_wfm:>8.3f} {p_wfm:>6.3f}  {r_cwf:>8.3f} {p_cwf:>6.3f}")

    all_c_m  += list(zip(c_vals,  m_vals))
    all_wf_m += list(zip(wf_vals, m_vals))
    all_c_wf += list(zip(c_vals,  wf_vals))

print("-" * 90)
r_cm,  p_cm  = spearman([x for x,_ in all_c_m],  [y for _,y in all_c_m])
r_wfm, p_wfm = spearman([x for x,_ in all_wf_m], [y for _,y in all_wf_m])
r_cwf, p_cwf = spearman([x for x,_ in all_c_wf], [y for _,y in all_c_wf])
n_pooled = len(all_c_m)
print(f"{'Pooled':<25} {n_pooled:>3}  {r_cm:>8.3f} {p_cm:>6.3f}  {r_wfm:>8.3f} {p_wfm:>6.3f}  {r_cwf:>8.3f} {p_cwf:>6.3f}")
print()
print("Columns: r(C,M) = Spearman correlation COMET vs Meme-MQM")
print("         r(WF,M) = LLM whole-text MQM vs Meme-MQM")
print("         r(C,WF) = COMET vs LLM whole-text MQM")
print("Note: COMET is higher-is-better; LLM-WF and Meme-MQM are lower-is-better.")
print("      A negative r(C,M) means COMET ranks methods oppositely to Meme-MQM.")
