#!/usr/bin/env python3
# Generate LaTeX tables comparing COMET, LLM whole-text MQM, and meme-MQM.
# Table A: all methods x three metrics for ivan-vas English
# Table B: rank of 'given' across all five films on all three metrics
# Requires in LaTeX preamble: \\usepackage{booktabs}
# Usage: python code/metrics_comparison_latex.py --out experiments/films/results/metrics_comparison.tex

import argparse
import json
from pathlib import Path
from scipy.stats import spearmanr

METHOD_ORDER = [
    "zero", "characters", "list", "list-analysis", "meme-search",
    "examples", "narratives", "intertext", "summary",
    "given", "noise",
]

METHOD_DISPLAY = {
    "zero":         "zero",
    "characters":   "characters",
    "list":         "meme list",
    "list-analysis":"list+analysis",
    "meme-search":  "meme-search",
    "examples":     "examples",
    "narratives":   "narratives",
    "intertext":    "intertext",
    "summary":      "summary",
    "given":        r"\textit{given trans.}",
    "noise":        "added noise",
}

FILM_DISPLAY = {
    "ivan-vas":      r"\textit{Ivan Vasilievich}",
    "diamond-arm":   r"\textit{Diamond Arm}",
    "pokrov-gate":   r"\textit{Pokrovskie Vorota}",
    "carnival-night":r"\textit{Carnival Night}",
    "gentelmen":     r"\textit{Gentlemen of Fortune}",
}

METHOD_ALIASES = {"meme-list": "list", "meme_list": "list"}


def canon(name):
    return METHOD_ALIASES.get(name.strip(), name.strip())


def bold(s):
    return r"\textbf{" + s + "}"


def fmt(x):
    return f"{x:.3f}"


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
        scores = [json.loads(p.read_text())["summary"]["major_equiv_per_unit"]
                  for p in runs]
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


def rank_of(scores, key, higher_better):
    if key not in scores:
        return None, None
    val = scores[key]
    all_vals = list(scores.values())
    rank = sum(1 for v in all_vals if (v > val if higher_better else v < val)) + 1
    return rank, len(all_vals)


# ── Table A ───────────────────────────────────────────────────────────────────

def table_a(film="ivan-vas", lang_pair="Russian-English"):
    comet = load_comet(film)
    wf    = load_wholefile(film, lang_pair)
    meme  = load_meme_mqm(film, lang_pair)

    methods = [m for m in METHOD_ORDER if m in comet or m in wf or m in meme]

    best_comet = max((comet[m] for m in methods if m in comet), default=None)
    best_wf    = min((wf[m]    for m in methods if m in wf),    default=None)
    best_meme  = min((meme[m]  for m in methods if m in meme),  default=None)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Method} & \textbf{COMET}$\uparrow$ & \textbf{LLM-WF}$\downarrow$ & \textbf{Meme-MQM}$\downarrow$ \\",
        r"\midrule",
    ]

    for m in methods:
        if m == "given":
            lines.append(r"\midrule")

        c  = fmt(comet[m]) if m in comet else "--"
        wv = fmt(wf[m])    if m in wf    else "--"
        mv = fmt(meme[m])  if m in meme  else "--"

        if m in comet and comet[m] == best_comet:
            c = bold(c)
        if m in wf and wf[m] == best_wf:
            wv = bold(wv)
        if m in meme and meme[m] == best_meme:
            mv = bold(mv)

        label = METHOD_DISPLAY.get(m, m)
        lines.append(f"{label} & {c} & {wv} & {mv} \\\\")

    film_label = FILM_DISPLAY.get(film, film)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Comparison of COMET$\uparrow$, LLM whole-text MQM (LLM-WF$\downarrow$), and meme-focused MQM (Meme-MQM$\downarrow$) on "
        + film_label
        + r" (Russian$\to$English). Best value per column in \textbf{bold}. \textit{Given trans.}\ uses approved reference translations as prompts.}",
        r"\label{tab:metrics-comparison-ivan-vas}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Table B ───────────────────────────────────────────────────────────────────

def table_b():
    films = ["ivan-vas", "diamond-arm", "pokrov-gate", "carnival-night", "gentelmen"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Film} & \textbf{COMET rank} & \textbf{LLM-WF rank} & \textbf{Meme-MQM rank} \\",
        r"\midrule",
    ]

    for film in films:
        comet = load_comet(film)
        wf    = load_wholefile(film)
        meme  = load_meme_mqm(film)

        cr, cn = rank_of(comet, "given", higher_better=True)
        wr, wn = rank_of(wf,    "given", higher_better=False)
        mr, mn = rank_of(meme,  "given", higher_better=False)

        def rs(r, n):
            if r is None:
                return "--"
            s = f"{r}/{n}"
            return bold(s) if r == 1 else s

        label = FILM_DISPLAY.get(film, film)
        lines.append(f"{label} & {rs(cr,cn)} & {rs(wr,wn)} & {rs(mr,mn)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Rank of the \textit{given translations} method among all tested methods, per film. Rank 1 = best. COMET: higher = better; LLM-WF and Meme-MQM: lower error rate = better. \textbf{Bold} = best rank.}",
        r"\label{tab:given-ranks}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Table C: Spearman correlations ───────────────────────────────────────────

def spearman_r(xs, ys):
    if len(xs) < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(xs, ys)
    return r, p

def sig_marker(p):
    if p < 0.001: return r"$^{***}$"
    if p < 0.01:  return r"$^{**}$"
    if p < 0.05:  return r"$^{*}$"
    return ""

def table_c():
    films = ["ivan-vas", "diamond-arm", "pokrov-gate", "carnival-night", "gentelmen"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Film}} & \multicolumn{2}{c}{\textbf{COMET / Meme-MQM}} & \multicolumn{2}{c}{\textbf{LLM-WF / Meme-MQM}} & \multicolumn{2}{c}{\textbf{COMET / LLM-WF}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & $\rho$ & $p$ & $\rho$ & $p$ & $\rho$ & $p$ \\",
        r"\midrule",
    ]

    all_cm, all_wfm, all_cwf = [], [], []

    for film in films:
        comet = load_comet(film)
        wf    = load_wholefile(film)
        meme  = load_meme_mqm(film)
        common = sorted(set(comet) & set(wf) & set(meme))
        n = len(common)

        c_v  = [comet[m] for m in common]
        wf_v = [wf[m]    for m in common]
        m_v  = [meme[m]  for m in common]

        r_cm,  p_cm  = spearman_r(c_v,  m_v)
        r_wfm, p_wfm = spearman_r(wf_v, m_v)
        r_cwf, p_cwf = spearman_r(c_v,  wf_v)

        def cell(r, p):
            if r != r:  # nan
                return "-- & --"
            return f"{r:+.2f}{sig_marker(p)} & {p:.3f}"

        label = FILM_DISPLAY.get(film, film) + f" ($N$={n})"
        lines.append(f"{label} & {cell(r_cm,p_cm)} & {cell(r_wfm,p_wfm)} & {cell(r_cwf,p_cwf)} \\\\")

        all_cm  += list(zip(c_v,  m_v))
        all_wfm += list(zip(wf_v, m_v))
        all_cwf += list(zip(c_v,  wf_v))

    lines.append(r"\midrule")
    n_pool = len(all_cm)
    r_cm,  p_cm  = spearman_r([x for x,_ in all_cm],  [y for _,y in all_cm])
    r_wfm, p_wfm = spearman_r([x for x,_ in all_wfm], [y for _,y in all_wfm])
    r_cwf, p_cwf = spearman_r([x for x,_ in all_cwf], [y for _,y in all_cwf])
    lines.append(rf"Pooled ($N$={n_pool}) & {cell(r_cm,p_cm)} & {cell(r_wfm,p_wfm)} & {cell(r_cwf,p_cwf)} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Spearman rank correlations between evaluation metrics across translation methods. COMET is higher-is-better; LLM-WF and Meme-MQM are lower-is-better. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}",
        r"\label{tab:correlations}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Table D: LLM-WF coverage + meta-evaluation ───────────────────────────────

def table_d():
    films = ["ivan-vas", "diamond-arm", "pokrov-gate", "carnival-night", "gentelmen"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Film} & \textbf{Items} & \textbf{Issues} & \textbf{Match\%} & \textbf{Recall} & \textbf{Harsh\%} & \textbf{Align\%} \\",
        r"\midrule",
    ]

    totals = dict(items=0, issues=0, matched=0, pairs=0, harsh=0, align=0)

    for film in films:
        sidecar = Path(f"experiments/films/output/eval/llm-eval/{film}/Russian-English"
                       "/gpt-5.2-by-gpt-5.4/mqm-wholefile/gpt-5.2-wholefile.json")
        comp = Path(f"experiments/films/output/eval/llm-eval/{film}/Russian-English"
                    "/gpt-5.2-by-gpt-5.4/mqm-wholefile/reference_comparison.json")

        if not sidecar.exists():
            continue

        data = json.loads(sidecar.read_text())
        items = data["items"]
        n_items = len(items)

        # total issues from raw run files
        wf_root = sidecar.parent
        total_issues = 0
        for md in sorted(wf_root.iterdir()):
            if not md.is_dir(): continue
            for p in md.glob("run_*_eval_*.json"):
                total_issues += len(json.loads(p.read_text()).get("issues", []))

        # matched issues and recall from sidecar
        n_matched = 0
        n_items_flagged = 0
        for item in items:
            wf = item.get("llm_wholefile", {}).get("eng", {})
            flagged = False
            for issues in wf.values():
                n_matched += len(issues)
                if issues: flagged = True
            if flagged: n_items_flagged += 1

        match_pct = 100 * n_matched / total_issues if total_issues else 0
        recall = 100 * n_items_flagged / n_items if n_items else 0

        # meta-evaluation
        n_pairs = harsh_yes = align_yes = 0
        if comp.exists():
            cdata = json.loads(comp.read_text())
            for it in cdata["items"]:
                for m in it["methods"]:
                    if not m["flagged"]: continue
                    n_pairs += 1
                    a = m.get("assessment") or {}
                    if a.get("harshness") == "yes": harsh_yes += 1
                    if a.get("alignment") == "yes": align_yes += 1

        harsh_pct = 100 * harsh_yes / n_pairs if n_pairs else 0
        align_pct = 100 * align_yes / n_pairs if n_pairs else 0

        label = FILM_DISPLAY.get(film, film)
        lines.append(
            f"{label} & {n_items} & {total_issues} & {match_pct:.0f}\\% & "
            f"{recall:.0f}\\% & {harsh_pct:.0f}\\% & {align_pct:.0f}\\% \\\\"
        )

        totals["items"]   += n_items
        totals["issues"]  += total_issues
        totals["matched"] += n_matched
        totals["pairs"]   += n_pairs
        totals["harsh"]   += harsh_yes
        totals["align"]   += align_yes

    lines.append(r"\midrule")
    t = totals
    mp = 100 * t["matched"] / t["issues"] if t["issues"] else 0
    hp = 100 * t["harsh"]   / t["pairs"]  if t["pairs"]  else 0
    ap = 100 * t["align"]   / t["pairs"]  if t["pairs"]  else 0
    lines.append(
        f"Total & {t['items']} & {t['issues']} & {mp:.0f}\\% & "
        f"-- & {hp:.0f}\\% & {ap:.0f}\\% \\\\"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{LLM whole-text judge behaviour on culturally significant passages. "
        r"\textbf{Issues}: total MQM issues raised across all methods and runs. "
        r"\textbf{Match\%}: fraction of issues whose source quote overlapped with a cultural item. "
        r"\textbf{Recall}: fraction of cultural items flagged in at least one method. "
        r"\textbf{Harsh\%}: fraction of flagged item$\times$method pairs rated unnecessarily harsh by meta-evaluator. "
        r"\textbf{Align\%}: fraction rated as well-aligned with the reference analysis.}",
        r"\label{tab:llm-wf-coverage}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ta = table_a()
    tb = table_b()
    tc = table_c()
    td = table_d()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ta + "\n\n" + tb + "\n\n" + tc + "\n\n" + td + "\n", encoding="utf-8")
    print(f"Written: {args.out}")
    for label, t in [("TABLE A", ta), ("TABLE B", tb), ("TABLE C", tc), ("TABLE D", td)]:
        print(f"\n--- {label} ---\n")
        print(t)


if __name__ == "__main__":
    main()
