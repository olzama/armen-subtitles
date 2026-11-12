"""Reference-free neural QE metrics on the meme passages alone.

Scores each (meme source, rendering span) pair of the Meme-MQM evaluation files with the same models
as evaluate_neural_metrics.py (whole subtitle file), so that the metrics see exactly the units the
human and the Meme-MQM evaluators judged.

  cometkiwi  Unbabel/wmt22-cometkiwi-da        higher is better
  xcomet     Unbabel/XCOMET-XL (src+mt only)   higher is better
  metricx    google/metricx-24-hybrid-large-v2p6, QE mode  lower is better (0-25)

Usage:
    python code/evaluate_neural_metrics_memes.py --metric cometkiwi
    python code/evaluate_neural_metrics_memes.py --metric metricx --films ivan-vas --translators gpt-6-sol

Input:  experiments/films/output/eval/llm-eval/<film>/Russian-<tgt>/<translator>-by-claude-sonnet-5-5/mqm-memes-v2/<method>/run_<r>_eval_1.json
Output: experiments/films/output/eval/comet-eval/<film>/Russian-<tgt>/<translator>/meme_scores_<metric>.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_neural_metrics import METRICS, CometScorer, MetricXScorer, hf_login

EVAL = Path("experiments/films/output/eval")
JUDGE = "claude-sonnet-5-5"
FILMS = ["ivan-vas", "diamond-arm", "gentelmen", "carnival-night", "pokrov-gate"]
TRANSLATORS = ["gpt-6-sol", "claude-opus-5-5", "gpt-5.2"]
EXTRA = {"gpt-5.2": [("ivan-vas", "Galician")]}  # language pairs beyond Russian-English


def clean(text):
    return " ".join((text or "").split())


def jobs(films, translators):
    for model in translators:
        pairs = [(f, "English") for f in films] + [p for p in EXTRA.get(model, []) if p[0] in films]
        for film, tgt in pairs:
            d = EVAL / "llm-eval" / film / f"Russian-{tgt}" / f"{model}-by-{JUDGE}" / "mqm-memes-v2"
            if d.exists():
                yield film, tgt, model, d


def collect(d):
    """{method: {run: [(meme id, source, candidate)]}}, from eval_1 of each run (eval_2 holds the same spans);
    falls back to eval_2 when a run has only that."""
    out = {}
    for mdir in sorted(p for p in d.iterdir() if p.is_dir()):
        runs = {}
        for f in sorted(mdir.glob("run_*_eval_*.json")):
            r = f.name.split("_")[1]
            if r in runs and not f.name.endswith("_eval_1.json"):
                continue
            items = json.loads(f.read_text())["items"]
            runs[r] = [(str(it["id"]), clean(it.get("source")), clean(it.get("candidate"))) for it in items]
            ids = [i for i, _, _ in runs[r]]
            assert len(ids) == len(set(ids)), f"repeated meme ids in {f}"
        if runs:
            out[mdir.name] = runs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, choices=list(METRICS))
    ap.add_argument("--films", default=",".join(FILMS))
    ap.add_argument("--translators", default=",".join(TRANSLATORS))
    ap.add_argument("--force", action="store_true", help="rescore files that already exist")
    args = ap.parse_args()

    todo = []
    for film, tgt, model, d in jobs(args.films.split(","), args.translators.split(",")):
        out = EVAL / "comet-eval" / film / f"Russian-{tgt}" / model / f"meme_scores_{args.metric}.json"
        if out.exists() and not args.force:
            print(f"skip (exists): {out}")
            continue
        todo.append((film, tgt, model, collect(d), out))
    if not todo:
        return

    hf_login()
    name = METRICS[args.metric]
    scorer = MetricXScorer(name) if args.metric == "metricx" else CometScorer(name)

    for film, tgt, model, data, out in todo:
        # Score every non-empty pair of this translator and film in one call.
        flat = [(m, r, i, s, c) for m, runs in data.items() for r, items in runs.items() for i, s, c in items]
        keep = [x for x in flat if x[3] and x[4]]
        scores = scorer([(s, c) for _, _, _, s, c in keep]) if keep else []
        methods = {}
        for m, runs in data.items():
            methods[m] = {"runs": {}}
            for r, items in sorted(runs.items(), key=lambda kv: int(kv[0])):
                methods[m]["runs"][r] = {"mean": None, "n_items": 0, "n_skipped": sum(1 for _, s, c in items if not (s and c)), "items": {}}
        for (m, r, i, _, _), v in zip(keep, scores):
            methods[m]["runs"][r]["items"][i] = float(v)
        for m in methods.values():
            for x in m["runs"].values():
                vals = list(x["items"].values())
                x["n_items"] = len(vals)
                x["mean"] = sum(vals) / len(vals) if vals else None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "film": film, "pair": f"Russian-{tgt}", "trans_model": model, "metric": args.metric, "model": name,
            "higher_is_better": args.metric != "metricx", "unit": "meme span",
            "methods": methods}, ensure_ascii=False, indent=1))
        skipped = sum(x["n_skipped"] for m in methods.values() for x in m["runs"].values())
        print(f"{film} {tgt} {model}: {len(keep)} scored, {skipped} skipped -> {out}")


if __name__ == "__main__":
    main()
