#!/usr/bin/env python3
"""Whole-film neural QE metrics with segments aligned by subtitle entry.

Supersedes the pairing in evaluate_wholefile_comet.py, which sentence-tokenized
source and translation independently and zipped the two lists by position;
sentence counts differ across languages, so pairs drifted out of alignment
within the first few dozen sentences. Translations here preserve the source
SRT's entries and time codes, so each source entry is paired with the
translated entry carrying the same time code.

All metrics run reference-free (QE), since there is no full-film reference:
  cometkiwi  Unbabel/wmt22-cometkiwi-da        higher is better
  xcomet     Unbabel/XCOMET-XL (src+mt only)   higher is better
  metricx    google/metricx-24-hybrid-large-v2p6, QE mode  lower is better (0-25)

Usage:
    python code/evaluate_neural_metrics.py yaml-pipelines/films/ivan-vas-russian-english-0.1.yaml --metric xcomet
    python code/evaluate_neural_metrics.py ... --metric metricx --n-runs 3 --methods zero,given
Output: experiments/films/output/eval/comet-eval/<film>/<src>-<tgt>/<trans_model>/aligned_scores_<metric>.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pysrt
import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_wholefile_comet import load_config, translation_dir, source_srt_path

METRICS = {
    "cometkiwi": "Unbabel/wmt22-cometkiwi-da",
    "xcomet": "Unbabel/XCOMET-XL",
    "metricx": "google/metricx-24-hybrid-large-v2p6",
}
METRICX_TOKENIZER = "google/mt5-large"
METRICX_CODE = Path.home() / "src" / "metricx"  # git clone https://github.com/google-research/metricx


def entries(path):
    subs = pysrt.open(str(path), error_handling=pysrt.ERROR_PASS)
    return {(str(s.start), str(s.end)): (s.index, s.text.replace("\n", " ").strip()) for s in subs}


def aligned_pairs(src, tgt):
    """Source/translation pairs matched by time code, falling back to the entry number
    when a translation garbled the time codes but kept the numbering."""
    by_idx = {i: t for i, t in tgt.values()}
    pairs = []
    for k, (i, s) in src.items():
        t = tgt[k][1] if k in tgt else by_idx.get(i)
        if s and t:
            pairs.append((s, t))
    return pairs, len(src) - len(pairs)


def hf_login():
    from huggingface_hub import login
    login(token=Path("hf-token.txt").read_text().strip(), add_to_git_credential=False)


class CometScorer:
    def __init__(self, name):
        from comet import download_model, load_from_checkpoint
        self.model = load_from_checkpoint(download_model(name))
        self.gpus = 1 if torch.cuda.is_available() else 0
        self.batch_size = 16
        if "XCOMET" in name and self.gpus:
            # xCOMET-XL does not fit an 11 GB GPU in fp32.
            self.model = self.model.half()
            self.batch_size = 8

    def __call__(self, pairs):
        out = self.model.predict([{"src": s, "mt": t} for s, t in pairs], batch_size=self.batch_size, gpus=self.gpus)
        return list(out["scores"])


class MetricXScorer:
    def __init__(self, name, max_len=1536, batch_size=8):
        sys.path.insert(0, str(METRICX_CODE))
        import transformers
        from metricx24 import models
        self.tok = transformers.AutoTokenizer.from_pretrained(METRICX_TOKENIZER)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.MT5ForRegression.from_pretrained(name, torch_dtype=torch.float32).to(self.device).eval()
        self.max_len, self.batch_size = max_len, batch_size

    @torch.no_grad()
    def __call__(self, pairs):
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            texts = [f"source: {s} candidate: {t}" for s, t in pairs[i:i + self.batch_size]]
            enc = self.tok(texts, max_length=self.max_len, truncation=True, padding=True, return_tensors="pt")
            # MetricX is trained without the trailing EOS token.
            ids, mask = enc["input_ids"][:, :-1], enc["attention_mask"][:, :-1]
            out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
            scores.extend(out.predictions.float().cpu().tolist())
        return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to yaml pipeline config")
    parser.add_argument("--metric", choices=METRICS, required=True)
    parser.add_argument("--n-runs", type=int, default=3, help="Translation runs per method to score (default: 3)")
    parser.add_argument("--methods", default=None, help="Comma-separated method names (default: all in yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hf_login()
    model_name = METRICS[args.metric]
    print(f"Loading {model_name} ...")
    scorer = MetricXScorer(model_name) if args.metric == "metricx" else CometScorer(model_name)

    src = entries(source_srt_path(cfg))
    method_filter = set(args.methods.split(",")) if args.methods else None
    results = {}
    for method in cfg["methods"]:
        name = method["name"]
        if method_filter and name not in method_filter:
            continue
        run_files = sorted(translation_dir(cfg, name).glob("translation-*.txt"),
                           key=lambda p: int(p.stem.split("-")[1]))[: args.n_runs]
        if not run_files:
            print(f"  [{name}] no translation files, skipping")
            continue
        run_scores, n_pairs, n_unmatched = [], [], []
        for rf in run_files:
            pairs, missing = aligned_pairs(src, entries(rf))
            seg = scorer(pairs)
            run_scores.append(sum(seg) / len(seg))
            n_pairs.append(len(pairs))
            n_unmatched.append(missing)
            print(f"  [{name}] {rf.name}: {len(pairs)} pairs ({missing} unmatched), score={run_scores[-1]:.4f}")
        mean = sum(run_scores) / len(run_scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in run_scores) / len(run_scores)) if len(run_scores) > 1 else 0.0
        results[name] = {"run_scores": run_scores, "mean_score": round(mean, 6), "std_score": round(std, 6),
                         "n_pairs_per_run": n_pairs, "n_unmatched_per_run": n_unmatched}

    out_dir = (Path("experiments/films/output/eval/comet-eval") / cfg["film"]
               / f"{cfg['source_lang']}-{cfg['target_lang']}" / cfg["trans_model"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"aligned_scores_{args.metric}.json"
    out_path.write_text(json.dumps({
        "film": cfg["film"], "source_lang": cfg["source_lang"], "target_lang": cfg["target_lang"],
        "trans_model": cfg["trans_model"], "metric": args.metric, "model": model_name,
        "higher_is_better": args.metric != "metricx", "segment": "subtitle entry (time-code aligned)",
        "n_runs_per_method": args.n_runs, "methods": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
