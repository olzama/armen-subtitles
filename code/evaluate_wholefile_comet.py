#!/usr/bin/env python3
"""Evaluate translations with COMET-KIWI (reference-free quality estimation).

Reads a yaml pipeline config, picks N runs per method, sentence-tokenizes
source and translation texts with Stanza, then scores segment pairs with
COMET-KIWI and reports a per-method mean.

Usage:
    python code/evaluate_wholefile_comet.py yaml-pipelines/films/carnival-night-russian-english-0.03.yaml
    python code/evaluate_wholefile_comet.py yaml-pipelines/... --n-runs 2 --comet-model Unbabel/wmt23-cometkiwi-da-xl
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pysrt
import stanza
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from lang_utils import lang_code

# Stanza uses ISO 639-1 two-letter codes
STANZA_LANG = {
    "rus": "ru",
    "eng": "en",
    "glg": "gl",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def srt_to_plain(path):
    """Parse any SRT-format file and return concatenated plain text."""
    subs = pysrt.open(str(path))
    return " ".join(sub.text.replace("\n", " ").strip() for sub in subs)


def sentence_tokenize(text, nlp):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sentences if s.text.strip()]


def translation_dir(cfg, method_name):
    src, tgt = cfg["source_lang"], cfg["target_lang"]
    return (
        Path("experiments/films/output/translations")
        / cfg["film"]
        / f"{src}-{tgt}"
        / cfg["trans_model"]
        / method_name
        / "translations"
    )


def source_srt_path(cfg):
    film = cfg["film"]
    lc = lang_code(cfg["source_lang"])
    return Path("experiments/films/data") / film / "subs" / f"{film}-{lc}.srt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to yaml pipeline config")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of translation runs per method to score (default: 3)")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-cometkiwi-da",
                        help="COMET model name (default: wmt22-cometkiwi-da)")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated list of method names to evaluate (default: all in yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    film = cfg["film"]
    src_lang = cfg["source_lang"]
    tgt_lang = cfg["target_lang"]
    trans_model = cfg["trans_model"]

    src_code = lang_code(src_lang)
    tgt_code = lang_code(tgt_lang)
    src_stanza = STANZA_LANG.get(src_code)
    tgt_stanza = STANZA_LANG.get(tgt_code)
    if not src_stanza:
        sys.exit(f"No Stanza code configured for source language '{src_lang}' ({src_code})")
    if not tgt_stanza:
        sys.exit(f"No Stanza code configured for target language '{tgt_lang}' ({tgt_code})")

    print(f"Loading Stanza ({src_stanza}, {tgt_stanza}) ...")
    nlp_src = stanza.Pipeline(src_stanza, processors="tokenize", verbose=False)
    nlp_tgt = (nlp_src if src_stanza == tgt_stanza
               else stanza.Pipeline(tgt_stanza, processors="tokenize", verbose=False))

    print(f"Loading COMET model: {args.comet_model} ...")
    from comet import download_model, load_from_checkpoint
    with open("hf-token.txt") as f:
        hf_token = f.read().strip()
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    model_path = download_model(args.comet_model)
    comet_model = load_from_checkpoint(model_path)

    gpus = 0
    if torch.cuda.is_available():
        try:
            torch.cuda.get_device_properties(0)
            gpus = 1
        except RuntimeError:
            pass
    print(f"Using {'GPU' if gpus else 'CPU'} for COMET inference")

    src_srt = source_srt_path(cfg)
    if not src_srt.exists():
        sys.exit(f"Source SRT not found: {src_srt}")
    src_text = srt_to_plain(src_srt)
    src_sentences = sentence_tokenize(src_text, nlp_src)
    print(f"Source: {len(src_sentences)} sentences  ({src_srt.name})")

    method_filter = set(args.methods.split(",")) if args.methods else None

    method_results = {}

    for method in cfg["methods"]:
        name = method["name"]
        if method_filter and name not in method_filter:
            continue
        trans_d = translation_dir(cfg, name)
        if not trans_d.exists():
            print(f"  [{name}] directory not found — skipping")
            method_results[name] = {"error": "translations directory not found"}
            continue

        run_files = sorted(trans_d.glob("translation-*.txt"))[: args.n_runs]
        if not run_files:
            print(f"  [{name}] no translation files — skipping")
            method_results[name] = {"error": "no translation files found"}
            continue

        run_scores = []
        n_pairs_per_run = []

        for rf in run_files:
            tgt_text = srt_to_plain(rf)
            tgt_sentences = sentence_tokenize(tgt_text, nlp_tgt)
            pairs = list(zip(src_sentences, tgt_sentences))
            comet_input = [{"src": s, "mt": t} for s, t in pairs]
            output = comet_model.predict(comet_input, batch_size=64, gpus=gpus)
            score = float(output["system_score"])
            run_scores.append(score)
            n_pairs_per_run.append(len(pairs))
            print(f"  [{name}] {rf.name}: {len(pairs)} pairs, score={score:.4f}")

        mean = sum(run_scores) / len(run_scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in run_scores) / len(run_scores)) if len(run_scores) > 1 else 0.0
        method_results[name] = {
            "run_scores": run_scores,
            "mean_score": round(mean, 6),
            "std_score": round(std, 6),
            "n_pairs_per_run": n_pairs_per_run,
        }

    out_dir = (
        Path("experiments/films/output/eval/comet-eval")
        / film
        / f"{src_lang}-{tgt_lang}"
        / trans_model
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.comet_model.split("/")[-1]
    out_path = out_dir / f"comet_scores_{model_slug}.json"
    out_path.write_text(
        json.dumps(
            {
                "film": film,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "trans_model": trans_model,
                "comet_model": args.comet_model,
                "n_runs_per_method": args.n_runs,
                "methods": method_results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
