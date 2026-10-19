#!/usr/bin/env python3
"""Export the paper's dataset, outputs and code into a clean release directory.

Only the five films and the translators used in the paper are exported. Source
subtitles are not copied: sources.csv records each file's checksum and entry
count, and fetch_sources.py retrieves and verifies them. Files that need a
manual decision before release (chat logs, poll data with usernames) are not
exported.

Usage:
    python code/build_release.py [--out /mnt/kesha/soviet-meme-mt]
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

FILMS = ["ivan-vas", "diamond-arm", "gentelmen", "carnival-night", "pokrov-gate"]
MODELS = ["gpt-6-sol", "claude-opus-5-5", "gpt-5.2", "gpt-5.4-mini"]
DATA_FILES = ["reference.json", "meme-list.json", "list-analysis-all.json", "list-analysis-general-all.json"]
CODE = ["translate.py", "claude_cli.py", "lang_utils.py", "evaluate_mqm_parallel.py", "aggregate_mqm.py", "variance.py",
        "extract_meme_spans.py", "evaluate_neural_metrics.py", "evaluate_wholefile_mqm.py", "difficulty_ranking.py",
        "meme_search_coverage.py", "paper_tables.py", "process_srt.py", "fetch_sources.py"]
CONFIG_GLOBS = ["*-gpt6sol.yaml", "*-respan.yaml", "*-opus-check.yaml"]


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/mnt/kesha/soviet-meme-mt"))
    a = ap.parse_args()
    out, films_root = a.out, Path("experiments/films")
    n = 0

    rows = []
    for film in FILMS:
        d = films_root / "data" / film
        for name in DATA_FILES:
            if (d / name).exists():
                copy(d / name, out / "data" / film / name); n += 1
        for f in (d / "summaries").glob("*.txt"):
            copy(f, out / "data" / film / "summaries" / f.name); n += 1
        srt = d / "subs" / f"{film}-rus.srt"
        rows.append({"film": film, "file": srt.name, "sha256": sha256(srt),
                     "entries": srt.read_text(encoding="utf-8-sig").count("-->"), "opensubtitles_id": ""})
    with open(out / "sources.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    for f in (films_root / "prompts").rglob("*.txt"):
        copy(f, out / "prompts" / f.relative_to(films_root / "prompts")); n += 1

    tr = films_root / "output" / "translations"
    for film in FILMS:
        for model in MODELS:
            mapped = tr / film / f"{model}.json"
            if mapped.exists():
                copy(mapped, out / "translations" / film / f"{model}.memes.json"); n += 1
            for pair in (tr / film).glob(f"Russian-*/{model}"):
                for f in pair.glob("*/translations/translation-*.txt"):
                    if "." in f.parent.parent.name:  # set-aside folders such as list.pre-segfix
                        continue
                    rel = f.relative_to(tr / film)
                    copy(f, (out / "translations" / film / rel).with_suffix(".srt")); n += 1
                for f in pair.glob("meme-search/meme-analysis/*.txt"):
                    copy(f, out / "translations" / film / f.relative_to(tr / film)); n += 1

    ev = films_root / "output" / "eval"
    for film in FILMS:
        for f in (ev / "llm-eval" / film).glob("Russian-*/*-by-*/mqm-memes/**/*.json"):
            if any(f"/{m}-by-" in str(f) for m in MODELS):
                copy(f, out / "eval" / f.relative_to(ev)); n += 1
        for f in (ev / "comet-eval" / film).glob("*/*/aligned_scores_*.json"):
            copy(f, out / "eval" / f.relative_to(ev)); n += 1
        for f in (ev / "difficulty" / film).glob("*.json"):
            copy(f, out / "eval" / f.relative_to(ev)); n += 1
    if (ev / "difficulty" / "meme_search_coverage.json").exists():
        copy(ev / "difficulty" / "meme_search_coverage.json", out / "eval" / "difficulty" / "meme_search_coverage.json")

    for name in CODE:
        if Path("code", name).exists():
            copy(Path("code", name), out / "code" / name); n += 1
    copy(Path("run_pipeline.py"), out / "run_pipeline.py")
    copy(Path("requirements.txt"), out / "requirements.txt")
    copy(Path("docs/release_README.md"), out / "README.md")
    for g in CONFIG_GLOBS:
        for f in Path("yaml-pipelines/films").glob(g):
            copy(f, out / "configs" / f.name); n += 1
    print(f"exported {n} files to {out}")


if __name__ == "__main__":
    main()
