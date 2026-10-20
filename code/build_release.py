#!/usr/bin/env python3
"""Export the paper's dataset, outputs and code into a clean release directory.

Only the five films and the translators used in the paper are exported. Source
subtitles are not copied: sources.csv records each file's checksum and entry
count, and fetch_sources.py retrieves and verifies them. Human evaluation
sessions are exported without the evaluator profiles, with neutral evaluator
labels. Files that need a manual decision before release (chat logs, poll data
with usernames, evaluator notes) are not exported.

Usage:
    python code/build_release.py [--out /mnt/kesha/soviet-meme-mt]
"""

import argparse
import csv
import re
import hashlib
import json
import shutil
from pathlib import Path

FILMS = ["ivan-vas", "diamond-arm", "gentelmen", "carnival-night", "pokrov-gate"]
MODELS = ["gpt-6-sol", "claude-opus-5-5", "gpt-5.2", "gpt-5.4-mini"]
DATA_FILES = ["reference.json", "meme-list.json", "list-analysis-all.json", "list-analysis-general-all.json"]
# GPT-5.2 translations re-scored at the spans the human evaluator saw (diagnostic in the paper).
EXTRA_MODELS = ["gpt-5.2-humanspans"]
CODE = ["translate.py", "claude_cli.py", "lang_utils.py", "evaluate_mqm_parallel.py", "aggregate_mqm.py", "variance.py",
        "extract_meme_spans.py", "verify_spans.py", "evaluate_neural_metrics.py", "evaluate_wholefile_mqm.py",
        "difficulty_ranking.py", "meme_search_coverage.py", "compute_human_eval_summary.py", "paper_tables.py",
        "process_srt.py", "fetch_sources.py"]
CONFIG_GLOBS = [f"{film}-*.yaml" for film in FILMS]
COVERAGE = "meme_search_coverage-by-claude-sonnet-5-5.json"
# Human evaluation: evaluator IDs as recorded by the evaluation tool -> labels in the release.
EVALUATORS = {"oz-6dd2": "evaluator-en", "test4-5d9b": "evaluator-en", "olga-50fb": "evaluator-en",
              "sienna-fcc8": "evaluator-gl"}


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def anonymized(path):
    """A human evaluation file without the evaluator profile, with neutral evaluator labels."""
    def strip(x):  # evaluator profiles appear as "evaluator_meta" in sessions and "meta" in summaries
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("evaluator_meta", "meta")}
        return [strip(v) for v in x] if isinstance(x, list) else x
    d = strip(json.loads(path.read_text(encoding="utf-8")))
    text = json.dumps(d, ensure_ascii=False, indent=2)
    for k, v in EVALUATORS.items():
        text = text.replace(k, v)
    return text


def human_eval_files(ev):
    """Summaries used in the paper, and the session files of the evaluators they were built from."""
    for s in sorted((ev / "human-eval").rglob("summary_human_*.json")):
        if "backup" in s.parts:
            continue
        yield s
        ids = set(json.loads(s.read_text(encoding="utf-8")).get("evaluators", []))
        for f in s.parent.glob("*.json"):
            if f.name.startswith(("human_", "session_")):
                d = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(d, dict) and d.get("evaluator_id") in ids:
                    yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/mnt/kesha/soviet-meme-mt"))
    a = ap.parse_args()
    out, films_root = a.out, Path("experiments/films")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; export into a new directory")
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
        for model in MODELS + EXTRA_MODELS:
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
        for kind in ("mqm-memes", "mqm-wholefile"):
            for f in (ev / "llm-eval" / film).glob(f"Russian-*/*-by-*/{kind}/**/*.json"):
                if any(f"/{m}-by-" in str(f) for m in MODELS + EXTRA_MODELS):
                    copy(f, out / "eval" / f.relative_to(ev)); n += 1
        for f in (ev / "comet-eval" / film).glob("*/*/aligned_scores_*.json"):
            copy(f, out / "eval" / f.relative_to(ev)); n += 1
        for f in (ev / "difficulty" / film).glob("*.json"):
            copy(f, out / "eval" / f.relative_to(ev)); n += 1
    copy(ev / "difficulty" / COVERAGE, out / "eval" / "difficulty" / COVERAGE); n += 1
    seen = set()
    for f in human_eval_files(ev):
        if f in seen:
            continue
        seen.add(f)
        dst = out / "eval" / f.relative_to(ev)
        for k, v in EVALUATORS.items():
            dst = dst.with_name(dst.name.replace(k, v))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(anonymized(f), encoding="utf-8"); n += 1

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
