#!/usr/bin/env python3
"""Export the paper's dataset, outputs and code into a clean release directory.

Only the five films and the translators used in the paper are exported. The
release includes the Russian source subtitles (from OpenSubtitles.org) and every
full-film translation, except superseded ones set aside in folders whose name
contains a dot (e.g. "list.pre-segfix"). sources.json records each source file's
OpenSubtitles ID, checksums, and the recipe that rebuilds the exact file from
the download; fetch_sources.py retrieves, rebuilds and verifies them. Human evaluation
sessions are exported without the evaluator profiles, with neutral evaluator
labels. Raw poll data (with voters' usernames) and evaluator notes are not exported.

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
# GPT-5.2 translations re-scored at the spans the human evaluator saw (diagnostic in the paper).
EXTRA_MODELS = ["gpt-5.2-humanspans"]
CODE = ["translate.py", "claude_cli.py", "lang_utils.py", "evaluate_mqm_parallel.py", "aggregate_mqm.py", "variance.py",
        "extract_meme_spans.py", "verify_spans.py", "evaluate_neural_metrics.py", "evaluate_neural_metrics_memes.py", "evaluate_wholefile_mqm.py",
        "difficulty_ranking.py", "meme_search_coverage.py", "compute_human_eval_summary.py", "paper_tables.py",
        "process_srt.py", "fetch_sources.py", "make_source_recipes.py"]
CONFIG_GLOBS = [f"{film}-*.yaml" for film in FILMS]
COVERAGE = "meme_search_coverage-by-claude-sonnet-5-5.json"
# Logs of the human-in-the-loop construction step (kept for Ivan Vasilievich).
HIL_LOGS = {"ivan-vas": {"HIL-chat.txt": "analysis-and-references-chat.txt", "Referencias HIL": "galician-notes.txt"}}
# Human evaluation: evaluator IDs as recorded by the evaluation tool -> labels in the release.
EVALUATORS = {"oz-6dd2": "evaluator-en", "test4-5d9b": "evaluator-en", "olga-50fb": "evaluator-en",
              "sienna-fcc8": "evaluator-gl"}








def clean_memes_json(path, film_dir):
    """Passage file with only the conditions that have translations (drops stale aliases such as meme-list)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if path.stem in EXTRA_MODELS:  # condition names follow the human evaluation sessions
        return json.dumps(d, ensure_ascii=False, indent=2)
    have = {m.name for m in film_dir.glob(f"Russian-*/{path.stem}/*") if (m / "translations").is_dir() and "." not in m.name}
    for it in d["items"]:
        for code, methods in it.get("translations", {}).items():
            it["translations"][code] = {m: r for m, r in methods.items() if m in have}
    return json.dumps(d, ensure_ascii=False, indent=2)


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

    for film in FILMS:
        d = films_root / "data" / film
        for name in DATA_FILES:
            if (d / name).exists():
                copy(d / name, out / "data" / film / name); n += 1
        for f in (d / "summaries").glob("*.txt"):
            copy(f, out / "data" / film / "summaries" / f.name); n += 1
        for name, dst in HIL_LOGS.get(film, {}).items():
            copy(d / name, out / "data" / film / "hil" / dst); n += 1
    sources = json.loads((films_root / "data" / "opensubtitles_sources.json").read_text(encoding="utf-8"))
    for s in sources:  # the recipes must still rebuild the files the dataset refers to
        sub = films_root / "data" / s["film"] / "subs" / s["file"]
        assert s["sha256"] == sha256(sub), s["film"]
        copy(sub, out / "data" / s["film"] / "subs" / s["file"]); n += 1
    (out / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"); n += 1

    # Meme-MQM: the release carries the prompt the paper reports (mqm-memes-v2.txt), as mqm-memes.txt.
    for f in (films_root / "prompts").rglob("*.txt"):
        rel = f.relative_to(films_root / "prompts")
        if rel.as_posix() == "eval/mqm-memes.txt":
            continue
        copy(f, out / "prompts" / Path(rel.as_posix().replace("mqm-memes-v2.txt", "mqm-memes.txt"))); n += 1

    tr = films_root / "output" / "translations"
    for film in FILMS:
        for model in MODELS + EXTRA_MODELS:
            mapped = tr / film / f"{model}.json"
            if mapped.exists():
                dst = out / "translations" / film / f"{model}.memes.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(clean_memes_json(mapped, tr / film), encoding="utf-8"); n += 1
            for pair in (tr / film).glob(f"Russian-*/{model}"):
                # Full-film translations; folders such as "list.pre-segfix" hold superseded ones.
                for f in pair.glob("*/translations/translation-*.*"):
                    if "." not in f.parent.parent.name:
                        copy(f, out / "translations" / film / f.relative_to(tr / film)); n += 1
                # The model's own meme analysis; its raw web-search results are not distributed.
                for f in pair.glob("meme-search/meme-analysis/meme-analysis.txt"):
                    copy(f, out / "translations" / film / f.relative_to(tr / film)); n += 1

    ev = films_root / "output" / "eval"
    for film in FILMS:
        for kind in ("mqm-memes-v2", "mqm-wholefile"):  # Meme-MQM: the scores the paper reports, as mqm-memes/
            for f in (ev / "llm-eval" / film).glob(f"Russian-*/*-by-*/{kind}/**/*.json"):
                if not any(f"/{m}-by-" in str(f) for m in MODELS + EXTRA_MODELS):
                    continue
                cond = f.relative_to(ev).parts[5] if len(f.relative_to(ev).parts) > 6 else None
                pair, translator = f.relative_to(ev).parts[2], f.relative_to(ev).parts[3].split("-by-")[0]
                if cond and translator in MODELS and ("." in cond or not (tr / film / pair / translator / cond / "translations").is_dir()):
                    continue  # superseded ("list.pre-segfix") or former condition names ("meme-list") with no released translations
                dst = out / "eval" / Path(f.relative_to(ev).as_posix().replace("/mqm-memes-v2/", "/mqm-memes/"))
                copy(f, dst); n += 1
        for f in [*(ev / "comet-eval" / film).glob("*/*/aligned_scores_*.json"),
                  *(ev / "comet-eval" / film).glob("*/*/meme_scores_*.json")]:  # whole film; meme passages only
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
    copy(Path("docs/release_LICENSE"), out / "LICENSE")
    copy(Path("docs/release_LICENSE-DATA.md"), out / "LICENSE-DATA.md")
    copy(Path("docs/release_zenodo.json"), out / ".zenodo.json")
    copy(Path("docs/release_CITATION.cff"), out / "CITATION.cff")
    for g in CONFIG_GLOBS:
        for f in Path("yaml-pipelines/films").glob(g):
            copy(f, out / "configs" / f.name); n += 1
    print(f"exported {n} files to {out}")


if __name__ == "__main__":
    main()
