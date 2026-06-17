#!/usr/bin/env python3
"""Build JSON segment maps from subtitle files for the web evaluation tool.

Run from the repo root. Must be run before launching the web tool if you want
the "Show context" buttons to work.

Russian source subtitles — pass the .srt file:
    python code/build_subs_json.py experiments/films/data/ivan-vas/subs/ivan-vas-rus.srt
    Writes: web/data/<film>/subs.json

Translated subtitles — pass the model directory:
    python code/build_subs_json.py experiments/films/output/translations/carnival-night/Russian-English/gpt-5.2
    Scans:  <model_dir>/<method>/translations/translation-1.{srt,txt}
    Writes: web/data/<film>/gpt-5.2-<method>-subs.json  (one file per method)
    Film name is inferred as the grandparent of the lang directory (e.g. carnival-night).

    For older layouts without a lang sub-directory, you can also pass a JSON file
    that sits alongside the model directory:
    python code/build_subs_json.py experiments/films/output/translations/ivan-vas/gpt-5.2.json
    Film name is inferred as the parent directory of the JSON file.

Output format: {"1": "text", "2": "text", ...}  (segment number -> text)
"""

import argparse
import json
import sys
from pathlib import Path

import pysrt

SUPPORTED_EXTENSIONS = {".srt", ".txt"}


def parse_srt_file(path: Path) -> dict:
    """Parse an SRT or SRT-like TXT file and return {index: text}."""
    raw = path.read_text(encoding="utf-8-sig")

    # Some .txt exports prepend a bare "srt" line — strip it.
    lines = raw.splitlines()
    if lines and lines[0].strip().lower() == "srt":
        raw = "\n".join(lines[1:])

    subs = pysrt.SubRipFile.from_string(raw)
    result = {}
    for sub in subs:
        text = sub.text.replace("\r\n", "\n").replace("\r", "\n").strip()
        result[sub.index] = text
    return result


def write_json(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_source_subs(srt_path: Path) -> None:
    # Infer film name: .../experiments/films/data/<film>/subs/<file>.srt
    film = srt_path.parts[-3]  # grandparent of the subs/ dir

    print(f"Reading: {srt_path}")
    result = parse_srt_file(srt_path)

    out_path = Path("web/data") / film / "subs.json"
    write_json(result, out_path)
    print(f"Written {len(result)} segments -> {out_path}")


def scan_method_dirs(base: Path, film: str, model: str) -> None:
    method_dirs = sorted(d for d in base.iterdir() if d.is_dir())
    if not method_dirs:
        print(f"Error: no method subdirectories found in {base}", file=sys.stderr)
        sys.exit(1)

    written = 0
    for method_dir in method_dirs:
        trans_subdir = method_dir / "translations"
        if not trans_subdir.is_dir():
            continue

        # Use the first run (lowest-numbered file) as the context source
        candidates = sorted(
            p for p in trans_subdir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not candidates:
            continue

        run1 = candidates[0]
        print(f"Reading: {run1}")
        try:
            result = parse_srt_file(run1)
        except Exception as e:
            print(f"  Warning: could not parse {run1}: {e}", file=sys.stderr)
            continue

        out_path = Path("web/data") / film / f"{model}-{method_dir.name}-subs.json"
        write_json(result, out_path)
        print(f"Written {len(result)} segments -> {out_path}")
        written += 1

    if written == 0:
        print(f"Warning: no translation files found under {base} — check that method subdirectories contain a translations/ folder with .srt or .txt files.", file=sys.stderr)


def build_translation_subs(json_path: Path) -> None:
    # Infer film and model from path: .../translations/<film>/<model>.json
    model = json_path.stem
    film  = json_path.parent.name

    base = json_path.parent / model
    if not base.is_dir():
        print(f"Error: translations directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    scan_method_dirs(base, film, model)


def build_translation_subs_from_dir(model_dir: Path) -> None:
    # Layout: .../translations/<film>/<lang>/<model>/
    model = model_dir.name
    film  = model_dir.parent.parent.name  # skip the lang sub-directory

    scan_method_dirs(model_dir, film, model)


def main():
    parser = argparse.ArgumentParser(
        description="Build JSON segment maps from subtitle files for the web evaluation tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Path to a source .srt file, a translations .json file, or a model directory",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        build_translation_subs_from_dir(path)
    elif not path.is_file():
        print(f"Error: path not found: {path}", file=sys.stderr)
        sys.exit(1)
    elif path.suffix.lower() == ".json":
        build_translation_subs(path)
    elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
        build_source_subs(path)
    else:
        print(f"Error: unsupported file type: {path.suffix}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
