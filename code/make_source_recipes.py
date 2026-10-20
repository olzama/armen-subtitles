#!/usr/bin/env python3
"""Record how each local Russian source subtitle file derives from its OpenSubtitles original.

For each film, compares the downloaded OpenSubtitles file with the local file
the dataset's segment numbers refer to, and writes the recipe that
fetch_sources.py applies: byte-order mark, line endings, entry-number offset,
and the remaining line edits. Each recipe is checked to rebuild the local file
byte for byte.

Usage:
    python code/make_source_recipes.py <download_dir> [--out experiments/films/data/opensubtitles_sources.json]
<download_dir> holds the downloaded files as <film>-<file_id>.srt.
"""

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_sources import apply_recipe, lines

# OpenSubtitles file IDs of the files the local copies derive from (matched by content, 2026-10-20).
SOURCES = {
    "ivan-vas": {"imdb_id": "tt0070233", "opensubtitles_file_id": 5113169},
    "diamond-arm": {"imdb_id": "tt0062759", "opensubtitles_file_id": 641602},
    "gentelmen": {"imdb_id": "tt0068519", "opensubtitles_file_id": 5061259},
    "carnival-night": {"imdb_id": "tt0049397", "opensubtitles_file_id": 10562978},
    "pokrov-gate": {"imdb_id": "tt0083465", "opensubtitles_file_id": 4542393},
}


def recipe_for(raw, local):
    first = lambda ls: next(int(l) for k, l in enumerate(ls) if re.fullmatch(r"\d+", l) and "-->" in ls[k + 1])
    off = first(lines(local)) - first(lines(raw))
    recipe = {"bom": local.startswith(b"\xef\xbb\xbf"), "crlf": b"\r\n" in local, "renumber_offset": off, "line_edits": []}
    base = lines(apply_recipe(raw, {**recipe, "bom": False, "crlf": False}))
    target = lines(local)
    recipe["line_edits"] = [[i1, i2, target[j1:j2]] for op, i1, i2, j1, j2
                            in difflib.SequenceMatcher(None, base, target, autojunk=False).get_opcodes() if op != "equal"]
    return recipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("download_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("experiments/films/data/opensubtitles_sources.json"))
    a = ap.parse_args()
    out = []
    for film, meta in SOURCES.items():
        local_path = Path(f"experiments/films/data/{film}/subs/{film}-rus.srt")
        local = local_path.read_bytes()
        raw = (a.download_dir / f"{film}-{meta['opensubtitles_file_id']}.srt").read_bytes()
        recipe = recipe_for(raw, local)
        assert apply_recipe(raw, recipe) == local, film
        out.append({"film": film, "file": local_path.name, **meta,
                    "download_sha256": hashlib.sha256(raw).hexdigest(), "sha256": hashlib.sha256(local).hexdigest(),
                    "entries": local.count(b"-->"), "recipe": recipe})
        edits = sum(i2 - i1 + len(new) for i1, i2, new in recipe["line_edits"])
        print(f"{film}: file {meta['opensubtitles_file_id']}, offset {recipe['renumber_offset']}, "
              f"{len(recipe['line_edits'])} edits ({edits} lines), rebuilds the local file exactly")
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
