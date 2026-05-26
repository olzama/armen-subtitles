#!/usr/bin/env python3
"""
Spellcheck Galician subtitle translations using hunspell (gl_ES).
Outputs flagged words with frequency, provenance, and context.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import hunspell

ROOT = Path(__file__).parent.parent
TRANSLATIONS_ROOT = ROOT / "films/output/translations/ivan-vas/Russian-Galician/gpt-5.2"
DIC = "/usr/share/hunspell/gl_ES.dic"
AFF = "/usr/share/hunspell/gl_ES.aff"
OUTPUT_FILE = ROOT / "films/output/spellcheck_galician.json"

# SRT subtitle block: index, timestamp, text lines
SRT_BLOCK_RE = re.compile(
    r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)",
    re.DOTALL,
)


def parse_srt(text):
    """Yield (subtitle_index, timestamp, subtitle_text) tuples."""
    for m in SRT_BLOCK_RE.finditer(text.replace("\r\n", "\n").strip()):
        yield int(m.group(1)), m.group(2), m.group(3).strip()


def tokenize(text):
    """Extract lowercase alphabetic tokens, keeping accented chars."""
    return re.findall(r"[a-záéíóúàèìòùâêîôûãõüïñç]+", text.lower())


def hunspell_check(words, hs):
    """Return the subset of words that hunspell does not recognise."""
    return {w for w in words if not hs.spell(w)}


def collect_texts():
    """
    Walk every method/translations/ directory and collect subtitle data.
    Returns list of dicts: {method, run, subtitle_idx, timestamp, text, tokens}
    """
    records = []
    for method_dir in sorted(TRANSLATIONS_ROOT.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "mapping":
            continue
        trans_dir = method_dir / "translations"
        if not trans_dir.exists():
            continue
        for srt_file in sorted(trans_dir.glob("translation-*.txt")):
            run = int(re.search(r"translation-(\d+)", srt_file.name).group(1))
            raw = srt_file.read_text(encoding="utf-8-sig")
            for idx, ts, text in parse_srt(raw):
                records.append(
                    {
                        "method": method_dir.name,
                        "run": run,
                        "subtitle_idx": idx,
                        "timestamp": ts,
                        "text": text,
                        "tokens": tokenize(text),
                    }
                )
    return records


def main():
    print("Collecting subtitle texts...", file=sys.stderr)
    records = collect_texts()
    print(f"  {len(records)} subtitle blocks loaded.", file=sys.stderr)

    # Gather all unique words to spellcheck in one hunspell call
    all_words = set(tok for r in records for tok in r["tokens"])
    print(f"  {len(all_words)} unique tokens to check.", file=sys.stderr)

    print("Running hunspell...", file=sys.stderr)
    hs = hunspell.HunSpell(DIC, AFF)
    flagged_set = hunspell_check(all_words, hs)
    print(f"  {len(flagged_set)} words flagged.", file=sys.stderr)

    # Build per-word report
    # word -> {frequency, occurrences: [{method, run, subtitle_idx, timestamp, context}]}
    report = defaultdict(lambda: {"frequency": 0, "occurrences": []})

    for r in records:
        flagged_in_block = set(r["tokens"]) & flagged_set
        for word in flagged_in_block:
            report[word]["frequency"] += r["tokens"].count(word)
            report[word]["occurrences"].append(
                {
                    "method": r["method"],
                    "run": r["run"],
                    "subtitle_idx": r["subtitle_idx"],
                    "timestamp": r["timestamp"],
                    "context": r["text"],
                }
            )

    # Sort by descending frequency
    output = {
        "dictionary": "gl_ES",
        "total_flagged_types": len(report),
        "words": {
            word: data
            for word, data in sorted(
                report.items(), key=lambda x: -x[1]["frequency"]
            )
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output written to {OUTPUT_FILE}", file=sys.stderr)

    # Brief console summary
    print(f"\nTop 20 flagged words:")
    for word, data in list(output["words"].items())[:20]:
        print(f"  {word:25s} freq={data['frequency']}")


if __name__ == "__main__":
    main()
