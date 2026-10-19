#!/usr/bin/env python3
"""Check that the meme spans stored in <model>.json are exactly the spans that were scored.

The evaluator records the candidate text it scored for each meme. LLM trimming
is not perfectly deterministic, so re-running span extraction after scoring can
leave stored spans that differ from the scored ones. With --sync, the stored
mapping is set to the scored text (the scored text is what the results describe).

Usage:
    python code/audit_spans.py [--sync]
"""

import argparse
import glob
import json
import re
from pathlib import Path

from lang_utils import lang_code

JUDGE = {"English": "gpt-5-mini", "Galician": "gpt-5.4-mini", "Spanish": "gpt-5.4-mini"}


def n(t):
    return re.sub(r"\s+", " ", t or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    a = ap.parse_args()
    for mapped in sorted(Path("experiments/films/output/translations").glob("*/*.json")):
        if mapped.name.endswith(".pre-span.json"):
            continue
        film, model = mapped.parent.name, mapped.stem
        data = json.loads(mapped.read_text(encoding="utf-8"))
        items = {str(i["id"]): i for i in data.get("items", [])}
        changed = 0
        for tgt, judge in JUDGE.items():
            code = lang_code(tgt)
            root = Path(f"experiments/films/output/eval/llm-eval/{film}/Russian-{tgt}/{model}-by-{judge}/mqm-memes")
            mism = 0
            for f in glob.glob(str(root / "*" / "run_*_eval_1.json")):
                method, run = f.split("/")[-2], re.search(r"run_(\d+)_", f).group(1)
                for it in json.loads(Path(f).read_text(encoding="utf-8"))["items"]:
                    stored = items.get(str(it["id"]), {}).get("translations", {}).get(code, {}).get(method, {}).get(run)
                    if n(stored) != n(it.get("candidate")):
                        mism += 1
                        if a.sync:
                            items[str(it["id"])]["translations"][code][method][run] = it["candidate"]
                            changed += 1
            if root.exists():
                print(f"{film:15s} {model:16s} {tgt:9s} mismatches: {mism}")
        if a.sync and changed:
            mapped.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
