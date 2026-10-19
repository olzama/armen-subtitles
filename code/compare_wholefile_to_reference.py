#!/usr/bin/env python3
"""Meta-evaluate LLM whole-text MQM judgments against reference analysis.

For each item × method where the LLM flagged issues, calls an LLM to assess:
  1. Whether the judgment aligns with the reference analysis
  2. Whether it is unnecessarily harsh given how close the translation is to the reference

Items with no matched issues are recorded as not_flagged without an LLM call.

Usage:
    python code/compare_wholefile_to_reference.py \\
        experiments/films/output/eval/llm-eval/ivan-vas/Russian-English/gpt-5.2-by-gpt-5.4/mqm-wholefile/gpt-5.2-wholefile.json \\
        --lang eng \\
        --out experiments/films/output/eval/llm-eval/ivan-vas/Russian-English/gpt-5.2-by-gpt-5.4/mqm-wholefile/reference_comparison.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import openai

RATES = {
    "gpt-5.4":      {"input": 2.50 / 1_000_000, "cached_input": 0.25 / 1_000_000, "output": 15.00 / 1_000_000},
    "gpt-5.4-mini": {"input": 0.75 / 1_000_000, "cached_input": 0.075 / 1_000_000, "output": 4.50 / 1_000_000},
    "gpt-5-mini":   {"input": 0.25 / 1_000_000, "output": 2.00 / 1_000_000},
}

SYSTEM_MSG = (
    "You are an expert in literary translation quality assessment. "
    "You will evaluate whether an LLM translation judge's complaints about a subtitle translation "
    "are well-founded, given a reference translation and an expert analysis of the passage."
)

PROMPT_TEMPLATE = """\
You are reviewing an LLM judge's complaints about a subtitle translation of a Russian film passage.

## Source (Russian)
{source}

## Reference translation ({lang})
{reference}

## Expert analysis of this passage
{analysis}

## Actual translation being judged (method: {method})
{translation}

## LLM judge's issues
{issues}

Please assess the LLM judge's complaints by answering these two questions:

1. ALIGNMENT: Do the complaints align with what the expert analysis identifies as the real translation challenge for this passage? (yes / partial / no)

2. HARSHNESS: Is the LLM being unnecessarily harsh? Specifically: does it flag problems that the reference translation itself would also be "guilty" of, or does it penalize a reasonable creative choice that differs from the reference but is still valid? (yes / partial / no)

Respond in JSON:
{{
  "alignment": "yes|partial|no",
  "alignment_note": "one sentence",
  "harshness": "yes|partial|no",
  "harshness_note": "one sentence"
}}
"""


def load_key(path):
    return Path(path).read_text(encoding="utf-8").strip()


def call_llm(prompt, client, model):
    rate = RATES[model]
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    usage = response.usage
    in_tok = usage.prompt_tokens
    out_tok = usage.completion_tokens
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    non_cached = in_tok - cached
    cost = (
        non_cached * rate["input"]
        + cached * rate.get("cached_input", rate["input"])
        + out_tok * rate["output"]
    )
    # strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw, cost


def parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw, "parse_error": True}


def first_translation(method_runs):
    if not method_runs:
        return None
    for k in ["1", "01"]:
        if k in method_runs:
            return method_runs[k]
    try:
        key = str(min(int(k) for k in method_runs))
        return method_runs.get(key)
    except (ValueError, TypeError):
        return next(iter(method_runs.values()), None)


def format_issues(issues):
    lines = []
    for i, iss in enumerate(issues, 1):
        sev = iss.get("severity", "?")
        cat = iss.get("category", "?")
        quote = iss.get("source_quote", "")
        just = iss.get("justification", "")
        lines.append(f"{i}. [{sev}/{cat}] source: \"{quote}\" — {just}")
    return "\n".join(lines)


def main():
    import re as _re
    global re
    re = _re

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wholefile_json", help="Path to gpt-5.2-wholefile.json sidecar")
    parser.add_argument("--lang", required=True, help="Target language code (e.g. eng, glg)")
    parser.add_argument("--model", default="gpt-5.4-mini", help="LLM for meta-evaluation (default: gpt-5.4-mini)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling the API")
    args = parser.parse_args()

    data = json.loads(Path(args.wholefile_json).read_text(encoding="utf-8"))
    items = data.get("items", [])

    client = None
    if not args.dry_run:
        client = openai.OpenAI(api_key=load_key("LYS-API-key.txt"))

    total_cost = 0.0
    results = []

    for item in items:
        item_id = item["id"]
        source = item.get("original", {}).get("rus", "")
        reference = item.get("reference", {}).get(args.lang, "") or ""
        analysis_parts = []
        general = item.get("analysis", {}).get("general", {}).get("text", "")
        if general:
            analysis_parts.append(general)
        lang_specific = item.get("analysis", {}).get("language_specific", {}).get(args.lang, {})
        if isinstance(lang_specific, dict):
            if lang_specific.get("text"):
                analysis_parts.append(lang_specific["text"])
            if lang_specific.get("nb"):
                analysis_parts.append("NB: " + lang_specific["nb"])
        analysis = " ".join(analysis_parts)

        llm_wf = item.get("llm_wholefile", {}).get(args.lang, {})
        translations_by_method = item.get("translations", {}).get(args.lang, {})

        method_results = []
        for method, issues in sorted(llm_wf.items()):
            translation = first_translation(translations_by_method.get(method, {}))

            if not issues:
                method_results.append({
                    "method": method,
                    "flagged": False,
                    "assessment": None,
                    "cost_usd": 0.0,
                })
                continue

            issues_text = format_issues(issues)
            prompt = PROMPT_TEMPLATE.format(
                source=source,
                lang=args.lang,
                reference=reference,
                analysis=analysis,
                method=method,
                translation=translation or "(no translation available)",
                issues=issues_text,
            )

            if args.dry_run:
                print(f"\n--- item {item_id} / {method} ---")
                print(prompt[:600] + "...")
                assessment = {"dry_run": True}
                cost = 0.0
            else:
                raw, cost = call_llm(prompt, client, args.model)
                assessment = parse_json(raw)
                total_cost += cost
                time.sleep(0.3)

            method_results.append({
                "method": method,
                "flagged": True,
                "n_issues": len(issues),
                "issues": issues,
                "translation": translation,
                "assessment": assessment,
                "cost_usd": round(cost, 6),
            })

        results.append({
            "id": item_id,
            "character": item.get("character", ""),
            "source": source,
            "reference": reference,
            "analysis": analysis,
            "methods": method_results,
        })

    output = {
        "lang": args.lang,
        "eval_model": args.model,
        "total_cost_usd": round(total_cost, 4),
        "items": results,
    }
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Summary
    flagged = [(r, m) for r in results for m in r["methods"] if m["flagged"]]
    print(f"\n{len(flagged)} flagged item×method pairs across {len(items)} items")
    for r, m in flagged:
        a = m.get("assessment") or {}
        align = a.get("alignment", "?")
        harsh = a.get("harshness", "?")
        print(f"  id={r['id']:2d} [{m['method']:15s}]  alignment={align}  harshness={harsh}")
    print(f"\nTotal cost: ${total_cost:.4f}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
