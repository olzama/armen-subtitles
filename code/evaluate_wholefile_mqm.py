#!/usr/bin/env python3
"""Whole-file MQM evaluation using an LLM judge (reference-free).

Reads a yaml pipeline config, picks N translation runs per method, and for
each run performs E LLM evaluation calls on the full source + translation text.
Outputs per-run JSON files in the same format as evaluate_mqm_parallel.py so
that aggregate_mqm.py and variance.py can be reused without changes.

Normalization unit: subtitle entry count (number of SRT entries in the source).

Usage:
    python code/evaluate_wholefile_mqm.py yaml-pipelines/films/ivan-vas-russian-english-0.1.yaml
    python code/evaluate_wholefile_mqm.py ... --n-runs 3 --eval-runs 3 --methods zero,characters
    python code/evaluate_wholefile_mqm.py ... --eval-model gpt-5.2 --dry-run
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import openai
import pysrt
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from lang_utils import lang_code
from claude_cli import call_claude

RATES = {
    "gpt-5.2":    {"input": 1.75 / 1_000_000, "output": 14.00 / 1_000_000, "max_chars": 100_000},
    "gpt-5.4":    {"input": 2.50 / 1_000_000, "cached_input": 0.25 / 1_000_000, "output": 15.00 / 1_000_000, "max_chars": 100_000},
    "gpt-5-mini": {"input": 0.25 / 1_000_000, "output":  2.00 / 1_000_000, "max_chars": 50_000},
    "gpt-5.4-mini": {"input": 0.75 / 1_000_000, "cached_input": 0.075 / 1_000_000, "output": 4.50 / 1_000_000, "max_chars": 200_000},
    # Claude runs through the `claude -p` CLI on a subscription; cost is the CLI's API-equivalent estimate.
    "claude-opus-5-5":   {"input": 0.0, "output": 0.0, "max_chars": 100_000},
    "claude-sonnet-5-5": {"input": 0.0, "output": 0.0, "max_chars": 100_000},
    "claude-haiku-5-5":  {"input": 0.0, "output": 0.0, "max_chars": 100_000},
}

SEVERITY_WEIGHTS = {"major": 5, "minor": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_key(path):
    return Path(path).read_text(encoding="utf-8").strip()


def srt_entries(path):
    """Return list of (text, index) from an SRT-format file."""
    subs = pysrt.open(str(path))
    return [(sub.text.replace("\n", " ").strip(), sub.index) for sub in subs]


def entries_to_text(entries):
    return " ".join(text for text, _ in entries if text)


def source_srt_path(cfg):
    film = cfg["film"]
    lc = lang_code(cfg["source_lang"])
    return Path("experiments/films/data") / film / "subs" / f"{film}-{lc}.srt"


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


def output_dir(cfg, method_name, eval_model):
    src, tgt = cfg["source_lang"], cfg["target_lang"]
    tag = f"{cfg['trans_model']}-by-{eval_model}"
    return (
        Path("experiments/films/output/eval/llm-eval")
        / cfg["film"]
        / f"{src}-{tgt}"
        / tag
        / "mqm-wholefile"
        / method_name
    )


def chunk_entries(src_entries, tgt_entries, max_chars, prompt_overhead=2000):
    """Split parallel entry lists into chunks that fit within max_chars."""
    budget = max_chars - prompt_overhead
    chunks = []
    cur_src, cur_tgt, cur_len = [], [], 0

    for (s_text, s_idx), (t_text, t_idx) in zip(src_entries, tgt_entries):
        pair_len = len(s_text) + len(t_text) + 2
        if cur_src and cur_len + pair_len > budget:
            chunks.append((cur_src[:], cur_tgt[:]))
            cur_src, cur_tgt, cur_len = [], [], 0
        cur_src.append((s_text, s_idx))
        cur_tgt.append((t_text, t_idx))
        cur_len += pair_len

    if cur_src:
        chunks.append((cur_src, cur_tgt))
    return chunks


def build_prompt(prompt_template, src_text, tgt_text, src_lang, tgt_lang):
    header = (
        f"SOURCE LANGUAGE: {src_lang}\n"
        f"TARGET LANGUAGE: {tgt_lang}\n\n"
        f"--- SOURCE TEXT ---\n{src_text}\n\n"
        f"--- TRANSLATION ---\n{tgt_text}\n\n"
        f"--- INSTRUCTIONS ---\n"
    )
    return header + prompt_template


def call_gpt(content, client, model, system_msg):
    rate = RATES[model]
    response = client.chat.completions.create(
        model=model,
        temperature=1.0,
        top_p=1,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": content},
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
    return raw, in_tok, out_tok, cost


def parse_issues(raw):
    """Extract issues list from LLM response, with basic JSON repair."""
    text = raw.strip()
    # strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    issues = data.get("issues", [])
    # normalise severity to lowercase
    for issue in issues:
        issue["severity"] = issue.get("severity", "").lower()
    return issues


def score_issues(issues, n_entries):
    counts = {"major": 0, "minor": 0}
    for issue in issues:
        sev = issue.get("severity", "").lower()
        if sev in counts:
            counts[sev] += 1
    total_points = sum(counts[s] * SEVERITY_WEIGHTS[s] for s in counts)
    major_equiv = total_points / SEVERITY_WEIGHTS["major"]
    per_entry = major_equiv / n_entries if n_entries > 0 else 0.0
    return counts, total_points, major_equiv, per_entry


def run_eval(src_entries, tgt_entries, prompt_template, client, model,
             src_lang, tgt_lang, max_chars, system_msg):
    """
    Run one LLM evaluation call. If text is too long, split into chunks,
    evaluate each, and merge issues.
    """
    chunks = chunk_entries(src_entries, tgt_entries, max_chars)
    all_issues = []
    total_in, total_out, total_cost = 0, 0, 0.0

    for src_chunk, tgt_chunk in chunks:
        src_text = entries_to_text(src_chunk)
        tgt_text = entries_to_text(tgt_chunk)
        content = build_prompt(prompt_template, src_text, tgt_text, src_lang, tgt_lang)
        if model.startswith("claude"):
            raw, in_tok, out_tok, cost = call_claude(content, model, system_msg)
        else:
            raw, in_tok, out_tok, cost = call_gpt(content, client, model, system_msg)
        issues = parse_issues(raw)
        all_issues.extend(issues)
        total_in += in_tok
        total_out += out_tok
        total_cost += cost

    return all_issues, total_in, total_out, total_cost


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to yaml pipeline config")
    parser.add_argument("--eval-model", default="gpt-5.4",
                        help="LLM to use for evaluation (default: gpt-5.4)")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of translation runs per method (default: 3)")
    parser.add_argument("--eval-runs", type=int, default=3,
                        help="Number of LLM eval calls per translation run (default: 3)")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated method names to evaluate (default: all in yaml)")
    parser.add_argument("--prompt", default="experiments/films/prompts/eval/mqm-wholefile.txt",
                        help="Path to eval prompt template")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without making API calls")
    args = parser.parse_args()

    if args.eval_model not in RATES:
        sys.exit(f"Unknown eval model '{args.eval_model}'. Known: {list(RATES)}")

    cfg = load_config(args.config)
    film = cfg["film"]
    src_lang = cfg["source_lang"]
    tgt_lang = cfg["target_lang"]
    trans_model = cfg["trans_model"]
    max_chars = RATES[args.eval_model]["max_chars"]

    method_filter = set(args.methods.split(",")) if args.methods else None
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")
    system_msg = "Expert in literary translation quality assessment using MQM framework."

    client = None
    if not args.dry_run:
        client = openai.OpenAI(api_key=load_key("LYS-API-key.txt"))

    src_srt = source_srt_path(cfg)
    if not src_srt.exists():
        sys.exit(f"Source SRT not found: {src_srt}")
    src_entries = srt_entries(src_srt)
    n_src_entries = len(src_entries)
    print(f"{film}  {src_lang}→{tgt_lang}  trans={trans_model}  eval={args.eval_model}")
    print(f"Source: {n_src_entries} entries  ({src_srt.name})")

    total_cost = 0.0

    for method in cfg["methods"]:
        name = method["name"]
        if method_filter and name not in method_filter:
            continue

        trans_d = translation_dir(cfg, name)
        if not trans_d.exists():
            print(f"  [{name}] translations dir not found — skipping")
            continue

        run_files = sorted(trans_d.glob("translation-*.txt"))[: args.n_runs]
        if not run_files:
            print(f"  [{name}] no translation files — skipping")
            continue

        out_d = output_dir(cfg, name, args.eval_model)
        if not args.dry_run:
            out_d.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{name}]  {len(run_files)} translation runs × {args.eval_runs} eval runs")

        for rf in run_files:
            run_num = re.search(r"translation-(\d+)", rf.name).group(1)
            tgt_entries_list = srt_entries(rf)

            for e in range(1, args.eval_runs + 1):
                out_path = out_d / f"run_{run_num}_eval_{e}.json"
                if out_path.exists():
                    print(f"    run {run_num} eval {e}: already exists — skipping")
                    continue

                if args.dry_run:
                    src_text = entries_to_text(src_entries)
                    tgt_text = entries_to_text(tgt_entries_list)
                    combined = len(src_text) + len(tgt_text)
                    n_chunks = len(chunk_entries(src_entries, tgt_entries_list, max_chars))
                    print(f"    run {run_num} eval {e}: {combined:,} chars → {n_chunks} chunk(s) [dry-run]")
                    continue

                try:
                    issues, in_tok, out_tok, cost = run_eval(
                        src_entries, tgt_entries_list, prompt_template,
                        client, args.eval_model, src_lang, tgt_lang,
                        max_chars, system_msg
                    )
                except Exception as ex:
                    print(f"    run {run_num} eval {e}: ERROR — {ex}")
                    continue

                counts, total_pts, major_equiv, per_entry = score_issues(issues, n_src_entries)
                total_cost += cost

                record = {
                    "dataset_name": film,
                    "method": name,
                    "run": run_num,
                    "eval_run": e,
                    "evaluator": args.eval_model,
                    "translator": trans_model,
                    "n_subtitle_entries": n_src_entries,
                    "issues": issues,
                    "summary": {
                        "major_count": counts["major"],
                        "minor_count": counts["minor"],
                        "total_points": total_pts,
                        "major_equiv": major_equiv,
                        "major_equiv_per_unit": per_entry,
                    },
                    "usage": {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "cost_usd": cost,
                    },
                }
                out_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"    run {run_num} eval {e}: "
                    f"{counts['major']}maj/{counts['minor']}min  "
                    f"score={per_entry:.4f}  cost=${cost:.4f}"
                )
                time.sleep(0.5)

    print(f"\nTotal API cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
