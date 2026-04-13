#!/usr/bin/env python3
"""
Interactive human MQM evaluation of subtitle translations.

Usage:
    python evaluate_human.py <film1> [<film2> ...] --trans-model <model> --evaluator-id <id>

Session: films/output/eval/human-eval/human-<evaluator_id>/session.json
Export:  films/output/eval/human-eval/<film>/<trans_model>-by-human-<evaluator_id>/<method>/run_<run>_eval_1.json
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


# ─────────────────────────────────────────────
# ISSUE PARSING
# ─────────────────────────────────────────────

SEVERITY_MAP = {
    "ma":       "major",
    "maj":      "major",
    "major":    "major",
    "mi":       "minor",
    "min":      "minor",
    "minor":    "minor",
}

CATEGORY_MAP = {
    "a":           "accuracy",
    "acc":         "accuracy",
    "accuracy":    "accuracy",
    "f":           "fluency",
    "flu":         "fluency",
    "fluency":     "fluency",
    "s":           "style",
    "sty":         "style",
    "style":       "style",
    "t":           "terminology",
    "ter":         "terminology",
    "term":        "terminology",
    "terminology": "terminology",
    "o":           "other",
    "oth":         "other",
    "other":       "other",
}

ISSUE_HELP = (
    "  severity: ma=major  mi=minor\n"
    "  category: a=accuracy  f=fluency  s=style  t=terminology  o=other\n"
    "  format  : <severity> <category> <justification>\n"
    "  0=no issues   s=skip"
)


def parse_issue_line(line):
    """Parse 'ma a justification text' into an issue dict.

    Returns (issue_dict, None) on success, (None, error_str) on failure.
    """
    parts = line.split(None, 2)
    if len(parts) < 2:
        return None, "need at least severity and category"

    severity = SEVERITY_MAP.get(parts[0].lower())
    if severity is None:
        return None, f"unknown severity '{parts[0]}' — use: c, ma, mi"

    category = CATEGORY_MAP.get(parts[1].lower())
    if category is None:
        return None, f"unknown category '{parts[1]}' — use: a, f, s, t, o"

    justification = parts[2].strip() if len(parts) > 2 else ""

    return {
        "severity":      severity,
        "category":      category,
        "span":          "",
        "justification": justification,
    }, None


# ─────────────────────────────────────────────
# MQM SCORING (mirrors evaluate_mqm_parallel)
# ─────────────────────────────────────────────

SEVERITY_WEIGHTS = {"major": 5, "minor": 1}


def score_issues(issues):
    """Return major_equiv_per_unit score for a single item's issues list."""
    points = sum(
        SEVERITY_WEIGHTS.get(iss.get("severity", "").lower(), 0)
        for iss in issues
        if iss.get("category") != "no-issue"
    )
    return points / SEVERITY_WEIGHTS["major"]


def compute_mqm_summary(items_with_issues):
    counts = {s: 0 for s in SEVERITY_WEIGHTS}
    for item in items_with_issues:
        for iss in item.get("issues", []):
            sev = iss.get("severity", "").lower()
            if sev in counts and iss.get("category") != "no-issue":
                counts[sev] += 1

    total_points = sum(counts[s] * SEVERITY_WEIGHTS[s] for s in counts)
    n = len(items_with_issues)
    penalty_per_unit = total_points / n if n > 0 else 0.0
    major_equiv = penalty_per_unit / SEVERITY_WEIGHTS["major"]

    return {
        "counts": counts,
        "total_points": total_points,
        "meaning_units": n,
        "penalty_per_unit": penalty_per_unit,
        "major_equiv_per_unit": major_equiv,
        "interpretation": (
            f"{penalty_per_unit:.2f} MQM penalty points per meaning unit "
            f"(≈ {major_equiv:.2f} major issues per unit)"
        ),
    }


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_film_data(film_name, trans_model):
    """Load translations JSON for a single (film, model) pair."""
    path = Path("films/output/translations") / film_name / f"{trans_model}.json"
    if not path.exists():
        raise FileNotFoundError(f"Translations not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "items" not in data:
        raise ValueError(f"{path}: expected a JSON object with 'items'.")
    return data


def discover_trans_models(film_name):
    """Return all translation model names available for a film."""
    trans_dir = Path("films/output/translations") / film_name
    if not trans_dir.is_dir():
        return []
    return sorted(p.stem for p in trans_dir.glob("*.json"))


def load_eval_config(session_dir):
    """Load researcher-created config file, or return None if absent.

    Expected format:
        {
          "films": {
            "ivan-vas":    ["gpt-5.2", "gpt-5-mini"],
            "diamond-arm": ["gpt-5.2"]
          }
        }

    Returns a dict mapping film_name -> [model, ...], or None.
    """
    path = session_dir / "config.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    films = raw.get("films")
    if not isinstance(films, dict):
        raise ValueError(f"{path}: 'films' must be a dict mapping film names to model lists.")
    for film, models in films.items():
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError(f"{path}: models for '{film}' must be a list of strings.")
    return {film: list(models) for film, models in films.items()}


# ─────────────────────────────────────────────
# SESSION BUILDING
# ─────────────────────────────────────────────

def _sample_runs(method_runs, max_runs, rng):
    """Return {method: [run_id, ...]} sampling at most max_runs per method."""
    result = {}
    for method, run_set in sorted(method_runs.items()):
        runs = sorted(run_set, key=lambda x: (len(x), x))
        if len(runs) > max_runs:
            runs = sorted(rng.sample(runs, max_runs), key=lambda x: (len(x), x))
        result[method] = runs
    return result


def _base_records_for_film(film_name, trans_model, data, method_filter, max_runs, rng):
    """Return list of {film, trans_model, item_id, method, run} for one (film, model) pair."""
    items = data["items"]

    method_runs = {}
    for item in items:
        for method, runs in item.get("translations", {}).get("eng", {}).items():
            if method_filter and method not in method_filter:
                continue
            method_runs.setdefault(method, set()).update(str(r) for r in runs)

    sampled = _sample_runs(method_runs, max_runs, rng)

    records = []
    for method, runs in sampled.items():
        for run in runs:
            missing = [
                item["id"]
                for item in items
                if str(run) not in item.get("translations", {}).get("eng", {}).get(method, {})
            ]
            if missing:
                print(
                    f"  Warning: film={film_name} model={trans_model} method={method} run={run} "
                    f"missing item(s) {missing}; skipping.",
                    file=sys.stderr,
                )
                continue
            for item in items:
                records.append({
                    "film":        film_name,
                    "trans_model": trans_model,
                    "item_id":     item["id"],
                    "method":      method,
                    "run":         run,
                })
    return records


def build_session_tasks(films_models_data, method_filter, max_runs, repeat_fraction, rng):
    """Build a shuffled task list with silent repeats.

    films_models_data: list of (film_name, trans_model, data) triples.
    Each task: {film, trans_model, item_id, method, run, is_repeat}
    """
    base = []
    for film_name, trans_model, data in films_models_data:
        base.extend(_base_records_for_film(film_name, trans_model, data, method_filter, max_runs, rng))

    rng.shuffle(base)

    tasks = [
        {"film": r["film"], "trans_model": r["trans_model"], "item_id": r["item_id"],
         "method": r["method"], "run": r["run"], "is_repeat": False}
        for r in base
    ]

    # Insert silent repeats in the second half
    n_repeats = max(0, round(len(base) * repeat_fraction))
    repeat_srcs = rng.sample(range(len(base)), min(n_repeats, len(base)))

    repeats = [
        {"film": base[i]["film"], "trans_model": base[i]["trans_model"],
         "item_id": base[i]["item_id"], "method": base[i]["method"],
         "run": base[i]["run"], "is_repeat": True}
        for i in repeat_srcs
    ]
    rng.shuffle(repeats)

    half = len(tasks) // 2
    for rt in repeats:
        pos = rng.randint(half, len(tasks))
        tasks.insert(pos, rt)

    return tasks


# ─────────────────────────────────────────────
# SESSION PERSISTENCE
# ─────────────────────────────────────────────

def load_session(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_session(session, path):
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_session(films_models_data, evaluator_id, method_filter, max_runs, repeat_fraction, rng):
    tasks = build_session_tasks(films_models_data, method_filter, max_runs, repeat_fraction, rng)
    films = sorted({film for film, _, _ in films_models_data})
    return {
        "evaluator_id":    evaluator_id,
        "films":           films,
        "runs_requested":  max_runs,
        "repeat_fraction": repeat_fraction,
        "created":         datetime.now(timezone.utc).isoformat(),
        "tasks":           tasks,
        "judgments":       {},   # str(task_idx) → {"issues": [...]}
        "skipped":         [],   # list of task indices (deferred)
    }


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

WIDTH = 60


def _rule(char="─"):
    return char * WIDTH


def _center(text):
    return text.center(WIDTH)


def display_task(task_num, total, n_done, n_skipped, film_titles, task, item):
    translation = (item.get("translations", {}).get("eng", {})
                       .get(task["method"], {}).get(str(task["run"]), ""))

    pct = int(100 * n_done / total) if total > 0 else 0
    bar_len = 20
    filled = int(bar_len * n_done / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    print()
    print(_rule("═"))
    print(_center(f"Item {task_num} of {total}   [{bar}] {pct}%"))
    if n_skipped:
        remaining = total - n_done - n_skipped
        print(_center(f"{n_done} done · {n_skipped} skipped · {remaining} remaining"))
    print(_rule("═"))
    print()

    film_display = film_titles.get(task["film"], task["film"])
    print(f"Film:      {film_display}")
    print(f"Character: {item.get('character', '—')}")
    print()
    print(_rule())
    print("Original:")
    for line in item.get("original", {}).get("rus", "").splitlines():
        print(f"  {line}")
    print()
    print("Translation:")
    for line in translation.splitlines():
        print(f"  {line}")
    print(_rule())
    print()

    return translation


def display_issue_prompt(can_go_back=False):
    print("Enter issues one per line, blank line to save:")
    print(ISSUE_HELP)
    if can_go_back:
        print("  b=back to edit previous item")
    print()


def _format_issues_brief(issues):
    if not issues:
        return "no issues"
    parts = []
    for iss in issues:
        just = iss.get("justification", "")
        label = f"{iss['severity']}/{iss['category']}"
        parts.append(f"{label}: {just[:40]}" if just else label)
    return " · ".join(parts)


def display_saved_summary(issues):
    print(f"  ✓ Saved: {_format_issues_brief(issues)}")
    print()


# ─────────────────────────────────────────────
# INTERACTIVE ISSUE COLLECTION
# ─────────────────────────────────────────────

def collect_issues(can_go_back=False):
    """Prompt the user to enter issues line by line.

    Returns (issues, action) where action is 'accept' | 'skip' | 'back'.
    """
    issues = []

    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return issues, "accept"

        if not raw:
            return issues, "accept"

        if raw.lower() == "s":
            return [], "skip"

        if raw.lower() == "b" and can_go_back:
            return [], "back"

        if raw == "0":
            return [], "accept"

        issue, err = parse_issue_line(raw)
        if err:
            print(f"  Error: {err}")
            print(f"  {ISSUE_HELP.splitlines()[0]}")
            continue

        issues.append(issue)


# ─────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────

def run_session(session, items_by_film, film_titles, session_path):
    tasks     = session["tasks"]
    judgments = session["judgments"]
    skipped   = set(session["skipped"])

    pending = [
        (i, t) for i, t in enumerate(tasks)
        if str(i) not in judgments and i not in skipped
    ]

    if not pending:
        print("All tasks complete (or skipped). Run with --export to write eval files.")
        return

    total  = len(pending)
    n_done = 0
    n_skip = len(skipped)

    # last_saved: (task_idx, issues) enabling single-step undo
    last_saved = None
    pos = 0

    while pos < len(pending):
        task_idx, task = pending[pos]

        # Skip tasks that were auto-filled since this loop started
        if str(task_idx) in judgments:
            n_done += 1
            pos += 1
            continue

        item = items_by_film[task["film"]][task["trans_model"]][task["item_id"]]

        display_task(
            task_num    = n_done + 1,
            total       = total,
            n_done      = n_done,
            n_skipped   = n_skip,
            film_titles = film_titles,
            task        = task,
            item        = item,
        )

        can_go_back = last_saved is not None
        display_issue_prompt(can_go_back=can_go_back)
        issues, action = collect_issues(can_go_back=can_go_back)

        if action == "back":
            prev_task_idx, prev_issues = last_saved
            del judgments[str(prev_task_idx)]
            session["judgments"] = judgments
            save_session(session, session_path)
            print(f"  ↩ Back. Previous entry was: {_format_issues_brief(prev_issues)}")
            print()
            last_saved = None
            n_done -= 1
            pos -= 1
            continue

        if action == "skip":
            skipped.add(task_idx)
            session["skipped"] = sorted(skipped)
            save_session(session, session_path)
            last_saved = None
            n_skip += 1
            pos += 1
            print("  Skipped.")
            continue

        # action == "accept"
        judgments[str(task_idx)] = {"issues": issues}
        session["judgments"] = judgments
        n_auto = auto_fill_from_consensus(session, items_by_film)
        save_session(session, session_path)
        display_saved_summary(issues)
        if n_auto:
            print(f"  ✦ Auto-filled {n_auto} task(s) with identical translation.")
        last_saved = (task_idx, issues)
        n_done += 1
        pos += 1

    remaining_skipped = sum(
        1 for i, t in enumerate(tasks)
        if str(i) not in judgments and i in skipped
    )
    if remaining_skipped:
        print(f"{remaining_skipped} skipped item(s) remain. Run again to evaluate them.")

    total_non_repeat = sum(1 for t in tasks if not t["is_repeat"])
    completed = sum(1 for i, t in enumerate(tasks)
                    if str(i) in judgments and not t["is_repeat"])
    print()
    print(f"Progress: {completed}/{total_non_repeat} non-repeat tasks judged.")


# ─────────────────────────────────────────────
# INCONSISTENCY REVIEW
# ─────────────────────────────────────────────

def _task_translation(task, items_by_film):
    item = items_by_film.get(task["film"], {}).get(task["trans_model"], {}).get(task["item_id"], {})
    return (item.get("translations", {}).get("eng", {})
                .get(task["method"], {}).get(str(task["run"]), ""))


def _issues_signature(issues):
    """Canonical form of an issues list: sorted (severity, category) pairs.
    Ignores justification text and order."""
    return tuple(sorted(
        (iss["severity"], iss["category"])
        for iss in issues
        if iss.get("category") != "no-issue"
    ))


def auto_fill_from_consensus(session, items_by_film):
    """If any translation text has been judged with the same issues 3+ times,
    auto-fill all remaining pending tasks that have that text.
    Returns the number of newly auto-filled tasks.
    """
    tasks     = session["tasks"]
    judgments = session["judgments"]

    # Build text → {sig: (count, issues)} from human (non-auto) judgments
    text_sigs = {}
    for i, task in enumerate(tasks):
        j = judgments.get(str(i))
        if not j or j.get("auto_filled"):
            continue
        text = _task_translation(task, items_by_film)
        if not text:
            continue
        sig = _issues_signature(j["issues"])
        bucket = text_sigs.setdefault(text, {})
        count, stored_issues = bucket.get(sig, (0, j["issues"]))
        bucket[sig] = (count + 1, stored_issues)

    # Collect texts whose top signature has 3+ votes
    consensus = {}   # text → issues
    for text, bucket in text_sigs.items():
        for sig, (count, issues) in bucket.items():
            if count >= 3:
                consensus[text] = issues
                break

    if not consensus:
        return 0

    # Auto-fill pending tasks
    n_filled = 0
    for i, task in enumerate(tasks):
        if str(i) in judgments:
            continue
        text = _task_translation(task, items_by_film)
        if text in consensus:
            judgments[str(i)] = {"issues": consensus[text], "auto_filled": True}
            n_filled += 1

    return n_filled


def find_inconsistencies(session, items_by_film):
    """Return list of inconsistency dicts of two kinds:

    type='repeat'        — silent repeat pair scored differently
    type='identical_text'— different (method, run) pairs with the exact same
                           translation text scored differently
    """
    tasks     = session["tasks"]
    judgments = session["judgments"]
    result    = []

    # ── Type 1: silent repeat pairs ──────────────────────────────────────────
    by_triple = {}   # (film, trans_model, item_id, method, run) → [idx, ...]
    for i, t in enumerate(tasks):
        if str(i) not in judgments:
            continue
        key = (t["film"], t["trans_model"], t["item_id"], t["method"], t["run"])
        by_triple.setdefault(key, []).append(i)

    for key, indices in by_triple.items():
        originals = [i for i in indices if not tasks[i]["is_repeat"]]
        repeats   = [i for i in indices if     tasks[i]["is_repeat"]]
        if len(originals) != 1 or len(repeats) != 1:
            continue
        a, b = originals[0], repeats[0]
        score_a = score_issues(judgments[str(a)]["issues"])
        score_b = score_issues(judgments[str(b)]["issues"])
        if score_a != score_b:
            result.append({
                "type":         "repeat",
                "task_indices": [a, b],
                "film": key[0], "trans_model": key[1], "item_id": key[2],
                "scores":       [score_a, score_b],
            })

    # ── Type 2: identical translation text across different (method, run) ────
    # Only non-repeat judged tasks; group by (film, trans_model, item_id)
    by_item = {}
    for i, t in enumerate(tasks):
        if str(i) not in judgments or t["is_repeat"]:
            continue
        key = (t["film"], t["trans_model"], t["item_id"])
        by_item.setdefault(key, []).append(i)

    for (film, trans_model, item_id), indices in by_item.items():
        if len(indices) < 2:
            continue
        # Group by exact translation text
        by_text = {}
        for i in indices:
            text = _task_translation(tasks[i], items_by_film)
            by_text.setdefault(text, []).append(i)

        for text, text_indices in by_text.items():
            if len(text_indices) < 2:
                continue
            for a, b in combinations(text_indices, 2):
                score_a = score_issues(judgments[str(a)]["issues"])
                score_b = score_issues(judgments[str(b)]["issues"])
                if score_a != score_b:
                    result.append({
                        "type":             "identical_text",
                        "task_indices":     [a, b],
                        "film":             film,
                        "trans_model":      trans_model,
                        "item_id":          item_id,
                        "translation_text": text,
                        "scores":           [score_a, score_b],
                    })

    return result


def review_inconsistencies(session, items_by_film, film_titles, session_path):
    """Show conflicting judgments and ask the evaluator to resolve each one."""
    inconsistencies = find_inconsistencies(session, items_by_film)
    if not inconsistencies:
        return

    resolutions = session.setdefault("inconsistency_resolutions", {})
    # An inconsistency is resolved when task_indices[0] has an entry
    unresolved = [inc for inc in inconsistencies
                  if str(inc["task_indices"][0]) not in resolutions]
    if not unresolved:
        return

    tasks     = session["tasks"]
    judgments = session["judgments"]

    n_repeat = sum(1 for inc in unresolved if inc["type"] == "repeat")
    n_text   = sum(1 for inc in unresolved if inc["type"] == "identical_text")
    kinds = []
    if n_repeat: kinds.append(f"{n_repeat} silent repeat(s)")
    if n_text:   kinds.append(f"{n_text} identical translation(s) scored differently")

    print()
    print(_rule("═"))
    print(_center("INCONSISTENCY REVIEW"))
    print(_rule("═"))
    print(f"\nFound {len(unresolved)} inconsistency(ies): {', '.join(kinds)}.")
    print("For each, choose which judgment to keep. Both original scores are recorded.\n")

    def _show_judgment(label, issues, score):
        print(f"  {label}  (score {score:.1f}):")
        if issues:
            for iss in issues:
                just = iss.get("justification", "")
                print(f"    [{iss['severity']}] {iss['category']}"
                      + (f": {just}" if just else ""))
        else:
            print("    (no issues)")

    for n, inc in enumerate(unresolved, 1):
        idx_a, idx_b = inc["task_indices"]
        film, trans_model, item_id = inc["film"], inc["trans_model"], inc["item_id"]
        item = items_by_film[film][trans_model][item_id]

        issues_a = judgments[str(idx_a)]["issues"]
        issues_b = judgments[str(idx_b)]["issues"]

        # For repeat type, translation comes from the original task's (method, run)
        # For identical_text, both texts are the same — use either
        if inc["type"] == "identical_text":
            translation = inc["translation_text"]
            kind_label  = "identical translation text, scored differently"
        else:
            t = tasks[idx_a]
            translation = _task_translation(t, items_by_film)
            kind_label  = "silent repeat, scored differently"

        print(_rule())
        print(f"  {n} of {len(unresolved)}  —  {kind_label}")
        print(f"  {film_titles.get(film, film)}  |  {item.get('character', '—')}")
        print()
        print("  Original:")
        for line in item.get("original", {}).get("rus", "").splitlines():
            print(f"    {line}")
        print()
        print("  Translation:")
        for line in translation.splitlines():
            print(f"    {line}")
        print()
        _show_judgment("Pass 1", issues_a, inc["scores"][0])
        print()
        _show_judgment("Pass 2", issues_b, inc["scores"][1])
        print()
        print("  1=keep first   2=keep second   e=enter new judgment")

        while True:
            try:
                raw = input("> ").strip()
            except EOFError:
                raw = "1"

            if raw == "1":
                chosen = {"issues": issues_a, "source": "first"}
            elif raw == "2":
                chosen = {"issues": issues_b, "source": "second"}
            elif raw.lower() == "e":
                print()
                display_issue_prompt(can_go_back=False)
                new_issues, action = collect_issues(can_go_back=False)
                if action != "accept":
                    continue
                chosen = {"issues": new_issues, "source": "manual"}
            else:
                print("  Please enter 1, 2, or e.")
                continue

            # Apply to both task indices (matters for identical_text where both are exported)
            resolutions[str(idx_a)] = chosen
            resolutions[str(idx_b)] = chosen
            break

        session["inconsistency_resolutions"] = resolutions
        save_session(session, session_path)
        print(f"  ✓ Resolved: {_format_issues_brief(chosen['issues'])}")
        print()

    print("Inconsistency review complete.")


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

def export_session(session, items_by_film, base_out):
    """Write per-film, per-method eval files and a noise summary.

    base_out: films/output/eval/human-eval/human-<evaluator_id>/
    Film eval files go to:
        films/output/eval/human-eval/<film>/<trans_model>-by-human-<evaluator_id>/<method>/run_<run>_eval_1.json
    Noise file goes to:
        base_out/human_eval_noise.json
    """
    tasks        = session["tasks"]
    judgments    = session["judgments"]
    evaluator_id = session["evaluator_id"]

    resolutions = session.get("inconsistency_resolutions", {})

    # Group judged tasks by (film, trans_model, method, run)
    groups = {}
    for idx_str, judgment in judgments.items():
        task = tasks[int(idx_str)]
        key = (task["film"], task["trans_model"], task["method"], task["run"])
        groups.setdefault(key, {"first": [], "repeat": []})
        slot = "repeat" if task["is_repeat"] else "first"
        groups[key][slot].append({
            "task_idx": int(idx_str),
            "item_id":  task["item_id"],
            "issues":   judgment["issues"],
        })

    n_exported = 0
    noise_pairs = []

    for (film, trans_model, method, run), slots in sorted(groups.items()):
        first_entries  = sorted(slots["first"],  key=lambda e: e["item_id"])
        repeat_entries = sorted(slots["repeat"], key=lambda e: e["item_id"])

        if not first_entries:
            continue

        film_items = items_by_film.get(film, {}).get(trans_model, {})

        items_out = []
        for entry in first_entries:
            item = film_items.get(entry["item_id"], {})
            candidate = (item.get("translations", {}).get("eng", {})
                             .get(method, {}).get(str(run), ""))
            # Use resolved judgment if the evaluator reconciled an inconsistency
            res = resolutions.get(str(entry["task_idx"]))
            issues = res["issues"] if res else entry["issues"]
            items_out.append({
                "id":        item.get("id"),
                "character": item.get("character"),
                "source":    item.get("original", {}).get("rus", ""),
                "reference": item.get("reference", {}).get("eng", ""),
                "analysis":  item.get("analysis", ""),
                "candidate": candidate,
                "issues":    issues,
            })

        summary = compute_mqm_summary(items_out)

        record = {
            "method":     method,
            "run":        run,
            "eval_run":   1,
            "evaluator":  f"human-{evaluator_id}",
            "translator": trans_model,
            "summary":    summary,
            "items":      items_out,
        }

        eval_dir_name = f"{trans_model}-by-human-{evaluator_id}"
        out_dir = Path("films/output/eval/human-eval") / film / eval_dir_name / method
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"run_{run}_eval_1.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n_exported += 1
        print(f"  Wrote {out_path}")

        # Collect noise pairs (always use original issues, not resolved, for noise estimation)
        first_by_item  = {e["item_id"]: e for e in first_entries}
        for rep_entry in repeat_entries:
            fst = first_by_item.get(rep_entry["item_id"])
            if fst is None:
                continue
            res = resolutions.get(str(fst["task_idx"]))
            noise_pairs.append({
                "film":           film,
                "trans_model":    trans_model,
                "item_id":        rep_entry["item_id"],
                "method":         method,
                "run":            run,
                "first_score":    score_issues(fst["issues"]),
                "repeat_score":   score_issues(rep_entry["issues"]),
                "delta":          abs(score_issues(fst["issues"]) - score_issues(rep_entry["issues"])),
                "resolved_score": score_issues(res["issues"]) if res else None,
                "resolution":     res["source"] if res else None,
            })

    # Write noise summary
    if noise_pairs:
        deltas = [p["delta"] for p in noise_pairs]
        mean_delta = sum(deltas) / len(deltas)
        diffs = [p["first_score"] - p["repeat_score"] for p in noise_pairs]
        if len(diffs) >= 2:
            mean_d = sum(diffs) / len(diffs)
            var_d  = sum((d - mean_d) ** 2 for d in diffs) / (len(diffs) - 1)
            within_person_sd = math.sqrt(var_d / 2)
        else:
            within_person_sd = None

        noise_record = {
            "n_pairs":               len(noise_pairs),
            "mean_abs_delta":        mean_delta,
            "within_person_eval_sd": within_person_sd,
            "pairs":                 noise_pairs,
        }
        noise_path = base_out / "human_eval_noise.json"
        noise_path.write_text(
            json.dumps(noise_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  Wrote {noise_path}")

    print(f"\nExported {n_exported} eval file(s).")


# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive human MQM evaluation of subtitle translations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Config:   films/output/eval/human-eval/human-<evaluator_id>/config.json  (researcher creates this)\n"
            "Session:  films/output/eval/human-eval/human-<evaluator_id>/session.json\n"
            "Export:   films/output/eval/human-eval/<film>/<trans_model>-by-human-<evaluator_id>/\n\n"
            "Config format:\n"
            '  {"films": {"ivan-vas": ["gpt-5.2", "gpt-5-mini"], "diamond-arm": ["gpt-5.2"]}}\n\n'
            "Issue format (one per line):\n"
            "  <severity> <category> <justification>\n"
            "  severity: ma=major  mi=minor\n"
            "  category: a=accuracy  f=fluency  s=style  t=terminology  o=other\n"
            "  0=no issues   s=skip   b=back to edit previous\n"
        ),
    )
    parser.add_argument(
        "film_names", nargs="*",
        help=(
            "Film directory name(s) to evaluate (e.g. ivan-vas diamond-arm). "
            "Ignored when a config.json exists in the session directory; "
            "required when it does not."
        ),
    )
    parser.add_argument(
        "--evaluator-id", required=True,
        help="Evaluator identifier; seeds the shuffle and names the output directory.",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Max translation runs to sample per method (default: 3).",
    )
    parser.add_argument(
        "--repeat-fraction", type=float, default=0.10,
        help="Fraction of tasks to silently repeat for noise estimation (default: 0.10).",
    )
    parser.add_argument(
        "--methods", default=None,
        help="Comma-separated list of methods to include (default: all).",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export completed judgments to eval files and exit.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    evaluator_id = args.evaluator_id.strip()
    method_filter = (
        set(m.strip() for m in args.methods.split(",") if m.strip())
        if args.methods else None
    )

    # Session directory is always derived from evaluator_id only
    session_dir  = Path("films/output/eval/human-eval") / f"human-{evaluator_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / "session.json"

    # If resuming an existing session, skip setup entirely
    if session_path.exists():
        session = load_session(session_path)
        print(f"Resuming session: {session_path}")

        # Reload data for the films recorded in the session
        films_models_data = _load_films_models_from_session(session)
        if films_models_data is None:
            sys.exit(1)

    else:
        # New session: resolve films + models from config, or from CLI args
        config = load_eval_config(session_dir)

        if config is not None:
            film_models_map = config           # film → [model, ...]
            if args.film_names:
                print(
                    "Note: config.json found; ignoring film_names argument.",
                    file=sys.stderr,
                )
        elif args.film_names:
            # No config: use CLI film names + auto-discovery (researcher convenience mode)
            film_models_map = {}
            for film in args.film_names:
                models = discover_trans_models(film)
                if not models:
                    print(f"Error: no translation files found for film '{film}'.", file=sys.stderr)
                    sys.exit(1)
                film_models_map[film] = models
        else:
            print(
                "Error: no config.json found and no film names provided.\n"
                f"Create {session_dir}/config.json or pass film names on the command line.",
                file=sys.stderr,
            )
            sys.exit(1)

        films_models_data = []
        for film_name, models in film_models_map.items():
            for trans_model in models:
                try:
                    data = load_film_data(film_name, trans_model)
                except FileNotFoundError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)
                films_models_data.append((film_name, trans_model, data))

        seed = hash(evaluator_id) & 0xFFFFFFFF
        rng  = random.Random(seed)
        session = make_session(
            films_models_data, evaluator_id,
            method_filter, args.runs, args.repeat_fraction, rng,
        )
        save_session(session, session_path)
        n_main = sum(1 for t in session["tasks"] if not t["is_repeat"])
        n_rep  = sum(1 for t in session["tasks"] if t["is_repeat"])
        print(f"New session: {n_main} tasks + {n_rep} silent repeats → {session_path}")

    # Build lookup structures from the resolved films_models_data
    film_titles = {}
    items_by_film = {}
    for film_name, trans_model, data in films_models_data:
        if film_name not in film_titles:
            film_titles[film_name] = data.get("title", film_name)
        items_by_film.setdefault(film_name, {})[trans_model] = {
            item["id"]: item for item in data["items"]
        }

    if args.export:
        print(f"\nExporting...")
        export_session(session, items_by_film, session_dir)
        return

    try:
        run_session(session, items_by_film, film_titles, session_path)
    except KeyboardInterrupt:
        print("\n\nSession saved. Run again to continue.")
        return

    # Auto-export when all non-repeat tasks are judged (skipped ones are OK)
    judgments = session["judgments"]
    skipped   = set(session["skipped"])
    all_done  = all(
        str(i) in judgments or i in skipped
        for i, t in enumerate(session["tasks"])
        if not t["is_repeat"]
    )
    if all_done:
        review_inconsistencies(session, items_by_film, film_titles, session_path)

        print(f"\nAll tasks done. Exporting...")
        export_session(session, items_by_film, session_dir)


def _load_films_models_from_session(session):
    """Re-load translation data for all (film, trans_model) pairs recorded in the session."""
    pairs = sorted({(t["film"], t["trans_model"]) for t in session["tasks"]})
    result = []
    for film_name, trans_model in pairs:
        try:
            data = load_film_data(film_name, trans_model)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return None
        result.append((film_name, trans_model, data))
    return result


if __name__ == "__main__":
    main()
