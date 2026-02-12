#!/usr/bin/env python3
"""
Compare two folders of eval JSON (.txt) files that contain repeated runs.

Each eval file is expected to be JSON with shape:
{
  "items": [
    {"id": <int|str>, "issues": [{"severity": "...", "category": "..."}, ...], ...},
    ...
  ],
  "summary": {...}  # ignored
}

This script aggregates per-folder distributions over:
- severity
- category
- (severity, category) joint

Then compares folder A vs folder B per item id and reports:
- majority severity/category changes
- Jensen–Shannon divergence (severity/category/joint)
- delta in expected penalty points per run (based on severity weights)

Outputs:
- differences.csv (per-id comparison)
- summary.json (overall summary + top differing ids)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any


SEVERITY_WEIGHTS = {
    "no-issue": 0,
    "minor": 1,
    "major": 5,
    "critical": 10,
}


def safe_read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON in {path}: {e}") from e


def list_eval_files(folder: str, pattern: str) -> List[str]:
    folder = os.path.abspath(folder)
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    return [p for p in paths if os.path.isfile(p)]


def normalize_id(x: Any) -> str:
    # Normalize ids to strings to avoid 1 vs "1" mismatches
    return str(x)


def majority_label(counter: Counter, tie_breaker: str = "lex") -> Optional[str]:
    """
    Returns the most common label. If ties:
      - tie_breaker="lex": pick lexicographically smallest label for determinism.
      - tie_breaker="none": return None on tie.
    """
    if not counter:
        return None
    most = counter.most_common()
    top_count = most[0][1]
    tied = [lbl for lbl, c in most if c == top_count]
    if len(tied) == 1:
        return tied[0]
    if tie_breaker == "none":
        return None
    return sorted(tied)[0]


def normalize_dist(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def kl_div(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    """KL(P||Q) for discrete distributions represented as dicts; uses epsilon smoothing on Q only."""
    s = 0.0
    for k, pv in p.items():
        if pv <= 0:
            continue
        qv = q.get(k, 0.0)
        qv = max(qv, eps)
        s += pv * math.log(pv / qv, 2)
    return s


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen–Shannon divergence in bits, between 0 and 1 (for base-2 logs) if supports match."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    # Ensure complete support
    p2 = {k: p.get(k, 0.0) for k in keys}
    q2 = {k: q.get(k, 0.0) for k in keys}
    m = {k: 0.5 * (p2[k] + q2[k]) for k in keys}
    return 0.5 * kl_div(p2, m) + 0.5 * kl_div(q2, m)


def expected_penalty_per_run(sev_counter: Counter, n_runs: int) -> float:
    """
    Expected penalty points per run for an item, computed from aggregated severity counts across runs.
    If n_runs==0 -> 0.
    """
    if n_runs <= 0:
        return 0.0
    total = 0.0
    for sev, cnt in sev_counter.items():
        w = SEVERITY_WEIGHTS.get(sev, 0)
        total += w * cnt
    # cnt is aggregated across runs, so divide by runs
    return total / n_runs


@dataclass
class FolderAgg:
    folder: str
    n_files: int
    # per id
    severity: Dict[str, Counter]
    category: Dict[str, Counter]
    joint: Dict[str, Counter]  # keys are "severity|category"


def parse_eval_file(path: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns: {id -> list of (severity, category) issues in that file}
    """
    data = safe_read_json(path)
    items = data.get("items", [])
    out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for it in items:
        if "id" not in it:
            continue
        _id = normalize_id(it["id"])
        issues = it.get("issues", []) or []
        for iss in issues:
            sev = (iss.get("severity") or "").strip()
            cat = (iss.get("category") or "").strip()
            if not sev or not cat:
                continue
            out[_id].append((sev, cat))
    return out


def aggregate_folder(folder: str, pattern: str) -> FolderAgg:
    paths = list_eval_files(folder, pattern)
    sev_map: Dict[str, Counter] = defaultdict(Counter)
    cat_map: Dict[str, Counter] = defaultdict(Counter)
    joint_map: Dict[str, Counter] = defaultdict(Counter)

    for p in paths:
        per_file = parse_eval_file(p)  # {id -> [(sev,cat),...]}
        # Note: if an id is absent in a run, we simply have no observations for it in that file.
        for _id, pairs in per_file.items():
            for sev, cat in pairs:
                sev_map[_id][sev] += 1
                cat_map[_id][cat] += 1
                joint_map[_id][f"{sev}|{cat}"] += 1

    return FolderAgg(
        folder=os.path.abspath(folder),
        n_files=len(paths),
        severity=sev_map,
        category=cat_map,
        joint=joint_map,
    )


def compare_folders(
    A: FolderAgg,
    B: FolderAgg,
    js_threshold: float,
    tie_breaker: str,
) -> Tuple[List[dict], dict]:
    ids = sorted(set(A.severity.keys()) | set(B.severity.keys()) |
                 set(A.category.keys()) | set(B.category.keys()) |
                 set(A.joint.keys()) | set(B.joint.keys()))

    rows: List[dict] = []

    changed_sev = 0
    changed_cat = 0
    flagged = 0

    js_sev_vals = []
    js_cat_vals = []
    js_joint_vals = []

    for _id in ids:
        sevA = A.severity.get(_id, Counter())
        sevB = B.severity.get(_id, Counter())
        catA = A.category.get(_id, Counter())
        catB = B.category.get(_id, Counter())
        jA = A.joint.get(_id, Counter())
        jB = B.joint.get(_id, Counter())

        maj_sev_A = majority_label(sevA, tie_breaker=tie_breaker)
        maj_sev_B = majority_label(sevB, tie_breaker=tie_breaker)
        maj_cat_A = majority_label(catA, tie_breaker=tie_breaker)
        maj_cat_B = majority_label(catB, tie_breaker=tie_breaker)

        sev_dist_A = normalize_dist(sevA)
        sev_dist_B = normalize_dist(sevB)
        cat_dist_A = normalize_dist(catA)
        cat_dist_B = normalize_dist(catB)
        joint_dist_A = normalize_dist(jA)
        joint_dist_B = normalize_dist(jB)

        js_sev = js_divergence(sev_dist_A, sev_dist_B)
        js_cat = js_divergence(cat_dist_A, cat_dist_B)
        js_joint = js_divergence(joint_dist_A, joint_dist_B)

        js_sev_vals.append(js_sev)
        js_cat_vals.append(js_cat)
        js_joint_vals.append(js_joint)

        exp_pen_A = expected_penalty_per_run(sevA, A.n_files) if A.n_files else 0.0
        exp_pen_B = expected_penalty_per_run(sevB, B.n_files) if B.n_files else 0.0
        delta_pen = exp_pen_A - exp_pen_B

        sev_change = (maj_sev_A != maj_sev_B) and (maj_sev_A is not None or maj_sev_B is not None)
        cat_change = (maj_cat_A != maj_cat_B) and (maj_cat_A is not None or maj_cat_B is not None)

        if sev_change:
            changed_sev += 1
        if cat_change:
            changed_cat += 1

        is_flagged = sev_change or cat_change or (js_joint >= js_threshold) or (js_sev >= js_threshold) or (js_cat >= js_threshold)
        if is_flagged:
            flagged += 1

        row = {
            "id": _id,
            "A_majority_severity": maj_sev_A,
            "B_majority_severity": maj_sev_B,
            "A_majority_category": maj_cat_A,
            "B_majority_category": maj_cat_B,
            "A_total_issues": sum(sevA.values()),
            "B_total_issues": sum(sevB.values()),
            "A_expected_penalty_per_run": round(exp_pen_A, 6),
            "B_expected_penalty_per_run": round(exp_pen_B, 6),
            "delta_expected_penalty_per_run_A_minus_B": round(delta_pen, 6),
            "JS_severity": round(js_sev, 6),
            "JS_category": round(js_cat, 6),
            "JS_joint": round(js_joint, 6),
            "severity_changed": int(sev_change),
            "category_changed": int(cat_change),
            "flagged": int(is_flagged),
            # Compact distributions for inspection/debugging
            "A_severity_dist": json.dumps(sev_dist_A, ensure_ascii=False, sort_keys=True),
            "B_severity_dist": json.dumps(sev_dist_B, ensure_ascii=False, sort_keys=True),
            "A_category_dist": json.dumps(cat_dist_A, ensure_ascii=False, sort_keys=True),
            "B_category_dist": json.dumps(cat_dist_B, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)

    # Sort: most different first (joint divergence, then penalty delta magnitude)
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["flagged"], r["JS_joint"], abs(r["delta_expected_penalty_per_run_A_minus_B"])),
        reverse=True,
    )

    def mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    summary = {
        "folder_A": A.folder,
        "folder_B": B.folder,
        "n_files_A": A.n_files,
        "n_files_B": B.n_files,
        "n_ids_union": len(ids),
        "n_ids_flagged": flagged,
        "n_ids_majority_severity_changed": changed_sev,
        "n_ids_majority_category_changed": changed_cat,
        "mean_JS_severity": round(mean(js_sev_vals), 6),
        "mean_JS_category": round(mean(js_cat_vals), 6),
        "mean_JS_joint": round(mean(js_joint_vals), 6),
        "js_threshold": js_threshold,
        "tie_breaker": tie_breaker,
        "top_20_ids_by_JS_joint": [
            {"id": r["id"], "JS_joint": r["JS_joint"], "severity_changed": r["severity_changed"], "category_changed": r["category_changed"]}
            for r in rows_sorted[:20]
        ],
    }

    return rows_sorted, summary


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two folders of repeated-run eval JSON files.")
    ap.add_argument("--a", required=True, help="Folder A path (first set of runs).")
    ap.add_argument("--b", required=True, help="Folder B path (second set of runs).")
    ap.add_argument("--pattern", default="eval-*.txt", help="Glob pattern for eval files inside each folder.")
    ap.add_argument("--out_csv", default="differences.csv", help="Output CSV path.")
    ap.add_argument("--out_json", default="summary.json", help="Output summary JSON path.")
    ap.add_argument("--js_threshold", type=float, default=0.10, help="Flag an item if JS divergence >= threshold.")
    ap.add_argument(
        "--tie_breaker",
        choices=["lex", "none"],
        default="lex",
        help="How to handle ties for majority labels. 'lex' is deterministic; 'none' yields None on ties.",
    )
    args = ap.parse_args()

    A = aggregate_folder(args.a, args.pattern)
    B = aggregate_folder(args.b, args.pattern)

    if A.n_files == 0:
        raise SystemExit(f"No files matched pattern '{args.pattern}' in folder A: {A.folder}")
    if B.n_files == 0:
        raise SystemExit(f"No files matched pattern '{args.pattern}' in folder B: {B.folder}")

    rows, summary = compare_folders(A, B, js_threshold=args.js_threshold, tie_breaker=args.tie_breaker)

    write_csv(args.out_csv, rows)
    write_json(args.out_json, summary)

    # Console summary (compact)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {os.path.abspath(args.out_csv)}")
    print(f"Wrote: {os.path.abspath(args.out_json)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
