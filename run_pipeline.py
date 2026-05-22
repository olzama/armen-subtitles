#!/usr/bin/env python3
"""Pipeline driver for subtitle translation experiments.

Usage:
    python run_pipeline.py experiments/ivan-vas-russian-galician.yaml --status
    python run_pipeline.py experiments/ivan-vas-russian-galician.yaml --step translate [--parallel]
    python run_pipeline.py experiments/ivan-vas-russian-galician.yaml --step eval
    python run_pipeline.py experiments/ivan-vas-russian-galician.yaml --step aggregate
    python run_pipeline.py experiments/ivan-vas-russian-galician.yaml --step variance
    python run_pipeline.py experiments/ivan-vas-russian-galician.yaml  # runs all non-interactive steps
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
sys.path.insert(0, str(Path(__file__).parent / "code"))
from lang_utils import lang_code


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def translation_dir(cfg, method_name):
    src, tgt = cfg["source_lang"], cfg["target_lang"]
    return Path("films/output/translations") / cfg["film"] / f"{src}-{tgt}" / cfg["trans_model"] / method_name


def translation_status(cfg):
    """Return dict: method_name -> (found_runs, expected_runs)."""
    status = {}
    for m in cfg["methods"]:
        d = translation_dir(cfg, m["name"]) / "translations"
        found = len(list(d.glob("translation-*.txt"))) if d.exists() else 0
        status[m["name"]] = (found, m["n_runs"])
    return status


def mapped_json_path(cfg):
    return Path("films/output/translations") / cfg["film"] / f"{cfg['trans_model']}.json"


def unmapped_translations(cfg):
    """Return dict of method -> list of run IDs present on disk but absent from the mapped JSON."""
    mapped = mapped_json_path(cfg)
    if not mapped.exists():
        # Nothing mapped at all — every existing translation is unmapped
        missing = {}
        for m in cfg["methods"]:
            d = translation_dir(cfg, m["name"]) / "translations"
            runs = [f.stem.split("-")[1] for f in d.glob("translation-*.txt")] if d.exists() else []
            if runs:
                missing[m["name"]] = runs
        return missing

    with open(mapped) as f:
        data = json.load(f)
    items = data.get("items", data) if isinstance(data, dict) else data
    target_lang = lang_code(cfg["target_lang"])

    # Collect mapped run IDs per method from the first item that has them
    mapped_runs = {}
    for item in items:
        for lang, methods in item.get("translations", {}).items():
            if lang.lower() != target_lang:
                continue
            for method, runs in methods.items():
                mapped_runs.setdefault(method, set()).update(str(r) for r in runs)

    missing = {}
    for m in cfg["methods"]:
        d = translation_dir(cfg, m["name"]) / "translations"
        disk_runs = {f.stem.split("-")[1] for f in d.glob("translation-*.txt")} if d.exists() else set()
        absent = disk_runs - mapped_runs.get(m["name"], set())
        if absent:
            missing[m["name"]] = sorted(absent, key=lambda x: int(x))
    return missing


def eval_dir(cfg):
    src, tgt = cfg["source_lang"], cfg["target_lang"]
    combo = f"{cfg['trans_model']}-by-{cfg['eval_model']}"
    prompt_name = Path(cfg["eval_prompt"]).stem
    return Path("films/output/eval/llm-eval") / cfg["film"] / f"{src}-{tgt}" / combo / prompt_name


def eval_status(cfg):
    """Return dict: method_name -> eval_runs_found (across all translation runs)."""
    d = eval_dir(cfg)
    status = {}
    for m in cfg["methods"]:
        method_dir = d / m["name"]
        found = len(list(method_dir.glob("run_*_eval_*.json"))) if method_dir.exists() else 0
        status[m["name"]] = found
    return status


def aggregate_done(cfg):
    return (eval_dir(cfg) / "aggregated_summary.json").exists()


# ---------------------------------------------------------------------------
# Print status
# ---------------------------------------------------------------------------

def print_status(cfg, config_path):
    print(f"\n=== {cfg['film']}  {cfg['source_lang']} -> {cfg['target_lang']}  ({cfg['trans_model']}) ===\n")

    print("STEP 1: Translate")
    t_status = translation_status(cfg)
    all_translated = True
    for name, (found, expected) in t_status.items():
        done = found >= expected
        mark = "ok" if done else "!!"
        print(f"  [{mark}] {name}: {found}/{expected} runs")
        if not done:
            all_translated = False
    print()

    print("STEP 2: Map translations (interactive)")
    missing = unmapped_translations(cfg)
    if missing:
        for method, runs in missing.items():
            print(f"  [!!] {method}: runs {', '.join(runs)} not mapped")
        print(f"       Run: python code/map_translation_segments.py {cfg['film']} {cfg['trans_model']} {cfg['source_lang']} {cfg['target_lang']}")
    else:
        print(f"  [ok] All translations mapped.")
    print()

    print("STEP 3: Evaluate")
    e_status = eval_status(cfg)
    all_evaled = True
    for name, found in e_status.items():
        expected = cfg["eval_runs"] * cfg["methods"][[m["name"] for m in cfg["methods"]].index(name)]["n_runs"]
        done = found >= expected
        mark = "ok" if done else "!!"
        print(f"  [{mark}] {name}: {found}/{expected} eval files")
        if not done:
            all_evaled = False
    print()

    print("STEP 4: Aggregate")
    if aggregate_done(cfg):
        print(f"  [ok] {eval_dir(cfg) / 'aggregated_summary.json'}")
    else:
        print(f"  [!!] aggregated_summary.json not found")
    print()

    print("STEP 5: Variance + extra YAML")
    print(f"  python run_pipeline.py {config_path} --step variance")
    print()


# ---------------------------------------------------------------------------
# Step: translate
# ---------------------------------------------------------------------------

def build_translate_cmd(cfg, method, n_runs=None):
    cmd = [
        "python", "code/translate.py",
        cfg["film"],
        method["name"],
        cfg["trans_model"],
        str(cfg["temperature"]),
        str(n_runs if n_runs is not None else method["n_runs"]),
        cfg["source_lang"],
        cfg["target_lang"],
    ]
    if method.get("prompt"):
        cmd += ["--prompt", method["prompt"]]
    if method.get("summary"):
        cmd += ["--summary", method["summary"]]
    if method.get("unit_list"):
        cmd += ["--unit_list", method["unit_list"]]
    if method.get("given_trans"):
        cmd += ["--given_trans", method["given_trans"]]
    if method.get("lang_prompt"):
        cmd += ["--lang-prompt"]
    return cmd


def run_translate(cfg, only_missing=True, parallel=False):
    t_status = translation_status(cfg)
    to_run = []
    for method in cfg["methods"]:
        name = method["name"]
        found, expected = t_status[name]
        if only_missing and found >= expected:
            print(f"  [skip] {name} ({found}/{expected} runs already exist)")
            continue
        start = found + 1 if only_missing and found > 0 else None
        missing = expected - found if start else expected
        cmd = build_translate_cmd(cfg, method, n_runs=missing)
        if start:
            cmd += ["--start-num", str(start)]
            print(f"  [run] {name}: adding runs {start}-{expected}")
        else:
            print(f"  [run] {name}: {expected} runs")
        print("   $", " ".join(cmd))
        to_run.append((name, cmd))

    if not to_run:
        return

    map_cmd = (f"python code/map_translation_segments.py "
               f"{cfg['film']} {cfg['trans_model']} {cfg['source_lang']} {cfg['target_lang']}")
    print(f"\n  New translations will be produced. You can run mapping in a second terminal now:")
    print(f"    {map_cmd}\n")
    sys.stdout.flush()

    if not parallel:
        for name, cmd in to_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  [ERROR] {name} failed (exit {result.returncode})")
                sys.exit(1)
    else:
        print(f"\n  Launching {len(to_run)} methods in parallel...\n")
        procs = [(name, subprocess.Popen(cmd)) for name, cmd in to_run]
        failed = []
        for name, proc in procs:
            proc.wait()
            if proc.returncode != 0:
                failed.append(name)
        if failed:
            print(f"  [ERROR] These methods failed: {', '.join(failed)}")
            sys.exit(1)
        print(f"\n  All methods completed.")


# ---------------------------------------------------------------------------
# Step: eval
# ---------------------------------------------------------------------------

def run_eval(cfg):
    mapped = mapped_json_path(cfg)
    if not mapped.exists():
        print(f"[ERROR] Mapped JSON not found: {mapped}")
        print(f"Run the interactive mapping step first:")
        print(f"  python code/map_translation_segments.py {cfg['film']} {cfg['trans_model']} {cfg['source_lang']} {cfg['target_lang']}")
        sys.exit(1)

    combo = f"{cfg['trans_model']}-by-{cfg['eval_model']}"
    cmd = [
        "python", "code/evaluate_mqm_parallel.py",
        cfg["film"],
        cfg["source_lang"],
        cfg["target_lang"],
        cfg["trans_model"],
        cfg["eval_model"],
        str(cfg["eval_runs"]),
        cfg["eval_prompt"],
    ]
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Evaluation failed (exit {result.returncode})")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step: aggregate
# ---------------------------------------------------------------------------

def run_aggregate(cfg):
    cmd = [
        "python", "code/aggregate_mqm.py",
        cfg["film"],
        cfg["trans_model"],
        cfg["eval_model"],
        cfg["source_lang"],
        cfg["target_lang"],
        Path(cfg["eval_prompt"]).stem,
    ]
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Aggregation failed (exit {result.returncode})")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step: variance
# ---------------------------------------------------------------------------

def run_variance(cfg):
    cmd = [
        "python", "code/variance.py",
        cfg["film"],
        cfg["trans_model"],
        cfg["eval_model"],
        str(cfg["variance_delta"]),
        cfg["source_lang"],
        cfg["target_lang"],
        Path(cfg["eval_prompt"]).stem,
    ]
    print("  $", " ".join(cmd))
    subprocess.run(cmd)


# ---------------------------------------------------------------------------
# Update n_runs in YAML for methods not meeting delta
# ---------------------------------------------------------------------------

def update_yaml_n_runs(cfg, config_path, max_extra_runs=10):
    comparison_path = eval_dir(cfg) / "method_comparison.json"
    if not comparison_path.exists():
        print("  No method_comparison.json found; skipping n_runs update.")
        return

    with open(comparison_path) as f:
        comparison = json.load(f)

    methods_data = {m["method"]: m for m in comparison.get("methods", [])}

    updated = []
    already_scheduled = []
    for m in cfg["methods"]:
        name = m["name"]
        if name not in methods_data:
            continue
        mdata = methods_data[name]
        sensitivity = mdata.get("sensitivity", {})
        current_T = mdata["num_translations"]
        if sensitivity.get("meets_delta_target", True):
            # Sync n_runs to the T at which target was met, so status reflects reality.
            if current_T > m["n_runs"]:
                updated.append((name, m["n_runs"], current_T, 0))
                m["n_runs"] = current_T
            continue
        min_T = sensitivity.get("min_T_required_at_current_E")
        if min_T is None:
            print(f"  [{name}] does not meet delta but min_T is undetermined; skipping.")
            continue
        additional = min_T - current_T
        if additional <= 0:
            continue
        new_n_runs = current_T + min(additional, max_extra_runs)
        if new_n_runs <= m["n_runs"]:
            already_scheduled.append((name, current_T, m["n_runs"]))
            continue
        updated.append((name, m["n_runs"], new_n_runs, additional))
        m["n_runs"] = new_n_runs

    if not updated and not already_scheduled:
        print("  All methods meet delta target; no updates needed.")
        return

    if already_scheduled:
        print("  Methods below target with n_runs already scheduled (no YAML change needed):")
        for name, current_T, n_runs in already_scheduled:
            print(f"    {name}: T={current_T}, n_runs already set to {n_runs} — run translate + eval to close the gap")

    if not updated:
        return

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n  Updated {config_path}:")
    for name, old, new, additional in updated:
        if additional == 0:
            print(f"    {name}: n_runs {old} -> {new} (synced to actual T, target met)")
        else:
            print(f"    {name}: n_runs {old} -> {new} (+{min(additional, max_extra_runs)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline driver for subtitle translation experiments.")
    parser.add_argument("config", type=Path, help="Path to experiment YAML config")
    parser.add_argument("--status", action="store_true", help="Show pipeline status and exit")
    parser.add_argument("--step", choices=["translate", "map", "eval", "aggregate", "variance"],
                        help="Run a specific pipeline step only")
    parser.add_argument("--parallel", action="store_true",
                        help="Launch all translation methods in parallel (translate step only)")
    parser.add_argument("--max-extra-runs", type=int, default=6,
                        help="Max increase in T per variance cycle per method (default: 6); override to add more aggressively")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.status:
        print_status(cfg, args.config)
        return

    if args.step == "translate":
        print("\n--- Translate ---")
        run_translate(cfg, parallel=args.parallel)
    elif args.step == "map":
        print("\n--- Map (interactive) ---")
        print(f"Run this command manually:")
        print(f"  python code/map_translation_segments.py {cfg['film']} {cfg['trans_model']} {cfg['source_lang']} {cfg['target_lang']}")
    elif args.step == "eval":
        print("\n--- Evaluate ---")
        run_eval(cfg)
    elif args.step == "aggregate":
        print("\n--- Aggregate ---")
        run_aggregate(cfg)
    elif args.step == "variance":
        print("\n--- Variance ---")
        run_variance(cfg)
        print("\n--- Generating extra YAML (if needed) ---")
        update_yaml_n_runs(cfg, args.config, max_extra_runs=args.max_extra_runs)
    else:
        # Run all non-interactive steps, pausing at map
        map_cmd = f"python code/map_translation_segments.py {cfg['film']} {cfg['trans_model']} {cfg['source_lang']} {cfg['target_lang']}"

        print("\n--- Step 1: Translate ---")
        run_translate(cfg, parallel=args.parallel)

        print("\n--- Step 2: Map (interactive — must be done manually) ---")
        missing = unmapped_translations(cfg)
        if missing:
            print(f"  [!!] Unmapped translations (will be skipped by eval until mapped):")
            for method, runs in missing.items():
                print(f"       {method}: runs {', '.join(runs)}")
            print(f"  Run mapping now or in parallel:  {map_cmd}")
        else:
            print(f"  [ok] All translations mapped.")

        print("\n--- Step 3: Evaluate ---")
        run_eval(cfg)

        print("\n--- Step 4: Aggregate ---")
        run_aggregate(cfg)

        print("\n--- Step 5: Variance ---")
        run_variance(cfg)
        print("\n--- Generating extra YAML (if needed) ---")
        update_yaml_n_runs(cfg, args.config, max_extra_runs=args.max_extra_runs)

        if missing:
            print(f"\n  [!!] {sum(len(v) for v in missing.values())} translation(s) were not mapped and excluded from the above results.")
            print(f"  Map them and re-run to include them:")
            print(f"    {map_cmd}")


if __name__ == "__main__":
    main()
