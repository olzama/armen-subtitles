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
    mapped = mapped_json_path(cfg)
    if mapped.exists():
        print(f"  [ok] {mapped}")
    else:
        print(f"  [!!] {mapped} not found")
        print(f"       Run: python code/map_translation_segments.py {cfg['film']} {cfg['trans_model']} {cfg['source_lang']} {cfg['target_lang']}")
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

def build_translate_cmd(cfg, method):
    cmd = [
        "python", "code/translate.py",
        cfg["film"],
        method["name"],
        cfg["trans_model"],
        str(cfg["temperature"]),
        str(method["n_runs"]),
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
        cmd = build_translate_cmd(cfg, method)
        if start:
            cmd += ["--start-num", str(start)]
            print(f"  [run] {name}: adding runs {start}-{expected}")
        else:
            print(f"  [run] {name}: {expected} runs")
        print("   $", " ".join(cmd))
        to_run.append((name, cmd))

    if not to_run:
        return

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
# Step: generate extra YAML for methods not meeting delta
# ---------------------------------------------------------------------------

def generate_extra_yaml(cfg, config_path, max_extra_runs=10):
    comparison_path = eval_dir(cfg) / "method_comparison.json"
    if not comparison_path.exists():
        print("  No method_comparison.json found; skipping extra YAML generation.")
        return

    with open(comparison_path) as f:
        comparison = json.load(f)

    methods_data = {m["method"]: m for m in comparison.get("methods", [])}
    method_configs = {m["name"]: m for m in cfg["methods"]}

    extra_methods = []
    for name, mdata in methods_data.items():
        sensitivity = mdata.get("sensitivity", {})
        if sensitivity.get("meets_delta_target", True):
            continue
        min_T = sensitivity.get("min_T_required_at_current_E")
        if min_T is None:
            print(f"  [{name}] does not meet delta but min_T is undetermined; skipping.")
            continue
        current_T = mdata["num_translations"]
        additional = min_T - current_T
        if additional <= 0:
            continue
        if name not in method_configs:
            print(f"  [{name}] not found in config methods; skipping.")
            continue
        extra_method = dict(method_configs[name])
        extra_method["n_runs"] = current_T + min(additional, max_extra_runs)
        extra_methods.append(extra_method)

    if not extra_methods:
        print("  All methods meet delta target; no extra YAML needed.")
        return

    base = Path(config_path).stem
    parent = Path(config_path).parent
    extra_path = parent / f"{base}-extra.yaml"

    extra_cfg = {k: v for k, v in cfg.items() if k != "methods"}
    extra_cfg["methods"] = extra_methods

    with open(extra_path, "w") as f:
        yaml.dump(extra_cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n  Extra YAML written: {extra_path}")
    for m, mdata in zip(extra_methods, [methods_data[m["name"]] for m in extra_methods]):
        additional = m["n_runs"] - mdata["num_translations"]
        print(f"    {m['name']}: {additional} additional run(s) (total n_runs: {m['n_runs']})")


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
    parser.add_argument("--max-extra-runs", type=int, default=10,
                        help="Cap on additional translation runs per method in the extra YAML (default: 10)")
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
        generate_extra_yaml(cfg, args.config, max_extra_runs=args.max_extra_runs)
    else:
        # Run all non-interactive steps, pausing at map
        print("\n--- Step 1: Translate ---")
        run_translate(cfg, parallel=args.parallel)

        print("\n--- Step 2: Map (interactive — must be done manually) ---")
        mapped = mapped_json_path(cfg)
        if not mapped.exists():
            print(f"  Mapped JSON not found. Run:")
            print(f"    python code/map_translation_segments.py {cfg['film']} {cfg['trans_model']} {cfg['source_lang']} {cfg['target_lang']}")
            print(f"  Then re-run this script to continue with eval/aggregate.")
            return
        else:
            print(f"  [ok] {mapped} exists, skipping.")

        print("\n--- Step 3: Evaluate ---")
        run_eval(cfg)

        print("\n--- Step 4: Aggregate ---")
        run_aggregate(cfg)

        print("\n--- Step 5: Variance ---")
        run_variance(cfg)
        print("\n--- Generating extra YAML (if needed) ---")
        generate_extra_yaml(cfg, args.config, max_extra_runs=args.max_extra_runs)


if __name__ == "__main__":
    main()
