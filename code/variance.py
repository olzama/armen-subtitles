import sys, json, os
import statistics
import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import glob


def load_existing_eval_results(output_dir):
    rows = []

    json_files = glob.glob(os.path.join(output_dir, "*.json"))

    for path in json_files:
        fname = os.path.basename(path)

        if "_eval_" not in fname:
            continue

        translation_id = fname.split("_eval_")[0]
        eval_id = fname.split("_eval_")[1].split(".")[0]

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        score = data["summary"]["major_equiv_per_unit"]

        rows.append({
            "translation": translation_id,
            "eval_run": int(eval_id),
            "score": score
        })

    if not rows:
        raise ValueError("No evaluation JSON files found.")

    return pd.DataFrame(rows)


def estimate_variance_components(df):
    model = smf.mixedlm("score ~ 1", df, groups=df["translation"])
    result = model.fit()

    var_translation = float(result.cov_re.iloc[0, 0])
    var_evaluation = float(result.scale)

    total_var = var_translation + var_evaluation

    return {
        "var_translation": var_translation,
        "var_evaluation": var_evaluation,
        "total_variance": total_var,
        "prop_translation": (var_translation / total_var) if total_var > 0 else 0.0,
        "prop_evaluation": (var_evaluation / total_var) if total_var > 0 else 0.0,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python variance.py <evaluation_folder> <delta>")
        sys.exit(1)

    eval_folder = sys.argv[1]
    delta = float(sys.argv[2])

    df = load_existing_eval_results(eval_folder)

    T = df["translation"].nunique()
    E_current = df.groupby("translation").size().mean()

    print("Loaded data:")
    print(f"  Translations (T): {T}")
    print(f"  Eval runs per translation (E): {E_current:.2f}")
    print(f"  Total observations: {len(df)}")

    vc = estimate_variance_components(df)

    var_trans = vc["var_translation"]
    var_eval = vc["var_evaluation"]

    print("\nEstimated variance components:")
    print(f"  Translation variance: {var_trans:.6f}")
    print(f"  Evaluation variance:  {var_eval:.6f}")
    print(f"  Proportion translation: {vc['prop_translation']:.2%}")
    print(f"  Proportion evaluation:  {vc['prop_evaluation']:.2%}")

    # -------------------------
    # Current precision
    # -------------------------

    var_method = var_trans / T + var_eval / (T * E_current)
    se_method = math.sqrt(var_method)

    se_diff = math.sqrt(2 * var_method)
    ci_diff_halfwidth = 1.96 * se_diff

    print("\nCurrent precision:")
    print(f"  SE(method mean): {se_method:.6f}")
    print(f"  SE(method difference): {se_diff:.6f}")
    print(f"  95% CI half-width for method difference: {ci_diff_halfwidth:.6f}")

    # -------------------------
    # Required E for target delta
    # -------------------------

    # We want:
    # 1.96 * sqrt(2 * (var_trans/T + var_eval/(T*E))) <= delta
    # Solve for E

    term_fixed = var_trans / T
    target_var = (delta / 1.96) ** 2 / 2

    if target_var <= term_fixed:
        print("\nWARNING:")
        print("  Translation variance alone exceeds target precision.")
        print("  Increasing E will not achieve this delta.")
        print("  You must increase T (number of translations).")
        return

    required_E = var_eval / (T * (target_var - term_fixed))
    required_E = math.ceil(required_E)

    print("\nTarget comparison precision:")
    print(f"  Desired detectable delta: {delta}")
    print(f"  Required 95% CI half-width <= {delta}")
    print(f"  Required evaluation runs per translation (E): {required_E}")

    print("\nInterpretation:")
    print(f"  With current design, you can distinguish differences "
          f"larger than about {ci_diff_halfwidth:.3f}.")
    print(f"  To reliably distinguish {delta}, you need E ≈ {required_E}.")


if __name__ == "__main__":
    main()