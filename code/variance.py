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

        # Expect format: translation-8.txt_eval_3.json
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

def recommend_eval_runs(var_evaluation, target_se=0.3):
    # SE = sqrt(var_evaluation / E)
    # Solve for E
    E = var_evaluation / (target_se ** 2)
    return math.ceil(E)

def recommend_E_by_diminishing_returns(var_eval, r=0.05, E_max=50):
    # r = 0.05 means "stop when adding one more run improves SE by < 5%"
    if var_eval <= 0:
        return 1

    def se(E):
        return math.sqrt(var_eval / E)

    for E in range(1, E_max):
        rel_improve = (se(E) - se(E + 1)) / se(E)
        if rel_improve < r:
            return E
    return E_max

def variance_components_anova(df):
    # df: columns translation, score (major_equiv_per_unit)
    g = df.groupby("translation")["score"]
    means = g.mean().to_numpy()
    vars_within = g.var(ddof=1).to_numpy()
    ns = g.size().to_numpy()

    E = float(np.mean(ns))  # assumes roughly balanced
    sigma2_eval = float(np.mean(vars_within))

    # variance of translation means includes eval noise / E
    sigma2_trans = float(np.var(means, ddof=1) - sigma2_eval / E)
    if sigma2_trans < 0:
        sigma2_trans = 0.0

    return sigma2_trans, sigma2_eval, E

def main():
    if len(sys.argv) < 2:
        print("Usage: python variance_analysis.py <evaluation_folder> [target_se]")
        sys.exit(1)

    eval_folder = sys.argv[1]

    if not os.path.isdir(eval_folder):
        raise ValueError("Provided path is not a folder.")

    target_se = float(sys.argv[2]) if len(sys.argv) > 2 else None

    print("Loading evaluation results...")
    df = load_existing_eval_results(eval_folder)

    print(f"Loaded {len(df)} evaluation observations.")
    print(f"Translations: {df['translation'].nunique()}")
    print(f"Evaluation runs per translation (avg): "
          f"{df.groupby('translation').size().mean():.2f}")

    sigma2_trans_a, sigma2_eval_a, E = variance_components_anova(df)
    print("\n=== ANOVA variance components (check) ===")
    print(f"Translation variance: {sigma2_trans_a:.4f}")
    print(f"Evaluation variance:  {sigma2_eval_a:.4f}")

    print("\nEstimating variance components...")
    vc = estimate_variance_components(df)

    print("\n=== Variance Decomposition ===")
    print(f"Translation variance: {vc['var_translation']:.4f}")
    print(f"Evaluation variance:  {vc['var_evaluation']:.4f}")
    print(f"Total variance:       {vc['total_variance']:.4f}")
    print(f"Proportion translation: {vc['prop_translation']:.2%}")
    print(f"Proportion evaluation:  {vc['prop_evaluation']:.2%}")

    # Average number of evaluation runs per translation
    E_current = df.groupby("translation").size().mean()
    var_eval = vc["var_evaluation"]
    se_mean = math.sqrt(var_eval / E_current)
    print("\n=== Current Precision of Mean Evaluation Score ===")
    print(f"Evaluation runs per translation (avg): {E_current:.2f}")
    print(f"SE(mean) [major_equiv_per_unit]: {se_mean:.4f}")
    print(f"Approx. 95% CI half-width: {1.96 * se_mean:.4f}")

    if target_se is not None:
        recommended_E = recommend_eval_runs(
            vc["var_evaluation"],
            target_se=target_se
        )
        print("\n=== Evaluation Replication Recommendation (Target SE) ===")
        print(f"Target SE(mean): {target_se}")
        print(f"Recommended eval runs per translation: {recommended_E}")

    else:
        recommended_E_dr = recommend_E_by_diminishing_returns(var_eval, r=0.05)
        print("\n=== Evaluation Replication Recommendation (Diminishing Returns) ===")
        print("Stop when adding one more run improves SE by < 5%")
        print(f"Recommended eval runs per translation: {recommended_E_dr}")


if __name__ == "__main__":
    main()