"""
notebooks/anomaly_detection/pipeline_1_2_3/wire_temporal_window_and_risk_factors.py

Wires the two notebooks/decision_support subfolders into the actual
DEPLOYED pipeline (the calibrated plain and fair RISK_SCORE from section
5c), instead of leaving them as standalone historical notebooks scored
against final.csv's own bundled RISK_SCORE.

PART A -- completes temporal_window/ (Track 1, the cost model / interval
policies). Sections 5d-5e-8 of this README already wired in
snapshot_adaptive (recommend_interval, used everywhere) and
trajectory_adaptive (recommend_interval_trajectory, section 8). This adds
fixed_policy (1_fixed_policy.ipynb: the same interval for everyone, no
patient information used) so all three of temporal_window's
patient-information policies are compared side by side, on the actual
deployed scores. (forecast_adaptive, 4_forecast_adaptive.ipynb, is left
out -- it needs CONVERSION_PROB_3M/6M/12M multi-horizon forecast columns
this pipeline's fallback cascade never produces; building those would be
a genuinely new stage-2 extension, not just "wiring in" an existing one.)

PART B -- wires in risk_factors/ (Track 3, attribution). 1_global.ipynb's
own method -- Lasso (signed, linear) and Random Forest (feature_importances_)
regressed against RISK_SCORE, plus SHAP on the forest -- reused here
(du.fit_lasso_baseline, du.fit_forest_baseline, du.compute_shap_importance,
all already in decision_util.py) but applied to the DEPLOYED plain and
fair calibrated RISK_SCORE, not the notebook's own historical one. This
directly extends the proxy-discrimination question explainability_fairness.py
already asked about the OLD constant-shift fairness fix (Part 3: "does the
correction change the model's reasoning?") to the actual deployed
ExponentiatedGradient mechanism: does the fair model lean on a genuinely
different set of biomarkers than the plain one, at the population level?

Usage:
    python3 wire_temporal_window_and_risk_factors.py
Outputs (written next to this script):
    - wire_temporal_window_and_risk_factors_report.txt
    - risk_factors_attribution_plain_vs_fair.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fairness_expgrad_pipeline as fep  # noqa: E402

REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402

RATIO = 20.0
MENU = (3, 6, 12)
SAFE_INTERVAL_MONTHS = 6
PROTECTED_DOMAIN = lambda df: {"PTGENDER": (1, 2), "PTEDUCAT_BUCKET": (0, 1),
                                "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique()))}


def policy_metrics(df, risk_col, recommended, label, lines):
    cmodel = du.ConversionCostModel(check_cost=1.0, missed_conversion_cost=RATIO,
                                     safe_interval_months=SAFE_INTERVAL_MONTHS,
                                     reference_interval_months=12.0)
    total_cost, _, _ = cmodel.cost(rid_ids=df["RID"].values, risk_scores=df[risk_col].values,
                                    threshold=0.5, interval_months=recommended, return_margin=False)
    didi = du.compute_didi(df, recommended, PROTECTED_DOMAIN(df))
    outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
    outcomes["RECOMMENDED_INTERVAL"] = recommended
    worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
    margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
    catch_rate = (margin >= 0).mean() * 100
    lines.append(f"  {label:38s} cost={total_cost:9.1f}  DIDI={didi:.3f}  catch={catch_rate:5.1f}%")


def part_a_fixed_policy(lines):
    lines.append("=== PART A: temporal_window/1_fixed_policy.ipynb, wired to the deployed scores ===")
    lines.append(f"Fixed policy: every patient gets the same {SAFE_INTERVAL_MONTHS}-month interval, no")
    lines.append("patient information used at all -- the zero-information floor DIDI is 0 by")
    lines.append("construction; cost/catch are what a completely undifferentiated policy costs.")
    lines.append("")
    df = pd.read_csv(os.path.join(HERE, "calibrated_risk_scores.csv"))
    df["EXAMDATE_DX"] = pd.to_datetime(df["EXAMDATE_DX"])

    for score_col, label in [("RISK_SCORE_plain_cal", "Plain (no fairness), calibrated"),
                              ("RISK_SCORE_fair_cal", "Fair (EG), calibrated")]:
        lines.append(f"-- {label} --")
        fixed_rec = np.full(len(df), float(SAFE_INTERVAL_MONTHS))
        snapshot_rec = du.recommend_interval(df[score_col].values, interval_menu_months=MENU,
                                              check_cost=1.0, missed_conversion_cost=RATIO,
                                              reference_interval_months=12.0)
        traj = du.compute_risk_trajectory(df, id_col="RID", date_col="EXAMDATE_DX", risk_col=score_col)
        traj_rec = du.recommend_interval_trajectory(traj[score_col].values, traj["RISK_SLOPE"].values,
                                                      interval_menu_months=MENU, check_cost=1.0,
                                                      missed_conversion_cost=RATIO, reference_interval_months=12.0)
        policy_metrics(df, score_col, fixed_rec, "Fixed (1_fixed_policy)", lines)
        policy_metrics(df, score_col, snapshot_rec, "Snapshot adaptive (2_snapshot_adaptive)", lines)
        policy_metrics(traj, score_col, traj_rec, "Trajectory adaptive (3_trajectory_adaptive)", lines)
        lines.append("")
    lines.append("All three of temporal_window's patient-information policies are now wired to")
    lines.append("the actual deployed pipeline (forecast_adaptive excluded -- needs multi-horizon")
    lines.append("forecast columns this pipeline doesn't produce). Read the progression fixed ->")
    lines.append("snapshot -> trajectory as the real, cumulative value of each successive piece of")
    lines.append("patient information: none, current risk level, current risk level plus direction.")
    lines.append("")


def part_b_attribution(lines):
    lines.append("=== PART B: risk_factors/1_global.ipynb's method, wired to the deployed scores ===")
    lines.append("Lasso (signed, linear) and Random Forest (feature_importances_) regressed")
    lines.append("against RISK_SCORE, plus SHAP on the forest -- three vantage points on 'what")
    lines.append("drives this score, at the population level', applied to BOTH the plain and the")
    lines.append("fair calibrated score, to see whether the fairness mechanism leans on a")
    lines.append("genuinely different set of biomarkers, not just a shifted output.")
    lines.append("")

    scores = pd.read_csv(os.path.join(HERE, "calibrated_risk_scores.csv"))
    scores["EXAMDATE_DX"] = pd.to_datetime(scores["EXAMDATE_DX"])
    panel = fep.load_panel()
    feature_cols = fep.EXTENDED_A
    panel_feats = panel[["RID", "EXAMDATE_DX"] + feature_cols].copy()
    panel_feats["EXAMDATE_DX"] = pd.to_datetime(panel_feats["EXAMDATE_DX"])
    panel_feats = panel_feats.drop_duplicates(subset=["RID", "EXAMDATE_DX"])
    df = scores.merge(panel_feats, on=["RID", "EXAMDATE_DX"], how="inner").dropna(subset=feature_cols)
    lines.append(f"n={len(df)} rows with the full extended+anomaly feature set available.")
    lines.append("")

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (score_col, label) in zip(axes, [("RISK_SCORE_plain_cal", "Plain, calibrated"),
                                              ("RISK_SCORE_fair_cal", "Fair (EG), calibrated")]):
        lasso_model, lasso_scaler = du.fit_lasso_baseline(df[feature_cols], df[score_col], alpha=0.01)
        lasso_r2 = du.evaluate_regression(lasso_model, df[feature_cols], df[score_col], scaler=lasso_scaler)
        weights = du.top_lasso_weights(lasso_model, feature_cols, top_n=len(feature_cols))

        forest_model = du.fit_forest_baseline(df[feature_cols], df[score_col], n_estimators=200, max_depth=4)
        forest_r2 = du.evaluate_regression(forest_model, df[feature_cols], df[score_col])
        forest_importances = pd.Series(forest_model.feature_importances_, index=feature_cols) \
            .sort_values(ascending=False)
        shap_sample = df[feature_cols].sample(n=min(300, len(df)), random_state=42)
        mean_abs_shap, _ = du.compute_shap_importance(forest_model, shap_sample, feature_cols)
        shap_series = mean_abs_shap.set_index("feature")["mean_abs_shap"].sort_values(ascending=False)

        lines.append(f"-- {label} -- Lasso R2={lasso_r2['r2']:.3f} MAE={lasso_r2['mae']:.3f}  |  "
                     f"Forest R2={forest_r2['r2']:.3f} MAE={forest_r2['mae']:.3f}")
        lines.append("   Lasso weights (signed, standardized) -- low R2 above means these are")
        lines.append("   near-meaningless here, shown for completeness only:")
        lines.append("   " + weights.to_string(index=False, float_format=lambda v: f"{v:+.4f}").replace("\n", "\n   "))
        lines.append("   Forest importances:")
        lines.append("   " + forest_importances.to_string(float_format=lambda v: f"{v:.4f}").replace("\n", "\n   "))
        lines.append("   SHAP mean |value| (on the same forest, 300-row sample):")
        lines.append("   " + shap_series.to_string(float_format=lambda v: f"{v:.4f}").replace("\n", "\n   "))
        lines.append("")
        results[score_col] = {"lasso": weights.set_index("feature")["weight"],
                               "forest": forest_importances, "shap": shap_series}

        ax.barh(forest_importances.index, forest_importances.values, color="#4C72B0")
        ax.invert_yaxis()
        ax.set_title(f"{label}\nForest feature importance (R2={forest_r2['r2']:.2f})")
        ax.set_xlabel("Importance")

    plt.suptitle("risk_factors/1_global.ipynb wired to the deployed pipeline:\n"
                 "what drives the plain vs. the fair calibrated RISK_SCORE")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(HERE, "risk_factors_attribution_plain_vs_fair.png"), dpi=150)
    plt.close()

    forest_cmp = pd.DataFrame({
        "plain": results["RISK_SCORE_plain_cal"]["forest"],
        "fair": results["RISK_SCORE_fair_cal"]["forest"],
    })
    forest_cmp["delta"] = forest_cmp["fair"] - forest_cmp["plain"]
    lines.append("-- Forest importance delta (fair minus plain) --")
    lines.append(forest_cmp.sort_values("delta", ascending=False)
                 .to_string(float_format=lambda v: f"{v:+.4f}"))
    lines.append("")
    top_shift = forest_cmp["delta"].abs().idxmax()
    lines.append(f"Largest shift in what drives the score: {top_shift} "
                 f"({forest_cmp.loc[top_shift, 'delta']:+.4f}).")
    lines.append("If PTAU/TAU/AMYGDALA_NORM-style biomarkers stay dominant for both and the")
    lines.append("shifts are small, that confirms explainability_fairness.py's earlier finding")
    lines.append("(with the OLD constant-shift fix) still holds for the ACTUAL deployed")
    lines.append("ExponentiatedGradient mechanism too: fairness correction changes the OUTPUT")
    lines.append("distribution, not which biomarkers the score is actually built from. A large")
    lines.append("shift would be a genuinely new finding worth flagging -- it would mean the fair")
    lines.append("model isn't just a recalibrated version of the plain one, it reasons differently.")


def main():
    lines = []
    part_a_fixed_policy(lines)
    part_b_attribution(lines)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "wire_temporal_window_and_risk_factors_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
