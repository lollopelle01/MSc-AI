"""
notebooks/anomaly_detection/pipeline_1_2_3/fairness_expgrad_equalized_odds.py

Section 9 found a real problem with the deployed fairness mechanism
(ExponentiatedGradient under DemographicParity): it appears to satisfy the
constraint partly by leaning on NOMINAL_MONTH (visit timing) instead of
genuine biomarkers, since DemographicParity only asks for equal AVERAGE
outcomes across groups, by whatever means gets there cheapest.

This tries the concrete fix proposed in response: refit under
EqualizedOdds instead. EqualizedOdds requires equal true-positive AND
false-positive rates PER GROUP, conditioned on the real label
(EVENT_AT_VISIT) -- a constraint that can't be satisfied by leaning on a
feature uncorrelated with genuine risk, because doing so would hurt
accuracy for a group's real positives and negatives alike, not just shift
an average. Everything else (base estimator, cost-aligned objective,
fallback cascade, Platt calibration) stays identical to
calibrate_fair_and_plain_risk_scores.py, so this is a clean, one-variable
comparison: DemographicParity vs EqualizedOdds, same everything else.

Usage:
    python3 fairness_expgrad_equalized_odds.py
Outputs (written next to this script):
    - fairness_expgrad_equalized_odds_report.txt
    - equalized_odds_attribution_comparison.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, ErrorRate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fairness_expgrad_pipeline as fep  # noqa: E402
from calibration_and_group_accuracy_check import per_group_metrics  # noqa: E402

REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402

FEATURE_SETS = {"core": fep.CORE, "extended": fep.EXTENDED,
                 "core+anomaly": fep.CORE_A, "extended+anomaly": fep.EXTENDED_A}
FALLBACK_ORDER = [("extended+anomaly", "expgrad"), ("extended", "expgrad"),
                   ("core+anomaly", "expgrad"), ("core", "expgrad"), ("core", "logistic")]
RATIO = fep.MISSED_CONVERSION_COST  # 20.0, unchanged -- only the constraint changes


def fit_expgrad_eo(train_df, cols, protected_cols):
    """Exact copy of fep.fit_expgrad's structure -- same base estimator, same
    cost-aligned objective -- with EqualizedOdds swapped in for
    DemographicParity, the one variable this script changes."""
    from sklearn.ensemble import RandomForestClassifier
    base_est = RandomForestClassifier(n_estimators=fep.EG_N_ESTIMATORS, max_depth=6,
                                       class_weight="balanced", random_state=42)
    eg = ExponentiatedGradient(
        estimator=base_est, constraints=EqualizedOdds(),
        objective=ErrorRate(costs={"fp": 1.0, "fn": fep.MISSED_CONVERSION_COST / fep.CHECK_COST}),
        eps=fep.EG_EPS, max_iter=fep.EG_MAX_ITER,
    )
    eg.fit(train_df[cols], train_df["EVENT_AT_VISIT"], sensitive_features=train_df[protected_cols])
    return fep.ExpGradWrapper(eg, cols)


def fit_logistic_fallback(fit_df):
    train_core = fep.at_risk_complete(fit_df, fep.CORE)
    scaler = StandardScaler().fit(train_core[fep.CORE])
    logistic = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                   class_weight="balanced", random_state=42)
    logistic.fit(scaler.transform(train_core[fep.CORE]), train_core["EVENT_AT_VISIT"])
    return scaler, logistic


def build_eo_cascade(fit_df, apply_dfs, score_col):
    fitted = {}
    for tier_name, cols in FEATURE_SETS.items():
        train_df = fep.at_risk_complete(fit_df, cols + fep.PROTECTED)
        if len(train_df) < 30:
            continue
        fitted[tier_name] = fit_expgrad_eo(train_df, cols, fep.PROTECTED)
    scaler, logistic = fit_logistic_fallback(fit_df)

    outputs = []
    for apply_df in apply_dfs:
        df = apply_df.copy()
        df[score_col] = np.nan
        remaining = df[score_col].isna()
        for tier_name, kind in FALLBACK_ORDER:
            cols = FEATURE_SETS.get(tier_name, fep.CORE)
            eligible = remaining & df[cols].notna().all(axis=1)
            if not eligible.any():
                continue
            X = df.loc[eligible, cols]
            if kind == "logistic":
                proba = logistic.predict_proba(scaler.transform(X))[:, 1]
            else:
                proba = fitted[tier_name].predict_proba(X)[:, 1]
            df.loc[eligible, score_col] = proba
            remaining = df[score_col].isna()
        outputs.append(df.dropna(subset=[score_col]).copy())
    return outputs


def platt_calibrate(calib_df, test_df, raw_col, calibrated_col):
    lr = LogisticRegression()
    lr.fit(calib_df[[raw_col]].to_numpy(), calib_df["EVENT_AT_VISIT"].to_numpy())
    test_df = test_df.copy()
    test_df[calibrated_col] = lr.predict_proba(test_df[[raw_col]].to_numpy())[:, 1]
    return test_df


def policy_metrics(df, risk_col, recommended, protected, label, lines):
    cmodel = du.ConversionCostModel(check_cost=1.0, missed_conversion_cost=RATIO,
                                     safe_interval_months=6, reference_interval_months=12.0)
    total_cost, _, _ = cmodel.cost(rid_ids=df["RID"].values, risk_scores=df[risk_col].values,
                                    threshold=0.5, interval_months=recommended, return_margin=False)
    didi = du.compute_didi(df, recommended, protected)
    outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
    outcomes["RECOMMENDED_INTERVAL"] = recommended
    worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
    margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
    catch_rate = (margin >= 0).mean() * 100
    auc = roc_auc_score(df["EVENT_AT_VISIT"], df[risk_col])
    lines.append(f"  {label:34s} cost={total_cost:9.1f}  DIDI={didi:.3f}  catch={catch_rate:5.1f}%  AUC={auc:.3f}")


def main():
    panel = fep.load_panel()
    train_panel, test_panel = du.subject_train_test_split(panel, test_fraction=0.25, random_state=42)
    fit_subset, calib_subset = du.subject_train_test_split(train_panel, test_fraction=0.2, random_state=1)

    lines = []
    lines.append("=== ExponentiatedGradient under EqualizedOdds vs. DemographicParity ===")
    lines.append("Same base estimator, same cost-aligned objective, same fallback cascade,")
    lines.append("same Platt calibration -- only the fairness constraint changes.")
    lines.append("")

    calib_eo, test_eo = build_eo_cascade(fit_subset, [calib_subset, test_panel], "RISK_SCORE_eo")
    test_eo = platt_calibrate(calib_eo, test_eo, "RISK_SCORE_eo", "RISK_SCORE_eo_cal")

    old = pd.read_csv(os.path.join(HERE, "calibrated_risk_scores.csv"))
    old["EXAMDATE_DX"] = old["EXAMDATE_DX"].astype(str)
    test_eo["EXAMDATE_DX"] = test_eo["EXAMDATE_DX"].astype(str)
    old = old.drop_duplicates(subset=["RID", "EXAMDATE_DX"])
    test_eo_dedup = test_eo.drop_duplicates(subset=["RID", "EXAMDATE_DX"])
    cmp_df = test_eo_dedup.merge(old[["RID", "EXAMDATE_DX", "RISK_SCORE_plain_cal", "RISK_SCORE_fair_cal"]],
                                   on=["RID", "EXAMDATE_DX"], how="inner")
    lines.append(f"n={len(cmp_df)} rows common to both files.")
    lines.append("")

    protected = fep.make_protected(panel)
    lines.append("=== Policy comparison, all decided at ratio=20 ===")
    for col, label in [("RISK_SCORE_eo_cal", "Fair (EqualizedOdds), calibrated"),
                        ("RISK_SCORE_fair_cal", "Fair (DemographicParity), calibrated -- section 5c"),
                        ("RISK_SCORE_plain_cal", "Plain (no fairness), calibrated")]:
        recommended = du.recommend_interval(cmp_df[col].values, interval_menu_months=(3, 6, 12),
                                             check_cost=1.0, missed_conversion_cost=RATIO,
                                             reference_interval_months=12.0)
        policy_metrics(cmp_df, col, recommended, protected, label, lines)
    lines.append("")

    lines.append("=== Per-group AUC / calibration gap, EqualizedOdds model ===")
    gtab_eo = per_group_metrics(cmp_df, protected, score_col="RISK_SCORE_eo_cal")
    lines.append(gtab_eo.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("")

    # ---- Attribution: did switching the constraint actually fix the NOMINAL_MONTH proxy issue? ----
    print("\n".join(lines))  # flush what's computed so far -- the expensive part is done
    with open(os.path.join(HERE, "fairness_expgrad_equalized_odds_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    lines.append("=== Attribution: does EqualizedOdds still lean on NOMINAL_MONTH? ===")
    feature_cols = fep.EXTENDED_A
    # cmp_df already carries every panel column (it derives from test_panel via
    # build_eo_cascade's apply_df.copy(), which never drops columns) -- no
    # merge needed, and merging panel again would have collided on these exact
    # column names (suffixed _x/_y), which is what broke the first run.
    attr_df = cmp_df.dropna(subset=feature_cols)
    lines.append(f"n={len(attr_df)} rows with full extended+anomaly features.")
    lines.append("")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    shap_results = {}
    for ax, (col, label) in zip(axes, [("RISK_SCORE_plain_cal", "Plain"),
                                        ("RISK_SCORE_fair_cal", "Fair (DemographicParity)"),
                                        ("RISK_SCORE_eo_cal", "Fair (EqualizedOdds)")]):
        forest_model = du.fit_forest_baseline(attr_df[feature_cols], attr_df[col], n_estimators=200, max_depth=4)
        r2 = du.evaluate_regression(forest_model, attr_df[feature_cols], attr_df[col])
        shap_sample = attr_df[feature_cols].sample(n=min(300, len(attr_df)), random_state=42)
        mean_abs_shap, _ = du.compute_shap_importance(forest_model, shap_sample, feature_cols)
        shap_series = mean_abs_shap.set_index("feature")["mean_abs_shap"].sort_values(ascending=False)
        shap_results[label] = shap_series
        lines.append(f"-- {label} -- Forest R2={r2['r2']:.3f}")
        lines.append("   SHAP mean |value|: " + shap_series.to_string(float_format=lambda v: f"{v:.4f}").replace("\n", " | "))
        lines.append("")
        ax.barh(shap_series.index, shap_series.values, color="#55A868" if "Equalized" in label else "#4C72B0")
        ax.invert_yaxis()
        ax.set_title(f"{label}\n(R2={r2['r2']:.2f})")
        ax.set_xlabel("Mean |SHAP|")

    plt.suptitle("Does switching DemographicParity -> EqualizedOdds fix the NOMINAL_MONTH reliance?")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(HERE, "equalized_odds_attribution_comparison.png"), dpi=150)
    plt.close()

    nm_plain = shap_results["Plain"].get("NOMINAL_MONTH", 0.0)
    nm_dp = shap_results["Fair (DemographicParity)"].get("NOMINAL_MONTH", 0.0)
    nm_eo = shap_results["Fair (EqualizedOdds)"].get("NOMINAL_MONTH", 0.0)
    suvr_plain = shap_results["Plain"].get("SUMMARY_SUVR", 0.0)
    suvr_dp = shap_results["Fair (DemographicParity)"].get("SUMMARY_SUVR", 0.0)
    suvr_eo = shap_results["Fair (EqualizedOdds)"].get("SUMMARY_SUVR", 0.0)
    lines.append(f"NOMINAL_MONTH SHAP: plain={nm_plain:.4f}  DP-fair={nm_dp:.4f}  EO-fair={nm_eo:.4f}")
    lines.append(f"SUMMARY_SUVR  SHAP: plain={suvr_plain:.4f}  DP-fair={suvr_dp:.4f}  EO-fair={suvr_eo:.4f}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "fairness_expgrad_equalized_odds_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
