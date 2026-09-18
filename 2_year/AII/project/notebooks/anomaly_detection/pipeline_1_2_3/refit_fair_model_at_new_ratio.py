"""
notebooks/anomaly_detection/pipeline_1_2_3/refit_fair_model_at_new_ratio.py

Section 5d found that re-thresholding the FAIR (ExponentiatedGradient)
model by raising MISSED_CONVERSION_COST after the fact does not cleanly
recover catch rate -- DIDI climbs back toward the uncorrected baseline
before catch rate gets there, because the fairness constraint was trained
at ratio=20's implicit decision threshold, not the new one. The proposed
fix was to REFIT ExponentiatedGradient directly at the new ratio, instead
of re-thresholding a model fit for a different one. This does that.

fit_expgrad's own objective is `ErrorRate(costs={"fp": 1.0,
"fn": MISSED_CONVERSION_COST / CHECK_COST})` -- it reads those two names
as module globals at call time, so this script temporarily overrides
fairness_expgrad_pipeline.MISSED_CONVERSION_COST before calling
fep.fit_expgrad (and restores it immediately after), rather than
duplicating fit_expgrad's body with a new parameter -- same function,
same fitting logic, different cost input.

TARGET_RATIO=100 because that's exactly the ratio
retune_cost_ratio_after_calibration.py found recovers the plain model's
original 89.7% catch rate -- the natural "does refitting at the ratio
that worked for the plain model also work for the fair one" question.

Usage:
    python3 refit_fair_model_at_new_ratio.py
Outputs (written next to this script):
    - refit_at_new_ratio_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
TARGET_RATIO = 100
ORIGINAL_RATIO = fep.MISSED_CONVERSION_COST  # 20, restored after fitting


def fit_logistic_fallback(fit_df):
    train_core = fep.at_risk_complete(fit_df, fep.CORE)
    scaler = StandardScaler().fit(train_core[fep.CORE])
    logistic = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                   class_weight="balanced", random_state=42)
    logistic.fit(scaler.transform(train_core[fep.CORE]), train_core["EVENT_AT_VISIT"])
    return scaler, logistic


def build_fair_cascade_at_ratio(fit_df, apply_dfs, ratio, score_col):
    fitted = {}
    fep.MISSED_CONVERSION_COST = ratio  # fit_expgrad reads this as a module global
    try:
        for tier_name, cols in FEATURE_SETS.items():
            train_df = fep.at_risk_complete(fit_df, cols + fep.PROTECTED)
            if len(train_df) < 30:
                continue
            fitted[tier_name] = fep.fit_expgrad(train_df, cols, fep.PROTECTED)
    finally:
        fep.MISSED_CONVERSION_COST = ORIGINAL_RATIO  # restore immediately, don't leak state

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


def main():
    panel = fep.load_panel()
    train_panel, test_panel = du.subject_train_test_split(panel, test_fraction=0.25, random_state=42)
    fit_subset, calib_subset = du.subject_train_test_split(train_panel, test_fraction=0.2, random_state=1)

    lines = []
    lines.append(f"=== Refitting ExponentiatedGradient directly at ratio={TARGET_RATIO} "
                 f"(vs. re-thresholding the ratio={ORIGINAL_RATIO}-fit model) ===")
    lines.append("")

    calib_new, test_new = build_fair_cascade_at_ratio(
        fit_subset, [calib_subset, test_panel], TARGET_RATIO, "RISK_SCORE_fair_r100")
    test_new = platt_calibrate(calib_new, test_new, "RISK_SCORE_fair_r100", "RISK_SCORE_fair_r100_cal")

    # Save immediately -- refitting ExponentiatedGradient at a new ratio is
    # the expensive part (~3 min). Merge/compare against
    # calibrated_risk_scores.csv in a SEPARATE fast script
    # (compare_refit_vs_rethreshold.py) instead of doing it inline here, so
    # a merge-key bug (there was one -- EXAMDATE_DX dtype mismatch, and
    # before that a RID-only merge that silently fanned out to 19629 rows)
    # never requires re-running the slow refit to fix.
    keep_cols = ["RID", "EXAMDATE_DX", "EVENT_AT_VISIT", "DIAGNOSIS",
                 "PTGENDER", "PTEDUCAT_BUCKET", "PTMARRY",
                 "RISK_SCORE_fair_r100", "RISK_SCORE_fair_r100_cal"]
    test_new[keep_cols].to_csv(os.path.join(HERE, "refit_at_ratio100_scores.csv"), index=False)

    lines = []
    lines.append(f"=== Refit ExponentiatedGradient at ratio={TARGET_RATIO} ===")
    lines.append(f"Wrote refit_at_ratio100_scores.csv ({len(test_new)} rows). "
                 f"Run compare_refit_vs_rethreshold.py next for the policy comparison.")
    gtab = per_group_metrics(test_new, fep.make_protected(panel), score_col="RISK_SCORE_fair_r100_cal")
    lines.append("")
    lines.append("Per-group AUC / calibration gap, REFIT-at-100 model:")
    lines.append(gtab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "refit_at_new_ratio_report.txt"), "w") as f:
        f.write(report + "\n")
    return


if __name__ == "__main__":
    main()
