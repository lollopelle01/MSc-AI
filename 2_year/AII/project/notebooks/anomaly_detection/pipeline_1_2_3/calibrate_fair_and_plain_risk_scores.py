"""
notebooks/anomaly_detection/pipeline_1_2_3/calibrate_fair_and_plain_risk_scores.py

Two follow-ups to section 5b's findings:

  1. Calibrate the FAIR (ExponentiatedGradient-corrected) RISK_SCORE --
     the one actually deployed in the pipeline -- whose UNcalibrated gap
     (part B of calibration_and_group_accuracy_check.py: ~0.47-0.50) is
     worse than the plain model's original 0.208 gap that Part A already
     fixed. Does the same recalibration technique work here too, and does
     it also touch the per-group calibration problem section 5b found
     (gap roughly doubling per group under the fairness fix)?

  2. Build the RISK_SCORE the pipeline would produce WITHOUT the fairness
     mechanism at all -- same soft-fallback cascade, plain, unconstrained
     RandomForestClassifier per tier instead of ExponentiatedGradient --
     calibrated the same way, so "fair, calibrated" and "plain (no
     fairness), calibrated" are a fair, apples-to-apples comparison on
     DIDI, cost, catch rate, AUC, and per-group calibration -- not a
     calibrated model vs an uncalibrated one.

Calibration here is a single Platt (sigmoid) fit -- one LogisticRegression
on the raw score, 1 feature -- rather than sklearn's
CalibratedClassifierCV(cv=5) that Part A used. cv=5 would mean refitting
each ExponentiatedGradient reduction 5 MORE times per tier, on top of the
fit already needed -- far too slow for something that already takes
~3 minutes once. A single held-out calibration split is the same
technique CalibratedClassifierCV(method="sigmoid") applies internally
(sklearn's own cv="prefit" mode is exactly this idea) -- appropriate for
an estimator that is expensive to refit.

Usage:
    python3 calibrate_fair_and_plain_risk_scores.py
Outputs (written next to this script):
    - risk_score_calibration_comparison.png
    - risk_score_calibration_comparison_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
FALLBACK_ORDER_FAIR = [("extended+anomaly", "expgrad"), ("extended", "expgrad"),
                        ("core+anomaly", "expgrad"), ("core", "expgrad"), ("core", "logistic")]
FALLBACK_ORDER_PLAIN = [("extended+anomaly", "forest"), ("extended", "forest"),
                         ("core+anomaly", "forest"), ("core", "forest"), ("core", "logistic")]


def fit_logistic_fallback(fit_df):
    train_core = fep.at_risk_complete(fit_df, fep.CORE)
    scaler = StandardScaler().fit(train_core[fep.CORE])
    logistic = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                   class_weight="balanced", random_state=42)
    logistic.fit(scaler.transform(train_core[fep.CORE]), train_core["EVENT_AT_VISIT"])
    return scaler, logistic


def build_cascade(fit_df, apply_dfs, order, score_col, is_fair):
    """Fits one soft-fallback cascade (fair or plain) on fit_df, then scores
    every DataFrame in apply_dfs with it. Same fallback structure
    fairness_expgrad_pipeline.py / decision_mechanism.py / wire_full_pipeline.py
    all already use -- reproduced here (not reimplemented differently) so a
    fit-subset version and the deployed full-train version agree in method."""
    fitted = {}
    for tier_name, cols in FEATURE_SETS.items():
        needed = cols + fep.PROTECTED if is_fair else cols
        train_df = fep.at_risk_complete(fit_df, needed)
        if len(train_df) < 30:
            continue
        if is_fair:
            fitted[tier_name] = fep.fit_expgrad(train_df, cols, fep.PROTECTED)
        else:
            model = RandomForestClassifier(n_estimators=300, max_depth=6,
                                            class_weight="balanced", random_state=42)
            model.fit(train_df[cols], train_df["EVENT_AT_VISIT"])
            fitted[tier_name] = model

    scaler, logistic = fit_logistic_fallback(fit_df)

    outputs = []
    for apply_df in apply_dfs:
        df = apply_df.copy()
        df[score_col] = np.nan
        remaining = df[score_col].isna()
        for tier_name, kind in order:
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
    # Held out purely to fit Platt scaling -- never touched by model fitting,
    # so calibration is measured on data the model has not memorized.
    fit_subset, calib_subset = du.subject_train_test_split(train_panel, test_fraction=0.2, random_state=1)

    lines = []
    lines.append("=== Calibrating both ends of the fairness comparison ===")
    lines.append(f"fit_subset: {fit_subset['RID'].nunique()} patients (fits every model).")
    lines.append(f"calib_subset: {calib_subset['RID'].nunique()} patients, held out only for Platt scaling.")
    lines.append(f"test_panel: {test_panel['RID'].nunique()} patients, untouched until final scoring.")
    lines.append("")

    calib_fair, test_fair = build_cascade(fit_subset, [calib_subset, test_panel],
                                           FALLBACK_ORDER_FAIR, "RISK_SCORE_fair", is_fair=True)
    calib_plain, test_plain = build_cascade(fit_subset, [calib_subset, test_panel],
                                             FALLBACK_ORDER_PLAIN, "RISK_SCORE_plain", is_fair=False)

    test_fair = platt_calibrate(calib_fair, test_fair, "RISK_SCORE_fair", "RISK_SCORE_fair_cal")
    test_plain = platt_calibrate(calib_plain, test_plain, "RISK_SCORE_plain", "RISK_SCORE_plain_cal")

    common_idx = test_fair.index.intersection(test_plain.index)
    cmp_df = test_fair.loc[common_idx].copy()
    cmp_df["RISK_SCORE_plain"] = test_plain.loc[common_idx, "RISK_SCORE_plain"]
    cmp_df["RISK_SCORE_plain_cal"] = test_plain.loc[common_idx, "RISK_SCORE_plain_cal"]
    lines.append(f"Comparison test set: {len(cmp_df)} rows scored by both cascades "
                 f"({cmp_df['RID'].nunique()} patients).")
    lines.append("")

    variants = [
        ("RISK_SCORE_fair", "Fair (EG), UNcalibrated"),
        ("RISK_SCORE_fair_cal", "Fair (EG), calibrated"),
        ("RISK_SCORE_plain", "Plain (no fairness), UNcalibrated"),
        ("RISK_SCORE_plain_cal", "Plain (no fairness), calibrated"),
    ]

    lines.append("=== Overall: AUC, calibration gap, DIDI, cost, catch rate ===")
    calib_tables = {}
    for col, label in variants:
        y = cmp_df["EVENT_AT_VISIT"].to_numpy()
        p = cmp_df[col].to_numpy()
        auc = roc_auc_score(y, p)
        table, gap = du.calibration_curve_check(y, p, n_buckets=10)
        calib_tables[col] = table
        lines.append(f"-- {label} --")
        lines.append(f"   AUC={auc:.3f}   calibration_gap={gap:.3f}")
        fep.evaluate_policy(cmp_df, col, lines, "   policy")
    lines.append("")

    protected = fep.make_protected(panel)
    lines.append("=== Per-group AUC / calibration gap ===")
    group_tables = {}
    for col, label in variants:
        gtab = per_group_metrics(cmp_df, protected, score_col=col)
        group_tables[col] = gtab
        lines.append(f"-- {label} --")
        lines.append(gtab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        lines.append("")

    fair_delta = group_tables["RISK_SCORE_fair_cal"].merge(
        group_tables["RISK_SCORE_fair"], on=["attribute", "group"], suffixes=("_cal", "_raw"))
    fair_delta["calibration_gap_change"] = (
        fair_delta["calibration_gap_cal"] - fair_delta["calibration_gap_raw"])
    fair_delta["auc_change"] = fair_delta["auc_cal"] - fair_delta["auc_raw"]
    lines.append("=== Does calibrating the FAIR model also fix section 5b's per-group problem? ===")
    lines.append(fair_delta[["attribute", "group", "auc_change", "calibration_gap_change"]]
                 .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    lines.append("")

    cmp_fair_plain = group_tables["RISK_SCORE_fair_cal"].merge(
        group_tables["RISK_SCORE_plain_cal"], on=["attribute", "group"], suffixes=("_fair", "_plain"))
    cmp_fair_plain["auc_delta"] = cmp_fair_plain["auc_fair"] - cmp_fair_plain["auc_plain"]
    cmp_fair_plain["calibration_gap_delta"] = (
        cmp_fair_plain["calibration_gap_fair"] - cmp_fair_plain["calibration_gap_plain"])
    lines.append("=== Calibrated fair vs. calibrated plain (no fairness mechanism), per group ===")
    lines.append(cmp_fair_plain[["attribute", "group", "n_fair", "auc_delta", "calibration_gap_delta"]]
                 .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    colors = {"RISK_SCORE_fair": "#C44E52", "RISK_SCORE_fair_cal": "#8172B2",
              "RISK_SCORE_plain": "#DD8452", "RISK_SCORE_plain_cal": "#55A868"}
    for col, label in variants:
        t = calib_tables[col]
        ax.plot(t["mean_predicted"], t["observed_fraction"], marker="o",
                 color=colors[col], label=label, alpha=0.85)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of real events")
    ax.set_title("Overall calibration: all four variants")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")

    ax2 = axes[1]
    plot_df = cmp_fair_plain.copy()
    plot_df["label"] = plot_df["attribute"] + "=" + plot_df["group"].astype(str)
    x = np.arange(len(plot_df))
    width = 0.35
    ax2.bar(x - width / 2, plot_df["calibration_gap_fair"], width,
            label="Fair (EG), calibrated", color="#8172B2")
    ax2.bar(x + width / 2, plot_df["calibration_gap_plain"], width,
            label="Plain (no fairness), calibrated", color="#55A868")
    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_df["label"], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Per-group calibration gap")
    ax2.set_title("Per-group calibration, both now calibrated:\nfair vs. no fairness mechanism")
    ax2.legend(fontsize=8)

    plt.suptitle("Calibrating both ends: does recalibration also close the fairness\n"
                 "mechanism's per-group gap, and how does it compare to dropping fairness entirely?")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(HERE, "risk_score_calibration_comparison.png"), dpi=150)
    plt.close()

    keep_cols = ["RID", "EVENT_AT_VISIT", "EXAMDATE_DX", "DIAGNOSIS",
                 "PTGENDER", "PTEDUCAT_BUCKET", "PTMARRY",
                 "RISK_SCORE_fair", "RISK_SCORE_fair_cal",
                 "RISK_SCORE_plain", "RISK_SCORE_plain_cal"]
    cmp_df[keep_cols].to_csv(os.path.join(HERE, "calibrated_risk_scores.csv"), index=False)
    lines.append(f"Wrote calibrated_risk_scores.csv ({len(cmp_df)} rows) so downstream "
                 "analysis (e.g. re-tuning the cost ratio) can reuse these scores without "
                 "refitting ExponentiatedGradient.")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "risk_score_calibration_comparison_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
