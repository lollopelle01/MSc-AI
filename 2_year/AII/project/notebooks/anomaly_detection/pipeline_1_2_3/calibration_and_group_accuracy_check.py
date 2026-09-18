"""
notebooks/anomaly_detection/pipeline_1_2_3/calibration_and_group_accuracy_check.py

Two of the honest limitations surfaced while rehearsing, both checked
directly instead of left as caveats:

PART A -- Is the hazard model's calibration gap (0.242, from
hazard_survival_model.ipynb section 9) fixable? Standard post-hoc
recalibration (Platt/sigmoid scaling, CalibratedClassifierCV) is applied to
the exact deployed best tier (extended+anomaly/forest, the same model
wire_full_pipeline.py picks and fairness_expgrad_pipeline.py retrains under
a fairness constraint), and calibration_curve_check is rerun before/after
to see whether it actually helps, not just assumed to.

PART B -- DIDI's own blind spot: it checks whether GROUP AVERAGES match,
never whether the numbers are individually correct. This computes AUC and
calibration_gap SEPARATELY per protected-attribute group, for both the
plain (unconstrained) fallback RISK_SCORE and the fairness-corrected
(ExponentiatedGradient) one, so it's possible to see directly whether the
DIDI-driven fairness fix helped some groups' own accuracy, hurt others, or
left accuracy roughly untouched while only equalizing averages -- the
question DIDI alone cannot answer.

Usage:
    python3 calibration_and_group_accuracy_check.py
Outputs (written next to this script):
    - calibration_check_report.txt, calibration_before_after.png
    - group_accuracy_report.txt, group_accuracy_comparison.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fairness_expgrad_pipeline as fep  # noqa: E402

REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402

TIER_NAME = "extended+anomaly"
TIER_COLS = fep.EXTENDED_A


# ======================================================================
# PART A: Calibration -- is the 0.242 gap fixable with post-hoc recalibration?
# ======================================================================

def part_a_calibration(panel, train_panel, test_panel, lines):
    lines.append("=== PART A: Can the hazard model's calibration gap be fixed? ===")
    lines.append(f"Tier: {TIER_NAME}/forest (the deployed best single estimator, "
                 f"test AUC 0.807 from wire_full_pipeline.py)")
    lines.append("")

    train_df = fep.at_risk_complete(train_panel, TIER_COLS)
    test_df = fep.at_risk_complete(test_panel, TIER_COLS)

    # Uncalibrated (plain) model -- same config as wire_full_pipeline.py
    plain = RandomForestClassifier(n_estimators=300, max_depth=6,
                                    class_weight="balanced", random_state=42)
    plain.fit(train_df[TIER_COLS], train_df["EVENT_AT_VISIT"])
    proba_plain = plain.predict_proba(test_df[TIER_COLS])[:, 1]
    auc_plain = roc_auc_score(test_df["EVENT_AT_VISIT"], proba_plain)
    table_before, gap_before = du.calibration_curve_check(
        test_df["EVENT_AT_VISIT"].to_numpy(), proba_plain, n_buckets=10)

    # Calibrated version -- 5-fold sigmoid (Platt) calibration on TRAIN only,
    # never touching test, so this is a fair before/after comparison.
    calibrated = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, max_depth=6,
                                class_weight="balanced", random_state=42),
        method="sigmoid", cv=5,
    )
    calibrated.fit(train_df[TIER_COLS], train_df["EVENT_AT_VISIT"])
    proba_cal = calibrated.predict_proba(test_df[TIER_COLS])[:, 1]
    auc_cal = roc_auc_score(test_df["EVENT_AT_VISIT"], proba_cal)
    table_after, gap_after = du.calibration_curve_check(
        test_df["EVENT_AT_VISIT"].to_numpy(), proba_cal, n_buckets=10)

    lines.append(f"BEFORE (plain forest):       test AUC={auc_plain:.3f}  "
                 f"weighted mean |calibration gap|={gap_before:.3f}")
    lines.append(f"AFTER (sigmoid-calibrated):  test AUC={auc_cal:.3f}  "
                 f"weighted mean |calibration gap|={gap_after:.3f}")
    improvement = (gap_before - gap_after) / gap_before * 100
    lines.append(f"Calibration gap change: {improvement:+.1f}% "
                 f"({'improved' if improvement > 0 else 'worsened'})")
    lines.append("AUC change: "
                 f"{auc_plain:.3f} -> {auc_cal:.3f} "
                 f"({'preserved' if abs(auc_cal - auc_plain) < 0.01 else 'changed'} "
                 f"-- recalibration should not meaningfully change ranking)")
    lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, table, gap, label in [(axes[0], table_before, gap_before, "Before (plain)"),
                                    (axes[1], table_after, gap_after, "After (sigmoid-calibrated)")]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.plot(table["mean_predicted"], table["observed_fraction"], marker="o", color="#4C72B0")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction of real events")
        ax.set_title(f"{label}\nweighted |gap| = {gap:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    plt.suptitle(f"Calibration before/after post-hoc sigmoid recalibration ({TIER_NAME}/forest)")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(HERE, "calibration_before_after.png"), dpi=150)
    plt.close()

    return calibrated


# ======================================================================
# PART B: DIDI's blind spot -- per-group accuracy, plain vs fair model
# ======================================================================

def build_plain_fallback_risk_score(panel, train_panel, test_panel):
    """Same fallback cascade structure as fairness_expgrad_pipeline.py's
    build (and decision_mechanism.py's rebuild of it), but with PLAIN,
    unconstrained RandomForestClassifier per tier instead of
    ExponentiatedGradient -- i.e. wire_full_pipeline.py's own soft-fallback
    mechanism, reproduced here for a direct plain-vs-fair comparison."""
    feature_sets = {"core": fep.CORE, "extended": fep.EXTENDED,
                     "core+anomaly": fep.CORE_A, "extended+anomaly": fep.EXTENDED_A}
    fitted = {}
    for tier_name, cols in feature_sets.items():
        train_df = fep.at_risk_complete(train_panel, cols)
        if len(train_df) < 30:
            continue
        model = RandomForestClassifier(n_estimators=300, max_depth=6,
                                        class_weight="balanced", random_state=42)
        model.fit(train_df[cols], train_df["EVENT_AT_VISIT"])
        fitted[tier_name] = model

    train_core = fep.at_risk_complete(train_panel, fep.CORE)
    scaler = StandardScaler().fit(train_core[fep.CORE])
    logistic = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                   class_weight="balanced", random_state=42)
    logistic.fit(scaler.transform(train_core[fep.CORE]), train_core["EVENT_AT_VISIT"])

    FALLBACK_ORDER = [
        ("extended+anomaly", "forest"), ("extended", "forest"),
        ("core+anomaly", "forest"), ("core", "forest"), ("core", "logistic"),
    ]
    panel_fb = panel.copy()
    panel_fb["RISK_SCORE"] = np.nan
    remaining = panel_fb["RISK_SCORE"].isna()
    for tier_name, kind in FALLBACK_ORDER:
        cols = feature_sets.get(tier_name, fep.CORE)
        eligible = remaining & panel_fb[cols].notna().all(axis=1)
        if not eligible.any():
            continue
        X = panel_fb.loc[eligible, cols]
        if kind == "forest":
            proba = fitted[tier_name].predict_proba(X)[:, 1]
        else:
            proba = logistic.predict_proba(scaler.transform(X))[:, 1]
        panel_fb.loc[eligible, "RISK_SCORE"] = proba
        remaining = panel_fb["RISK_SCORE"].isna()

    _, test_fb = du.split_by_rid_membership(
        panel_fb.dropna(subset=["RISK_SCORE"]).copy(), set(train_panel["RID"]), set(test_panel["RID"])
    )  # fixed: re-splitting this filtered subset independently leaked most "test"
       # patients from train_panel -- see split_by_rid_membership docstring.
    return test_fb


def per_group_metrics(df, protected, score_col="RISK_SCORE"):
    rows = []
    for attr, domain in protected.items():
        for v in domain:
            mask = df[attr] == v
            if mask.sum() < 15:
                continue
            y = df.loc[mask, "EVENT_AT_VISIT"].to_numpy()
            p = df.loc[mask, score_col].to_numpy()
            if y.sum() < 3 or (y == 0).sum() < 3:
                auc = np.nan
            else:
                auc = roc_auc_score(y, p)
            _, gap = du.calibration_curve_check(y, p, n_buckets=5)
            rows.append({"attribute": attr, "group": v, "n": int(mask.sum()),
                         "auc": auc, "calibration_gap": gap})
    return pd.DataFrame(rows)


def part_b_group_accuracy(panel, train_panel, test_panel, lines):
    lines.append("=== PART B: DIDI's blind spot -- per-group accuracy, plain vs fair ===")
    lines.append("DIDI checks whether GROUP AVERAGE outcomes match. It says nothing about")
    lines.append("whether each group's own predictions are individually accurate. This")
    lines.append("checks that directly: AUC and calibration gap, computed SEPARATELY per")
    lines.append("protected-attribute group, for the plain fallback RISK_SCORE and the")
    lines.append("fairness-corrected (ExponentiatedGradient) one.")
    lines.append("")

    plain_test = build_plain_fallback_risk_score(panel, train_panel, test_panel)
    from decision_mechanism import build_risk_score as build_fair_risk_score
    fair_test = build_fair_risk_score()

    protected = fep.make_protected(panel)

    plain_metrics = per_group_metrics(plain_test, protected)
    fair_metrics = per_group_metrics(fair_test, protected)

    plain_metrics["model"] = "plain"
    fair_metrics["model"] = "fair (EG)"
    combined = pd.concat([plain_metrics, fair_metrics], ignore_index=True)

    lines.append("-- Plain (unconstrained) fallback RISK_SCORE --")
    lines.append(plain_metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("")
    lines.append("-- Fair (ExponentiatedGradient, cost-aligned) fallback RISK_SCORE --")
    lines.append(fair_metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("")

    merged = plain_metrics.merge(
        fair_metrics, on=["attribute", "group"], suffixes=("_plain", "_fair")
    )
    merged["auc_delta"] = merged["auc_fair"] - merged["auc_plain"]
    merged["calibration_gap_delta"] = merged["calibration_gap_fair"] - merged["calibration_gap_plain"]
    lines.append("-- Delta (fair minus plain) per group --")
    lines.append(merged[["attribute", "group", "auc_delta", "calibration_gap_delta"]]
                 .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    lines.append("")
    worst_auc_drop = merged.loc[merged["auc_delta"].idxmin()]
    lines.append(f"Largest AUC drop for any single group: {worst_auc_drop['attribute']}="
                 f"{worst_auc_drop['group']}, {worst_auc_drop['auc_delta']:+.3f}")
    lines.append("Read this alongside the DIDI numbers already reported: DIDI fell 76%")
    lines.append("(8.570 -> 2.098) from the same fairness correction. If per-group AUC and")
    lines.append("calibration gap stayed roughly flat across groups (small deltas here),")
    lines.append("that's direct evidence the fairness fix equalized OUTCOMES without")
    lines.append("degrading any single group's own prediction ACCURACY -- the exact")
    lines.append("question DIDI alone cannot answer.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"{r.attribute}={r.group}" for r in combined.itertuples()]
    combined["label"] = [f"{r.attribute}={r.group}" for r in combined.itertuples()]
    for ax, metric, title in [(axes[0], "auc", "Per-group AUC"),
                                (axes[1], "calibration_gap", "Per-group calibration gap")]:
        pivot = combined.pivot_table(index="label", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, color=["#C44E52", "#55A868"])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    plt.suptitle("DIDI's blind spot, checked directly: does the fairness fix\n"
                 "change per-group accuracy, or only per-group averages?")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(HERE, "group_accuracy_comparison.png"), dpi=150)
    plt.close()


def main():
    panel = fep.load_panel()
    train_panel, test_panel = du.subject_train_test_split(panel, test_fraction=0.25, random_state=42)

    lines_a = []
    part_a_calibration(panel, train_panel, test_panel, lines_a)
    report_a = "\n".join(lines_a)
    print(report_a)
    with open(os.path.join(HERE, "calibration_check_report.txt"), "w") as f:
        f.write(report_a + "\n")

    lines_b = []
    part_b_group_accuracy(panel, train_panel, test_panel, lines_b)
    report_b = "\n".join(lines_b)
    print("\n" + report_b)
    with open(os.path.join(HERE, "group_accuracy_report.txt"), "w") as f:
        f.write(report_b + "\n")


if __name__ == "__main__":
    main()
