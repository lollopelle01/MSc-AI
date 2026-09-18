"""
notebooks/anomaly_detection/pipeline_1_2_3/stage3_trajectory_and_learned_policy.py

Tries the two most promising "develop stage 3 further" directions named
while rehearsing (both already anticipated by the project itself --
recommend_interval's own docstring names both a trajectory-aware policy
and a learned/DFL-style one as the intended next steps beyond the
hardcoded grid search):

PART A -- recommend_interval_trajectory, already written in
decision_util.py but never wired into the deployed pipeline. Projects
risk forward using each patient's own observed risk slope
(compute_risk_trajectory) instead of treating today's snapshot as if it
held steady for the whole candidate interval. Reuses
calibrated_risk_scores.csv (section 5c) directly -- no refitting needed,
since RISK_SLOPE only needs the risk scores already computed there.

PART B -- a genuinely LEARNED interval-choice model: a multi-class
classifier trained to predict the interval directly from a patient's raw
features (not just their single risk number), using the current
closed-form policy's own output as bootstrap ground-truth labels. This
tests whether richer context than one scalar risk score can improve on
the formula's decisions. Since we don't have calibrated scores for the
ORIGINAL train_panel (only for test_panel, from section 5c), this uses a
nested nested nested split -- test_panel's own patients are further split
into clf_train/clf_test by RID, which is a legitimate, leakage-free
holdout (the classifier never sees clf_test patients, in either its
features or its bootstrap labels), it just isn't the original 25% split.

Usage:
    python3 stage3_trajectory_and_learned_policy.py
Outputs (written next to this script):
    - stage3_trajectory_and_learned_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fairness_expgrad_pipeline as fep  # noqa: E402

REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402

RATIO = 20.0  # the project's original ratio, used as the reference policy throughout
MENU = (3, 6, 12)
PROTECTED_DOMAIN = lambda df: {"PTGENDER": (1, 2), "PTEDUCAT_BUCKET": (0, 1),
                                "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique()))}


def policy_metrics(df, risk_col, recommended, label, lines):
    cmodel = du.ConversionCostModel(check_cost=1.0, missed_conversion_cost=RATIO,
                                     safe_interval_months=6, reference_interval_months=12.0)
    total_cost, _, _ = cmodel.cost(rid_ids=df["RID"].values, risk_scores=df[risk_col].values,
                                    threshold=0.5, interval_months=recommended, return_margin=False)
    didi = du.compute_didi(df, recommended, PROTECTED_DOMAIN(df))
    outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
    outcomes["RECOMMENDED_INTERVAL"] = recommended
    worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
    margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
    catch_rate = (margin >= 0).mean() * 100
    lines.append(f"  {label:38s} cost={total_cost:9.1f}  DIDI={didi:.3f}  catch={catch_rate:5.1f}%")
    return total_cost, didi, catch_rate


def part_a_trajectory(lines):
    lines.append("=== PART A: trajectory-aware policy (recommend_interval_trajectory) ===")
    lines.append("Reuses calibrated_risk_scores.csv -- no refitting.")
    lines.append("")
    df = pd.read_csv(os.path.join(HERE, "calibrated_risk_scores.csv"))
    df["EXAMDATE_DX"] = pd.to_datetime(df["EXAMDATE_DX"])

    for score_col, label in [("RISK_SCORE_plain_cal", "Plain (no fairness), calibrated"),
                              ("RISK_SCORE_fair_cal", "Fair (EG), calibrated")]:
        traj = du.compute_risk_trajectory(df, id_col="RID", date_col="EXAMDATE_DX", risk_col=score_col)
        coverage = traj["HAS_TRAJECTORY"].mean() * 100
        lines.append(f"-- {label} -- trajectory coverage: {coverage:.1f}% of rows have a prior visit")

        snapshot_rec = du.recommend_interval(traj[score_col].values, interval_menu_months=MENU,
                                              check_cost=1.0, missed_conversion_cost=RATIO,
                                              reference_interval_months=12.0)
        traj_rec = du.recommend_interval_trajectory(traj[score_col].values, traj["RISK_SLOPE"].values,
                                                      interval_menu_months=MENU, check_cost=1.0,
                                                      missed_conversion_cost=RATIO, reference_interval_months=12.0)
        changed = (snapshot_rec != traj_rec).mean() * 100
        lines.append(f"   decisions changed by adding trajectory: {changed:.1f}% of rows")
        policy_metrics(traj, score_col, snapshot_rec, "Snapshot only (current policy)", lines)
        policy_metrics(traj, score_col, traj_rec, "Trajectory-aware", lines)
        lines.append("")
    lines.append("Read the DIDI/cost/catch numbers above as: does knowing whether a patient's")
    lines.append("risk is rising or falling, not just its current level, actually change enough")
    lines.append("decisions to move the aggregate policy numbers, or does it only re-shuffle a")
    lines.append("small minority of borderline cases?")
    lines.append("")


def part_b_learned_policy(lines):
    lines.append("=== PART B: a LEARNED interval-choice model vs. the closed-form formula ===")
    lines.append("Nested split: test_panel's own patients further split into clf_train/clf_test")
    lines.append("by RID (leakage-free -- clf_test patients never touch fitting or labeling).")
    lines.append("")

    scores = pd.read_csv(os.path.join(HERE, "calibrated_risk_scores.csv"))
    scores["EXAMDATE_DX"] = pd.to_datetime(scores["EXAMDATE_DX"])

    panel = fep.load_panel()
    feature_cols = fep.EXTENDED_A  # richest tier: 7 biomarkers + AGE + PRIOR_DIAGNOSIS + NOMINAL_MONTH + ANOMALY_SCORE
    panel_feats = panel[["RID", "EXAMDATE_DX"] + feature_cols].copy()
    panel_feats["EXAMDATE_DX"] = pd.to_datetime(panel_feats["EXAMDATE_DX"])
    panel_feats = panel_feats.drop_duplicates(subset=["RID", "EXAMDATE_DX"])

    df = scores.merge(panel_feats, on=["RID", "EXAMDATE_DX"], how="inner").dropna(subset=feature_cols)
    lines.append(f"n={len(df)} rows with full extended+anomaly features available (of {len(scores)} scored rows).")

    clf_train, clf_test = du.subject_train_test_split(df, test_fraction=0.3, random_state=7)
    lines.append(f"clf_train: {clf_train['RID'].nunique()} patients ({len(clf_train)} rows). "
                 f"clf_test: {clf_test['RID'].nunique()} patients ({len(clf_test)} rows).")
    lines.append("")

    score_col = "RISK_SCORE_plain_cal"  # plain model, so this is about stage 3's own structure, not fairness
    model_feature_cols = feature_cols + [score_col]

    # Bootstrap ground-truth labels from the CURRENT closed-form policy, on each split separately.
    clf_train = clf_train.copy()
    clf_test = clf_test.copy()
    clf_train["Y_INTERVAL"] = du.recommend_interval(clf_train[score_col].values, interval_menu_months=MENU,
                                                      check_cost=1.0, missed_conversion_cost=RATIO,
                                                      reference_interval_months=12.0)
    clf_test["Y_INTERVAL"] = du.recommend_interval(clf_test[score_col].values, interval_menu_months=MENU,
                                                     check_cost=1.0, missed_conversion_cost=RATIO,
                                                     reference_interval_months=12.0)
    lines.append("clf_train label distribution: " +
                 clf_train["Y_INTERVAL"].value_counts().sort_index().to_dict().__repr__())
    lines.append("clf_test  label distribution: " +
                 clf_test["Y_INTERVAL"].value_counts().sort_index().to_dict().__repr__())
    lines.append("")

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    clf.fit(clf_train[model_feature_cols], clf_train["Y_INTERVAL"])
    learned_pred = clf.predict(clf_test[model_feature_cols])

    acc = accuracy_score(clf_test["Y_INTERVAL"], learned_pred)
    lines.append(f"Learned model's agreement with the formula's own labels on clf_test: {acc*100:.1f}%")
    importances = pd.Series(clf.feature_importances_, index=model_feature_cols).sort_values(ascending=False)
    lines.append("Feature importances (does the learned model actually use context beyond the risk score?):")
    lines.append(importances.to_string(float_format=lambda v: f"{v:.3f}"))
    lines.append("")

    formula_rec = clf_test["Y_INTERVAL"].values  # the formula's own decision, already computed above
    policy_metrics(clf_test, score_col, formula_rec, "Closed-form formula (current policy)", lines)
    policy_metrics(clf_test, score_col, learned_pred, "Learned classifier", lines)
    lines.append("")
    lines.append("If the learned model mostly just re-derives RISK_SCORE (that feature dominating")
    lines.append("importances) and matches the formula closely, that's a clean negative result:")
    lines.append("the extra raw features don't carry decision-relevant information the risk score")
    lines.append("hadn't already summarized. If other features carry real weight and the policy")
    lines.append("numbers move, that's evidence a learned policy could do genuinely more than the")
    lines.append("closed-form formula, at the cost of losing the formula's transparency (section 7's")
    lines.append("decision_mechanism.py could no longer explain 'why this interval' as three lines")
    lines.append("of arithmetic -- it would need SHAP or similar, same tradeoff stage 2 already has).")


def main():
    lines = []
    part_a_trajectory(lines)
    part_b_learned_policy(lines)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "stage3_trajectory_and_learned_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
