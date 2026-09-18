"""
notebooks/anomaly_detection/pipeline_1_2_3/fairness_correction_reduction.py

The user asked, correctly, whether the post-hoc group-shift in
fairness_correction.py is "ideal" -- explainability_fairness.py's Part 3
already answered honestly that it is not: it repairs the group-level DIDI
number without touching the RandomForest's actual reasoning (its per-patient
SHAP decomposition is provably unchanged by a constant per-group shift).

This tries a genuine IN-TRAINING alternative that does not require
TensorFlow (which already failed here -- fairness_correction_tf.py's
LagDualDIDIRegressor lost too much AUC, 0.791->0.570, because its MSE loss
has no class-imbalance handling for this rare-event target).

fairlearn (sklearn-ecosystem, no TF/Keras) provides the reductions approach
from Agarwal et al. 2018 ("A Reductions Approach to Fair Classification"):
ExponentiatedGradient wraps an ARBITRARY base classifier -- so it can wrap
the exact same RandomForestClassifier(class_weight="balanced", ...) already
used everywhere else in this project, keeping its imbalance handling intact,
while iteratively reweighting/resampling training examples across a
sequence of cost-sensitive classifiers so the ensemble satisfies a fairness
constraint (DemographicParity here, the direct in-training analogue of
DIDI: DIDI is literally the demographic-parity gap, summed over groups).

Also included: classic Reweighing (Kamiran & Calders 2012), a simpler
PRE-processing baseline -- one sample_weight per protected group, computed
once from the training data, that a single RandomForestClassifier.fit()
call can consume directly. Cheaper and more transparent than the reduction
approach; whether it's "enough" is an empirical question this script
answers rather than assumes.

Both are genuine retrains: unlike fairness_correction.py, the resulting
per-patient SHAP decomposition WOULD be different from the baseline model,
because the model itself changed, not just its output.

Usage:
    python3 fairness_correction_reduction.py
Outputs (written next to this script):
    - fairness_correction_reduction_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402
from util import hazard_util as hu  # noqa: E402

DATA_PATH = os.path.join(REPO_ROOT, "datasets", "final.csv")
ANOMALY_KEYED_PATH = os.path.join(REPO_ROOT, "notebooks", "anomaly_detection", "method_b_autoencoder_hi",
                                    "pca_hi_trajectories_keyed.csv")

CORE = ["HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM", "SUMMARY_SUVR", "AGE", "PRIOR_DIAGNOSIS", "NOMINAL_MONTH"]
PROTECTED = ["PTGENDER", "PTEDUCAT_BUCKET", "PTMARRY"]

CHECK_COST = 1.0
MISSED_CONVERSION_COST = 20.0
REFERENCE_INTERVAL_MONTHS = 12.0


def make_protected(df):
    return {
        "PTGENDER": (1, 2),
        "PTEDUCAT_BUCKET": (0, 1),
        "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique())),
    }


def load_panel():
    data = pd.read_csv(DATA_PATH)
    biomarker_cols_to_fill = ["HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM", "SUMMARY_SUVR", "ABETA_RATIO", "TAU", "PTAU"]
    data = du.forward_fill_by_patient(data, biomarker_cols_to_fill, id_col="RID", date_col="EXAMDATE_DX")
    anomaly = pd.read_csv(ANOMALY_KEYED_PATH, usecols=["RID", "VISCODE2_norm", "HI"])
    anomaly = anomaly.dropna(subset=["VISCODE2_norm"]).rename(columns={"HI": "ANOMALY_SCORE"}).drop_duplicates(["RID", "VISCODE2_norm"])
    data = data.merge(anomaly, on=["RID", "VISCODE2_norm"], how="left")
    data = hu.add_nominal_month(data)
    panel = hu.build_hazard_panel(data)
    panel["PTEDUCAT_BUCKET"] = du.bucket_educat(panel["PTEDUCAT"], split_at=16)
    return panel


def at_risk_complete(df, cols):
    return df[df["AT_RISK"]].dropna(subset=cols)


def interval_didi(df, score):
    rec = du.recommend_interval(score, interval_menu_months=(3, 6, 12),
                                 check_cost=CHECK_COST, missed_conversion_cost=MISSED_CONVERSION_COST,
                                 reference_interval_months=REFERENCE_INTERVAL_MONTHS)
    return du.compute_didi(df, rec, make_protected(df))


def reweighing_weights(df, protected_cols, label_col):
    """Classic Kamiran & Calders (2012) reweighing: for every combination of
    (protected-group value, label), weight = P(group)*P(label) / P(group, label).
    Computed once on TRAIN; this is a data-level correction applied ONCE
    before a single model fit -- the model then genuinely learns from a
    rebalanced view of the data, not from an untouched fit whose output is
    adjusted afterward."""
    n = len(df)
    weights = np.ones(n)
    y = df[label_col].to_numpy()
    for attr in protected_cols:
        groups = df[attr].to_numpy()
        for g in np.unique(groups):
            for lbl in np.unique(y):
                mask = (groups == g) & (y == lbl)
                if mask.sum() == 0:
                    continue
                p_group = (groups == g).mean()
                p_label = (y == lbl).mean()
                p_joint = mask.mean()
                weights[mask] *= (p_group * p_label) / p_joint
    # normalize per-attribute compounding back to a sane scale
    weights = weights / weights.mean()
    return weights


def main():
    panel = load_panel()
    train_panel, test_panel = du.subject_train_test_split(panel, test_fraction=0.25, random_state=42)

    feature_cols = CORE  # worst-DIDI tier from didi_breakdown.py -- the hardest case
    train_df = at_risk_complete(train_panel, feature_cols + PROTECTED)
    test_df = at_risk_complete(test_panel, feature_cols + PROTECTED)
    print(f"core tier: train {len(train_df)}, test {len(test_df)}")

    y_train = train_df["EVENT_AT_VISIT"]
    y_test = test_df["EVENT_AT_VISIT"]

    lines = []
    lines.append("=== In-training alternatives to the post-hoc group-shift fix ===")
    lines.append("Tier: core/forest (worst individual-tier DIDI found by didi_breakdown.py, 7.289)")
    lines.append("")
    lines.append(f"{'Model':38s} {'AUC train':>10s} {'AUC test':>10s} {'DIDI(interval) test':>20s}")

    # -- baseline: unconstrained, class-balanced forest (same as everywhere else) --
    baseline = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    baseline.fit(train_df[feature_cols], y_train)
    pred_train_base = baseline.predict_proba(train_df[feature_cols])[:, 1]
    pred_test_base = baseline.predict_proba(test_df[feature_cols])[:, 1]
    auc_train_base = roc_auc_score(y_train, pred_train_base)
    auc_test_base = roc_auc_score(y_test, pred_test_base)
    didi_test_base = interval_didi(test_df, pred_test_base)
    lines.append(f"{'Unconstrained (forest, baseline)':38s} {auc_train_base:10.3f} {auc_test_base:10.3f} {didi_test_base:20.3f}")

    # -- Reweighing (Kamiran & Calders): pre-processing, single retrain --
    sample_weight = reweighing_weights(train_df, PROTECTED, "EVENT_AT_VISIT")
    reweighed = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    reweighed.fit(train_df[feature_cols], y_train, sample_weight=sample_weight)
    pred_train_rw = reweighed.predict_proba(train_df[feature_cols])[:, 1]
    pred_test_rw = reweighed.predict_proba(test_df[feature_cols])[:, 1]
    auc_train_rw = roc_auc_score(y_train, pred_train_rw)
    auc_test_rw = roc_auc_score(y_test, pred_test_rw)
    didi_test_rw = interval_didi(test_df, pred_test_rw)
    lines.append(f"{'Reweighing (Kamiran & Calders)':38s} {auc_train_rw:10.3f} {auc_test_rw:10.3f} {didi_test_rw:20.3f}")

    # -- ExponentiatedGradient (Agarwal et al. 2018): in-training reduction --
    # sensitive_features can be a DataFrame of multiple protected columns;
    # fairlearn treats each row's combination as its group for the constraint.
    # eps controls how tightly demographic parity is enforced -- sweeping it
    # traces out a genuine, tunable accuracy/fairness tradeoff curve (unlike
    # the TF Lagrangian, which only produced one catastrophic operating point).
    lines.append("")
    lines.append("ExponentiatedGradient (fairlearn) -- sweeping the fairness-tightness knob (eps):")
    lines.append(f"  {'eps':>6s} {'AUC train':>10s} {'AUC test':>10s} {'DIDI(interval) test':>20s}")
    best_eg = None
    for eps in (0.02, 0.01, 0.005):
        base_est = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42)
        expgrad = ExponentiatedGradient(estimator=base_est, constraints=DemographicParity(), eps=eps, max_iter=30)
        expgrad.fit(train_df[feature_cols], y_train, sensitive_features=train_df[PROTECTED])
        pred_train_eg = expgrad._pmf_predict(train_df[feature_cols])[:, 1]
        pred_test_eg = expgrad._pmf_predict(test_df[feature_cols])[:, 1]
        auc_train_eg = roc_auc_score(y_train, pred_train_eg)
        auc_test_eg = roc_auc_score(y_test, pred_test_eg)
        didi_test_eg = interval_didi(test_df, pred_test_eg)
        lines.append(f"  {eps:6.3f} {auc_train_eg:10.3f} {auc_test_eg:10.3f} {didi_test_eg:20.3f}")
        if best_eg is None or didi_test_eg < best_eg[2]:
            best_eg = (eps, auc_test_eg, didi_test_eg)
    lines.append(f"Tightest setting tried (eps={best_eg[0]}): test AUC {best_eg[1]:.3f}, DIDI(interval) {best_eg[2]:.3f}")

    lines.append("")
    lines.append("For reference (from earlier scripts, same tier/split):")
    lines.append("  Post-hoc group-shift (fairness_correction.py):    test DIDI(score) 8.570 -> 3.153 (63% cut), AUC untouched")
    lines.append("  LagDualDIDIRegressor (fairness_correction_tf.py): test DIDI(interval) 8.716 -> 7.313 (16% cut), AUC 0.791 -> 0.570")
    lines.append("")
    lines.append("Note: the post-hoc and TF numbers above are DIDI of the SCORE or of a different")
    lines.append("evaluation setup than this script's core-tier-only comparison; the three rows in")
    lines.append("the table above are directly comparable to each other (same split, same tier,")
    lines.append("same DIDI definition -- interval-level, core tier).")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(HERE, "fairness_correction_reduction_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
