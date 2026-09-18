"""
notebooks/anomaly_detection/pipeline_1_2_3/fairness_correction_tf.py

TensorFlow is now installed (see README), so this replaces the sklearn-only
post-hoc recalibration in fairness_correction.py with the team's own
in-syllabus technique: LagDualDIDIRegressor (07-ciml, lesson 2, Lagrangian
Approaches for Constraint Injection), already implemented in
util/decision_util.py and already used for point 3's own fairness track --
reused directly here, not reimplemented, so this is the same technique
Pelle used, applied to point 1/2's own worst-offending tier instead of
point 3's final recommendation.

didi_breakdown.py found DIDI varies a lot by fallback tier (1.357 to
7.289), with 'core/forest' -- the tier serving patients with the LEAST
complete records, and the one most correlated with protected attributes --
the worst offender by a wide margin. Rather than correcting the whole
blended RISK_SCORE after the fact (the sklearn version), this retrains
specifically that worst tier with the DIDI constraint built into training,
following the Lagrangian dual approach the lesson itself recommends
reporting as the main result (better accuracy than a fixed-alpha penalty
at the same fairness level).

Note on protected columns as inputs: util/decision_util.py's own docstring
for scale_excluding_protected is explicit that CstDIDIRegressor and
LagDualDIDIRegressor need the protected columns AS INPUTS (the constraint
reads them at training time to compute each group's mean prediction) --
different from the plain baseline models elsewhere in this project, which
exclude protected columns entirely. This is the documented, technique-
specific exception, not an inconsistency.

Usage:
    python3 fairness_correction_tf.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402
from util import hazard_util as hu  # noqa: E402

DATA_PATH = os.path.join(REPO_ROOT, "datasets", "final.csv")
ANOMALY_KEYED_PATH = os.path.join(REPO_ROOT, "notebooks", "anomaly_detection", "method_b_autoencoder_hi",
                                    "pca_hi_trajectories_keyed.csv")

CHECK_COST = 1.0
MISSED_CONVERSION_COST = 20.0
REFERENCE_INTERVAL_MONTHS = 12.0
SAFE_INTERVAL_MONTHS = 6

CORE = ["HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM", "SUMMARY_SUVR", "AGE", "PRIOR_DIAGNOSIS", "NOMINAL_MONTH"]
EXTENDED = CORE + ["TAU", "PTAU"]
CORE_A = CORE + ["ANOMALY_SCORE"]
EXTENDED_A = EXTENDED + ["ANOMALY_SCORE"]


def make_protected_named():
    return {"PTGENDER": (1, 2), "PTEDUCAT_BUCKET": (0, 1), "PTMARRY": None}  # PTMARRY domain filled in later


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


def main():
    panel = load_panel()
    train_panel, test_panel = du.subject_train_test_split(panel, test_fraction=0.25, random_state=42)

    protected_cols = ["PTGENDER", "PTEDUCAT_BUCKET", "PTMARRY"]
    feature_cols = CORE + protected_cols  # protected cols appended as extra inputs, required by the constraint

    def at_risk_complete(df, cols):
        return df[df["AT_RISK"]].dropna(subset=cols)

    train_core = at_risk_complete(train_panel, feature_cols)
    test_core = at_risk_complete(test_panel, feature_cols)
    print(f"Core tier (with protected cols appended): train {len(train_core)}, test {len(test_core)}")

    # -- unconstrained baseline (plain forest, same as before, for comparison) --
    baseline_rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    baseline_rf.fit(train_core[CORE], train_core["EVENT_AT_VISIT"])
    baseline_pred_train = baseline_rf.predict_proba(train_core[CORE])[:, 1]
    baseline_pred_test = baseline_rf.predict_proba(test_core[CORE])[:, 1]

    def protected_dict(df):
        return {"PTGENDER": (1, 2), "PTEDUCAT_BUCKET": (0, 1), "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique()))}

    didi_baseline_train = du.compute_didi(train_core, baseline_pred_train, protected_dict(train_core))
    didi_baseline_test = du.compute_didi(test_core, baseline_pred_test, protected_dict(test_core))
    print(f"Unconstrained core/forest DIDI: train {didi_baseline_train:.3f}, test {didi_baseline_test:.3f}")

    # -- LagDualDIDIRegressor, following 07-ciml lesson 2 exactly, reused from decision_util.py --
    X_train_scaled, X_test_scaled = du.scale_excluding_protected(
        train_core, test_core, feature_cols, protected_cols,
    )
    protected_columns_by_index = {
        feature_cols.index("PTGENDER"): (1, 2),
        feature_cols.index("PTEDUCAT_BUCKET"): (0, 1),
        feature_cols.index("PTMARRY"): tuple(sorted(train_core["PTMARRY"].dropna().unique())),
    }
    threshold = didi_baseline_train / 2.0  # same convention Pelle's own fairness notebook uses

    y_train = train_core["EVENT_AT_VISIT"].to_numpy(dtype="float32")
    y_test = test_core["EVENT_AT_VISIT"].to_numpy(dtype="float32")

    print(f"\nTraining LagDualDIDIRegressor (threshold={threshold:.3f}, hidden=(8,))...")
    lag_model = du.LagDualDIDIRegressor(
        input_dim=X_train_scaled.shape[1],
        protected_columns=protected_columns_by_index,
        threshold=threshold,
        hidden=(8,),
    )
    lag_model.fit(X_train_scaled, y_train, epochs=1500, verbose=0)
    print(f"Final trained alpha (Lagrangian multiplier): {lag_model.final_alpha:.4f}")

    pred_train_lag = np.clip(lag_model.predict(X_train_scaled), 0.0, 1.0)
    pred_test_lag = np.clip(lag_model.predict(X_test_scaled), 0.0, 1.0)

    didi_lag_train = du.compute_didi(train_core, pred_train_lag, protected_dict(train_core))
    didi_lag_test = du.compute_didi(test_core, pred_test_lag, protected_dict(test_core))

    auc_baseline_train = roc_auc_score(y_train, baseline_pred_train)
    auc_baseline_test = roc_auc_score(y_test, baseline_pred_test)
    auc_lag_train = roc_auc_score(y_train, pred_train_lag)
    auc_lag_test = roc_auc_score(y_test, pred_test_lag)

    # -- downstream: DIDI of the actual interval RECOMMENDATION, the number
    # comparable to didi_breakdown.py's 7.289 (that number was DIDI of the
    # recommended interval, 3/6/12 months -- a different scale than DIDI of
    # a raw 0-1 probability, which is what the numbers above measure) --
    def interval_didi(df, score):
        rec = du.recommend_interval(score, interval_menu_months=(3, 6, 12),
                                     check_cost=CHECK_COST, missed_conversion_cost=MISSED_CONVERSION_COST,
                                     reference_interval_months=REFERENCE_INTERVAL_MONTHS)
        return du.compute_didi(df, rec, protected_dict(df))

    interval_didi_baseline_test = interval_didi(test_core, baseline_pred_test)
    interval_didi_lag_test = interval_didi(test_core, pred_test_lag)

    lines = []
    lines.append("=== In-training fairness fix: LagDualDIDIRegressor on the 'core' tier ===")
    lines.append("(the worst individual-tier interval-level DIDI found by didi_breakdown.py: 7.289 on core/forest)")
    lines.append("")
    lines.append(f"{'Model':30s} {'AUC train':>10s} {'AUC test':>10s} {'DIDI(score) train':>18s} {'DIDI(score) test':>17s}")
    lines.append(f"{'Unconstrained (forest)':30s} {auc_baseline_train:10.3f} {auc_baseline_test:10.3f} "
                 f"{didi_baseline_train:18.3f} {didi_baseline_test:17.3f}")
    lines.append(f"{'LagDualDIDIRegressor':30s} {auc_lag_train:10.3f} {auc_lag_test:10.3f} "
                 f"{didi_lag_train:18.3f} {didi_lag_test:17.3f}")
    lines.append("")
    lines.append(f"DIDI of the downstream recommended INTERVAL (test split, comparable to the 7.289 figure):")
    lines.append(f"  Unconstrained (forest): {interval_didi_baseline_test:.3f}")
    lines.append(f"  LagDualDIDIRegressor:   {interval_didi_lag_test:.3f}")
    lines.append("")
    lines.append(f"Final trained Lagrangian multiplier (alpha): {lag_model.final_alpha:.4f}")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(HERE, "fairness_correction_tf_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
