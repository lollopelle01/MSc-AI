"""
notebooks/anomaly_detection/pipeline_1_2_3/fairness_correction.py

Fixes the fairness problem didi_breakdown.py found in the soft-fallback
RISK_SCORE: DIDI of 8.570 on the test split, driven partly by some fallback
tiers having much worse individual fairness (core/forest: DIDI 7.289) and
partly by tier assignment itself correlating with protected attributes
(DIDI of tier assignment: 2.923).

The team's own Lagrangian-constrained approach (07-ciml, lesson 2 --
CstDIDIRegressor / LagDualDIDIRegressor in util/decision_util.py) is the
in-syllabus technique for this, but it needs TensorFlow/Keras, which isn't
installed in this environment (the same constraint gio/method_b's own
autoencoder already had to work around). Instead, this applies a simpler,
sklearn-only POST-HOC RECALIBRATION: DIDI is literally defined as the sum,
over every protected group, of |group average prediction - global average
prediction| -- so directly re-centering each group's scores onto the global
mean attacks the exact quantity DIDI measures, rather than an indirect proxy
for it.

To keep this an honest, generalizing correction rather than something that
only erases the number on the data it was measured on: the group-mean
SHIFTS are computed once on the TRAIN split only, then the same shifts
(not re-computed) are applied to the TEST split, the same train/apply
discipline every model in this project already follows.

Corrects across all three protected attributes (PTGENDER, PTEDUCAT_BUCKET,
PTMARRY) iteratively, since removing one attribute's disparity can
reintroduce a bit of another's (they're not independent groupings of the
same patients) -- a few passes converges.

Usage:
    python3 fairness_correction.py
Outputs (written next to this script):
    - fairness_correction_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
N_CALIBRATION_PASSES = 8


def make_protected(df):
    return {
        "PTGENDER": (1, 2),
        "PTEDUCAT_BUCKET": (0, 1),
        "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique())),
    }


def fit_group_shifts(df, score_col, protected, n_passes=N_CALIBRATION_PASSES):
    """Computes, on TRAIN only, the additive shift each protected group
    needs so its mean matches the overall train mean, iterated a few
    passes since the three attributes overlap. Returns a list of
    (attribute, value, shift) applied in order at apply time."""
    scores = df[score_col].to_numpy(dtype=float).copy()
    global_mean = scores.mean()
    schedule = []
    for _ in range(n_passes):
        for attr, domain in protected.items():
            for v in domain:
                mask = df[attr].to_numpy() == v
                if mask.sum() == 0:
                    continue
                shift = global_mean - scores[mask].mean()
                scores[mask] += shift
                schedule.append((attr, v, shift))
    return schedule


def apply_group_shifts(df, score_col, schedule):
    scores = df[score_col].to_numpy(dtype=float).copy()
    for attr, v, shift in schedule:
        mask = df[attr].to_numpy() == v
        scores[mask] += shift
    return np.clip(scores, 0.0, 1.0)


def evaluate_policy(df, risk_col, lines, label):
    df = df.copy()
    cmodel = du.ConversionCostModel(
        check_cost=CHECK_COST, missed_conversion_cost=MISSED_CONVERSION_COST,
        safe_interval_months=SAFE_INTERVAL_MONTHS, reference_interval_months=REFERENCE_INTERVAL_MONTHS,
    )
    recommended = du.recommend_interval(
        df[risk_col].values, interval_menu_months=(3, 6, 12),
        check_cost=CHECK_COST, missed_conversion_cost=MISSED_CONVERSION_COST,
        reference_interval_months=REFERENCE_INTERVAL_MONTHS,
    )
    total_cost, over_threshold_count, _ = cmodel.cost(
        rid_ids=df["RID"].values, risk_scores=df[risk_col].values,
        threshold=0.5, interval_months=recommended, return_margin=False,
    )
    didi = du.compute_didi(df, recommended, make_protected(df))
    outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
    outcomes["RECOMMENDED_INTERVAL"] = recommended
    worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
    margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
    catch_rate = (margin >= 0).mean() * 100
    lines.append(f"  {label:28s} n={len(df):5d}  cost={total_cost:10.1f}  "
                 f"DIDI={didi:.3f}  catch={catch_rate:5.1f}%")
    return total_cost, didi, catch_rate


def main():
    data = pd.read_csv(DATA_PATH)
    biomarker_cols_to_fill = [
        "HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM",
        "SUMMARY_SUVR", "ABETA_RATIO", "TAU", "PTAU",
    ]
    data = du.forward_fill_by_patient(data, biomarker_cols_to_fill, id_col="RID", date_col="EXAMDATE_DX")
    anomaly = pd.read_csv(ANOMALY_KEYED_PATH, usecols=["RID", "VISCODE2_norm", "HI"])
    anomaly = anomaly.dropna(subset=["VISCODE2_norm"]).rename(columns={"HI": "ANOMALY_SCORE"}).drop_duplicates(["RID", "VISCODE2_norm"])
    data = data.merge(anomaly, on=["RID", "VISCODE2_norm"], how="left")
    data = hu.add_nominal_month(data)
    panel = hu.build_hazard_panel(data)

    CORE = ["HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM", "SUMMARY_SUVR", "AGE", "PRIOR_DIAGNOSIS", "NOMINAL_MONTH"]
    EXTENDED = CORE + ["TAU", "PTAU"]
    CORE_A = CORE + ["ANOMALY_SCORE"]
    EXTENDED_A = EXTENDED + ["ANOMALY_SCORE"]
    feature_sets = {"core": CORE, "extended": EXTENDED, "core+anomaly": CORE_A, "extended+anomaly": EXTENDED_A}

    train_panel, test_panel = du.subject_train_test_split(panel, test_fraction=0.25, random_state=42)

    def at_risk_complete(df, cols):
        return df[df["AT_RISK"]].dropna(subset=cols)

    fitted = {}
    for tier_name, cols in feature_sets.items():
        train_df = at_risk_complete(train_panel, cols)
        if len(train_df) < 30:
            continue
        scaler = StandardScaler().fit(train_df[cols])
        model_lr = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, class_weight="balanced", random_state=42)
        model_lr.fit(scaler.transform(train_df[cols]), train_df["EVENT_AT_VISIT"])
        fitted[(tier_name, "logistic")] = (model_lr, scaler, cols)
        model_rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
        model_rf.fit(train_df[cols], train_df["EVENT_AT_VISIT"])
        fitted[(tier_name, "forest")] = (model_rf, None, cols)

    FALLBACK_ORDER = [
        ("extended+anomaly", "forest"), ("extended", "forest"),
        ("core+anomaly", "forest"), ("core", "forest"), ("core", "logistic"),
    ]
    panel_fb = panel.copy()
    panel_fb["RISK_SCORE"] = np.nan
    panel_fb["RISK_SCORE_TIER"] = None
    remaining = panel_fb["RISK_SCORE"].isna()
    for tier_name, kind in FALLBACK_ORDER:
        if (tier_name, kind) not in fitted or not remaining.any():
            continue
        model, scaler, cols = fitted[(tier_name, kind)]
        eligible = remaining & panel_fb[cols].notna().all(axis=1)
        if not eligible.any():
            continue
        X = panel_fb.loc[eligible, cols]
        X_input = scaler.transform(X) if scaler is not None else X.to_numpy(dtype=float)
        panel_fb.loc[eligible, "RISK_SCORE"] = model.predict_proba(X_input)[:, 1]
        panel_fb.loc[eligible, "RISK_SCORE_TIER"] = f"{tier_name}/{kind}"
        remaining = panel_fb["RISK_SCORE"].isna()

    scoreable_fb = panel_fb.dropna(subset=["RISK_SCORE"]).copy()
    scoreable_fb["PTEDUCAT_BUCKET"] = du.bucket_educat(scoreable_fb["PTEDUCAT"], split_at=16)

    train_fb, test_fb = du.split_by_rid_membership(
        scoreable_fb, set(train_panel["RID"]), set(test_panel["RID"])
    )  # fixed: re-splitting this filtered subset independently leaked most "test"
       # patients from train_panel -- see split_by_rid_membership docstring.

    # fit the correction on TRAIN only
    protected = make_protected(train_fb)
    schedule = fit_group_shifts(train_fb, "RISK_SCORE", protected)

    train_fb["RISK_SCORE_FAIR"] = apply_group_shifts(train_fb, "RISK_SCORE", schedule)
    test_fb["RISK_SCORE_FAIR"] = apply_group_shifts(test_fb, "RISK_SCORE", schedule)

    lines = []
    lines.append("=== Fairness correction: post-hoc group recalibration (fit on train, applied to test) ===")
    lines.append(f"Protected attributes: {list(protected.keys())}")
    lines.append(f"Calibration passes: {N_CALIBRATION_PASSES}")
    lines.append("")
    lines.append("-- BEFORE correction --")
    evaluate_policy(train_fb, "RISK_SCORE", lines, "Train, before")
    evaluate_policy(test_fb, "RISK_SCORE", lines, "Test, before")
    lines.append("")
    lines.append("-- AFTER correction --")
    evaluate_policy(train_fb, "RISK_SCORE_FAIR", lines, "Train, after")
    evaluate_policy(test_fb, "RISK_SCORE_FAIR", lines, "Test, after")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "fairness_correction_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
