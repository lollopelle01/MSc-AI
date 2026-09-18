"""
notebooks/anomaly_detection/pipeline_1_2_3/didi_breakdown.py

The soft-fallback RISK_SCORE showed a much worse DIDI (8.570 test) than any
other version tested. Two very different explanations are possible, and
they lead to opposite conclusions:

  (A) The extra, lower-tier patients (scored by simpler fallback models)
      were always there and always this unfair -- the earlier "good" DIDI
      numbers were only ever computed on a narrower, more complete-data
      subset that happened to look fairer. If true: the pipeline isn't
      worse, the earlier fairness numbers were incomplete/optimistic, and
      this is a genuine, useful finding about data completeness masking
      disparity.

  (B) Which fallback tier a patient gets scored by is ITSELF correlated
      with a protected attribute (e.g. patients with less complete medical
      records skew toward a particular gender/education/marital-status
      group), so different protected groups are systematically being
      scored by DIFFERENT QUALITY models. If true: the fallback design
      itself introduces a new fairness problem (differential treatment by
      data completeness), separate from whatever disparity already existed
      in any single model, and that would be a real reason to reconsider
      using it as-is.

This checks both directly rather than guessing.

Usage:
    python3 didi_breakdown.py
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


def make_protected(df):
    return {
        "PTGENDER": (1, 2),
        "PTEDUCAT_BUCKET": (0, 1),
        "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique())),
    }


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

    print("=== Check A: is DIDI bad even WITHIN the top tier alone (extended+anomaly)? ===")
    top_only = scoreable_fb[scoreable_fb["RISK_SCORE_TIER"] == "extended+anomaly/forest"]
    recommended_top = du.recommend_interval(top_only["RISK_SCORE"].values, interval_menu_months=(3, 6, 12),
                                             check_cost=1.0, missed_conversion_cost=20.0, reference_interval_months=12.0)
    didi_top = du.compute_didi(top_only, recommended_top, make_protected(top_only))
    print(f"DIDI within top tier only (n={len(top_only)}): {didi_top:.3f}")

    print("\n=== Check A cont'd: DIDI within each individual tier ===")
    for tier in scoreable_fb["RISK_SCORE_TIER"].unique():
        sub = scoreable_fb[scoreable_fb["RISK_SCORE_TIER"] == tier]
        if len(sub) < 30:
            continue
        rec = du.recommend_interval(sub["RISK_SCORE"].values, interval_menu_months=(3, 6, 12),
                                     check_cost=1.0, missed_conversion_cost=20.0, reference_interval_months=12.0)
        didi_t = du.compute_didi(sub, rec, make_protected(sub))
        print(f"  {tier:24s} n={len(sub):5d}  DIDI={didi_t:.3f}")

    print("\n=== Check B: does tier assignment correlate with protected attributes? ===")
    print("(if yes, different groups are systematically scored by different-quality models)")
    for attr in ["PTGENDER", "PTEDUCAT_BUCKET", "PTMARRY"]:
        print(f"\n-- {attr} distribution by tier --")
        ct = pd.crosstab(scoreable_fb["RISK_SCORE_TIER"], scoreable_fb[attr], normalize="index")
        print(ct.to_string(float_format=lambda v: f"{v:.1%}"))

    print("\n=== Check B cont'd: overall DIDI of TIER ASSIGNMENT ITSELF (is who-gets-which-model fair?) ===")
    tier_rank = {t: i for i, t in enumerate(scoreable_fb["RISK_SCORE_TIER"].unique())}
    scoreable_fb["_tier_rank"] = scoreable_fb["RISK_SCORE_TIER"].map(tier_rank).astype(float)
    didi_tier_assignment = du.compute_didi(scoreable_fb, scoreable_fb["_tier_rank"].values, make_protected(scoreable_fb))
    print(f"DIDI of tier assignment across protected groups: {didi_tier_assignment:.3f}")


if __name__ == "__main__":
    main()
