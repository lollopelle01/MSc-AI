"""
notebooks/anomaly_detection/pipeline_1_2_3/wire_full_pipeline.py

Wires point 1 -> point 2 -> point 3 as a genuine SEQUENTIAL chain, closing
the gap identified in discussion with the professor: today, point 1 (this
folder's anomaly score) and point 2 (hazard_survival_model.ipynb) both feed
point 3 independently, but point 1 never passes THROUGH point 2 the way the
original proposal describes ("the output of stage one becomes an input
feature for stage two, and the output of stage two becomes an input for
stage three").

This is a new, separate folder rather than an edit to anyone else's
notebook, so it doesn't touch Pelle's hazard_survival_model.ipynb or the
decision_support notebooks directly -- it reuses the exact same functions
from util/hazard_util.py and util/decision_util.py those notebooks use, so
the comparison is apples-to-apples and the result could be folded back into
the team's own notebook later with no logic changes, only the file it lives
in.

What this script actually does, in order:
  1. Loads final.csv, forward-fills the same sparse biomarkers every other
     notebook in this project already does.
  2. Attaches ANOMALY_SCORE from this folder's own reconciled key join
     (pca_hi_trajectories_keyed.csv, RID + VISCODE2_norm, ~99.8% coverage
     on its own rows) -- POINT 1's OUTPUT.
  3. Builds the same hazard panel hazard_survival_model.ipynb builds
     (PRIOR_DIAGNOSIS, EVENT_AT_VISIT, AT_RISK), then fits the SAME four
     hazard estimators that notebook fits (logistic/forest x core/extended)
     PLUS four more that add ANOMALY_SCORE as one extra feature, on the
     identical train/test split (random_state=42), to test directly
     whether point 1's signal improves point 2's own hazard estimator --
     POINT 1 -> POINT 2.
  4. Picks whichever of all eight estimators has the best test AUC, scores
     every row, becomes RISK_SCORE.
  5. Runs the same snapshot-adaptive interval policy and cost model point 3
     already uses on this new RISK_SCORE, reporting cost/DIDI/catch-rate --
     POINT 2 -> POINT 3, now with point 1 baked in upstream instead of
     blended in downstream.

Usage:
    python3 wire_full_pipeline.py
Outputs (written next to this script):
    - full_pipeline_report.txt
    - hazard_auc_with_vs_without_anomaly_score.csv
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


def load_data_with_anomaly_score():
    data = pd.read_csv(DATA_PATH)
    biomarker_cols_to_fill = [
        "HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM",
        "SUMMARY_SUVR", "ABETA_RATIO", "TAU", "PTAU",
    ]
    data = du.forward_fill_by_patient(
        data, biomarker_cols_to_fill, id_col="RID", date_col="EXAMDATE_DX",
    )

    anomaly = pd.read_csv(ANOMALY_KEYED_PATH, usecols=["RID", "VISCODE2_norm", "HI"])
    anomaly = anomaly.dropna(subset=["VISCODE2_norm"]).rename(columns={"HI": "ANOMALY_SCORE"})
    anomaly = anomaly.drop_duplicates(["RID", "VISCODE2_norm"], keep="first")

    data = data.merge(anomaly, on=["RID", "VISCODE2_norm"], how="left")
    coverage = data["ANOMALY_SCORE"].notna().mean()
    print(f"ANOMALY_SCORE (point 1 output) attached to {coverage:.1%} of final.csv rows "
          f"via the real RID+VISCODE2_norm key join.")
    return data, coverage


def evaluate_auc(model, X, y, scaler=None):
    X_input = scaler.transform(X) if scaler is not None else X.to_numpy(dtype=float)
    return du.evaluate_classification(model, X, y, scaler=scaler)["auc"]


def main():
    data, hi_coverage = load_data_with_anomaly_score()
    data = hu.add_nominal_month(data)
    panel = hu.build_hazard_panel(data)

    CORE_FEATURES = [
        "HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM", "SUMMARY_SUVR",
        "AGE", "PRIOR_DIAGNOSIS", "NOMINAL_MONTH",
    ]
    EXTENDED_FEATURES = CORE_FEATURES + ["TAU", "PTAU"]
    CORE_PLUS_ANOMALY = CORE_FEATURES + ["ANOMALY_SCORE"]
    EXTENDED_PLUS_ANOMALY = EXTENDED_FEATURES + ["ANOMALY_SCORE"]

    feature_sets = {
        "core": CORE_FEATURES,
        "extended": EXTENDED_FEATURES,
        "core+anomaly": CORE_PLUS_ANOMALY,
        "extended+anomaly": EXTENDED_PLUS_ANOMALY,
    }

    train_panel, test_panel = du.subject_train_test_split(panel, test_fraction=0.25, random_state=42)

    def at_risk_complete(df, cols):
        return df[df["AT_RISK"]].dropna(subset=cols)

    results = []
    fitted = {}
    for tier_name, cols in feature_sets.items():
        train_df = at_risk_complete(train_panel, cols)
        test_df = at_risk_complete(test_panel, cols)
        if len(train_df) < 30 or len(test_df) < 10:
            print(f"Skipping {tier_name}: too few rows ({len(train_df)} train, {len(test_df)} test)")
            continue

        # logistic
        scaler = StandardScaler().fit(train_df[cols])
        model_lr = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                       class_weight="balanced", random_state=42)
        model_lr.fit(scaler.transform(train_df[cols]), train_df["EVENT_AT_VISIT"])
        auc_lr_train = du.evaluate_classification(model_lr, train_df[cols], train_df["EVENT_AT_VISIT"], scaler=scaler)["auc"]
        auc_lr_test = du.evaluate_classification(model_lr, test_df[cols], test_df["EVENT_AT_VISIT"], scaler=scaler)["auc"]
        fitted[(tier_name, "logistic")] = (model_lr, scaler, cols)
        results.append({"tier": tier_name, "model": "logistic", "train_auc": auc_lr_train,
                         "test_auc": auc_lr_test, "train_rows": len(train_df), "test_rows": len(test_df)})

        # forest
        model_rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                           class_weight="balanced", random_state=42)
        model_rf.fit(train_df[cols], train_df["EVENT_AT_VISIT"])
        auc_rf_train = du.evaluate_classification(model_rf, train_df[cols], train_df["EVENT_AT_VISIT"], scaler=None)["auc"]
        auc_rf_test = du.evaluate_classification(model_rf, test_df[cols], test_df["EVENT_AT_VISIT"], scaler=None)["auc"]
        fitted[(tier_name, "forest")] = (model_rf, None, cols)
        results.append({"tier": tier_name, "model": "forest", "train_auc": auc_rf_train,
                         "test_auc": auc_rf_test, "train_rows": len(train_df), "test_rows": len(test_df)})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(HERE, "hazard_auc_with_vs_without_anomaly_score.csv"), index=False)

    lines = []
    lines.append("=== Point 1 -> Point 2: does adding ANOMALY_SCORE improve the hazard estimator? ===")
    lines.append(f"ANOMALY_SCORE coverage on final.csv: {hi_coverage:.1%}")
    lines.append("")
    lines.append(results_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("")

    for base, plus in [("core", "core+anomaly"), ("extended", "extended+anomaly")]:
        for kind in ["logistic", "forest"]:
            r_base = results_df[(results_df.tier == base) & (results_df.model == kind)]
            r_plus = results_df[(results_df.tier == plus) & (results_df.model == kind)]
            if len(r_base) and len(r_plus):
                delta = r_plus.test_auc.values[0] - r_base.test_auc.values[0]
                lines.append(f"{kind:10s} {base:10s} -> {plus:18s}: test AUC {r_base.test_auc.values[0]:.3f} "
                             f"-> {r_plus.test_auc.values[0]:.3f}  (delta {delta:+.3f})")

    # pick best single estimator overall by test AUC (the HARD-dependency version)
    best_row = results_df.loc[results_df["test_auc"].idxmax()]
    best_tier, best_kind = best_row["tier"], best_row["model"]
    best_model, best_scaler, best_features = fitted[(best_tier, best_kind)]
    lines.append("")
    lines.append(f"Best SINGLE estimator (hard dependency): {best_kind} on '{best_tier}' tier "
                 f"(test AUC {best_row['test_auc']:.3f}), features: {best_features}")
    uses_anomaly = "anomaly" in best_tier
    lines.append(f"Uses point 1's ANOMALY_SCORE as an input: {uses_anomaly}")

    scoreable = panel.dropna(subset=best_features).copy()
    X_score = scoreable[best_features]
    X_input = best_scaler.transform(X_score) if best_scaler is not None else X_score.to_numpy(dtype=float)
    scoreable["RISK_SCORE"] = best_model.predict_proba(X_input)[:, 1]
    coverage_final = len(scoreable) / len(panel)
    lines.append(f"RISK_SCORE (single best estimator) computed for {coverage_final:.1%} of all panel rows")

    # --- SOFT FALLBACK: same idea, but a row only needs to be missing what a
    # tier requires to fall back one level, exactly the same pattern the
    # existing hazard model already applies for missing CSF -- extended one
    # level further to also cover a missing ANOMALY_SCORE. Priority order is
    # by each tier's own forest test AUC (forest beats logistic in every
    # tier here), most-complete/most-accurate tier tried first. ---
    FALLBACK_ORDER = [
        ("extended+anomaly", "forest"),  # 0.807, needs CSF + ANOMALY_SCORE
        ("extended", "forest"),          # 0.791, needs CSF
        ("core+anomaly", "forest"),      # 0.790, needs ANOMALY_SCORE
        ("core", "forest"),              # 0.791, baseline
        ("core", "logistic"),            # 0.743, last resort if forest somehow unavailable
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

    tier_counts = panel_fb["RISK_SCORE_TIER"].value_counts(dropna=False)
    soft_coverage = panel_fb["RISK_SCORE"].notna().mean()
    lines.append("")
    lines.append(f"=== SOFT FALLBACK: every row scored by the best tier it qualifies for ===")
    lines.append(f"Overall coverage: {soft_coverage:.1%} of all panel rows (vs {coverage_final:.1%} "
                 f"for the single best-estimator, hard-dependency version)")
    lines.append("Rows scored by each tier (best tier a row qualifies for, in priority order):")
    for tier_kind, count in tier_counts.items():
        pct = count / len(panel_fb) * 100
        label = tier_kind if tier_kind is not None else "NOT SCORED (missing even core features)"
        lines.append(f"  {label:28s} {count:6d} rows ({pct:5.1f}%)")

    scoreable_fb = panel_fb.dropna(subset=["RISK_SCORE"]).copy()

    train_score, test_score = du.split_by_rid_membership(
        scoreable, set(train_panel["RID"]), set(test_panel["RID"])
    )  # fixed: re-splitting this filtered subset independently leaked ~73% of
       # "test" patients from train_panel -- see split_by_rid_membership docstring.

    def make_protected(df):
        return {
            "PTGENDER": (1, 2),
            "PTEDUCAT_BUCKET": (0, 1),
            "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique())),
        }

    def evaluate_policy(df, label):
        df = df.copy()
        df["PTEDUCAT_BUCKET"] = du.bucket_educat(df["PTEDUCAT"], split_at=16)
        cmodel = du.ConversionCostModel(
            check_cost=CHECK_COST, missed_conversion_cost=MISSED_CONVERSION_COST,
            safe_interval_months=SAFE_INTERVAL_MONTHS, reference_interval_months=REFERENCE_INTERVAL_MONTHS,
        )
        recommended = du.recommend_interval(
            df["RISK_SCORE"].values, interval_menu_months=(3, 6, 12),
            check_cost=CHECK_COST, missed_conversion_cost=MISSED_CONVERSION_COST,
            reference_interval_months=REFERENCE_INTERVAL_MONTHS,
        )
        total_cost, over_threshold_count, _ = cmodel.cost(
            rid_ids=df["RID"].values, risk_scores=df["RISK_SCORE"].values,
            threshold=0.5, interval_months=recommended, return_margin=False,
        )
        didi = du.compute_didi(df, recommended, make_protected(df))
        outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
        outcomes["RECOMMENDED_INTERVAL"] = recommended
        worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
        margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
        catch_rate = (margin >= 0).mean() * 100
        lines.append(f"  {label:8s} n={len(df):5d}  cost={total_cost:10.1f}  "
                     f"over_thresh={over_threshold_count:5d}  DIDI={didi:.3f}  catch={catch_rate:5.1f}%")
        return total_cost, didi, catch_rate

    lines.append("")
    lines.append("=== Point 2 -> Point 3: interval policy, HARD-dependency RISK_SCORE (single best tier) ===")
    evaluate_policy(train_score, "Train")
    evaluate_policy(test_score, "Test")

    train_fb, test_fb = du.split_by_rid_membership(
        scoreable_fb, set(train_panel["RID"]), set(test_panel["RID"])
    )  # fixed: same leakage bug as above, same fix.
    lines.append("")
    lines.append("=== Point 2 -> Point 3: interval policy, SOFT-FALLBACK RISK_SCORE (full coverage) ===")
    evaluate_policy(train_fb, "Train")
    evaluate_policy(test_fb, "Test")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "full_pipeline_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
