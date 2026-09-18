"""
notebooks/anomaly_detection/final_pipeline_compare.py

Two things, both using the real, high-coverage key join instead of the
180-day merge_asof workaround:

1. Builds hi_trajectories_keyed.csv for the AUTOENCODER the same way
   final_key_join.py already built pca_hi_trajectories_keyed.csv for PCA
   (direct RID + nearest-EXAMDATE_DX match against final.csv, skipping the
   DXSUM-VISCODE2 intermediate), so both methods are compared at the same,
   much higher coverage instead of the autoencoder's own 67.1%.

2. Reruns approach_pipeline_anomaly.ipynb's own cost/DIDI/catch-rate
   comparison for both, this time joining on the real RID + VISCODE2_norm
   key rather than merge_asof, to see whether the higher coverage changes
   which method wins downstream.

Usage:
    python3 final_pipeline_compare.py
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", ".."))
from util import decision_util as du  # noqa: E402

DATA_DIR = os.path.join(HERE, "..", "..", "datasets")
AE_HI_PATH = os.path.join(HERE, "method_b_autoencoder_hi", "hi_trajectories.csv")
PCA_HI_PATH = os.path.join(HERE, "method_b_autoencoder_hi", "pca_hi_trajectories.csv")
WINDOW_DAYS = 180

CHECK_COST = 1.0
MISSED_CONVERSION_COST = 20.0
REFERENCE_INTERVAL_MONTHS = 12.0
SAFE_INTERVAL_MONTHS = 6
PIPELINE_WEIGHT = 0.2


def build_keyed(hi_path, out_path, label):
    final = pd.read_csv(f"{DATA_DIR}/final.csv", low_memory=False)
    final["EXAMDATE_DX"] = pd.to_datetime(final["EXAMDATE_DX"], errors="coerce")
    final_small = final[["RID", "VISCODE2_norm", "EXAMDATE_DX"]].dropna(subset=["EXAMDATE_DX"])

    hi = pd.read_csv(hi_path)
    hi["EXAMDATE"] = pd.to_datetime(hi["EXAMDATE"], errors="coerce")

    cand = hi.merge(final_small, on="RID", how="inner")
    cand["_gap"] = (cand["EXAMDATE"] - cand["EXAMDATE_DX"]).dt.days.abs()
    cand = cand[cand["_gap"] <= WINDOW_DAYS]
    cand = cand.sort_values("_gap").drop_duplicates(["RID", "EXAMDATE"], keep="first")

    keyed = hi.merge(
        cand[["RID", "EXAMDATE", "VISCODE2_norm", "_gap"]].rename(columns={"_gap": "MATCH_GAP_DAYS"}),
        on=["RID", "EXAMDATE"], how="left",
    )
    coverage = keyed["VISCODE2_norm"].notna().mean()
    print(f"{label}: {keyed['VISCODE2_norm'].notna().sum()}/{len(keyed)} rows keyed ({coverage:.1%})")
    keyed.to_csv(out_path, index=False)
    return coverage


def load_base_data():
    data = pd.read_csv(f"{DATA_DIR}/final.csv")
    biomarker_cols_to_fill = [
        "HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM",
        "SUMMARY_SUVR", "ABETA_RATIO", "TAU", "PTAU",
    ]
    data = du.forward_fill_by_patient(
        data, biomarker_cols_to_fill, id_col="RID", date_col="EXAMDATE_DX",
    )
    data["RISK_SCORE"] = du.placeholder_risk_score(data)
    return data


def attach_hi_by_key(data, keyed_hi_path):
    """Real join: RID + VISCODE2_norm, exact match, no date window at all."""
    hi = pd.read_csv(keyed_hi_path, usecols=["RID", "VISCODE2_norm", "HI"])
    hi = hi.dropna(subset=["VISCODE2_norm"])
    hi = hi.drop_duplicates(["RID", "VISCODE2_norm"], keep="first")
    out = data.merge(hi, on=["RID", "VISCODE2_norm"], how="left")
    coverage = out["HI"].notna().mean()
    return out, coverage


def add_pipeline_risk_score(df, hi_lo, hi_hi):
    df = df.copy()
    hi_norm = ((df["HI"] - hi_lo) / (hi_hi - hi_lo)).clip(0, 1)
    df["RISK_SCORE_PIPELINE"] = df["RISK_SCORE"]
    matched = hi_norm.notna()
    df.loc[matched, "RISK_SCORE_PIPELINE"] = (
        (1 - PIPELINE_WEIGHT) * df.loc[matched, "RISK_SCORE"]
        + PIPELINE_WEIGHT * hi_norm[matched]
    )
    return df


def make_protected(df):
    return {
        "PTGENDER": (1, 2),
        "PTEDUCAT_BUCKET": (0, 1),
        "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique())),
    }


def evaluate_policy(df, risk_col, cmodel, label):
    df = df.copy()
    df["PTEDUCAT_BUCKET"] = du.bucket_educat(df["PTEDUCAT"], split_at=16)
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
    print(f"  {label:26s} cost={total_cost:10.1f}  over_thresh={over_threshold_count:5d}  "
          f"DIDI={didi:.3f}  catch={catch_rate:5.1f}%")
    return total_cost, over_threshold_count, didi, catch_rate


def run_variant(name, keyed_hi_path):
    print(f"\n=== {name} (keyed join) ===")
    data = load_base_data()
    data, coverage = attach_hi_by_key(data, keyed_hi_path)
    print(f"HI coverage on final.csv rows (exact RID+VISCODE2_norm join): {coverage:.1%}")

    train_df, test_df = du.subject_train_test_split(data, test_fraction=0.25, random_state=42)
    hi_lo, hi_hi = train_df["HI"].quantile([0.05, 0.95])
    train_df = add_pipeline_risk_score(train_df, hi_lo, hi_hi)
    test_df = add_pipeline_risk_score(test_df, hi_lo, hi_hi)

    cmodel = du.ConversionCostModel(
        check_cost=CHECK_COST, missed_conversion_cost=MISSED_CONVERSION_COST,
        safe_interval_months=SAFE_INTERVAL_MONTHS, reference_interval_months=REFERENCE_INTERVAL_MONTHS,
    )
    results = {}
    for split_name, df in [("Train", train_df), ("Test", test_df)]:
        results[(split_name, "without")] = evaluate_policy(df, "RISK_SCORE", cmodel, f"{split_name}, without")
        results[(split_name, "with")] = evaluate_policy(df, "RISK_SCORE_PIPELINE", cmodel, f"{split_name}, with {name}")
    for split_name in ["Train", "Test"]:
        c0 = results[(split_name, "without")][0]
        c1 = results[(split_name, "with")][0]
        print(f"  {split_name}: cost change from adding {name} HI: {(c1-c0)/c0*100:+.2f}%")
    return results, coverage


if __name__ == "__main__":
    ae_keyed_path = os.path.join(HERE, "method_b_autoencoder_hi", "hi_trajectories_keyed.csv")
    pca_keyed_path = os.path.join(HERE, "method_b_autoencoder_hi", "pca_hi_trajectories_keyed.csv")

    print("--- Building keyed files ---")
    build_keyed(AE_HI_PATH, ae_keyed_path, "Autoencoder")
    if not os.path.exists(pca_keyed_path):
        build_keyed(PCA_HI_PATH, pca_keyed_path, "PCA")
    else:
        print("PCA keyed file already exists, reusing it.")

    ae_results, ae_cov = run_variant("Autoencoder", ae_keyed_path)
    pca_results, pca_cov = run_variant("PCA", pca_keyed_path)

    print("\n=== FINAL head-to-head, Test split, 'with pipeline' (real key join) ===")
    for name, res, cov in [("Autoencoder", ae_results, ae_cov), ("PCA", pca_results, pca_cov)]:
        cost, over, didi, catch = res[("Test", "with")]
        c0 = res[("Test", "without")][0]
        print(f"  {name:12s} final.csv HI coverage={cov:5.1%}  cost={cost:10.1f} "
              f"({(cost-c0)/c0*100:+.2f}% vs no-HI)  DIDI={didi:.3f}  catch={catch:5.1f}%")
