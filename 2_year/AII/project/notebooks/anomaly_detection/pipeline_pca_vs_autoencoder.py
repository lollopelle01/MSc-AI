"""
notebooks/anomaly_detection/pipeline_pca_vs_autoencoder.py

Reproduces approach_pipeline_anomaly.ipynb's own "without pipeline" vs "with
pipeline" comparison EXACTLY (same split, same cost model, same blend weight,
same evaluate_policy function), computed here so all three variants --
no HI, autoencoder HI, PCA HI -- come from one script and one run, avoiding
any confound from comparing against a stored CSV row produced by a different
notebook execution with potentially different splits/placeholder logic.

Usage (run from notebooks/pipeline/ so util's own relative imports resolve,
or adjust sys.path as below):
    python3 pipeline_pca_vs_autoencoder.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from util import decision_util as du  # noqa: E402

DATA_PATH = os.path.join("datasets", "final.csv")
AE_HI_PATH = os.path.join("notebooks", "anomaly_detection", "method_b_autoencoder_hi", "hi_trajectories.csv")
PCA_HI_PATH = os.path.join("notebooks", "anomaly_detection", "method_b_autoencoder_hi", "pca_hi_trajectories.csv")

CHECK_COST = 1.0
MISSED_CONVERSION_COST = 20.0
REFERENCE_INTERVAL_MONTHS = 12.0
SAFE_INTERVAL_MONTHS = 6
PIPELINE_WEIGHT = 0.2


def load_base_data():
    data = pd.read_csv(DATA_PATH)
    biomarker_cols_to_fill = [
        "HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM",
        "SUMMARY_SUVR", "ABETA_RATIO", "TAU", "PTAU",
    ]
    data = du.forward_fill_by_patient(
        data, biomarker_cols_to_fill, id_col="RID", date_col="EXAMDATE_DX",
    )
    data["RISK_SCORE"] = du.placeholder_risk_score(data)
    return data


def attach_hi(data, hi_path):
    hi = pd.read_csv(hi_path, usecols=["RID", "EXAMDATE", "HI"])
    hi["EXAMDATE"] = pd.to_datetime(hi["EXAMDATE"])
    hi = hi.sort_values("EXAMDATE").drop_duplicates(["RID", "EXAMDATE"], keep="first")

    has_date = data["EXAMDATE_DX"].notna()
    dated, undated = data[has_date].sort_values("EXAMDATE_DX").copy(), data[~has_date].copy()

    dated = pd.merge_asof(
        dated, hi,
        left_on="EXAMDATE_DX", right_on="EXAMDATE", by="RID",
        direction="nearest", tolerance=pd.Timedelta(days=180),
    )
    undated["EXAMDATE"] = pd.NaT
    undated["HI"] = np.nan
    out = pd.concat([dated, undated], ignore_index=True)
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
        df[risk_col].values,
        interval_menu_months=(3, 6, 12),
        check_cost=CHECK_COST,
        missed_conversion_cost=MISSED_CONVERSION_COST,
        reference_interval_months=REFERENCE_INTERVAL_MONTHS,
    )
    total_cost, over_threshold_count, _ = cmodel.cost(
        rid_ids=df["RID"].values,
        risk_scores=df[risk_col].values,
        threshold=0.5,
        interval_months=recommended,
        return_margin=False,
    )
    didi = du.compute_didi(df, recommended, make_protected(df))

    outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
    outcomes["RECOMMENDED_INTERVAL"] = recommended
    worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
    margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
    catch_rate = (margin >= 0).mean() * 100

    print(f"  {label:28s} total_cost={total_cost:10.1f}  over_thresh={over_threshold_count:5d}  "
          f"DIDI={didi:.3f}  catch_rate={catch_rate:5.1f}%")
    return total_cost, over_threshold_count, didi, catch_rate


def run_variant(name, hi_path):
    print(f"\n=== {name} ===")
    data = load_base_data()
    data, coverage = attach_hi(data, hi_path)
    print(f"HI coverage (180-day nearest-date merge): {coverage:.1%}")

    train_df, test_df = du.subject_train_test_split(data, test_fraction=0.25, random_state=42)

    hi_lo, hi_hi = train_df["HI"].quantile([0.05, 0.95])
    train_df = add_pipeline_risk_score(train_df, hi_lo, hi_hi)
    test_df = add_pipeline_risk_score(test_df, hi_lo, hi_hi)

    cmodel = du.ConversionCostModel(
        check_cost=CHECK_COST,
        missed_conversion_cost=MISSED_CONVERSION_COST,
        safe_interval_months=SAFE_INTERVAL_MONTHS,
        reference_interval_months=REFERENCE_INTERVAL_MONTHS,
    )

    results = {}
    for split_name, df in [("Train", train_df), ("Test", test_df)]:
        results[(split_name, "without")] = evaluate_policy(df, "RISK_SCORE", cmodel, f"{split_name}, without pipeline")
        results[(split_name, "with")] = evaluate_policy(df, "RISK_SCORE_PIPELINE", cmodel, f"{split_name}, with {name}")

    for split_name in ["Train", "Test"]:
        c_without = results[(split_name, "without")][0]
        c_with = results[(split_name, "with")][0]
        pct = (c_with - c_without) / c_without * 100
        print(f"  {split_name}: cost change from adding {name} HI: {pct:+.2f}%")

    return results


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ae_results = run_variant("Autoencoder", AE_HI_PATH)
    pca_results = run_variant("PCA", PCA_HI_PATH)

    print("\n=== Head-to-head on Test split, 'with pipeline' variant ===")
    for name, res in [("Autoencoder", ae_results), ("PCA", pca_results)]:
        cost, over, didi, catch = res[("Test", "with")]
        print(f"  {name:12s} cost={cost:10.1f}  DIDI={didi:.3f}  catch_rate={catch:5.1f}%")
