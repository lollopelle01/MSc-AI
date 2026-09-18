"""
notebooks/anomaly_detection/pipeline_1_2_3/compare_refit_vs_rethreshold.py

Fast, no-refitting comparison: does REFITTING ExponentiatedGradient at
ratio=100 (refit_at_ratio100_scores.csv) recover catch rate without
giving up DIDI, compared to just RE-THRESHOLDING the original ratio=20 fit
at ratio=100 (calibrated_risk_scores.csv, section 5d's finding that this
doesn't work cleanly)?

Usage:
    python3 compare_refit_vs_rethreshold.py
Outputs (written next to this script):
    - refit_vs_rethreshold_report.txt
"""

import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fairness_expgrad_pipeline as fep  # noqa: E402

REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402

TARGET_RATIO = 100
ORIGINAL_RATIO = 20


def policy_row(df, col, ratio):
    cmodel = du.ConversionCostModel(check_cost=1.0, missed_conversion_cost=ratio,
                                     safe_interval_months=6, reference_interval_months=12.0)
    recommended = du.recommend_interval(df[col].values, interval_menu_months=(3, 6, 12),
                                         check_cost=1.0, missed_conversion_cost=ratio,
                                         reference_interval_months=12.0)
    total_cost, _, _ = cmodel.cost(rid_ids=df["RID"].values, risk_scores=df[col].values,
                                    threshold=0.5, interval_months=recommended, return_margin=False)
    protected = {"PTGENDER": (1, 2), "PTEDUCAT_BUCKET": (0, 1),
                 "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique()))}
    didi = du.compute_didi(df, recommended, protected)
    outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
    outcomes["RECOMMENDED_INTERVAL"] = recommended
    worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
    margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
    catch_rate = (margin >= 0).mean() * 100
    auc = roc_auc_score(df["EVENT_AT_VISIT"], df[col])
    return total_cost, didi, catch_rate, auc


def main():
    old = pd.read_csv(os.path.join(HERE, "calibrated_risk_scores.csv"))
    new = pd.read_csv(os.path.join(HERE, "refit_at_ratio100_scores.csv"))
    for d in (old, new):
        d["EXAMDATE_DX"] = d["EXAMDATE_DX"].astype(str)

    # A handful of RID+EXAMDATE_DX collisions (same-day duplicate visit rows,
    # 5 of 2603 in both files) -- negligible, dropped rather than allowed to
    # fan out the merge.
    n_dup_old = old.duplicated(subset=["RID", "EXAMDATE_DX"]).sum()
    n_dup_new = new.duplicated(subset=["RID", "EXAMDATE_DX"]).sum()
    old = old.drop_duplicates(subset=["RID", "EXAMDATE_DX"])
    new = new.drop_duplicates(subset=["RID", "EXAMDATE_DX"])

    cmp_df = new.merge(old[["RID", "EXAMDATE_DX", "RISK_SCORE_fair_cal", "RISK_SCORE_plain_cal"]],
                        on=["RID", "EXAMDATE_DX"], how="inner")
    assert len(cmp_df) == len(new.merge(old[["RID", "EXAMDATE_DX"]], on=["RID", "EXAMDATE_DX"])), \
        "merge fanned out -- key still not unique after de-duplication"

    lines = []
    lines.append(f"=== Refit-at-{TARGET_RATIO} vs. re-thresholded-at-{TARGET_RATIO} vs. plain, "
                 f"all decided at ratio={TARGET_RATIO} ===")
    lines.append(f"n={len(cmp_df)} rows common to both files "
                 f"({n_dup_old} duplicate-key rows dropped from each side first).")
    lines.append("")

    variants = [
        (f"REFIT at ratio={TARGET_RATIO} (this script's question)", "RISK_SCORE_fair_r100_cal", TARGET_RATIO),
        (f"RE-THRESHOLDED only (ratio=20 fit, decided at {TARGET_RATIO} -- section 5d)", "RISK_SCORE_fair_cal", TARGET_RATIO),
        (f"Plain (no fairness), decided at {TARGET_RATIO}", "RISK_SCORE_plain_cal", TARGET_RATIO),
        ("Original fair fit, decided at its native ratio=20 (section 5c)", "RISK_SCORE_fair_cal", ORIGINAL_RATIO),
    ]
    for label, col, ratio in variants:
        cost, didi, catch, auc = policy_row(cmp_df, col, ratio)
        lines.append(f"-- {label} --")
        lines.append(f"   cost={cost:.1f}  DIDI={didi:.3f}  catch_rate={catch:.1f}%  AUC={auc:.3f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "refit_vs_rethreshold_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
