"""
notebooks/anomaly_detection/method_b_autoencoder_hi/min_visits_gain_check.py

Backs up the design choice to keep MIN_VISITS_PER_PATIENT=2 (reliability
over coverage) with a direct measurement: how much standalone anomaly-
detection performance would actually be gained by relaxing it to 1 -- using
Method B's PCA HI trajectory method itself (pca_hi_trajectory.py), not a
downstream pipeline number, so this isolates point 1 in isolation exactly
as the user asked.

Approach: build the longitudinal table once at min_visits=1 (the superset),
fit the SAME PCA baseline (on CN visits within that superset) used by
pca_hi_trajectory.py, then compare standalone CN-vs-MCI+AD AUC (last visit
per patient, since the marginal patients only ever have one) across three
populations:
  - "kept"     : RIDs with >=2 qualifying visits (today's actual population)
  - "expanded" : RIDs with >=1 qualifying visit (what MIN_VISITS_PER_PATIENT=1
                 would include)
  - "marginal" : RIDs with EXACTLY 1 qualifying visit (the newly-included
                 patients relaxing the threshold would add -- the ones
                 coverage_diagnosis.py counted as "984 patients gained")

Usage:
    python3 min_visits_gain_check.py
Outputs (written next to this script):
    - min_visits_gain_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "method_a_pca_density"))
from method_a import fit_pca_baseline, compute_t2_q, control_limits  # noqa: E402
from load_longitudinal import build_longitudinal_table  # noqa: E402


def last_visit_auc(df, mask_col_values, group):
    mask = (mask_col_values == "CN") | (mask_col_values == group)
    if mask.sum() < 10 or (mask_col_values[mask] == group).sum() < 3 or (mask_col_values[mask] == "CN").sum() < 3:
        return None, int(mask.sum())
    y = (mask_col_values[mask] == group).astype(int)
    return roc_auc_score(y, df.loc[mask, "HI"].to_numpy()), int(mask.sum())


def main():
    # Build at min_visits=1 -- the superset that contains both populations.
    df, feature_cols = build_longitudinal_table(min_visits=1)
    df = df.dropna(subset=["DIAGNOSIS_LABEL"]).reset_index(drop=True)

    visits_per_rid = df.groupby("RID").size()
    rids_kept = set(visits_per_rid[visits_per_rid >= 2].index)
    rids_marginal = set(visits_per_rid[visits_per_rid == 1].index)

    print(f"RIDs with >=2 visits (kept today): {len(rids_kept)}")
    print(f"RIDs with exactly 1 visit (marginal, newly included at min_visits=1): {len(rids_marginal)}")

    # Fit ONE PCA baseline on all CN visits in the full superset, so every
    # population below is scored by the exact same model -- isolates the
    # population/coverage effect, not a refitting artifact.
    X = df[feature_cols].to_numpy()
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").to_numpy()
    scaler, pca, _, cum_var = fit_pca_baseline(X[cn_mask])
    T2, Q = compute_t2_q(X, scaler, pca)
    t2_ucl, q_ucl = control_limits(T2[cn_mask], Q[cn_mask], pca.n_components_, cn_mask.sum())
    df["HI"] = T2 / t2_ucl + Q / q_ucl

    last = df.sort_values("EXAMDATE").groupby("RID").last().reset_index()

    def subset_auc_report(rid_set, label):
        sub = last[last["RID"].isin(rid_set)]
        lines = [f"-- {label} (n_patients={len(sub)}) --"]
        for group in ["MCI", "AD"]:
            auc, n = last_visit_auc(sub, sub["DIAGNOSIS_LABEL"].to_numpy(), group)
            if auc is None:
                lines.append(f"  CN vs {group}: insufficient data (n={n})")
            else:
                lines.append(f"  CN vs {group}: {auc:.3f}  (n={n})")
        mask_any = sub["DIAGNOSIS_LABEL"].isin(["CN", "MCI", "AD"])
        y_any = (sub.loc[mask_any, "DIAGNOSIS_LABEL"] != "CN").astype(int)
        if mask_any.sum() >= 10 and y_any.sum() >= 3 and (y_any == 0).sum() >= 3:
            auc_any = roc_auc_score(y_any, sub.loc[mask_any, "HI"])
            lines.append(f"  CN vs MCI+AD: {auc_any:.3f}  (n={mask_any.sum()})")
        else:
            lines.append(f"  CN vs MCI+AD: insufficient data (n={mask_any.sum()})")
        return lines

    lines = []
    lines.append("=== Standalone PCA anomaly-detection gain from relaxing MIN_VISITS_PER_PATIENT ===")
    lines.append("Same PCA baseline (fit once on all CN visits at min_visits=1) scores all three")
    lines.append("populations below -- differences reflect the POPULATION, not a refitting artifact.")
    lines.append("")
    lines.extend(subset_auc_report(rids_kept, "KEPT today (>=2 visits, current MIN_VISITS_PER_PATIENT)"))
    lines.append("")
    lines.extend(subset_auc_report(rids_kept | rids_marginal, "EXPANDED (>=1 visit, MIN_VISITS_PER_PATIENT=1)"))
    lines.append("")
    lines.extend(subset_auc_report(rids_marginal, "MARGINAL ONLY (exactly 1 visit -- the newly-included patients)"))
    lines.append("")
    lines.append("Diagnosis mix of the marginal (newly-included) group:")
    marg_mix = last[last["RID"].isin(rids_marginal)]["DIAGNOSIS_LABEL"].value_counts()
    for k, v in marg_mix.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Interpretation: compare KEPT vs EXPANDED -- if the aggregate AUC barely moves,")
    lines.append("relaxing the threshold adds coverage without adding standalone detection power,")
    lines.append("which is exactly the 'small gain' argument for keeping MIN_VISITS_PER_PATIENT=2.")
    lines.append("The MARGINAL-only AUC (if computable) shows directly how reliable a health index")
    lines.append("built from a single visit actually is for the patients it would newly reach.")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(HERE, "min_visits_gain_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
