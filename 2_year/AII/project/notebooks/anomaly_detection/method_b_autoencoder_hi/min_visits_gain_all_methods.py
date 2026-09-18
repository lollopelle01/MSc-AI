"""
notebooks/anomaly_detection/method_b_autoencoder_hi/min_visits_gain_all_methods.py

Extends min_visits_gain_check.py (PCA only) to all three anomaly-detection
methods this project compares -- PCA, GMM, and the autoencoder -- to answer:
does relaxing MIN_VISITS_PER_PATIENT (2 -> 1) buy real standalone detection
power for ANY of them, or is the "reliability over coverage" argument found
for PCA a general property of single-visit health indices, independent of
which density model computes them?

Same design as min_visits_gain_check.py: build the longitudinal table once
at min_visits=1 (the superset), fit each method's CN-only baseline ONCE on
that superset, then compare standalone CN-vs-MCI+AD AUC (last visit per
patient) across:
  - "kept"     : RIDs with >=2 qualifying visits (today's population)
  - "expanded" : RIDs with >=1 qualifying visit (relaxed threshold)
  - "marginal" : RIDs with EXACTLY 1 qualifying visit (newly included)

Usage:
    python3 min_visits_gain_all_methods.py
Outputs (written next to this script):
    - min_visits_gain_all_methods_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "method_a_pca_density"))
from method_a import fit_pca_baseline, compute_t2_q, control_limits  # noqa: E402
from load_longitudinal import build_longitudinal_table  # noqa: E402

RANDOM_STATE = 42
HIDDEN_DIM = 16
LATENT_DIM = 6


def fit_gmm_baseline(X_cn):
    scaler = StandardScaler().fit(X_cn)
    Xs = scaler.transform(X_cn)
    best_bic, best_gmm = np.inf, None
    for n_components in range(1, 6):
        for cov_type in ["diag", "full"]:
            gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type,
                                   random_state=RANDOM_STATE, reg_covar=1e-4, n_init=3)
            gmm.fit(Xs)
            bic = gmm.bic(Xs)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
    return scaler, best_gmm


def fit_autoencoder(X_cn):
    scaler = StandardScaler().fit(X_cn)
    Xs = scaler.transform(X_cn)
    ae = MLPRegressor(hidden_layer_sizes=(HIDDEN_DIM, LATENT_DIM, HIDDEN_DIM),
                       activation="relu", solver="adam", alpha=1e-3,
                       max_iter=2000, early_stopping=True,
                       n_iter_no_change=25, random_state=RANDOM_STATE)
    ae.fit(Xs, Xs)
    return scaler, ae


def last_visit_auc(sub, group):
    vals = sub["DIAGNOSIS_LABEL"].to_numpy()
    mask = (vals == "CN") | (vals == group)
    if mask.sum() < 10 or (vals[mask] == group).sum() < 3 or (vals[mask] == "CN").sum() < 3:
        return None, int(mask.sum())
    y = (vals[mask] == group).astype(int)
    return roc_auc_score(y, sub.loc[mask, "HI"].to_numpy()), int(mask.sum())


def subset_auc_report(last, rid_set, label):
    sub = last[last["RID"].isin(rid_set)]
    lines = [f"  -- {label} (n_patients={len(sub)}) --"]
    for group in ["MCI", "AD"]:
        auc, n = last_visit_auc(sub, group)
        lines.append(f"    CN vs {group}: {'insufficient data' if auc is None else f'{auc:.3f}'}  (n={n})")
    vals = sub["DIAGNOSIS_LABEL"].to_numpy()
    mask_any = np.isin(vals, ["CN", "MCI", "AD"])
    y_any = (vals[mask_any] != "CN").astype(int)
    if mask_any.sum() >= 10 and y_any.sum() >= 3 and (y_any == 0).sum() >= 3:
        auc_any = roc_auc_score(y_any, sub.loc[mask_any, "HI"])
        lines.append(f"    CN vs MCI+AD: {auc_any:.3f}  (n={mask_any.sum()})")
    else:
        lines.append(f"    CN vs MCI+AD: insufficient data (n={mask_any.sum()})")
    return lines


def main():
    df, feature_cols = build_longitudinal_table(min_visits=1)
    df = df.dropna(subset=["DIAGNOSIS_LABEL"]).reset_index(drop=True)

    visits_per_rid = df.groupby("RID").size()
    rids_kept = set(visits_per_rid[visits_per_rid >= 2].index)
    rids_marginal = set(visits_per_rid[visits_per_rid == 1].index)

    X = df[feature_cols].to_numpy()
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").to_numpy()

    lines = []
    lines.append("=== Does the min-visits coverage/reliability tradeoff hold for GMM and the autoencoder too? ===")
    lines.append(f"RIDs with >=2 visits (kept today): {len(rids_kept)}")
    lines.append(f"RIDs with exactly 1 visit (marginal): {len(rids_marginal)}")
    lines.append("Same longitudinal population and CN-fit-once-on-superset discipline as the PCA-only check.")
    lines.append("")

    # ---- PCA (repeat for a single consolidated report) ----
    scaler, pca, _, cum_var = fit_pca_baseline(X[cn_mask])
    T2, Q = compute_t2_q(X, scaler, pca)
    t2_ucl, q_ucl = control_limits(T2[cn_mask], Q[cn_mask], pca.n_components_, cn_mask.sum())
    df["HI_PCA"] = T2 / t2_ucl + Q / q_ucl

    # ---- GMM ----
    print("Fitting GMM baseline (BIC search over 1-5 components x diag/full)...", flush=True)
    scaler_g, gmm = fit_gmm_baseline(X[cn_mask])
    Xs_all_g = scaler_g.transform(X)
    nll = -gmm.score_samples(Xs_all_g)
    ucl_g = np.quantile(nll[cn_mask], 0.99)
    df["HI_GMM"] = nll / ucl_g
    print(f"  GMM selected: {gmm.n_components} components, {gmm.covariance_type} covariance", flush=True)

    # ---- Autoencoder (MLPRegressor) ----
    print("Fitting autoencoder baseline (MLPRegressor, may take a moment)...", flush=True)
    scaler_a, ae = fit_autoencoder(X[cn_mask])
    Xs_all_a = scaler_a.transform(X)
    Xhat = ae.predict(Xs_all_a)
    df["HI_AE"] = np.mean((Xs_all_a - Xhat) ** 2, axis=1)
    print("  done.", flush=True)

    last = df.sort_values("EXAMDATE").groupby("RID").last().reset_index()

    for method_label, hi_col in [("PCA", "HI_PCA"), ("GMM", "HI_GMM"), ("Autoencoder", "HI_AE")]:
        last_m = last.rename(columns={hi_col: "HI"})
        lines.append(f"### {method_label} ###")
        lines.extend(subset_auc_report(last_m, rids_kept, "KEPT today (>=2 visits)"))
        lines.extend(subset_auc_report(last_m, rids_kept | rids_marginal, "EXPANDED (>=1 visit)"))
        lines.extend(subset_auc_report(last_m, rids_marginal, "MARGINAL ONLY (exactly 1 visit)"))
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(HERE, "min_visits_gain_all_methods_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
