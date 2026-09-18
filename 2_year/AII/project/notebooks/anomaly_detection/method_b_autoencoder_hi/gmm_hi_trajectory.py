"""
notebooks/anomaly_detection/method_b_autoencoder_hi/gmm_hi_trajectory.py

Same idea as pca_hi_trajectory.py (keep Method B's longitudinal STRUCTURE --
per-visit HI trajectory + per-visit compressed representation for points 2/3
-- but swap the density model), this time using GMM: the method the course
itself teaches for higher-dimensional density estimation (see 02-ad-hd),
and the one that tied with PCA in the same-cohort, same-features comparison
in method_a_pca_density/compare_three_methods.py (both ~0.88-0.89 AUC,
clearly ahead of the autoencoder's ~0.81).

This script settles the remaining question: does that tie between PCA and
GMM hold up on the LONGITUDINAL cohort too (more patients, looser per-visit
matching, the actual data Method B's deliverable is built from)? If so,
either is a defensible choice to carry into points 2/3; if one pulls ahead
here, that's the one to standardize on.

HI here = negative log-likelihood under the GMM fit on CN visits, normalized
by the same idea as PCA's UCL (a 99th-percentile control limit fit on the
CN population) so the number is comparable in spirit to the other variants.
The "embedding" is the per-visit vector of responsibilities (posterior
probability of belonging to each Gaussian component) -- a natural GMM
analogue of PCA's component scores / the autoencoder's bottleneck.

Usage:
    python3 gmm_hi_trajectory.py
Outputs (written next to this script):
    - gmm_hi_trajectories.csv
    - gmm_hi_trajectory_mean.png
    - gmm_hi_slope_by_group.png
    - gmm_metrics.txt
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "method_a_pca_density"))
from load_longitudinal import build_longitudinal_table  # noqa: E402

RANDOM_STATE = 42
COLORS = {"CN": "#4C72B0", "MCI": "#DD8452", "AD": "#C44E52"}


def fit_gmm_baseline(X_cn):
    scaler = StandardScaler().fit(X_cn)
    Xs = scaler.transform(X_cn)
    best_bic, best_gmm, best_cfg = np.inf, None, None
    for n_components in range(1, 6):
        for cov_type in ["diag", "full"]:
            gmm = GaussianMixture(
                n_components=n_components, covariance_type=cov_type,
                random_state=RANDOM_STATE, reg_covar=1e-4, n_init=3,
            )
            gmm.fit(Xs)
            bic = gmm.bic(Xs)
            if bic < best_bic:
                best_bic, best_gmm, best_cfg = bic, gmm, (n_components, cov_type)
    return scaler, best_gmm, best_cfg, best_bic


def main():
    df, feature_cols = build_longitudinal_table()
    df = df.dropna(subset=["DIAGNOSIS_LABEL"]).reset_index(drop=True)
    baseline_date = df.groupby("RID")["EXAMDATE"].transform("min")
    df["YEARS_FROM_BASELINE"] = (df["EXAMDATE"] - baseline_date).dt.days / 365.25

    print(f"Loaded {len(df)} visits from {df['RID'].nunique()} patients.")

    X = df[feature_cols].values
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").values
    print(f"Fitting GMM baseline on {cn_mask.sum()} CN visits "
          f"({df.loc[cn_mask, 'RID'].nunique()} unique CN patients).")

    scaler, gmm, cfg, bic = fit_gmm_baseline(X[cn_mask])
    print(f"Selected: {cfg[0]} components, {cfg[1]} covariance (BIC={bic:.1f})")

    Xs_all = scaler.transform(X)
    nll = -gmm.score_samples(Xs_all)
    nll_cn = nll[cn_mask]
    ucl = np.quantile(nll_cn, 0.99)
    df["HI"] = nll / ucl

    resp = gmm.predict_proba(Xs_all)
    for i in range(cfg[0]):
        df[f"resp_{i}"] = resp[:, i]

    out_cols = (["RID", "VISCODE2", "EXAMDATE", "VISIT_INDEX", "YEARS_FROM_BASELINE",
                 "DIAGNOSIS_LABEL", "BASELINE_DIAGNOSIS", "HI"]
                + [f"resp_{i}" for i in range(cfg[0])])
    df[out_cols].to_csv("gmm_hi_trajectories.csv", index=False)

    lines = []
    lines.append(f"n_visits = {len(df)}, n_patients = {df['RID'].nunique()}")
    lines.append(f"CN visits used to fit GMM: {cn_mask.sum()} "
                 f"({df.loc[cn_mask, 'RID'].nunique()} patients)")
    lines.append(f"GMM config: {cfg[0]} components, {cfg[1]} covariance, BIC={bic:.1f}")
    lines.append("")
    lines.append("-- AUC using every visit --")
    for group in ["MCI", "AD"]:
        mask = cn_mask | (df["DIAGNOSIS_LABEL"] == group).values
        y = (df.loc[mask, "DIAGNOSIS_LABEL"] == group).astype(int)
        auc = roc_auc_score(y, df.loc[mask, "HI"])
        lines.append(f"  CN vs {group}: {auc:.3f}  (n={mask.sum()})")
    mask_any = df["DIAGNOSIS_LABEL"].isin(["CN", "MCI", "AD"])
    y_any = (df.loc[mask_any, "DIAGNOSIS_LABEL"] != "CN").astype(int)
    lines.append(f"  CN vs MCI+AD: {roc_auc_score(y_any, df.loc[mask_any, 'HI']):.3f}")

    last = df.sort_values("EXAMDATE").groupby("RID").last().reset_index()
    lines.append("")
    lines.append("-- AUC using only each patient's LAST visit --")
    for group in ["MCI", "AD"]:
        mask = (last["DIAGNOSIS_LABEL"] == "CN") | (last["DIAGNOSIS_LABEL"] == group)
        y = (last.loc[mask, "DIAGNOSIS_LABEL"] == group).astype(int)
        auc = roc_auc_score(y, last.loc[mask, "HI"])
        lines.append(f"  CN vs {group}: {auc:.3f}  (n={mask.sum()})")
    mask_any2 = last["DIAGNOSIS_LABEL"].isin(["CN", "MCI", "AD"])
    y_any2 = (last.loc[mask_any2, "DIAGNOSIS_LABEL"] != "CN").astype(int)
    lines.append(f"  CN vs MCI+AD: {roc_auc_score(y_any2, last.loc[mask_any2, 'HI']):.3f}")

    def slope(g):
        if len(g) < 2 or g["YEARS_FROM_BASELINE"].max() == 0:
            return np.nan
        return np.polyfit(g["YEARS_FROM_BASELINE"], g["HI"], 1)[0]

    slopes = df.groupby("RID").apply(slope, include_groups=False).rename("HI_SLOPE")
    slopes_df = df.drop_duplicates("RID")[["RID", "BASELINE_DIAGNOSIS"]].merge(
        slopes, on="RID"
    ).dropna()

    lines.append("")
    lines.append("-- Per-patient HI slope by baseline group --")
    for g in ["CN", "MCI", "AD"]:
        s = slopes_df.loc[slopes_df["BASELINE_DIAGNOSIS"] == g, "HI_SLOPE"]
        lines.append(f"  {g}: median={s.median():.4f}, mean={s.mean():.4f}, n={len(s)}")

    lines.append("")
    lines.append("-- AUC of slope itself as a per-patient predictor --")
    for group in ["MCI", "AD"]:
        mask = slopes_df["BASELINE_DIAGNOSIS"].isin(["CN", group])
        y = (slopes_df.loc[mask, "BASELINE_DIAGNOSIS"] == group).astype(int)
        auc = roc_auc_score(y, slopes_df.loc[mask, "HI_SLOPE"])
        lines.append(f"  CN vs {group}: {auc:.3f}")

    metrics_text = "\n".join(lines)
    print("\n" + metrics_text)
    with open("gmm_metrics.txt", "w") as f:
        f.write(metrics_text + "\n")

    order = ["CN", "MCI", "AD"]
    plt.figure(figsize=(7, 5))
    bins = np.arange(0, 8.5, 0.5)
    for group, color in COLORS.items():
        g = df[df["BASELINE_DIAGNOSIS"] == group].copy()
        g["bin"] = pd.cut(g["YEARS_FROM_BASELINE"], bins)
        stats = g.groupby("bin", observed=True)["HI"].agg(["mean", "std", "count"])
        stats = stats[stats["count"] >= 5]
        x = [iv.mid for iv in stats.index]
        plt.plot(x, stats["mean"], color=color, label=group, marker="o", ms=4)
        plt.fill_between(x, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                          color=color, alpha=0.15)
    plt.xlabel("Years from baseline visit")
    plt.ylabel("Mean HI (GMM -log-lik / UCL) +/- std")
    plt.title("Mean GMM-based HI trajectory by baseline diagnosis group")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gmm_hi_trajectory_mean.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    data = [slopes_df.loc[slopes_df["BASELINE_DIAGNOSIS"] == g, "HI_SLOPE"] for g in order]
    bp = plt.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.6)
    plt.axhline(0, color="k", lw=0.8, ls="--")
    plt.ylabel("HI slope (units/year)")
    plt.title("Per-patient GMM-HI trend by baseline diagnosis group")
    plt.tight_layout()
    plt.savefig("gmm_hi_slope_by_group.png", dpi=150)
    plt.close()

    print("\nWrote: gmm_hi_trajectories.csv, gmm_metrics.txt, "
          "gmm_hi_trajectory_mean.png, gmm_hi_slope_by_group.png")


if __name__ == "__main__":
    main()
