"""
notebooks/anomaly_detection/method_b_autoencoder_hi/pca_hi_trajectory.py

A variant of Method B that keeps its longitudinal STRUCTURE (per-visit HI
trajectory + per-visit compressed representation, meant to feed points 2 and
3) but swaps the underlying density model from the autoencoder to Method A's
PCA + Hotelling's T^2/Q. This was motivated by a direct, same-cohort test
(see the project conversation / metrics.txt in method_a_pca_density) showing
PCA outperforms the autoencoder as a pure anomaly detector on this tabular
feature set (CN vs AD AUC 0.88 for PCA vs 0.78 for the autoencoder, on the
exact same patients) -- not a cohort artifact, a genuine modeling-approach
difference.

Two outputs, same contract as method_b.py's autoencoder version:
  - HI(t): here, the combined T2/UCL + Q/UCL score per visit (instead of
    autoencoder reconstruction error) -> still the input point 2 (RUL) would
    consume.
  - a per-visit compressed representation -> here, the retained PCA
    component scores (14 dimensions, vs the autoencoder's 6-dim bottleneck)
    -> still the input point 3 (decision support) would consume, just
    higher-dimensional and linear/interpretable instead of learned/compact.

Usage:
    python3 pca_hi_trajectory.py
Outputs (written next to this script):
    - pca_hi_trajectories.csv
    - pca_hi_trajectory_mean.png
    - pca_hi_slope_by_group.png
    - metrics.txt (appended as pca_metrics.txt to avoid clobbering the
      autoencoder run's metrics.txt)
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "method_a_pca_density"))
from method_a import fit_pca_baseline, compute_t2_q, control_limits  # noqa: E402
from load_longitudinal import build_longitudinal_table  # noqa: E402

COLORS = {"CN": "#4C72B0", "MCI": "#DD8452", "AD": "#C44E52"}


def main():
    df, feature_cols = build_longitudinal_table()
    df = df.dropna(subset=["DIAGNOSIS_LABEL"]).reset_index(drop=True)
    baseline_date = df.groupby("RID")["EXAMDATE"].transform("min")
    df["YEARS_FROM_BASELINE"] = (df["EXAMDATE"] - baseline_date).dt.days / 365.25

    print(f"Loaded {len(df)} visits from {df['RID'].nunique()} patients.")

    X = df[feature_cols].values
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").values
    print(f"Fitting PCA baseline on {cn_mask.sum()} CN visits "
          f"({df.loc[cn_mask, 'RID'].nunique()} unique CN patients).")

    scaler, pca, _, cum_var = fit_pca_baseline(X[cn_mask])
    print(f"Retained {pca.n_components_} components "
          f"({cum_var[pca.n_components_-1]:.1%} variance).")

    T2, Q = compute_t2_q(X, scaler, pca)
    t2_ucl, q_ucl = control_limits(T2[cn_mask], Q[cn_mask], pca.n_components_, cn_mask.sum())
    df["HI"] = T2 / t2_ucl + Q / q_ucl

    Xs_all = scaler.transform(X)
    scores = pca.transform(Xs_all)
    for i in range(pca.n_components_):
        df[f"pc_{i}"] = scores[:, i]

    out_cols = (["RID", "VISCODE2", "EXAMDATE", "VISIT_INDEX", "YEARS_FROM_BASELINE",
                 "DIAGNOSIS_LABEL", "BASELINE_DIAGNOSIS", "HI"]
                + [f"pc_{i}" for i in range(pca.n_components_)])
    df[out_cols].to_csv("pca_hi_trajectories.csv", index=False)

    lines = []
    lines.append(f"n_visits = {len(df)}, n_patients = {df['RID'].nunique()}")
    lines.append(f"CN visits used to fit PCA: {cn_mask.sum()} "
                 f"({df.loc[cn_mask, 'RID'].nunique()} patients)")
    lines.append(f"Retained components: {pca.n_components_} "
                 f"({cum_var[pca.n_components_-1]:.1%} variance)")
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
    lines.append("-- Per-patient HI slope by baseline group (PCA-based HI) --")
    for g in ["CN", "MCI", "AD"]:
        s = slopes_df.loc[slopes_df["BASELINE_DIAGNOSIS"] == g, "HI_SLOPE"]
        lines.append(f"  {g}: median={s.median():.4f}, mean={s.mean():.4f}, n={len(s)}")

    lines.append("")
    lines.append("-- AUC of slope itself as a per-patient predictor (sanity check) --")
    for group in ["MCI", "AD"]:
        mask = slopes_df["BASELINE_DIAGNOSIS"].isin(["CN", group])
        y = (slopes_df.loc[mask, "BASELINE_DIAGNOSIS"] == group).astype(int)
        auc = roc_auc_score(y, slopes_df.loc[mask, "HI_SLOPE"])
        lines.append(f"  CN vs {group}: {auc:.3f}")

    metrics_text = "\n".join(lines)
    print("\n" + metrics_text)
    with open("pca_metrics.txt", "w") as f:
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
    plt.ylabel("Mean HI (PCA T2/Q) +/- std")
    plt.title("Mean PCA-based HI trajectory by baseline diagnosis group")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_hi_trajectory_mean.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    data = [slopes_df.loc[slopes_df["BASELINE_DIAGNOSIS"] == g, "HI_SLOPE"] for g in order]
    bp = plt.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.6)
    plt.axhline(0, color="k", lw=0.8, ls="--")
    plt.ylabel("HI slope (units/year)")
    plt.title("Per-patient PCA-HI trend by baseline diagnosis group")
    plt.tight_layout()
    plt.savefig("pca_hi_slope_by_group.png", dpi=150)
    plt.close()

    print("\nWrote: pca_hi_trajectories.csv, pca_metrics.txt, "
          "pca_hi_trajectory_mean.png, pca_hi_slope_by_group.png")


if __name__ == "__main__":
    main()
