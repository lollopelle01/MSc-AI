"""
notebooks/anomaly_detection/method_a_pca_density/method_a.py

Method A -- self-contained anomaly detection via PCA + Hotelling's T^2 / SPE
(the classic multivariate SPC / condition-monitoring approach to density
estimation in a high-dimensional space).

Idea: fit a "healthy operating region" from Cognitively Normal (CN) subjects'
baseline multimodal profile (demographics + structural MRI). Any patient's
deviation from that region is measured two ways:
  - T^2 (Hotelling's): Mahalanobis distance *within* the retained PCA
    subspace -- "an unusual combination of otherwise-normal variation".
  - Q / SPE (squared prediction error): residual *outside* the retained
    subspace -- "a pattern never seen in the healthy population".
Early-stage Alzheimer's is treated as an anomaly: MCI/AD patients should
score higher on T^2 and/or Q than CN patients.

This script is self-contained: it takes the multimodal feature table in and
produces anomaly scores + evaluation plots out. Nothing downstream (RUL,
decision support) needs to exist for this to run or be evaluated.

Usage:
    python3 method_a.py
Outputs (written next to this script):
    - pca_variance.png     cumulative explained variance vs #components
    - t2_q_control_chart.png   T2 vs Q scatter colored by diagnosis, with UCLs
    - anomaly_score_by_diagnosis.png   boxplot of combined score per group
    - results.csv          per-patient RID, DIAGNOSIS_LABEL, T2, Q, combined score
    - metrics.txt           ROC-AUC (CN vs MCI, CN vs AD, CN vs MCI+AD) + notes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from load_features import build_feature_table

RANDOM_STATE = 42
VARIANCE_TARGET = 0.90  # retain enough PCs to explain this much variance


def fit_pca_baseline(X_cn, variance_target=VARIANCE_TARGET):
    """Fit scaler + PCA on the CN ("healthy") subset only."""
    scaler = StandardScaler().fit(X_cn)
    Xs = scaler.transform(X_cn)

    pca_full = PCA(random_state=RANDOM_STATE).fit(Xs)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, variance_target) + 1)
    n_components = min(n_components, Xs.shape[1] - 1)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(Xs)
    return scaler, pca, pca_full, cum_var


def compute_t2_q(X, scaler, pca):
    """
    T^2: Mahalanobis distance in the retained PC subspace, normalized by
         each component's eigenvalue (variance) -- i.e. sum((score_i^2)/eigval_i).
    Q (SPE): squared reconstruction residual outside the retained subspace.
    """
    Xs = scaler.transform(X)
    scores = pca.transform(Xs)                 # (n, n_components)
    eigvals = pca.explained_variance_           # (n_components,)
    T2 = np.sum((scores ** 2) / eigvals, axis=1)

    X_reconstructed = pca.inverse_transform(scores)
    residual = Xs - X_reconstructed
    Q = np.sum(residual ** 2, axis=1)
    return T2, Q


def control_limits(T2_cn, Q_cn, n_components, n_samples, alpha=0.01):
    """
    Classic SPC control limits fit on the CN ("in control") population:
      - T^2 UCL from the F-distribution (Hotelling 1947).
      - Q UCL via the Jackson & Mudholkar (1979) approximation from the
        residual eigenvalues would require them explicitly; here we use the
        simpler, commonly-taught empirical percentile approach on the CN Q
        distribution instead, which is the standard practical shortcut when
        teaching/learning this method.
    """
    p, n = n_components, n_samples
    f_crit = stats.f.ppf(1 - alpha, p, n - p)
    t2_ucl = (p * (n - 1) / (n - p)) * f_crit
    q_ucl = np.quantile(Q_cn, 1 - alpha)
    return t2_ucl, q_ucl


def main():
    import matplotlib
    matplotlib.use("Agg")

    df, feature_cols = build_feature_table(feature_tier="core")
    print(f"Loaded {len(df)} patients, {len(feature_cols)} features.")
    print(df["DIAGNOSIS_LABEL"].value_counts())

    X = df[feature_cols].values
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").values

    scaler, pca, pca_full, cum_var = fit_pca_baseline(X[cn_mask])
    print(f"Retained {pca.n_components_} PCs "
          f"({VARIANCE_TARGET:.0%} variance target) out of {X.shape[1]} features.")

    T2, Q = compute_t2_q(X, scaler, pca)
    T2_cn, Q_cn = T2[cn_mask], Q[cn_mask]
    t2_ucl, q_ucl = control_limits(T2_cn, Q_cn, pca.n_components_, cn_mask.sum())

    # combined anomaly score: normalize each statistic by its CN-population
    # UCL and sum, so both contribute comparably regardless of raw scale.
    combined_score = T2 / t2_ucl + Q / q_ucl

    df["T2"] = T2
    df["Q"] = Q
    df["ANOMALY_SCORE"] = combined_score
    df[["RID", "DIAGNOSIS_LABEL", "T2", "Q", "ANOMALY_SCORE"]].to_csv(
        "results.csv", index=False
    )

    # --- evaluation: does the anomaly score separate CN from MCI/AD? ---
    lines = []
    lines.append(f"n_patients = {len(df)}  (CN={cn_mask.sum()}, "
                  f"MCI={(df['DIAGNOSIS_LABEL']=='MCI').sum()}, "
                  f"AD={(df['DIAGNOSIS_LABEL']=='AD').sum()})")
    lines.append(f"n_components retained = {pca.n_components_} "
                 f"(explains {cum_var[pca.n_components_-1]:.1%} variance)")
    lines.append(f"T2 UCL (99%) = {t2_ucl:.2f}, Q UCL (99%) = {q_ucl:.2f}")
    lines.append("")

    for group in ["MCI", "AD"]:
        mask = cn_mask | (df["DIAGNOSIS_LABEL"] == group).values
        y = (df.loc[mask, "DIAGNOSIS_LABEL"] == group).astype(int)
        if y.nunique() < 2:
            lines.append(f"CN vs {group}: skipped (only one class present)")
            continue
        auc = roc_auc_score(y, combined_score[mask])
        lines.append(f"ROC-AUC, CN vs {group} (combined T2+Q score): {auc:.3f}")

    mask_any = cn_mask | (df["DIAGNOSIS_LABEL"].isin(["MCI", "AD"])).values
    y_any = (df.loc[mask_any, "DIAGNOSIS_LABEL"] != "CN").astype(int)
    auc_any = roc_auc_score(y_any, combined_score[mask_any])
    lines.append(f"ROC-AUC, CN vs (MCI+AD combined): {auc_any:.3f}")

    metrics_text = "\n".join(lines)
    print("\n" + metrics_text)
    with open("metrics.txt", "w") as f:
        f.write(metrics_text + "\n")

    # --- plots ---
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker="o", ms=3)
    plt.axhline(VARIANCE_TARGET, color="gray", ls="--", lw=1,
                label=f"{VARIANCE_TARGET:.0%} target")
    plt.axvline(pca.n_components_, color="gray", ls=":", lw=1)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA on CN baseline: cumulative explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_variance.png", dpi=150)
    plt.close()

    colors = {"CN": "#4C72B0", "MCI": "#DD8452", "AD": "#C44E52"}
    plt.figure(figsize=(6, 5))
    for label, color in colors.items():
        m = (df["DIAGNOSIS_LABEL"] == label).values
        plt.scatter(T2[m], Q[m], s=14, alpha=0.6, label=label, color=color)
    plt.axvline(t2_ucl, color="k", ls="--", lw=1, label="99% UCL")
    plt.axhline(q_ucl, color="k", ls="--", lw=1)
    plt.xlabel("T$^2$ (Hotelling)")
    plt.ylabel("Q (SPE)")
    plt.title("PCA control chart: T$^2$ vs Q by diagnosis")
    plt.legend()
    plt.tight_layout()
    plt.savefig("t2_q_control_chart.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    order = ["CN", "MCI", "AD"]
    data = [combined_score[(df["DIAGNOSIS_LABEL"] == g).values] for g in order]
    bp = plt.boxplot(data, tick_labels=order, patch_artist=True)
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.6)
    plt.ylabel("Combined anomaly score (T2/UCL + Q/UCL)")
    plt.title("Anomaly score by diagnosis")
    plt.tight_layout()
    plt.savefig("anomaly_score_by_diagnosis.png", dpi=150)
    plt.close()

    print("\nWrote: results.csv, metrics.txt, pca_variance.png, "
          "t2_q_control_chart.png, anomaly_score_by_diagnosis.png")


if __name__ == "__main__":
    main()
