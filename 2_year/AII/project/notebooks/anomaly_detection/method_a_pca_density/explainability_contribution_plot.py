"""
notebooks/anomaly_detection/method_a_pca_density/explainability_contribution_plot.py

Explainability demo for Method A (point 1 / point 3 joint design ask, see
notebooks/decision_support/NOTES.md's "Fairness and explainability" section):
turns the PCA T^2 / Q anomaly score into a per-feature contribution plot for
one specific patient, i.e. answers "why did this patient get flagged", the
same question SHAP answers for point 3's RISK_SCORE (approach_attribution.ipynb)
but using PCA's own exact, closed-form decomposition instead of a model-
agnostic approximation -- this is the concrete reason PCA was chosen over the
GMM tie-break earlier (see three_method_metrics.txt): GMM's anomaly score has
no comparably simple per-feature decomposition.

Standard SPC decomposition (Westerhuis, Gurden & Smilde 1997/2000; MacGregor
& Kourti): for a standardized sample vector x with PCA scores t_k, loadings
p_jk and retained-component eigenvalues lambda_k,

    T2 contribution of feature j = sum_k (t_k / lambda_k) * p_jk * x_j
    Q  contribution of feature j = (x_j - xhat_j)^2   where xhat = reconstruction

Both are EXACT decompositions: summing the T2 contributions over all features
reproduces T2 exactly, and summing the Q contributions over all features
reproduces Q exactly (mod floating point). T2 contributions can be individually
negative (a feature can offset another's deviation inside the retained
subspace); Q contributions are always >= 0 (squared residual, deviation the
model doesn't explain at all).

Usage:
    python3 explainability_contribution_plot.py
Outputs (written next to this script):
    - contribution_plot_<RID>.png   for one AD and one CN patient
    - contribution_report.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_features import build_feature_table
from method_a import fit_pca_baseline, compute_t2_q, control_limits


def t2_contributions(x_std, scores, pca):
    """x_std: (p,) standardized sample. scores: (n_components,) its PCA scores.
    Returns (p,) exact per-feature contribution to this sample's T2."""
    lambdas = pca.explained_variance_[: pca.n_components_]
    loadings = pca.components_[: pca.n_components_]  # (n_components, p)
    # contribution_j = sum_k (t_k / lambda_k) * p_jk * x_j
    weight = (scores / lambdas) @ loadings  # (p,)
    return weight * x_std


def q_contributions(x_std, xhat_std):
    resid = x_std - xhat_std
    return resid ** 2


def main():
    df, feature_cols = build_feature_table(feature_tier="core")
    X = df[feature_cols].values
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").values

    scaler, pca, _, cum_var = fit_pca_baseline(X[cn_mask])
    T2, Q = compute_t2_q(X, scaler, pca)
    t2_ucl, q_ucl = control_limits(T2[cn_mask], Q[cn_mask], pca.n_components_, cn_mask.sum())
    hi = T2 / t2_ucl + Q / q_ucl
    df = df.copy()
    df["HI"] = hi
    df["T2"] = T2
    df["Q"] = Q

    Xs = scaler.transform(X)
    scores_all = pca.transform(Xs)
    Xhat_all = pca.inverse_transform(scores_all)

    def contributions_for_row(i):
        x_std = Xs[i]
        scores = scores_all[i, : pca.n_components_]
        xhat_std = Xhat_all[i]
        c_t2 = t2_contributions(x_std, scores, pca)
        c_q = q_contributions(x_std, xhat_std)
        return c_t2, c_q

    lines = []
    lines.append("PCA T2/Q contribution decomposition -- exactness check")
    lines.append("(sum of per-feature contributions must equal the sample's own T2 / Q)")
    lines.append("")

    # pick the highest-HI AD patient and a representative CN patient (median HI among CN)
    ad_df = df[df["DIAGNOSIS_LABEL"] == "AD"].sort_values("HI", ascending=False)
    cn_df = df[df["DIAGNOSIS_LABEL"] == "CN"].copy()
    cn_df["dist_to_median"] = (cn_df["HI"] - cn_df["HI"].median()).abs()
    cn_df = cn_df.sort_values("dist_to_median")

    targets = [
        ("AD_highest_HI", ad_df.iloc[0]),
        ("CN_typical", cn_df.iloc[0]),
    ]

    for label, row in targets:
        i = df.index.get_loc(row.name)
        c_t2, c_q = contributions_for_row(i)
        rid = df.loc[row.name, "RID"]
        diag = df.loc[row.name, "DIAGNOSIS_LABEL"]

        lines.append(f"-- {label}: RID={rid}, diagnosis={diag}, "
                     f"HI={row['HI']:.3f}, T2={row['T2']:.2f} (UCL={t2_ucl:.2f}), "
                     f"Q={row['Q']:.2f} (UCL={q_ucl:.2f}) --")
        lines.append(f"  sum(T2 contributions) = {c_t2.sum():.4f}  vs actual T2 = {row['T2']:.4f}")
        lines.append(f"  sum(Q contributions)  = {c_q.sum():.4f}  vs actual Q  = {row['Q']:.4f}")

        order_t2 = np.argsort(-np.abs(c_t2))[:8]
        order_q = np.argsort(-c_q)[:8]
        lines.append("  Top T2 contributors (can be + or -; + = pushes anomaly score up):")
        for j in order_t2:
            lines.append(f"    {feature_cols[j]:20s} {c_t2[j]:+.3f}")
        lines.append("  Top Q contributors (always >= 0; unexplained-by-model deviation):")
        for j in order_q:
            lines.append(f"    {feature_cols[j]:20s} {c_q[j]:+.3f}")
        lines.append("")

        # plot
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, order, contribs, title, color in [
            (axes[0], order_t2, c_t2, "T2 contribution (in-subspace deviation)", "#4C72B0"),
            (axes[1], order_q, c_q, "Q contribution (residual, unexplained by model)", "#C44E52"),
        ]:
            names = [feature_cols[j] for j in order][::-1]
            vals = [contribs[j] for j in order][::-1]
            colors = [color if v >= 0 else "#999999" for v in vals]
            ax.barh(names, vals, color=colors)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(title, fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
        fig.suptitle(f"Why is this patient's anomaly score what it is? "
                     f"RID {rid} ({diag}), HI={row['HI']:.2f}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"contribution_plot_{label}.png", dpi=150)
        plt.close()

    report = "\n".join(lines)
    print(report)
    with open("contribution_report.txt", "w") as f:
        f.write(report + "\n")

    print("\nWrote: contribution_plot_AD_highest_HI.png, contribution_plot_CN_typical.png, "
          "contribution_report.txt")


if __name__ == "__main__":
    main()
