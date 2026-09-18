"""
notebooks/anomaly_detection/method_a_pca_density/compare_three_methods.py

Autoencoder implementation: genuine Keras/TensorFlow (functional API,
ReLU Dense encoder to a bottleneck, linear Dense decoder, Adam + MSE,
EarlyStopping restoring best weights) -- built exactly as the course's own
lecture (02-ad-hd, lesson 4, "Autoencoders for Anomaly Detection") does it.
This replaces an earlier sklearn MLPRegressor workaround that was used only
because PyTorch/TensorFlow wasn't installed at the time; a side-by-side
rerun (keras_ae_experiment.py) showed the Keras version improves AUC by
+0.036 on every CN-vs-{MCI,AD,MCI+AD} comparison on this exact cohort, so
it was made the actual implementation here.

Three-way, same-cohort, same-features comparison of the density-estimation
methods relevant to point 1, aligned to what the course itself teaches for
this exact topic (checked directly against the AII-theory lecture
repositories):

  - 01-ad-de ("Anomaly Detection via Density Estimation") teaches KDE.
  - 02-ad-hd ("Anomaly Detection in Higher Dimensional Spaces") teaches GMM
    and Autoencoders. PCA / Hotelling's T2 never appears in either lecture
    block -- it's a legitimate industrial SPC technique, just not the one
    taught here.

So this script adds GMM (the professor's own method for higher-dimensional
density estimation) as a third method, alongside our existing PCA (Method A)
and Autoencoder (Method B), and compares all three under IDENTICAL
conditions: same 464-patient cross-sectional cohort, same 22 features, same
CN-only training population, same evaluation. This isolates "which density
model works best on this specific, relatively low-dimensional (22-feature)
tabular problem" from any cohort-size or feature-set confound -- earlier
comparisons (PCA vs Autoencoder) used different cohorts for A and B, this one
does not.

GMM model selection: n_components in {1..5} x covariance_type in
{'diag','full'} chosen by BIC on the CN training set only (never touching
MCI/AD labels), same spirit as sklearn's standard GMM model-selection
recipe and the course's own GMM notebook. 'diag' is included deliberately:
with ~230-270 CN training patients and 22 features, a 'full' covariance GMM
has O(d^2) parameters per component and can overfit; BIC will penalize that
if it doesn't pay for itself.

Usage:
    python3 compare_three_methods.py
Outputs (written next to this script):
    - three_method_comparison.png
    - three_method_comparison.csv
    - three_method_metrics.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from load_features import build_feature_table
from method_a import fit_pca_baseline, compute_t2_q, control_limits

RANDOM_STATE = 42
COLORS = {"CN": "#4C72B0", "MCI": "#DD8452", "AD": "#C44E52"}
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


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


def gmm_anomaly_score(gmm, scaler, X):
    Xs = scaler.transform(X)
    return -gmm.score_samples(Xs)  # negative log-likelihood


def fit_autoencoder_baseline(X_cn, latent_dim=6, hidden_dim=16, val_split=0.15):
    """Genuine Keras/TF autoencoder: functional API, ReLU Dense encoder down
    to a bottleneck, linear Dense decoder back to input dimensionality,
    Adam + MSE, EarlyStopping restoring best weights. Same 16-6-16 shape as
    the sklearn MLPRegressor it replaces."""
    scaler = StandardScaler().fit(X_cn)
    Xs = scaler.transform(X_cn).astype("float32")
    n_features = Xs.shape[1]

    ae_in = keras.Input(shape=(n_features,), dtype="float32")
    h1 = layers.Dense(hidden_dim, activation="relu")(ae_in)
    z = layers.Dense(latent_dim, activation="relu", name="bottleneck")(h1)
    h2 = layers.Dense(hidden_dim, activation="relu")(z)
    out = layers.Dense(n_features, activation="linear")(h2)
    ae = keras.Model(ae_in, out)
    ae.compile(optimizer="adam", loss="mse")

    cb = [callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor="val_loss")]
    history = ae.fit(
        Xs, Xs, validation_split=val_split, callbacks=cb,
        batch_size=32, epochs=500, verbose=0,
    )
    ae.n_epochs_ = len(history.history["loss"])
    ae.final_val_loss_ = min(history.history["val_loss"])
    return scaler, ae


def ae_anomaly_score(ae, scaler, X):
    Xs = scaler.transform(X).astype("float32")
    Xhat = ae.predict(Xs, verbose=0)
    return np.mean((Xs - Xhat) ** 2, axis=1)


def bootstrap_auc_ci(y, score, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    boots = []
    for _ in range(n_boot):
        bi = np.r_[rng.choice(idx_neg, len(idx_neg), replace=True),
                   rng.choice(idx_pos, len(idx_pos), replace=True)]
        boots.append(roc_auc_score(y[bi], score[bi]))
    return np.percentile(boots, [2.5, 97.5])


def main():
    df, feature_cols = build_feature_table(feature_tier="core")
    X = df[feature_cols].values
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").values
    print(f"Cohort: {len(df)} patients "
          f"(CN={cn_mask.sum()}, MCI={(df.DIAGNOSIS_LABEL=='MCI').sum()}, "
          f"AD={(df.DIAGNOSIS_LABEL=='AD').sum()}), {len(feature_cols)} features.")

    results = {}

    # -- PCA (Method A) --
    scaler_p, pca, _, cum_var = fit_pca_baseline(X[cn_mask])
    T2, Q = compute_t2_q(X, scaler_p, pca)
    t2_ucl, q_ucl = control_limits(T2[cn_mask], Q[cn_mask], pca.n_components_, cn_mask.sum())
    results["PCA (T2/Q)"] = T2 / t2_ucl + Q / q_ucl
    pca_info = f"{pca.n_components_} components, {cum_var[pca.n_components_-1]:.1%} variance"

    # -- GMM (professor's method for this topic) --
    scaler_g, gmm, gmm_cfg, gmm_bic = fit_gmm_baseline(X[cn_mask])
    results["GMM (-log-lik)"] = gmm_anomaly_score(gmm, scaler_g, X)
    gmm_info = f"{gmm_cfg[0]} components, {gmm_cfg[1]} covariance (BIC={gmm_bic:.1f})"

    # -- Autoencoder (Method B), retrained on THIS cohort for a fair comparison --
    scaler_a, ae = fit_autoencoder_baseline(X[cn_mask])
    results["Autoencoder (recon. error)"] = ae_anomaly_score(ae, scaler_a, X)
    ae_info = (f"16-6-16 Keras/TF Dense autoencoder (bottleneck=6), "
               f"{ae.n_epochs_} epochs (early stopped), best val_loss={ae.final_val_loss_:.4f}")

    print(f"\nPCA config: {pca_info}")
    print(f"GMM config: {gmm_info}")
    print(f"Autoencoder config: {ae_info}")

    lines = [f"Cohort: n={len(df)} (CN={cn_mask.sum()}, "
             f"MCI={(df.DIAGNOSIS_LABEL=='MCI').sum()}, AD={(df.DIAGNOSIS_LABEL=='AD').sum()}), "
             f"{len(feature_cols)} features (same for all 3 methods)",
             f"PCA:          {pca_info}",
             f"GMM:          {gmm_info}",
             f"Autoencoder:  {ae_info}",
             ""]

    rows = []
    for name, score in results.items():
        lines.append(f"-- {name} --")
        for group in ["MCI", "AD"]:
            mask = cn_mask | (df["DIAGNOSIS_LABEL"] == group).values
            y = (df.loc[mask, "DIAGNOSIS_LABEL"] == group).astype(int).values
            auc = roc_auc_score(y, score[mask])
            row = {"method": name, "comparison": f"CN vs {group}", "auc": auc, "n": mask.sum()}
            if group == "AD":
                ci = bootstrap_auc_ci(y, score[mask])
                row["ci_low"], row["ci_high"] = ci
                lines.append(f"  CN vs {group}: {auc:.3f}  (95% CI [{ci[0]:.3f}, {ci[1]:.3f}], n={mask.sum()})")
            else:
                lines.append(f"  CN vs {group}: {auc:.3f}  (n={mask.sum()})")
            rows.append(row)
        mask_any = cn_mask | df["DIAGNOSIS_LABEL"].isin(["MCI", "AD"]).values
        y_any = (df.loc[mask_any, "DIAGNOSIS_LABEL"] != "CN").astype(int).values
        auc_any = roc_auc_score(y_any, score[mask_any])
        lines.append(f"  CN vs MCI+AD: {auc_any:.3f}")
        rows.append({"method": name, "comparison": "CN vs MCI+AD", "auc": auc_any, "n": mask_any.sum()})
        lines.append("")

    results_df = pd.DataFrame(rows)
    results_df.to_csv("three_method_comparison.csv", index=False)

    lines.append("Winner by comparison:")
    for comp in ["CN vs MCI", "CN vs AD", "CN vs MCI+AD"]:
        sub = results_df[results_df["comparison"] == comp]
        best = sub.loc[sub["auc"].idxmax()]
        lines.append(f"  {comp}: {best['method']} (AUC={best['auc']:.3f})")

    metrics_text = "\n".join(lines)
    print("\n" + metrics_text)
    with open("three_method_metrics.txt", "w") as f:
        f.write(metrics_text + "\n")

    # -- plot: grouped bars of AUC by method x comparison --
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    comparisons = ["CN vs MCI", "CN vs AD", "CN vs MCI+AD"]
    methods = list(results.keys())
    method_colors = ["#4C72B0", "#55A868", "#C44E52"]
    x = np.arange(len(comparisons))
    width = 0.25
    for i, (m, c) in enumerate(zip(methods, method_colors)):
        vals = [results_df[(results_df.method == m) & (results_df.comparison == comp)]["auc"].values[0]
                for comp in comparisons]
        axes[0].bar(x + (i - 1) * width, vals, width, label=m, color=c, alpha=0.85)
    axes[0].axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparisons)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Method comparison -- same cohort, same 22 features")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.4, 1.0)

    order = ["CN", "MCI", "AD"]
    for i, (m, score) in enumerate(results.items()):
        df[f"_score_{i}"] = score
    ax = axes[1]
    positions = []
    data = []
    labels = []
    tick_positions = []
    for gi, g in enumerate(order):
        for mi, m in enumerate(methods):
            data.append(df.loc[df["DIAGNOSIS_LABEL"] == g, f"_score_{mi}"] /
                        df[f"_score_{mi}"].max())  # normalize 0-1 for visual comparability
            positions.append(gi * (len(methods) + 1) + mi)
        tick_positions.append(gi * (len(methods) + 1) + (len(methods) - 1) / 2)
    bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
    for patch, pos in zip(bp["boxes"], positions):
        gi = pos // (len(methods) + 1)
        mi = pos % (len(methods) + 1)
        patch.set_facecolor(method_colors[mi])
        patch.set_alpha(0.7)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(order)
    ax.set_ylabel("Anomaly score (min-max normalized per method)")
    ax.set_title("Score distributions by diagnosis (normalized for comparability)")
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=c, alpha=0.7, label=m) for c, m in zip(method_colors, methods)],
               fontsize=8)

    plt.tight_layout()
    plt.savefig("three_method_comparison.png", dpi=150)
    plt.close()

    print("\nWrote: three_method_comparison.csv, three_method_metrics.txt, "
          "three_method_comparison.png")


if __name__ == "__main__":
    main()
