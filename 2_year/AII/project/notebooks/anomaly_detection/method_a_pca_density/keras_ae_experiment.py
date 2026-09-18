"""
keras_ae_experiment.py

Standalone experiment: replace the sklearn MLPRegressor "autoencoder"
workaround (notebooks/anomaly_detection/method_b_autoencoder_hi/method_b.py and
notebooks/anomaly_detection/method_a_pca_density/compare_three_methods.py) with a genuine
Keras/TensorFlow autoencoder, built exactly the way the course's own lecture
("02-ad-hd / 4. Autoencoders for Anomaly Detection") does it: a functional-API
model with an Input layer, a ReLU Dense encoder down to a bottleneck, a linear
Dense decoder back to the input dimensionality, compiled with Adam + MSE, and
trained with an EarlyStopping callback restoring the best weights.

This reruns the exact same 464-patient, 22-feature, CN-only-training,
same-cohort comparison as compare_three_methods.py (PCA / GMM / Autoencoder),
swapping only the autoencoder's implementation, so the AUCs are directly
comparable to the existing three_method_metrics.txt.

Usage: python3 keras_ae_experiment.py
Run from inside notebooks/anomaly_detection/method_a_pca_density/ (same working directory
requirement as the original script, since load_features.py resolves
DATA_DIR relative to its own file, not the CWD).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from load_features import build_feature_table
from method_a import fit_pca_baseline, compute_t2_q, control_limits

RANDOM_STATE = 42
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
    return -gmm.score_samples(Xs)


def fit_sklearn_mlp_baseline(X_cn, latent_dim=6, hidden_dim=16):
    """The ORIGINAL implementation (sklearn MLPRegressor), kept here so this
    script can report it side by side with the new Keras model."""
    scaler = StandardScaler().fit(X_cn)
    Xs = scaler.transform(X_cn)
    ae = MLPRegressor(
        hidden_layer_sizes=(hidden_dim, latent_dim, hidden_dim),
        activation="relu", solver="adam", alpha=1e-3, max_iter=2000,
        early_stopping=True, n_iter_no_change=25, random_state=RANDOM_STATE,
    )
    ae.fit(Xs, Xs)
    return scaler, ae


def sklearn_ae_score(ae, scaler, X):
    Xs = scaler.transform(X)
    Xhat = ae.predict(Xs)
    return np.mean((Xs - Xhat) ** 2, axis=1)


def fit_keras_autoencoder(X_cn, latent_dim=6, hidden_dim=16, val_split=0.15):
    """Genuine Keras/TF autoencoder, same shape (hidden_dim-latent_dim-hidden_dim)
    as the sklearn MLPRegressor it replaces, built exactly as in the lecture:
    functional API, ReLU hidden layers, linear output, Adam + MSE, and an
    EarlyStopping callback restoring the best weights."""
    scaler = StandardScaler().fit(X_cn)
    Xs = scaler.transform(X_cn).astype("float32")

    n_features = Xs.shape[1]
    ae_x = keras.Input(shape=(n_features,), dtype="float32")
    ae_h1 = layers.Dense(hidden_dim, activation="relu")(ae_x)
    ae_z = layers.Dense(latent_dim, activation="relu", name="bottleneck")(ae_h1)
    ae_h2 = layers.Dense(hidden_dim, activation="relu")(ae_z)
    ae_y = layers.Dense(n_features, activation="linear")(ae_h2)
    ae = keras.Model(ae_x, ae_y)
    ae.compile(optimizer="adam", loss="mse")

    cb = [callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor="val_loss")]
    history = ae.fit(
        Xs, Xs, validation_split=val_split, callbacks=cb,
        batch_size=32, epochs=500, verbose=0,
    )
    n_epochs = len(history.history["loss"])
    final_val_loss = min(history.history["val_loss"])
    return scaler, ae, n_epochs, final_val_loss


def keras_ae_score(ae, scaler, X):
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


def auc_block(name, score, df, cn_mask, lines, rows):
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


def main():
    df, feature_cols = build_feature_table(feature_tier="core")
    X = df[feature_cols].values
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").values
    print(f"Cohort: {len(df)} patients "
          f"(CN={cn_mask.sum()}, MCI={(df.DIAGNOSIS_LABEL=='MCI').sum()}, "
          f"AD={(df.DIAGNOSIS_LABEL=='AD').sum()}), {len(feature_cols)} features.")
    print(f"TensorFlow {tf.__version__}, Keras {keras.__version__}")

    lines = [f"Cohort: n={len(df)} (CN={cn_mask.sum()}, "
             f"MCI={(df.DIAGNOSIS_LABEL=='MCI').sum()}, AD={(df.DIAGNOSIS_LABEL=='AD').sum()}), "
             f"{len(feature_cols)} features (same as compare_three_methods.py / three_method_metrics.txt)",
             f"TensorFlow {tf.__version__}, Keras {keras.__version__}", ""]
    rows = []

    # -- PCA (unchanged, for reference) --
    scaler_p, pca, _, cum_var = fit_pca_baseline(X[cn_mask])
    T2, Q = compute_t2_q(X, scaler_p, pca)
    t2_ucl, q_ucl = control_limits(T2[cn_mask], Q[cn_mask], pca.n_components_, cn_mask.sum())
    score_pca = T2 / t2_ucl + Q / q_ucl
    lines.append(f"PCA: {pca.n_components_} components, {cum_var[pca.n_components_-1]:.1%} variance")

    # -- GMM (unchanged, for reference) --
    scaler_g, gmm, gmm_cfg, gmm_bic = fit_gmm_baseline(X[cn_mask])
    score_gmm = gmm_anomaly_score(gmm, scaler_g, X)
    lines.append(f"GMM: {gmm_cfg[0]} components, {gmm_cfg[1]} covariance (BIC={gmm_bic:.1f})")

    # -- OLD: sklearn MLPRegressor "autoencoder" --
    scaler_mlp, mlp_ae = fit_sklearn_mlp_baseline(X[cn_mask])
    score_mlp = sklearn_ae_score(mlp_ae, scaler_mlp, X)
    lines.append(f"sklearn MLPRegressor AE (OLD): 16-6-16, {mlp_ae.n_iter_} iters, loss={mlp_ae.loss_:.4f}")

    # -- NEW: genuine Keras/TF autoencoder --
    scaler_k, keras_ae, n_epochs, val_loss = fit_keras_autoencoder(X[cn_mask])
    score_keras = keras_ae_score(keras_ae, scaler_k, X)
    lines.append(f"Keras/TF AE (NEW): 16-6-16 Dense, {n_epochs} epochs (early stopped), "
                 f"best val_loss={val_loss:.4f}")
    lines.append("")

    auc_block("PCA (T2/Q)", score_pca, df, cn_mask, lines, rows)
    auc_block("GMM (-log-lik)", score_gmm, df, cn_mask, lines, rows)
    auc_block("Autoencoder -- sklearn MLPRegressor (OLD)", score_mlp, df, cn_mask, lines, rows)
    auc_block("Autoencoder -- Keras/TF (NEW)", score_keras, df, cn_mask, lines, rows)

    results_df = pd.DataFrame(rows)
    results_df.to_csv("keras_ae_comparison.csv", index=False)

    lines.append("Winner by comparison:")
    for comp in ["CN vs MCI", "CN vs AD", "CN vs MCI+AD"]:
        sub = results_df[results_df["comparison"] == comp]
        best = sub.loc[sub["auc"].idxmax()]
        lines.append(f"  {comp}: {best['method']} (AUC={best['auc']:.3f})")

    lines.append("")
    lines.append("Keras AE vs sklearn MLPRegressor AE, delta AUC (positive = Keras better):")
    for comp in ["CN vs MCI", "CN vs AD", "CN vs MCI+AD"]:
        old = results_df[(results_df.method == "Autoencoder -- sklearn MLPRegressor (OLD)") &
                          (results_df.comparison == comp)]["auc"].values[0]
        new = results_df[(results_df.method == "Autoencoder -- Keras/TF (NEW)") &
                          (results_df.comparison == comp)]["auc"].values[0]
        lines.append(f"  {comp}: {new - old:+.3f}  (old={old:.3f}, new={new:.3f})")

    metrics_text = "\n".join(lines)
    print("\n" + metrics_text)
    with open("keras_ae_metrics.txt", "w") as f:
        f.write(metrics_text + "\n")

    print("\nWrote: keras_ae_comparison.csv, keras_ae_metrics.txt")


if __name__ == "__main__":
    main()
