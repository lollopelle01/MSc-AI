"""
notebooks/anomaly_detection/method_b_autoencoder_hi/method_b.py

Method B -- pipeline-oriented anomaly detection via an autoencoder trained
on the CN ("healthy") population, producing two outputs that are meant to be
consumed by the rest of the framework rather than being an end in themselves:

  1. HI(t): a per-visit Health Index (reconstruction error) that traces a
     degradation trajectory per patient over their ADNI follow-up -- the
     direct analogue of a bearing/turbofan degradation curve in classic PHM
     (prognostics & health management) pipelines. This is what point 2 (RUL
     / time-to-conversion) is meant to consume.
  2. A latent embedding per visit (the autoencoder's bottleneck
     representation) -- a compact, learned patient representation meant to
     feed point 3 (decision-support / risk stratification) so that stage
     doesn't have to re-derive features from scratch.

Autoencoder implementation: genuine Keras/TensorFlow (functional API, ReLU
Dense encoder down to a bottleneck, linear Dense decoder back to input
dimensionality, Adam + MSE, EarlyStopping restoring best weights) -- built
exactly the way the course's own lecture (02-ad-hd, lesson 4, "Autoencoders
for Anomaly Detection") does it. This replaces an earlier sklearn
MLPRegressor workaround that was used only because PyTorch/TensorFlow
wasn't installed at the time; a side-by-side rerun on the Point 1 cohort
(keras_ae_experiment.py, in method_a_pca_density/) showed the Keras version
improves AUC by +0.036 on every comparison, so it was made the actual
implementation here as well.

Usage:
    python3 method_b.py
Outputs (written next to this script):
    - hi_trajectories.csv       per-visit RID, VISIT_INDEX, YEARS_FROM_BASELINE,
                                 DIAGNOSIS_LABEL, BASELINE_DIAGNOSIS, HI,
                                 latent_0..latent_{LATENT_DIM-1}
    - hi_trajectory_sample.png  spaghetti plot of individual patient HI(t)
    - hi_trajectory_mean.png    mean HI(t) +/- std by baseline diagnosis group
    - hi_by_diagnosis.png       boxplot of per-visit HI by diagnosis (cross-sectional)
    - hi_slope_by_group.png     per-patient HI slope (trend) by baseline group
    - metrics.txt               AUCs (per-visit and last-visit-only) + notes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from load_longitudinal import build_longitudinal_table

RANDOM_STATE = 42
LATENT_DIM = 6
HIDDEN_DIM = 16
COLORS = {"CN": "#4C72B0", "MCI": "#DD8452", "AD": "#C44E52"}
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def fit_autoencoder(X_cn, val_split=0.15):
    """Genuine Keras/TF autoencoder: functional API, ReLU Dense encoder down
    to a bottleneck, linear Dense decoder back to input dimensionality,
    Adam + MSE, EarlyStopping restoring best weights. Returns the full
    autoencoder plus a small encoder sub-model sharing its weights, used by
    encode() to read off the bottleneck activations directly (no need to
    replicate the forward pass manually, unlike the old sklearn version)."""
    scaler = StandardScaler().fit(X_cn)
    Xs = scaler.transform(X_cn).astype("float32")
    n_features = Xs.shape[1]

    ae_in = keras.Input(shape=(n_features,), dtype="float32")
    h1 = layers.Dense(HIDDEN_DIM, activation="relu")(ae_in)
    z = layers.Dense(LATENT_DIM, activation="relu", name="bottleneck")(h1)
    h2 = layers.Dense(HIDDEN_DIM, activation="relu")(z)
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

    encoder = keras.Model(ae_in, z)
    return scaler, ae, encoder


def encode(encoder, Xs):
    return encoder.predict(Xs.astype("float32"), verbose=0)  # shape (n, LATENT_DIM)


def reconstruction_error(ae, Xs):
    Xhat = ae.predict(Xs.astype("float32"), verbose=0)
    return np.mean((Xs - Xhat) ** 2, axis=1)


def main():
    import matplotlib
    matplotlib.use("Agg")

    df, feature_cols = build_longitudinal_table()
    df = df.dropna(subset=["DIAGNOSIS_LABEL"]).reset_index(drop=True)
    print(f"Loaded {len(df)} visits from {df['RID'].nunique()} patients.")
    print(df["DIAGNOSIS_LABEL"].value_counts())

    baseline_date = df.groupby("RID")["EXAMDATE"].transform("min")
    df["YEARS_FROM_BASELINE"] = (df["EXAMDATE"] - baseline_date).dt.days / 365.25

    X = df[feature_cols].values
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").values
    print(f"Training autoencoder on {cn_mask.sum()} CN visits "
          f"({df.loc[cn_mask, 'RID'].nunique()} unique CN patients).")

    scaler, ae, encoder = fit_autoencoder(X[cn_mask])
    Xs_all = scaler.transform(X)
    df["HI"] = reconstruction_error(ae, Xs_all)
    latent = encode(encoder, Xs_all)
    for i in range(LATENT_DIM):
        df[f"latent_{i}"] = latent[:, i]

    out_cols = (["RID", "VISCODE2", "EXAMDATE", "VISIT_INDEX", "YEARS_FROM_BASELINE",
                 "DIAGNOSIS_LABEL", "BASELINE_DIAGNOSIS", "HI"]
                + [f"latent_{i}" for i in range(LATENT_DIM)])
    df[out_cols].to_csv("hi_trajectories.csv", index=False)

    # --- evaluation ---
    lines = []
    lines.append(f"n_visits = {len(df)}, n_patients = {df['RID'].nunique()}")
    lines.append(f"CN visits used to fit autoencoder: {cn_mask.sum()} "
                 f"({df.loc[cn_mask, 'RID'].nunique()} patients)")
    lines.append(f"Autoencoder: {HIDDEN_DIM}-{LATENT_DIM}-{HIDDEN_DIM} Keras/TF Dense, "
                 f"{ae.n_epochs_} epochs (early stopped), best val_loss={ae.final_val_loss_:.4f}")
    lines.append("")
    lines.append("-- AUC using every visit (per-visit DIAGNOSIS_LABEL as target) --")
    lines.append("NOTE: patients contribute multiple correlated visits here, so")
    lines.append("this inflates apparent sample size (pseudo-replication). See the")
    lines.append("last-visit-only AUC below for a per-patient, non-repeated check.")
    for group in ["MCI", "AD"]:
        mask = (df["DIAGNOSIS_LABEL"] == "CN") | (df["DIAGNOSIS_LABEL"] == group)
        y = (df.loc[mask, "DIAGNOSIS_LABEL"] == group).astype(int)
        auc = roc_auc_score(y, df.loc[mask, "HI"])
        lines.append(f"  CN vs {group}: {auc:.3f}  (n={mask.sum()})")
    mask_any = df["DIAGNOSIS_LABEL"].isin(["CN", "MCI", "AD"])
    y_any = (df.loc[mask_any, "DIAGNOSIS_LABEL"] != "CN").astype(int)
    lines.append(f"  CN vs MCI+AD: {roc_auc_score(y_any, df.loc[mask_any, 'HI']):.3f}")

    last_visit = df.sort_values("EXAMDATE").groupby("RID").last().reset_index()
    lines.append("")
    lines.append("-- AUC using only each patient's LAST visit (one row per patient) --")
    for group in ["MCI", "AD"]:
        mask = (last_visit["DIAGNOSIS_LABEL"] == "CN") | (last_visit["DIAGNOSIS_LABEL"] == group)
        y = (last_visit.loc[mask, "DIAGNOSIS_LABEL"] == group).astype(int)
        if y.nunique() < 2:
            lines.append(f"  CN vs {group}: skipped (one class only)")
            continue
        auc = roc_auc_score(y, last_visit.loc[mask, "HI"])
        lines.append(f"  CN vs {group}: {auc:.3f}  (n={mask.sum()})")
    mask_any = last_visit["DIAGNOSIS_LABEL"].isin(["CN", "MCI", "AD"])
    y_any = (last_visit.loc[mask_any, "DIAGNOSIS_LABEL"] != "CN").astype(int)
    lines.append(f"  CN vs MCI+AD: {roc_auc_score(y_any, last_visit.loc[mask_any, 'HI']):.3f}")

    # per-patient HI slope (trend): does HI rise faster for MCI/AD than CN?
    def slope(g):
        if len(g) < 2 or g["YEARS_FROM_BASELINE"].max() == 0:
            return np.nan
        return np.polyfit(g["YEARS_FROM_BASELINE"], g["HI"], 1)[0]

    slopes = df.groupby("RID").apply(slope, include_groups=False).rename("HI_SLOPE")
    slopes_df = df.drop_duplicates("RID")[["RID", "BASELINE_DIAGNOSIS"]].merge(
        slopes, on="RID"
    ).dropna()
    lines.append("")
    lines.append("-- Per-patient HI slope (HI units / year) by baseline group --")
    for g in ["CN", "MCI", "AD"]:
        s = slopes_df.loc[slopes_df["BASELINE_DIAGNOSIS"] == g, "HI_SLOPE"]
        lines.append(f"  {g}: median={s.median():.4f}, mean={s.mean():.4f}, n={len(s)}")

    metrics_text = "\n".join(lines)
    print("\n" + metrics_text)
    with open("metrics.txt", "w") as f:
        f.write(metrics_text + "\n")

    # --- plots ---
    rng = np.random.default_rng(RANDOM_STATE)
    plt.figure(figsize=(7, 5))
    for group, color in COLORS.items():
        rids = df.loc[df["BASELINE_DIAGNOSIS"] == group, "RID"].unique()
        sample = rng.choice(rids, size=min(8, len(rids)), replace=False)
        for i, rid in enumerate(sample):
            g = df[df["RID"] == rid].sort_values("YEARS_FROM_BASELINE")
            plt.plot(g["YEARS_FROM_BASELINE"], g["HI"], color=color, alpha=0.6,
                      lw=1.2, marker="o", ms=3,
                      label=group if i == 0 else None)
    plt.xlabel("Years from baseline visit")
    plt.ylabel("HI (reconstruction error)")
    plt.title("Individual patient HI trajectories (sample)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hi_trajectory_sample.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    bins = np.arange(0, 8.5, 0.5)
    for group, color in COLORS.items():
        g = df[df["BASELINE_DIAGNOSIS"] == group].copy()
        g["bin"] = pd.cut(g["YEARS_FROM_BASELINE"], bins)
        stats = g.groupby("bin", observed=True)["HI"].agg(["mean", "std", "count"])
        stats = stats[stats["count"] >= 5]
        x = [interval.mid for interval in stats.index]
        plt.plot(x, stats["mean"], color=color, label=group, marker="o", ms=4)
        plt.fill_between(x, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                          color=color, alpha=0.15)
    plt.xlabel("Years from baseline visit")
    plt.ylabel("Mean HI +/- std")
    plt.title("Mean HI trajectory by baseline diagnosis group")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hi_trajectory_mean.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    order = ["CN", "MCI", "AD"]
    data = [df.loc[df["DIAGNOSIS_LABEL"] == g, "HI"] for g in order]
    bp = plt.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.6)
    plt.ylabel("HI (reconstruction error)")
    plt.title("Per-visit HI by diagnosis (cross-sectional, all visits)")
    plt.tight_layout()
    plt.savefig("hi_by_diagnosis.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    data = [slopes_df.loc[slopes_df["BASELINE_DIAGNOSIS"] == g, "HI_SLOPE"] for g in order]
    bp = plt.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.6)
    plt.axhline(0, color="k", lw=0.8, ls="--")
    plt.ylabel("HI slope (units/year)")
    plt.title("Per-patient HI trend by baseline diagnosis group")
    plt.tight_layout()
    plt.savefig("hi_slope_by_group.png", dpi=150)
    plt.close()

    print("\nWrote: hi_trajectories.csv, metrics.txt, hi_trajectory_sample.png, "
          "hi_trajectory_mean.png, hi_by_diagnosis.png, hi_slope_by_group.png")


if __name__ == "__main__":
    main()
