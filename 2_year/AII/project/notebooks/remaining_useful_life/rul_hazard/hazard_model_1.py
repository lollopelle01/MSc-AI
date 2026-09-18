"""
notebooks/remaining_useful_life/rul_hazard/hazard_model_1.py

Classification variant of ../rul_model_1.py: instead of regressing RUL_YEARS
directly (only defined for converters, or converters + a horizon cap for
everyone else, and currently trained on converters only), this fits a
discrete-time hazard classifier -- one row per MCI visit, EVENT_AT_VISIT = 1
only at the visit immediately preceding a clean MCI -> AD-dementia
transition, 0 at every other visit, converter or not, any amount of
follow-up. See hazard_panel.py for the full reasoning and panel definition.

Same single-visit level features as rul_model_1.py (no rate-of-change --
that's hazard_model_2.py's job), same modality on/off switch, same
patient-grouped CV. Output is HAZARD = P(EVENT_AT_VISIT | features) at each
visit, evaluated by AUC (discrimination) and a bucketed calibration gap
(whether HAZARD = 0.3 really means ~30%), unlike rul_model_1.py's MAE/RMSE.

Because every visit's hazard chains into a survival curve, an RUL number can
still be derived from it (hazard_panel.forecast_survival_curves +
survival_to_rul) for anyone who wants one -- validated here against the true
RUL_YEARS_TRUE, converters only, the same true label rul_model_1.py regresses
against directly.

Run:  cd notebooks/remaining_useful_life/rul_hazard && python hazard_model_1.py
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*ChainedAssignment.*", category=FutureWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rul_model_1 import (  # noqa: E402
    DEMOG_COLS, MODALITY_COLS, sample_patient_series, plot_true_vs_pred_series,
)
import hazard_panel as hp  # noqa: E402

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
USE_DEMOGRAPHICS = True
MODEL = "forest"             # "forest" (RandomForestClassifier) | "logistic" (LogisticRegression)
N_SPLITS = 5
RANDOM_STATE = 0
RUL_METHOD = "median"        # "median" | "expected" -- see hazard_panel.survival_to_rul
PLOT_N_CONVERTERS = 20        # converters shown in the continuous true-RUL-vs-hazard plot

RUN_ALL = True
MODALITIES = ["mri", "pet", "csf"]
EXPERIMENTS = [[], ["mri"], ["pet"], ["csf"], ["mri", "pet", "csf"]]

TEMPORAL_COLS = ["NEXT_GAP_MONTHS", "MCI_DURATION_MONTHS"]


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
def make_model():
    if MODEL == "forest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                       n_jobs=-1, class_weight="balanced")
    if MODEL == "logistic":
        return LogisticRegression(penalty="l1", solver="liblinear",
                                   class_weight="balanced", max_iter=2000)
    raise ValueError("MODEL must be 'forest' or 'logistic'")


def feature_columns(modalities):
    return TEMPORAL_COLS + (DEMOG_COLS if USE_DEMOGRAPHICS else []) + [
        c for m in modalities for c in MODALITY_COLS[m]
    ]


def _impute_scale(X_tr, X_te, scale):
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_tr = np.where(np.isnan(X_tr), med, X_tr)
    X_te = np.where(np.isnan(X_te), med, X_te)
    scaler = None
    if scale:
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        sd[sd == 0] = 1.0
        X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
        scaler = (mu, sd)
    return X_tr, X_te, med, scaler


def _oof_predict(df, modalities):
    """
    Grouped-CV out-of-fold hazard probabilities, plus, per fold, a
    predict_fn closing over that fold's own fitted model/imputation
    median/scaler -- reused unchanged by forecast_survival_curves so
    forecasting a row always uses the model that never saw it in training.
    """
    feat_cols = feature_columns(modalities)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["EVENT_AT_VISIT"].to_numpy(dtype=float)
    groups = df["RID"].to_numpy()
    scale = (MODEL == "logistic")

    oof_prob = np.zeros(len(y))
    fold_info = []
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_te, med, scaler = _impute_scale(X[tr].copy(), X[te].copy(), scale)
        model = make_model()
        model.fit(X_tr, y[tr])
        oof_prob[te] = model.predict_proba(X_te)[:, 1]

        def predict_fn(step_X, model=model, med=med, scaler=scaler):
            step_X = np.where(np.isnan(step_X), med, step_X)
            if scaler is not None:
                mu, sd = scaler
                step_X = (step_X - mu) / sd
            return model.predict_proba(step_X)[:, 1]

        fold_info.append((te, predict_fn))

    return feat_cols, y, oof_prob, fold_info, X


def evaluate(df, modalities):
    feat_cols, y, oof_prob, _, _ = _oof_predict(df, modalities)
    _, gap = hp.calibration_table(y, oof_prob)
    return {
        "modalities": "+".join(modalities) if modalities else "(demographics only)",
        "n_features": len(feat_cols),
        "n_rows": len(y),
        "event_rate": y.mean(),
        "AUC": roc_auc_score(y, oof_prob),
        "calibration_gap": gap,
    }


# --------------------------------------------------------------------------- #
# Derived RUL, validated against RUL_YEARS_TRUE on converters                 #
# --------------------------------------------------------------------------- #
def derived_rul_years(df, modalities):
    feat_cols, _, _, fold_info, X = _oof_predict(df, modalities)

    rul_months = np.full(len(df), np.nan)
    for te, predict_fn in fold_info:
        S = hp.forecast_survival_curves(predict_fn, X[te], feat_cols)
        result = hp.survival_to_rul(S, method=RUL_METHOD)
        rul_months[te] = result[0] if RUL_METHOD == "expected" else result
    return rul_months / 12.0


# --------------------------------------------------------------------------- #
# True RUL vs. raw hazard probability, top converters concatenated over time  #
# --------------------------------------------------------------------------- #
def top_converter_series(conv, n_patients=PLOT_N_CONVERTERS):
    """
    Among converter visits (conv = df[df["CONVERTED"]], with a HAZARD_PROB
    column already attached), pick the n_patients patients with the longest
    visit sequences. For each, return (rid, elapsed_years, y_true_rul,
    hazard_prob) in chronological order -- elapsed_years is time since that
    patient's first converter visit (real EXAMDATE gaps, not a visit index).
    """
    sizes = conv.groupby("RID").size().sort_values(ascending=False)
    chosen = sizes.head(n_patients).index

    series = []
    for rid in chosen:
        pos = np.flatnonzero((conv["RID"] == rid).to_numpy())
        pos = pos[np.argsort(conv["EXAMDATE"].to_numpy()[pos])]
        examdate = conv["EXAMDATE"].to_numpy()[pos]
        elapsed_years = (examdate - examdate[0]) / np.timedelta64(1, "D") / 365.25
        series.append((
            str(rid), elapsed_years,
            conv["RUL_YEARS_TRUE"].to_numpy()[pos],
            conv["HAZARD_PROB"].to_numpy()[pos],
        ))
    return series


def plot_true_rul_vs_hazard_time(series, out_path, title, gap_years=1.0):
    """
    Concatenate each selected converter's true RUL (years) and predicted
    hazard probability (before any thresholding/classification) along one
    continuous time axis. True RUL and hazard sit on separate y-axes -- years
    vs. a 0-1 probability -- since what's meaningful is the hazard rising as
    true RUL approaches zero, not their absolute overlap. Dashed verticals
    mark where one patient's history ends and the next begins.
    """
    x_all, true_all, hazard_all, ticks, seps = [], [], [], [], []
    offset = 0.0
    for _, t, y_true_i, hazard_i in series:
        x = offset + (t - t[0])
        x_all.extend(x.tolist() + [np.nan])
        true_all.extend(y_true_i.tolist() + [np.nan])
        hazard_all.extend(hazard_i.tolist() + [np.nan])
        ticks.append((x[0] + x[-1]) / 2)
        offset = x[-1] + gap_years
        seps.append(offset - gap_years / 2)

    fig, ax1 = plt.subplots(figsize=(16, 4))
    ax1.plot(x_all, true_all, marker="o", markersize=3, label="True RUL (years)", color="tab:blue")
    ax1.set_ylabel("True RUL (years)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(x_all, hazard_all, marker="o", markersize=3, label="Hazard probability", color="tab:orange")
    ax2.set_ylabel("Hazard probability", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1)

    for b in seps[:-1]:
        ax1.axvline(b, color="grey", linestyle="--", linewidth=0.7)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([rid for rid, *_ in series], rotation=90)
    ax1.set_xlabel("elapsed time per patient (years since first converter visit), concatenated")
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    df, _ = hp.build_hazard_dataset()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"MCI visits: {len(df)}  patients: {df['RID'].nunique()}  "
          f"event visits: {int(df['EVENT_AT_VISIT'].sum())}  "
          f"converter visits: {int(df['CONVERTED'].sum())}")
    print(f"model: {MODEL}   forecast step: {hp.STEP_MONTHS}mo   "
          f"forecast horizon: {hp.STEP_MONTHS * hp.MAX_STEPS / 12:.0f}y\n")

    experiments = EXPERIMENTS if RUN_ALL else [MODALITIES]
    results = pd.DataFrame(evaluate(df, m) for m in experiments)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(os.path.join(out_dir, "hazard_results_1.csv"), index=False)

    rul_pred_years = derived_rul_years(df, MODALITIES)
    conv = df[df["CONVERTED"]].copy()
    conv["RUL_PRED_YEARS"] = rul_pred_years[df["CONVERTED"].to_numpy()]
    resolved = conv["RUL_PRED_YEARS"].notna()

    print(f"\nDerived RUL ({RUL_METHOD}) vs RUL_YEARS_TRUE, converter visits only:")
    print(f"  resolved within {hp.STEP_MONTHS * hp.MAX_STEPS / 12:.0f}y horizon: "
          f"{resolved.sum()} / {len(conv)} ({resolved.mean():.1%})")
    if resolved.any():
        mae = mean_absolute_error(conv.loc[resolved, "RUL_YEARS_TRUE"], conv.loc[resolved, "RUL_PRED_YEARS"])
        print(f"  MAE (resolved rows): {mae:.3f} years")

    conv = conv.reset_index(drop=True)
    series = sample_patient_series(conv, conv["RUL_YEARS_TRUE"].to_numpy(), conv["RUL_PRED_YEARS"].to_numpy())
    plot_true_vs_pred_series(
        series, os.path.join(out_dir, "hazard_model_1_pred_vs_true.png"),
        f"Hazard model 1 ({MODEL}, {'+'.join(MODALITIES)}): true vs. derived RUL (converters only)",
    )

    _, _, oof_prob, _, _ = _oof_predict(df, MODALITIES)
    hazard_conv = df[df["CONVERTED"]].copy()
    hazard_conv["HAZARD_PROB"] = oof_prob[df["CONVERTED"].to_numpy()]
    hazard_conv = hazard_conv.reset_index(drop=True)

    conv_series = top_converter_series(hazard_conv)
    plot_true_rul_vs_hazard_time(
        conv_series, os.path.join(out_dir, "hazard_model_1_top_converters_time.png"),
        f"Hazard model 1 ({MODEL}, {'+'.join(MODALITIES)}): true RUL vs. raw hazard probability over time -- "
        f"top {len(conv_series)} converters by sequence length",
    )


if __name__ == "__main__":
    main()
