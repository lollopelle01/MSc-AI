"""
notebooks/remaining_useful_life/rul_model_2.py

Rate-of-change variant of the simple RUL model.

Model 1 (rul_model_1.py) predicts RUL from the biomarker values at a single
visit. Model 2 reuses the exact same per-visit dataset, then expands each
patient's timeline into one row per ordered pair of visits (i before j, no more
than MAX_GAP_MONTHS apart -- every such pair, not only consecutive ones):

    <feat>     = biomarker value at the later visit          (level)
    d_<feat>   = later value - earlier value                 (change)
    GAP_MONTHS = months elapsed between the two visits
    target     = RUL_YEARS at the later visit

Giving the model both the current level and how much it moved (plus the gap it
moved over) lets it use "small and shrinking fast" rather than either alone.

Two regressors are run on this same delta dataset and compared:
    evaluate()      - linear / random forest (see MODEL)
    evaluate_mlp()  - a small non-linear encoder (MLP)
A recurrent net is not applicable here: these rows are independent visit pairs,
not per-patient sequences.

Dataset shaping (diagnosis timeline, modality matching, forward/backward fill)
is imported unchanged from rul_model_1. Same simplicity, same modality switch:
edit MODALITIES / EXPERIMENTS to compare mri / pet / csf contributions.

Run:  cd leo && python rul_model_2.py
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*ChainedAssignment.*", category=FutureWarning)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor

from rul_model_1 import (
    build_dataset, MODALITY_COLS, DEMOG_COLS, HORIZON_YEARS, HORIZON_LABEL, MATCH_WINDOW_DAYS,
    PLOT_N_PATIENTS, sample_patient_series, plot_true_vs_pred_series,
)

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
MAX_GAP_MONTHS = 36          # skip visit pairs further apart than this
USE_DEMOGRAPHICS = True      # AGE, PTGENDER, PTEDUCAT (later visit) as context
MODEL = "rf"                 # "rf" (RandomForestRegressor) | "linear" (Ridge)
N_SPLITS = 5
RANDOM_STATE = 0

RUN_ALL = True               # True: loop EXPERIMENTS; False: single MODALITIES run
MODALITIES = ["mri", "pet", "csf"]
EXPERIMENTS = [[], ["mri"], ["pet"], ["csf"], ["mri", "pet", "csf"]]

DAYS_PER_MONTH = 30.44


# --------------------------------------------------------------------------- #
# Visit-pair level + change features                                           #
# --------------------------------------------------------------------------- #
def build_delta_dataset(df, feature_cols):
    """
    One row per ordered pair of visits (i before j) of the same patient, up to
    MAX_GAP_MONTHS apart -- every such pair, not just consecutive visits.

    <feat>     = value at the later visit (j)
    d_<feat>   = value(j) - value(i)
    GAP_MONTHS = months between visit i and visit j
    RUL_YEARS  = RUL at the later visit (what we predict)
    """
    df = df.sort_values(["RID", "EXAMDATE"]).reset_index(drop=True)

    pairs = df.add_suffix("_i").merge(
        df.add_suffix("_j"), left_on="RID_i", right_on="RID_j"
    )
    gap_days = (pairs["EXAMDATE_j"] - pairs["EXAMDATE_i"]).dt.days
    pairs = pairs[(gap_days > 0) & (gap_days <= MAX_GAP_MONTHS * DAYS_PER_MONTH)]

    out = pd.DataFrame({
        "RID": pairs["RID_i"].to_numpy(),
        "EXAMDATE": pairs["EXAMDATE_j"].to_numpy(),
        "GAP_MONTHS": (pairs["EXAMDATE_j"] - pairs["EXAMDATE_i"]).dt.days / DAYS_PER_MONTH,
        "RUL_YEARS": pairs["RUL_YEARS_j"].to_numpy(),
        "CONVERTED": pairs["CONVERTED_j"].to_numpy(),
    })
    for c in feature_cols:
        out[c] = pairs[c + "_j"].to_numpy()                                  # level
        out["d_" + c] = (pairs[c + "_j"] - pairs[c + "_i"]).to_numpy()       # change
    for c in DEMOG_COLS:                       # static / near-static: later visit
        out[c] = pairs[c + "_j"].to_numpy()

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Evaluation (patient-grouped CV, shared by both regressors)                    #
# --------------------------------------------------------------------------- #
def feature_columns(modalities):
    cols = ["GAP_MONTHS"] + (DEMOG_COLS if USE_DEMOGRAPHICS else [])
    mod_feats = [c for m in modalities for c in MODALITY_COLS[m]]
    return cols + mod_feats + ["d_" + c for c in mod_feats]  # levels + changes


def _impute_scale(X_tr, X_te, scale):
    """Fill NaN with training-fold medians; optionally standardize."""
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_tr = np.where(np.isnan(X_tr), med, X_tr)
    X_te = np.where(np.isnan(X_te), med, X_te)
    if scale:
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        sd[sd == 0] = 1.0
        X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
    return X_tr, X_te


def _oof(df, modalities, estimator_fn, scale):
    """Out-of-fold predictions + mean-predictor baseline for one estimator."""
    feat_cols = feature_columns(modalities)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["RUL_YEARS"].to_numpy(dtype=float)
    groups = df["RID"].to_numpy()

    oof_pred = np.zeros(len(y))
    oof_base = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_te = _impute_scale(X[tr].copy(), X[te].copy(), scale)
        est = estimator_fn()
        est.fit(X_tr, y[tr])
        oof_pred[te] = est.predict(X_te)
        oof_base[te] = y[tr].mean()

    return feat_cols, y, oof_pred, oof_base


def _cv_oof(df, modalities, estimator_fn, scale):
    feat_cols, y, oof_pred, oof_base = _oof(df, modalities, estimator_fn, scale)
    return {
        "modalities": "+".join(modalities) if modalities else "(gap + demographics only)",
        "n_features": len(feat_cols),
        "n_rows": len(y),
        "MAE": mean_absolute_error(y, oof_pred),
        "RMSE": mean_squared_error(y, oof_pred) ** 0.5,
        "MAE_baseline": mean_absolute_error(y, oof_base),
    }


def _regressor_estimator():
    if MODEL == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    if MODEL == "linear":
        return Ridge()
    raise ValueError("MODEL must be 'rf' or 'linear'")


def _mlp_estimator():
    """
    A small non-linear encoder (multi-layer perceptron) on the same pairwise
    delta dataset -- an alternative to the linear / forest models above.

    Inputs are standardized; the two hidden layers (32 -> 16, ReLU) compress the
    level+change features into a non-linear representation before the RUL
    regression head. Kept deliberately small given the ~4-5k training rows.
    """
    return MLPRegressor(
        hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
        max_iter=1000, early_stopping=True, n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )


def evaluate(df, modalities):
    """Linear (Ridge) or Random Forest, selected by MODEL."""
    return {"model": MODEL, **_cv_oof(df, modalities, _regressor_estimator, scale=(MODEL == "linear"))}


def evaluate_mlp(df, modalities):
    return {"model": "mlp", **_cv_oof(df, modalities, _mlp_estimator, scale=True)}


def oof_predictions(df, modalities):
    """True RUL plus out-of-fold predictions from both regressors, for plotting."""
    _, y, main_pred, _ = _oof(df, modalities, _regressor_estimator, scale=(MODEL == "linear"))
    _, _, mlp_pred, _ = _oof(df, modalities, _mlp_estimator, scale=True)
    return y, {MODEL: main_pred, "mlp": mlp_pred}


# --------------------------------------------------------------------------- #
# True vs. predicted RUL, a handful of patients concatenated                  #
# --------------------------------------------------------------------------- #
def _dedupe_by_visit(df, y, pred):
    """
    Collapse visit-pair rows down to one row per (RID, later EXAMDATE) -- the
    pair with the smallest GAP_MONTHS, i.e. the most recent prior visit -- so
    each patient can be treated as a normal per-visit timeline for plotting.
    """
    d = pd.DataFrame({
        "RID": df["RID"].to_numpy(), "EXAMDATE": df["EXAMDATE"].to_numpy(),
        "GAP_MONTHS": df["GAP_MONTHS"].to_numpy(), "y": y, "pred": pred,
    })
    d = d.sort_values("GAP_MONTHS").drop_duplicates(["RID", "EXAMDATE"], keep="first")
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    visits, biomarker_cols = build_dataset()
    df = build_delta_dataset(visits, biomarker_cols)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    n_conv = int(df["CONVERTED"].sum())
    print(f"visit pairs: {len(df)}  patients: {df['RID'].nunique()}  "
          f"converter pairs: {n_conv}  non-converter pairs: {len(df) - n_conv}")
    print(f"model: {MODEL}(+mlp)   horizon: {HORIZON_LABEL}   "
          f"match window: {MATCH_WINDOW_DAYS}d   max gap: {MAX_GAP_MONTHS}mo")
    print(f"median gap: {df['GAP_MONTHS'].median():.1f} months\n")

    experiments = EXPERIMENTS if RUN_ALL else [MODALITIES]
    rows = [evaluate(df, m) for m in experiments]
    rows += [evaluate_mlp(df, m) for m in experiments]
    results = pd.DataFrame(rows)

    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(os.path.join(out_dir, "rul_results_2.csv"), index=False)

    y, preds_by_model = oof_predictions(df, MODALITIES)
    for name, pred in preds_by_model.items():
        d = _dedupe_by_visit(df, y, pred)
        series = sample_patient_series(d, d["y"].to_numpy(), d["pred"].to_numpy())
        plot_true_vs_pred_series(
            series, os.path.join(out_dir, f"rul_model_2_{name}_pred_vs_true.png"),
            f"Model 2 ({name}, {'+'.join(MODALITIES)}): true vs. predicted RUL",
        )


if __name__ == "__main__":
    main()
