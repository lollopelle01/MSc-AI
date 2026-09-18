"""
notebooks/remaining_useful_life/rul_hazard/hazard_model_2.py

Classification variant of ../rul_model_2.py: same visit-pair, level+change+
gap dataset (biomarker value at the later visit, its change since an earlier
one, and how many months separate them), but the target is EVENT_AT_VISIT --
did *this* MCI visit turn out to be the one immediately preceding a clean
AD-dementia transition -- instead of RUL_YEARS. Pairs are restricted to the
same clean MCI run (hazard_panel.RUN_ID): pairing across a reversion to CN,
or across a different episode entirely, would mix rate-of-change signal
across declines that aren't the same one.

Two regressors from rul_model_2.py become two classifiers here: RandomForest
/ LogisticRegression (see MODEL) and a small MLP (evaluate_mlp), compared the
same way via AUC + a bucketed calibration gap instead of MAE/RMSE. Derived
RUL (see hazard_panel.py) is validated against RUL_YEARS_TRUE the same way
hazard_model_1.py does.

PAIR_GAP_MONTHS (between the pair's earlier and later visit) stays frozen
during the forecast walk below -- it describes the historical window the
frozen delta features were computed over, not a future interval. Only
NEXT_GAP_MONTHS and MCI_DURATION_MONTHS advance (see
hazard_panel.forecast_survival_curves).

Run:  cd notebooks/remaining_useful_life/rul_hazard && python hazard_model_2.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*ChainedAssignment.*", category=FutureWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rul_model_1 import DEMOG_COLS, MODALITY_COLS, sample_patient_series, plot_true_vs_pred_series  # noqa: E402
import hazard_panel as hp  # noqa: E402

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
MAX_GAP_MONTHS = 36          # skip visit pairs further apart than this
USE_DEMOGRAPHICS = True
MODEL = "forest"             # "forest" (RandomForestClassifier) | "logistic" (LogisticRegression)
N_SPLITS = 5
RANDOM_STATE = 0
RUL_METHOD = "median"

RUN_ALL = True
MODALITIES = ["mri", "pet", "csf"]
EXPERIMENTS = [[], ["mri"], ["pet"], ["csf"], ["mri", "pet", "csf"]]

DAYS_PER_MONTH = 30.44
CARRY_COLS = ["NEXT_GAP_MONTHS", "MCI_DURATION_MONTHS", "CONVERTED", "RUL_YEARS_TRUE"]


# --------------------------------------------------------------------------- #
# Visit-pair level + change features, within the same clean MCI run only      #
# --------------------------------------------------------------------------- #
def build_delta_dataset(df, feature_cols):
    df = df.sort_values(["RID", "EXAMDATE"]).reset_index(drop=True)

    pairs = df.add_suffix("_i").merge(
        df.add_suffix("_j"), left_on=["RID_i", "RUN_ID_i"], right_on=["RID_j", "RUN_ID_j"]
    )
    gap_days = (pairs["EXAMDATE_j"] - pairs["EXAMDATE_i"]).dt.days
    pairs = pairs[(gap_days > 0) & (gap_days <= MAX_GAP_MONTHS * DAYS_PER_MONTH)]

    out = pd.DataFrame({
        "RID": pairs["RID_i"].to_numpy(),
        "EXAMDATE": pairs["EXAMDATE_j"].to_numpy(),
        "PAIR_GAP_MONTHS": (pairs["EXAMDATE_j"] - pairs["EXAMDATE_i"]).dt.days / DAYS_PER_MONTH,
        "EVENT_AT_VISIT": pairs["EVENT_AT_VISIT_j"].to_numpy(),
    })
    for c in CARRY_COLS:
        out[c] = pairs[c + "_j"].to_numpy()
    for c in feature_cols:
        out[c] = pairs[c + "_j"].to_numpy()                                  # level
        out["d_" + c] = (pairs[c + "_j"] - pairs[c + "_i"]).to_numpy()       # change
    for c in DEMOG_COLS:
        out[c] = pairs[c + "_j"].to_numpy()

    return out.reset_index(drop=True)


def feature_columns(modalities):
    cols = ["PAIR_GAP_MONTHS", "NEXT_GAP_MONTHS", "MCI_DURATION_MONTHS"] + \
           (DEMOG_COLS if USE_DEMOGRAPHICS else [])
    mod_feats = [c for m in modalities for c in MODALITY_COLS[m]]
    return cols + mod_feats + ["d_" + c for c in mod_feats]


# --------------------------------------------------------------------------- #
# Evaluation (patient-grouped CV, shared by both classifiers)                  #
# --------------------------------------------------------------------------- #
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


def _oof(df, modalities, estimator_fn, scale):
    feat_cols = feature_columns(modalities)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["EVENT_AT_VISIT"].to_numpy(dtype=float)
    groups = df["RID"].to_numpy()

    oof_prob = np.zeros(len(y))
    fold_info = []
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_te, med, scaler = _impute_scale(X[tr].copy(), X[te].copy(), scale)
        est = estimator_fn()
        est.fit(X_tr, y[tr])
        oof_prob[te] = est.predict_proba(X_te)[:, 1]

        def predict_fn(step_X, est=est, med=med, scaler=scaler):
            step_X = np.where(np.isnan(step_X), med, step_X)
            if scaler is not None:
                mu, sd = scaler
                step_X = (step_X - mu) / sd
            return est.predict_proba(step_X)[:, 1]

        fold_info.append((te, predict_fn))

    return feat_cols, y, oof_prob, fold_info, X


def _cv_oof(df, modalities, estimator_fn, scale):
    feat_cols, y, oof_prob, _, _ = _oof(df, modalities, estimator_fn, scale)
    _, gap = hp.calibration_table(y, oof_prob)
    return {
        "modalities": "+".join(modalities) if modalities else "(gap + demographics only)",
        "n_features": len(feat_cols),
        "n_rows": len(y),
        "event_rate": y.mean(),
        "AUC": roc_auc_score(y, oof_prob),
        "calibration_gap": gap,
    }


def _classifier_estimator():
    if MODEL == "forest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                       n_jobs=-1, class_weight="balanced")
    if MODEL == "logistic":
        return LogisticRegression(penalty="l1", solver="liblinear",
                                   class_weight="balanced", max_iter=2000)
    raise ValueError("MODEL must be 'forest' or 'logistic'")


def _mlp_estimator():
    return MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
        max_iter=1000, early_stopping=True, n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )


def evaluate(df, modalities):
    return {"model": MODEL, **_cv_oof(df, modalities, _classifier_estimator, scale=(MODEL == "logistic"))}


def evaluate_mlp(df, modalities):
    return {"model": "mlp", **_cv_oof(df, modalities, _mlp_estimator, scale=True)}


# --------------------------------------------------------------------------- #
# Derived RUL, validated against RUL_YEARS_TRUE on converters                  #
# --------------------------------------------------------------------------- #
def derived_rul_years(df, modalities, estimator_fn, scale):
    feat_cols, _, _, fold_info, X = _oof(df, modalities, estimator_fn, scale)
    rul_months = np.full(len(df), np.nan)
    for te, predict_fn in fold_info:
        S = hp.forecast_survival_curves(predict_fn, X[te], feat_cols)
        result = hp.survival_to_rul(S, method=RUL_METHOD)
        rul_months[te] = result[0] if RUL_METHOD == "expected" else result
    return rul_months / 12.0


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    visits, biomarker_cols = hp.build_hazard_dataset()
    df = build_delta_dataset(visits, biomarker_cols)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"visit pairs: {len(df)}  patients: {df['RID'].nunique()}  "
          f"event pairs: {int(df['EVENT_AT_VISIT'].sum())}")
    print(f"model: {MODEL}(+mlp)   max pair gap: {MAX_GAP_MONTHS}mo   "
          f"forecast step: {hp.STEP_MONTHS}mo")
    print(f"median pair gap: {df['PAIR_GAP_MONTHS'].median():.1f} months\n")

    experiments = EXPERIMENTS if RUN_ALL else [MODALITIES]
    rows = [evaluate(df, m) for m in experiments]
    rows += [evaluate_mlp(df, m) for m in experiments]
    results = pd.DataFrame(rows)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(os.path.join(out_dir, "hazard_results_2.csv"), index=False)

    for name, estimator_fn, scale in [(MODEL, _classifier_estimator, MODEL == "logistic"),
                                       ("mlp", _mlp_estimator, True)]:
        rul_pred_years = derived_rul_years(df, MODALITIES, estimator_fn, scale)

        # one row per (RID, EXAMDATE) -- the pair with the smallest PAIR_GAP_MONTHS
        # -- so each patient can be treated as a normal per-visit timeline for plotting
        d = pd.DataFrame({
            "RID": df["RID"], "EXAMDATE": df["EXAMDATE"], "PAIR_GAP_MONTHS": df["PAIR_GAP_MONTHS"],
            "CONVERTED": df["CONVERTED"], "RUL_YEARS_TRUE": df["RUL_YEARS_TRUE"],
            "RUL_PRED_YEARS": rul_pred_years,
        })
        d = d.sort_values("PAIR_GAP_MONTHS").drop_duplicates(["RID", "EXAMDATE"], keep="first")
        conv = d[d["CONVERTED"]].reset_index(drop=True)
        resolved = conv["RUL_PRED_YEARS"].notna()

        print(f"\n[{name}] derived RUL vs RUL_YEARS_TRUE, converter visits only: "
              f"resolved {resolved.sum()} / {len(conv)} ({resolved.mean():.1%})")
        if resolved.any():
            mae = mean_absolute_error(conv.loc[resolved, "RUL_YEARS_TRUE"], conv.loc[resolved, "RUL_PRED_YEARS"])
            print(f"  MAE (resolved rows): {mae:.3f} years")

        series = sample_patient_series(conv, conv["RUL_YEARS_TRUE"].to_numpy(), conv["RUL_PRED_YEARS"].to_numpy())
        plot_true_vs_pred_series(
            series, os.path.join(out_dir, f"hazard_model_2_{name}_pred_vs_true.png"),
            f"Hazard model 2 ({name}, {'+'.join(MODALITIES)}): true vs. derived RUL (converters only)",
        )


if __name__ == "__main__":
    main()
