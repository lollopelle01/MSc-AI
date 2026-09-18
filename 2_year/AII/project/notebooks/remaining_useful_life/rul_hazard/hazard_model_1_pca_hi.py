"""
notebooks/remaining_useful_life/rul_hazard/hazard_model_1_pca_hi.py

Extends hazard_model_1_cost_sensitive.py's balanced/cost_sensitive comparison
(RUL_METHODS.md section 10) with one more variable: whether
notebooks/anomaly_detection/method_b_autoencoder_hi/pca_hi_trajectory.py's per-visit PCA/Hotelling
T^2+Q health index (HI) is included as an extra hazard-model input feature,
alongside the usual mri+pet+csf modality columns.

pca_hi_trajectory.py fits Method A's PCA "healthy operating region" (on
Cognitively Normal visits) but keeps Method B's LONGITUDINAL structure: one
HI value per (RID, visit), not one per patient. That per-visit shape is what
makes it mergeable onto this hazard panel the same way MRI/PET/CSF already
are -- by nearest EXAMDATE within MATCH_WINDOW_DAYS (rul_model_1._asof_merge,
reused unchanged), then forward/backward-filled per patient like every other
modality column in hazard_panel.build_hazard_dataset. This gives far better
coverage than merging method_a.py's cross-sectional, one-row-per-patient
score by RID alone would (measured directly below).

The HI itself is recomputed here via method_a.py's own fit_pca_baseline /
compute_t2_q / control_limits, fit on pca_hi_trajectory.py's own longitudinal
CN visits -- the exact same computation pca_hi_trajectory.py performs, not a
reimplementation, so this stays in sync with both scripts.

Run:  cd notebooks/remaining_useful_life/rul_hazard && python hazard_model_1_pca_hi.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*ChainedAssignment.*", category=FutureWarning)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                              "anomaly_detection", "method_a_pca_density"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                              "anomaly_detection", "method_b_autoencoder_hi"))
import hazard_panel as hp  # noqa: E402
from rul_model_1 import _asof_merge  # noqa: E402
from hazard_model_1 import feature_columns, _impute_scale  # noqa: E402
from hazard_model_1_cost_sensitive import (  # noqa: E402
    make_model, expected_cost, sweep_best_threshold, DECISION_THRESHOLD, WEIGHTINGS,
)
from method_a import fit_pca_baseline, compute_t2_q, control_limits  # noqa: E402
from load_longitudinal import build_longitudinal_table  # noqa: E402

MODEL = "forest"             # matches hazard_model_1_cost_sensitive.py's default
N_SPLITS = 5
MODALITIES = ["mri", "pet", "csf"]
HI_COL = "PCA_HI"


# --------------------------------------------------------------------------- #
# pca_hi_trajectory.py's per-visit HI, recomputed and attached to the panel   #
# --------------------------------------------------------------------------- #
def compute_pca_hi():
    """One row per (RID, EXAMDATE): pca_hi_trajectory.py's combined T2/UCL +
    Q/UCL health index, using method_a's PCA-on-CN-baseline fit unchanged,
    fit on this longitudinal table's own CN visits."""
    long_df, feat_cols = build_longitudinal_table()
    X = long_df[feat_cols].values
    cn_mask = (long_df["DIAGNOSIS_LABEL"] == "CN").values

    scaler, pca, _, _ = fit_pca_baseline(X[cn_mask])
    T2, Q = compute_t2_q(X, scaler, pca)
    t2_ucl, q_ucl = control_limits(T2[cn_mask], Q[cn_mask], pca.n_components_, cn_mask.sum())
    long_df[HI_COL] = T2 / t2_ucl + Q / q_ucl

    return long_df[["RID", "EXAMDATE", HI_COL]]


def attach_pca_hi(df):
    hi = compute_pca_hi()
    out = _asof_merge(df, hi)
    out = out.sort_values(["RID", "EXAMDATE"])
    out[HI_COL] = out.groupby("RID")[HI_COL].ffill()
    out[HI_COL] = out.groupby("RID")[HI_COL].bfill()

    has_hi = out[HI_COL].notna()
    print(f"PCA-HI coverage: {out.loc[has_hi, 'RID'].nunique()} / "
          f"{df['RID'].nunique()} patients, {has_hi.mean():.1%} of {len(out)} visit-rows.")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Same grouped-CV evaluation as hazard_model_1_cost_sensitive.py, with an     #
# optional extra HI_COL feature                                               #
# --------------------------------------------------------------------------- #
def _oof_predict(df, modalities, weighting, with_hi):
    feat_cols = feature_columns(modalities) + ([HI_COL] if with_hi else [])
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["EVENT_AT_VISIT"].to_numpy(dtype=float)
    groups = df["RID"].to_numpy()
    scale = (MODEL == "logistic")

    oof_prob = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_te, _, _ = _impute_scale(X[tr].copy(), X[te].copy(), scale)
        model = make_model(weighting)
        model.fit(X_tr, y[tr])
        oof_prob[te] = model.predict_proba(X_te)[:, 1]
    return feat_cols, y, oof_prob


def evaluate(df, modalities, weighting, with_hi):
    feat_cols, y, oof_prob = _oof_predict(df, modalities, weighting, with_hi)
    _, gap = hp.calibration_table(y, oof_prob)
    cost_fixed, fn_fixed, fp_fixed = expected_cost(y, oof_prob, threshold=DECISION_THRESHOLD)
    cost_best, t_best, fn_best, fp_best = sweep_best_threshold(y, oof_prob)
    return {
        "weighting": weighting,
        "+pca_hi": with_hi,
        "n_features": len(feat_cols),
        "AUC": roc_auc_score(y, oof_prob),
        "calibration_gap": gap,
        "cost@0.5": cost_fixed,
        "fn@0.5": fn_fixed,
        "fp@0.5": fp_fixed,
        "best_threshold": t_best,
        "cost@best": cost_best,
        "fn@best": fn_best,
        "fp@best": fp_best,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    df, _ = hp.build_hazard_dataset()
    df = attach_pca_hi(df)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    rows = [
        evaluate(df, MODALITIES, weighting, with_hi)
        for weighting in WEIGHTINGS
        for with_hi in (False, True)
    ]
    results = pd.DataFrame(rows)

    print(f"\nmri+pet+csf, {MODEL}, with vs. without pca_hi_trajectory's per-visit PCA-HI:")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(os.path.join(out_dir, "hazard_results_1_pca_hi.csv"), index=False)


if __name__ == "__main__":
    main()
