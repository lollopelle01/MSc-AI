"""
notebooks/anomaly_detection/method_a_pca_density/window_sensitivity.py

Sensitivity check on MATCH_WINDOW_DAYS (the diagnosis<->MRI date-matching
window in load_features.py), designed to separate two different effects
that a naive "just rerun with window=90 vs window=180" comparison would
conflate:

  (a) Evaluation-set composition effect: a tighter window excludes patients
      whose nearest MRI is far from their diagnosis date. Those excluded
      patients might differ systematically (e.g. denser follow-up visits
      could correlate with disease severity), so a full-cohort AUC
      difference between window sizes could just reflect *who got scored*,
      not whether tighter alignment produces a better anomaly signal.

  (b) Healthy-baseline pollution effect: the PCA "healthy" model is fit only
      on CN patients. If CN patients with loosely-matched (far-in-time) MRI
      get included in that baseline fit, their less-relevant scans could
      blur the learned healthy region, making anomalies harder to detect
      for everyone -- independent of which patients are being *evaluated*.

To isolate (b) from (a), this script fixes the EVALUATION set to the
tightly-matched cohort (gap <= 90 days) in both comparisons, and only
varies which cohort's CN patients were used to FIT the PCA baseline:
  - Model_90:  baseline fit on CN patients with gap <= 90 days
  - Model_180: baseline fit on CN patients with gap <= 180 days (superset,
               includes some CN with looser matches)
Both models are then scored on the SAME fixed evaluation set (gap <= 90).
If their AUCs are close, the window size mostly doesn't matter for baseline
quality. If Model_90 clearly outperforms Model_180 on the same eval set,
that's real evidence the window size affects anomaly-detection quality, not
just cohort composition.

For reference, the naive (a)-confounded comparison (full window=90 cohort
vs full window=180 cohort, different eval sets each time) is also reported.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from load_features import build_feature_table
from method_a import fit_pca_baseline, compute_t2_q, control_limits

WIDE_WINDOW = 180
TIGHT_WINDOW = 90


def score_with_model(df, feature_cols, cn_mask_fit, eval_df):
    """Fit PCA baseline on df[cn_mask_fit], score eval_df, return combined score array."""
    X_fit = df.loc[cn_mask_fit, feature_cols].values
    scaler, pca, _, _ = fit_pca_baseline(X_fit)
    T2_cn, Q_cn = compute_t2_q(X_fit, scaler, pca)
    t2_ucl, q_ucl = control_limits(T2_cn, Q_cn, pca.n_components_, cn_mask_fit.sum())

    X_eval = eval_df[feature_cols].values
    T2, Q = compute_t2_q(X_eval, scaler, pca)
    return T2 / t2_ucl + Q / q_ucl


def auc_report(df, score_col_values, label):
    df = df.copy()
    df["_score"] = score_col_values
    out = {}
    for group in ["MCI", "AD"]:
        mask = (df["DIAGNOSIS_LABEL"] == "CN") | (df["DIAGNOSIS_LABEL"] == group)
        y = (df.loc[mask, "DIAGNOSIS_LABEL"] == group).astype(int)
        if y.nunique() < 2:
            out[f"CN_vs_{group}"] = None
            continue
        out[f"CN_vs_{group}"] = roc_auc_score(y, df.loc[mask, "_score"])
    mask_any = df["DIAGNOSIS_LABEL"].isin(["CN", "MCI", "AD"])
    y_any = (df.loc[mask_any, "DIAGNOSIS_LABEL"] != "CN").astype(int)
    out["CN_vs_MCI+AD"] = roc_auc_score(y_any, df.loc[mask_any, "_score"])
    print(f"\n[{label}] n={len(df)} "
          f"(CN={ (df.DIAGNOSIS_LABEL=='CN').sum() }, "
          f"MCI={(df.DIAGNOSIS_LABEL=='MCI').sum()}, "
          f"AD={(df.DIAGNOSIS_LABEL=='AD').sum()})")
    for k, v in out.items():
        print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: n/a")
    return out


def main():
    # full table at the wide window, with per-patient MRI_EXAMDATE_GAP_DAYS
    df, feature_cols = build_feature_table(feature_tier="core", window_days=WIDE_WINDOW)
    tight_mask = df["MRI_EXAMDATE_GAP_DAYS"] <= TIGHT_WINDOW

    print("=" * 70)
    print("PART 1 -- naive comparison (different eval cohorts each time)")
    print("=" * 70)
    df_tight = df[tight_mask].reset_index(drop=True)
    cn_tight = (df_tight["DIAGNOSIS_LABEL"] == "CN").values
    score_tight_naive = score_with_model(df_tight, feature_cols, cn_tight, df_tight)
    auc_report(df_tight, score_tight_naive, f"window<={TIGHT_WINDOW}d, fit+eval on same tight cohort")

    cn_wide = (df["DIAGNOSIS_LABEL"] == "CN").values
    score_wide_naive = score_with_model(df, feature_cols, cn_wide, df)
    auc_report(df, score_wide_naive, f"window<={WIDE_WINDOW}d, fit+eval on same wide cohort")

    print("\n" + "=" * 70)
    print("PART 2 -- controlled comparison (SAME fixed eval set = tight cohort,")
    print("          only the CN training population for the baseline differs)")
    print("=" * 70)

    # Model_90: baseline fit on CN within the tight cohort
    score_eval_tight_fit90 = score_with_model(
        df_tight, feature_cols, cn_tight, df_tight
    )
    r90 = auc_report(df_tight, score_eval_tight_fit90,
                      "eval=tight cohort, baseline fit on tight CN (Model_90)")

    # Model_180: baseline fit on CN within the full (wide) cohort, but
    # evaluated on the SAME tight cohort as above
    score_eval_tight_fit180 = score_with_model(
        df, feature_cols, cn_wide, df_tight
    )
    r180 = auc_report(df_tight, score_eval_tight_fit180,
                       "eval=tight cohort, baseline fit on wide CN (Model_180)")

    print("\n" + "=" * 70)
    print("Interpretation")
    print("=" * 70)
    delta_ad = None
    if r90.get("CN_vs_AD") is not None and r180.get("CN_vs_AD") is not None:
        delta_ad = r90["CN_vs_AD"] - r180["CN_vs_AD"]
    delta_any = r90["CN_vs_MCI+AD"] - r180["CN_vs_MCI+AD"]
    print(f"On the SAME evaluation patients, "
          f"AUC(CN vs AD): Model_90 - Model_180 = "
          f"{delta_ad:+.3f}" if delta_ad is not None else "AUC(CN vs AD): n/a")
    print(f"On the SAME evaluation patients, "
          f"AUC(CN vs MCI+AD): Model_90 - Model_180 = {delta_any:+.3f}")
    print("A near-zero delta means the wider match window does not "
          "meaningfully pollute the healthy baseline -- the earlier "
          "full-cohort AUC differences were mostly a cohort-composition "
          "effect, not a baseline-quality effect. A clearly positive delta "
          "means tighter matching genuinely improves anomaly detection, "
          "independent of who gets evaluated.")


if __name__ == "__main__":
    main()
