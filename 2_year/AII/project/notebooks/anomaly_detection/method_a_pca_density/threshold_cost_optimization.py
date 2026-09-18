"""
notebooks/anomaly_detection/method_a_pca_density/threshold_cost_optimization.py

Applies the course's own cost-based threshold selection methodology
(01-ad-de, lecture 4: define alarm/missed costs, encapsulate in a small
cost class, optimize the threshold via a line search) to the three-way
method comparison in compare_three_methods.py, instead of the flat 99th
percentile control limit used everywhere else in this project.

AUC is threshold-independent, so the earlier PCA/GMM/Autoencoder ranking
(0.890 / 0.881 / 0.814 CN-vs-AD) could in principle hide a different story
once each method's alarm signal is actually turned into a binary decision
at its own best operating point. This checks that directly: fit each
method's own optimal threshold (per the course's line-search recipe) and
compare the resulting cost, catch rate (sensitivity), and false-alarm rate
side by side, at more than one alarm/missed cost ratio (since the lecture
is explicit that these costs are assumptions that should be tested under
more than one setting, not committed to a single guess).

Following the lecture's own precedent exactly: it optimizes the threshold
on a "validation set" that is explicitly allowed to include the training
data, because -- as the lecture states -- this validation set isn't
warding off overfitting, it's tuning one extra scalar parameter. The same
logic applies here (threshold tuning, not model fitting), so the full
464-patient cohort is used for the line search, same as the lecture's own
simplest, explicitly-labeled-as-imperfect version ("yes, we are cheating a
bit... but it works").

Usage:
    python3 threshold_cost_optimization.py
Outputs (written next to this script):
    - threshold_cost_optimization.png
    - threshold_cost_optimization_report.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_features import build_feature_table
from method_a import fit_pca_baseline, compute_t2_q, control_limits
from compare_three_methods import (
    fit_gmm_baseline, gmm_anomaly_score, fit_autoencoder_baseline, ae_anomaly_score,
)

COST_RATIOS = [5, 10, 20]  # c_missed / c_alarm -- swept, not committed to one guess
C_ALARM = 1.0


class ADSimpleCostModel:
    """Direct adaptation of 01-ad-de's ADSimpleCostModel to a per-patient
    (non-time-series) binary detection setting: no 'late detection' cost,
    since there is no time-windowed sequence here, just alarm vs missed."""

    def __init__(self, c_alarm, c_missed):
        self.c_alarm = c_alarm
        self.c_missed = c_missed

    def cost(self, scores, y_true, thr):
        pred = (scores >= thr).astype(int)
        false_alarms = int(((pred == 1) & (y_true == 0)).sum())
        missed = int(((pred == 0) & (y_true == 1)).sum())
        total = self.c_alarm * false_alarms + self.c_missed * missed
        return total, false_alarms, missed


def opt_thr(scores, y_true, cmodel, thr_range):
    costs = [cmodel.cost(scores, y_true, t)[0] for t in thr_range]
    best_idx = int(np.argmin(costs))
    return thr_range[best_idx], costs[best_idx]


def evaluate_at_threshold(scores, y_true, thr):
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else np.nan
    return sensitivity, false_alarm_rate, tp, fn, fp, tn


def main():
    df, feature_cols = build_feature_table(feature_tier="core")
    X = df[feature_cols].values
    cn_mask = (df["DIAGNOSIS_LABEL"] == "CN").values
    print(f"Cohort: {len(df)} patients (CN={cn_mask.sum()}, "
          f"MCI={(df.DIAGNOSIS_LABEL=='MCI').sum()}, AD={(df.DIAGNOSIS_LABEL=='AD').sum()})")

    methods = {}

    scaler_p, pca, _, cum_var = fit_pca_baseline(X[cn_mask])
    T2, Q = compute_t2_q(X, scaler_p, pca)
    t2_ucl, q_ucl = control_limits(T2[cn_mask], Q[cn_mask], pca.n_components_, cn_mask.sum())
    methods["PCA"] = T2 / t2_ucl + Q / q_ucl

    scaler_g, gmm, gmm_cfg, gmm_bic = fit_gmm_baseline(X[cn_mask])
    methods["GMM"] = gmm_anomaly_score(gmm, scaler_g, X)

    scaler_a, ae = fit_autoencoder_baseline(X[cn_mask])
    methods["Autoencoder"] = ae_anomaly_score(ae, scaler_a, X)

    lines = []
    lines.append("Cost-based threshold optimization (01-ad-de line-search recipe)")
    lines.append(f"Cost model: c_alarm = {C_ALARM}, c_missed swept over {COST_RATIOS} x c_alarm")
    lines.append("(sensitivity analysis, per the lecture's own caution that costs are ")
    lines.append(" assumptions to be tested under more than one setting, not a single guess)")
    lines.append("")

    fig, axes = plt.subplots(1, len(COST_RATIOS), figsize=(5 * len(COST_RATIOS), 4.5), sharey=False)
    if len(COST_RATIOS) == 1:
        axes = [axes]

    for comparison, group in [("CN vs AD", "AD"), ("CN vs MCI+AD", None)]:
        lines.append(f"=== {comparison} ===")
        if group is not None:
            mask = cn_mask | (df["DIAGNOSIS_LABEL"] == group).values
            y_true_full = (df["DIAGNOSIS_LABEL"] == group).astype(int).values
        else:
            mask = cn_mask | df["DIAGNOSIS_LABEL"].isin(["MCI", "AD"]).values
            y_true_full = (df["DIAGNOSIS_LABEL"] != "CN").astype(int).values

        for ratio in COST_RATIOS:
            cmodel = ADSimpleCostModel(c_alarm=C_ALARM, c_missed=C_ALARM * ratio)
            lines.append(f"-- c_missed / c_alarm = {ratio} --")
            for name, score in methods.items():
                s = score[mask]
                y = y_true_full[mask]
                thr_range = np.quantile(score[cn_mask], np.linspace(0.5, 0.999, 500))
                best_thr, best_cost = opt_thr(s, y, cmodel, thr_range)
                sens, far, tp, fn, fp, tn = evaluate_at_threshold(s, y, best_thr)
                thr_pctile = (score[cn_mask] <= best_thr).mean() * 100
                lines.append(
                    f"    {name:12s} optimal_thr@{thr_pctile:5.1f}pctile(CN)  "
                    f"cost={best_cost:6.1f}  sensitivity={sens:.3f}  false_alarm_rate={far:.3f}  "
                    f"(TP={tp} FN={fn} FP={fp} TN={tn})"
                )
        lines.append("")

    # plot: cost vs threshold percentile, CN vs AD, for each cost ratio
    mask_ad = cn_mask | (df["DIAGNOSIS_LABEL"] == "AD").values
    y_ad = (df["DIAGNOSIS_LABEL"] == "AD").astype(int).values
    colors = {"PCA": "#4C72B0", "GMM": "#55A868", "Autoencoder": "#C44E52"}
    for ax, ratio in zip(axes, COST_RATIOS):
        cmodel = ADSimpleCostModel(c_alarm=C_ALARM, c_missed=C_ALARM * ratio)
        for name, score in methods.items():
            thr_range = np.quantile(score[cn_mask], np.linspace(0.5, 0.999, 200))
            pctiles = np.linspace(50, 99.9, 200)
            costs = [cmodel.cost(score[mask_ad], y_ad[mask_ad], t)[0] for t in thr_range]
            ax.plot(pctiles, costs, label=name, color=colors[name])
            best_i = int(np.argmin(costs))
            ax.scatter([pctiles[best_i]], [costs[best_i]], color=colors[name], zorder=5, s=40)
        ax.set_xlabel("Threshold (percentile of CN training distribution)")
        ax.set_ylabel("Total cost")
        ax.set_title(f"CN vs AD, c_missed/c_alarm = {ratio}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("threshold_cost_optimization.png", dpi=150)
    plt.close()

    report = "\n".join(lines)
    print(report)
    with open("threshold_cost_optimization_report.txt", "w") as f:
        f.write(report + "\n")
    print("\nWrote: threshold_cost_optimization.png, threshold_cost_optimization_report.txt")


if __name__ == "__main__":
    main()
