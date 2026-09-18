"""
notebooks/anomaly_detection/method_a_pca_density/threshold_cost_optimization_literature.py

threshold_cost_optimization.py sweeps c_missed/c_alarm over 5, 10, and 20 --
a documented but ARBITRARY sensitivity range, chosen only to satisfy the
lecture's own instruction not to commit to a single guess. This variant
replaces that arbitrary sweep with a literature-anchored one, reusing the
exact same real-world cost anchor Pelle's own decision-support notebooks
already use for stage 3 (util/decision_util.py's cost_model_for):

  - 2.8x -- point estimate from a 2026 medRxiv preprint on the societal
    cost of Alzheimer's disease by diagnosis timing (~62,000 EUR/year for
    an early MCI patient vs. ~175,000 EUR/year for a late, severe AD
    patient).
  - 1.5x and 4.0x -- the low and high end of the broader range reported in
    "Global Societal Burden of Alzheimer's Disease by Severity" (Neurology
    and Therapy), a targeted review of 81 studies (2013-2024), which finds
    the mild-to-severe cost ratio typically falls between about 1.4 and
    3.6, with at least one study reporting a fourfold difference.

Reframing note, stated honestly rather than glossed over: that literature
measures the cost ratio between an EARLY and a LATE diagnosis of the same
patient -- i.e. exactly what a MISSED anomaly-detection alarm causes, since
a patient missed at today's (earlier, cheaper) stage is later caught, if at
all, only once they progress to a more advanced (costlier) one. It is not a
literal "cost of one false alarm" figure -- no health-economics literature
prices "one unnecessary follow-up scan" the way this project's own
C_ALARM=1 unit does -- so c_alarm is kept at 1 (one routine workup unit,
the same convention as everywhere else in this project) and only the
missed/alarm RATIO is replaced with this real anchor. This is the same
honest-approximation the lecture itself models: use the closest available
real number, state exactly what it does and doesn't cover, and show the
range rather than resting on one point estimate.

Usage:
    python3 threshold_cost_optimization_literature.py
Outputs (written next to this script):
    - threshold_cost_optimization_literature.png
    - threshold_cost_optimization_literature_report.txt
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_features import build_feature_table
from method_a import fit_pca_baseline, compute_t2_q, control_limits
from compare_three_methods import (
    fit_gmm_baseline, gmm_anomaly_score, fit_autoencoder_baseline, ae_anomaly_score,
)
from threshold_cost_optimization import ADSimpleCostModel, opt_thr, evaluate_at_threshold

# Literature-anchored ratios, replacing the arbitrary [5, 10, 20] sweep.
COST_RATIOS = [1.5, 2.8, 4.0]
COST_RATIO_LABELS = {
    1.5: "1.5x (low end, Neurology & Therapy review)",
    2.8: "2.8x (medRxiv point estimate -- same anchor as stage 3's cost_model_for)",
    4.0: "4.0x (high end, Neurology & Therapy review)",
}
C_ALARM = 1.0


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
    lines.append("Cost-based threshold optimization, LITERATURE-ANCHORED cost ratios")
    lines.append("(replaces threshold_cost_optimization.py's arbitrary [5, 10, 20] sweep)")
    lines.append("")
    lines.append("Source: 2.8x is the point estimate from a 2026 medRxiv preprint on the")
    lines.append("societal cost of Alzheimer's disease by diagnosis timing (~62,000 EUR/yr")
    lines.append("early MCI vs. ~175,000 EUR/yr late severe AD). 1.5x/4.0x are the low/high")
    lines.append("end of 'Global Societal Burden of Alzheimer's Disease by Severity'")
    lines.append("(Neurology and Therapy, 81 studies, 2013-2024, typical ratio 1.4-3.6x).")
    lines.append("This is the SAME literature anchor stage 3's cost_model_for() already uses")
    lines.append("for MCI_OR_WORSE_COST_MULTIPLIER -- reused here for stage 1's own threshold")
    lines.append("instead of an unrelated, arbitrary sweep, so both stages rest on one")
    lines.append("consistent, citable real-world cost story instead of two disconnected ones.")
    lines.append("")
    lines.append("Reframing caveat, stated honestly: this literature prices EARLY vs LATE")
    lines.append("diagnosis of the same disease, which is exactly what a missed anomaly alarm")
    lines.append("causes (later, costlier detection) -- it is not a literal 'one false alarm'")
    lines.append("price, so c_alarm is kept at 1 unit (one routine workup) and only the")
    lines.append("missed/alarm RATIO is replaced with this real anchor.")
    lines.append("")

    fig, axes = plt.subplots(1, len(COST_RATIOS), figsize=(5 * len(COST_RATIOS), 4.5), sharey=False)

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
            lines.append(f"-- c_missed / c_alarm = {COST_RATIO_LABELS[ratio]} --")
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
        ax.set_title(f"CN vs AD, c_missed/c_alarm = {ratio}x (literature)")
        ax.legend(fontsize=8)
    plt.suptitle("Literature-anchored cost ratios (medRxiv + Neurology and Therapy),\n"
                 "replacing the arbitrary 5/10/20 sweep", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("threshold_cost_optimization_literature.png", dpi=150)
    plt.close()

    report = "\n".join(lines)
    print(report)
    with open("threshold_cost_optimization_literature_report.txt", "w") as f:
        f.write(report + "\n")
    print("\nWrote: threshold_cost_optimization_literature.png, "
          "threshold_cost_optimization_literature_report.txt")


if __name__ == "__main__":
    main()
