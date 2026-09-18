"""
notebooks/anomaly_detection/pipeline_1_2_3/retune_cost_ratio_after_calibration.py

Follow-up to section 5c: calibrating RISK_SCORE dropped catch rate for
both models (fair: 89.7% -> 65.8%; plain: 89.7% -> 69.2%) because
recommend_interval's expected-cost formula uses the score's raw
magnitude, not just its rank, and MISSED_CONVERSION_COST/CHECK_COST was
never re-tuned once the score it multiplies became accurate instead of
inflated. This is a parameter question, not a data question: it sweeps
MISSED_CONVERSION_COST (CHECK_COST fixed at 1.0, so the value IS the
ratio) against the already-calibrated scores in calibrated_risk_scores.csv
-- no refitting needed -- and reports cost/DIDI/catch-rate at each ratio,
for both the calibrated fair and calibrated plain scores, to find the
ratio that recovers a catch rate close to the original 89.7% and see what
it costs to get there.

Usage:
    python3 retune_cost_ratio_after_calibration.py
Outputs (written next to this script):
    - cost_ratio_retune_report.txt, cost_ratio_retune.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fairness_expgrad_pipeline as fep  # noqa: E402

REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402

CHECK_COST = 1.0
REFERENCE_INTERVAL_MONTHS = 12.0
SAFE_INTERVAL_MONTHS = 6
# 20 is the original, pre-calibration ratio. Swept well past it since a
# lower (correctly-scaled) risk needs a HIGHER ratio to recommend the same
# interval it used to under an inflated score.
RATIOS = [20, 30, 40, 60, 80, 100, 130, 160, 200]
ORIGINAL_CATCH_RATE_TARGET = 89.7  # from the uncalibrated policy, section 5c


def evaluate_at_ratio(df, risk_col, ratio, protected):
    cmodel = du.ConversionCostModel(
        check_cost=CHECK_COST, missed_conversion_cost=ratio,
        safe_interval_months=SAFE_INTERVAL_MONTHS,
        reference_interval_months=REFERENCE_INTERVAL_MONTHS,
    )
    recommended = du.recommend_interval(
        df[risk_col].values, interval_menu_months=(3, 6, 12),
        check_cost=CHECK_COST, missed_conversion_cost=ratio,
        reference_interval_months=REFERENCE_INTERVAL_MONTHS,
    )
    total_cost, _, _ = cmodel.cost(
        rid_ids=df["RID"].values, risk_scores=df[risk_col].values,
        threshold=0.5, interval_months=recommended, return_margin=False,
    )
    didi = du.compute_didi(df, recommended, protected)
    outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
    outcomes["RECOMMENDED_INTERVAL"] = recommended
    worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
    margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
    catch_rate = (margin >= 0).mean() * 100
    return {"ratio": ratio, "cost": total_cost, "DIDI": didi, "catch_rate": catch_rate,
            "pct_3mo": float((recommended == 3).mean() * 100)}


def main():
    df = pd.read_csv(os.path.join(HERE, "calibrated_risk_scores.csv"))
    protected = {
        "PTGENDER": (1, 2),
        "PTEDUCAT_BUCKET": (0, 1),
        "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique())),
    }

    lines = []
    lines.append("=== Re-tuning MISSED_CONVERSION_COST after calibration ===")
    lines.append(f"n={len(df)} rows, loaded from calibrated_risk_scores.csv (no refitting).")
    lines.append(f"CHECK_COST fixed at {CHECK_COST}, so MISSED_CONVERSION_COST IS the ratio.")
    lines.append(f"Original policy (uncalibrated scores, ratio=20) caught {ORIGINAL_CATCH_RATE_TARGET}%.")
    lines.append("")

    results = {}
    for label, col in [("Fair (EG), calibrated", "RISK_SCORE_fair_cal"),
                        ("Plain (no fairness), calibrated", "RISK_SCORE_plain_cal")]:
        rows = [evaluate_at_ratio(df, col, r, protected) for r in RATIOS]
        rtab = pd.DataFrame(rows)
        results[label] = rtab
        lines.append(f"-- {label} --")
        lines.append(rtab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        # first ratio that reaches or exceeds the original catch rate
        hit = rtab[rtab["catch_rate"] >= ORIGINAL_CATCH_RATE_TARGET]
        if len(hit):
            row = hit.iloc[0]
            lines.append(f"  -> ratio={row['ratio']:.0f} recovers catch_rate={row['catch_rate']:.1f}% "
                         f"(target {ORIGINAL_CATCH_RATE_TARGET}%) at cost={row['cost']:.1f}, "
                         f"DIDI={row['DIDI']:.3f}")
        else:
            lines.append(f"  -> none of the ratios tried reached {ORIGINAL_CATCH_RATE_TARGET}% "
                         f"(max {rtab['catch_rate'].max():.1f}% at ratio={rtab.loc[rtab['catch_rate'].idxmax(), 'ratio']:.0f})")
        lines.append("")

    lines.append("Read this alongside section 5c: at ratio=20 (unchanged), calibration cut cost")
    lines.append("roughly 3x (16630->5123 fair, 11317->4926 plain) by removing the inflation an")
    lines.append("uncalibrated score was adding to the missed-conversion term. Raising the ratio")
    lines.append("recovers catch rate by spending some of that savings back -- deliberately, on")
    lines.append("a score that is now actually trustworthy, rather than by accident on one that")
    lines.append("wasn't. The DIDI column shows whether the fairness benefit survives re-tuning.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"Fair (EG), calibrated": "#8172B2", "Plain (no fairness), calibrated": "#55A868"}
    for label, rtab in results.items():
        axes[0].plot(rtab["ratio"], rtab["catch_rate"], marker="o", color=colors[label], label=label)
        axes[1].plot(rtab["ratio"], rtab["cost"], marker="o", color=colors[label], label=label)
    axes[0].axhline(ORIGINAL_CATCH_RATE_TARGET, linestyle="--", color="gray",
                     label=f"Original (uncalibrated, ratio=20): {ORIGINAL_CATCH_RATE_TARGET}%")
    axes[0].set_xlabel("MISSED_CONVERSION_COST / CHECK_COST ratio")
    axes[0].set_ylabel("Catch rate (%)")
    axes[0].set_title("Catch rate recovers as the ratio rises")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("MISSED_CONVERSION_COST / CHECK_COST ratio")
    axes[1].set_ylabel("Total cost (units)")
    axes[1].set_title("...at a real, rising cost")
    axes[1].legend(fontsize=8)
    plt.suptitle("Re-tuning the cost ratio after calibration: catch rate is recoverable,\n"
                 "it just needs its own deliberate setting instead of an accidental one")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(HERE, "cost_ratio_retune.png"), dpi=150)
    plt.close()

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "cost_ratio_retune_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
