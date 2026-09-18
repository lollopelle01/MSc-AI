"""
notebooks/anomaly_detection/pipeline_1_2_3/oracle_regret_analysis.py

Answers "how do we measure whether the interval DECISIONS are actually
relevant" directly, rather than relying on cost/DIDI/catch numbers that
can look fine even when a policy has degenerated toward giving everyone
the same interval (the ratio=160 near-uniform case from section 5d/8, and
the fixed policy's trivial DIDI=0 from section 9, both showed this).

Three checks, run for every policy variant already built this session:

1. INTERVAL DISTRIBUTION -- what share of patients get 3 / 6 / 12 months.
   A policy piled up on one value, regardless of its cost/DIDI numbers, is
   not using patient information.

2. SPEARMAN CORRELATION between RISK_SCORE and RECOMMENDED_INTERVAL --
   does the decision actually track the score's own ranking? Near zero
   means the score's discriminative power (its AUC) isn't reaching the
   decisions at all.

3. REGRET against an ORACLE -- the standard Predict-then-Optimize /
   Decision-Focused-Learning evaluation (08-dfl-tf, already cited by this
   project's own recommend_interval docstring). The oracle feeds
   EVENT_AT_VISIT itself (the real, realized 0/1 outcome, i.e. a
   hypothetical perfect predictor) through the EXACT SAME
   recommend_interval / ConversionCostModel formula every other policy
   uses -- not a different method, the same one, with perfect information
   instead of a prediction. regret_pct = (policy_cost - oracle_cost) /
   (naive_cost - oracle_cost) * 100, where naive is the fixed,
   zero-information policy (section 9) -- 0% is oracle-optimal, 100% is no
   better than not personalizing at all.

Usage:
    python3 oracle_regret_analysis.py
Outputs (written next to this script):
    - oracle_regret_report.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fairness_expgrad_pipeline as fep  # noqa: E402

REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402

RATIO = 20.0
MENU = (3, 6, 12)
SAFE_INTERVAL_MONTHS = 6
PROTECTED_DOMAIN = lambda df: {"PTGENDER": (1, 2), "PTEDUCAT_BUCKET": (0, 1),
                                "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique()))}


def evaluate(df, risk_col, recommended, label, lines, results):
    cmodel = du.ConversionCostModel(check_cost=1.0, missed_conversion_cost=RATIO,
                                     safe_interval_months=SAFE_INTERVAL_MONTHS,
                                     reference_interval_months=12.0)
    total_cost, _, _ = cmodel.cost(rid_ids=df["RID"].values, risk_scores=df[risk_col].values,
                                    threshold=0.5, interval_months=recommended, return_margin=False)
    didi = du.compute_didi(df, recommended, PROTECTED_DOMAIN(df))
    outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
    outcomes["RECOMMENDED_INTERVAL"] = recommended
    worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
    margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
    catch_rate = (margin >= 0).mean() * 100

    dist = pd.Series(recommended).value_counts(normalize=True).reindex(MENU, fill_value=0) * 100
    corr, pval = spearmanr(df[risk_col].values, recommended)

    lines.append(f"  {label:38s} cost={total_cost:9.1f}  DIDI={didi:.3f}  catch={catch_rate:5.1f}%  "
                 f"spearman(risk,interval)={corr:+.3f}  3mo={dist[3]:.0f}%/6mo={dist[6]:.0f}%/12mo={dist[12]:.0f}%")
    results[label] = {"cost": total_cost, "didi": didi, "catch": catch_rate, "corr": corr,
                       "dist": dist, "recommended": recommended}
    return total_cost


def main():
    df = pd.read_csv(os.path.join(HERE, "calibrated_risk_scores.csv"))
    df["EXAMDATE_DX"] = pd.to_datetime(df["EXAMDATE_DX"])

    lines = []
    lines.append("=== Oracle regret analysis: are the interval decisions actually relevant? ===")
    lines.append(f"n={len(df)} rows (test set, common to every policy compared here).")
    lines.append("")

    results = {}

    # --- Naive baseline: fixed policy, everyone gets the same interval ---
    fixed_rec = np.full(len(df), float(SAFE_INTERVAL_MONTHS))
    naive_cost = evaluate(df, "RISK_SCORE_plain_cal", fixed_rec, "Fixed (naive baseline)", lines, results)

    # --- Oracle: feed the REAL outcome through the same formula ---
    # EVENT_AT_VISIT is the wrong target here and gave a nonsensical result on
    # the first run (oracle catch rate LOWER than the real models') -- it's
    # whether THIS visit already reflects a worsening from the prior one (the
    # label stage 2 trains on), not whether the patient's NEXT real visit
    # will show worsening, which is what catch_rate actually checks
    # (compute_diagnosis_worsening's DIAGNOSIS_WORSENED_NEXT). A genuine
    # oracle needs to be perfect information about the SAME thing catch_rate
    # measures, not about a different, related label. Rows with no recorded
    # next visit (HAS_NEXT_VISIT=False, right-censored) have no known future
    # outcome -- oracle risk defaults to 0 for those (no evidence of
    # worsening), same as any real model would have to assume.
    outcomes_for_oracle = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX",
                                                           diagnosis_col="DIAGNOSIS")
    df["ORACLE_RISK"] = (outcomes_for_oracle["HAS_NEXT_VISIT"] &
                          outcomes_for_oracle["DIAGNOSIS_WORSENED_NEXT"]).astype(float).values
    lines.append(f"Oracle target: {df['ORACLE_RISK'].mean()*100:.1f}% of rows have a real forward "
                 f"worsening event to catch (vs. EVENT_AT_VISIT's {df['EVENT_AT_VISIT'].mean()*100:.1f}%, "
                 f"the wrong, backward-looking label a first attempt at this used -- caught and fixed since "
                 f"it gave an oracle with a LOWER catch rate than the real models, which is not possible for "
                 f"genuine perfect information on the metric actually being optimized).")
    lines.append("")
    oracle_rec = du.recommend_interval(df["ORACLE_RISK"].values, interval_menu_months=MENU,
                                        check_cost=1.0, missed_conversion_cost=RATIO,
                                        reference_interval_months=12.0)
    oracle_cost = evaluate(df, "ORACLE_RISK", oracle_rec, "ORACLE (perfect information)", lines, results)
    lines.append("")

    def regret(cost):
        return (cost - oracle_cost) / (naive_cost - oracle_cost) * 100

    # --- Real policies ---
    lines.append("-- Snapshot-adaptive policies --")
    policy_costs = {}
    for score_col, label in [("RISK_SCORE_plain_cal", "Plain, calibrated"),
                              ("RISK_SCORE_fair_cal", "Fair (DemographicParity), calibrated")]:
        rec = du.recommend_interval(df[score_col].values, interval_menu_months=MENU,
                                     check_cost=1.0, missed_conversion_cost=RATIO,
                                     reference_interval_months=12.0)
        cost = evaluate(df, score_col, rec, label, lines, results)
        policy_costs[label] = cost
    lines.append("")

    lines.append("-- Trajectory-adaptive policies (section 8) --")
    for score_col, label in [("RISK_SCORE_plain_cal", "Plain, trajectory-aware"),
                              ("RISK_SCORE_fair_cal", "Fair (DP), trajectory-aware")]:
        traj = du.compute_risk_trajectory(df, id_col="RID", date_col="EXAMDATE_DX", risk_col=score_col)
        rec = du.recommend_interval_trajectory(traj[score_col].values, traj["RISK_SLOPE"].values,
                                                interval_menu_months=MENU, check_cost=1.0,
                                                missed_conversion_cost=RATIO, reference_interval_months=12.0)
        cost = evaluate(traj, score_col, rec, label, lines, results)
        policy_costs[label] = cost
    lines.append("")

    lines.append("=== Regret vs. the oracle (0% = oracle-optimal, 100% = no better than the naive fixed policy) ===")
    lines.append(f"Oracle cost:  {oracle_cost:9.1f}  (perfect information, catch={results['ORACLE (perfect information)']['catch']:.1f}%)")
    lines.append(f"Naive cost:   {naive_cost:9.1f}  (fixed 6mo for everyone, DIDI=0 by construction)")
    lines.append("")
    for label, cost in policy_costs.items():
        r = regret(cost)
        lines.append(f"  {label:38s} cost={cost:9.1f}  regret={r:6.1f}%")
    lines.append("")

    lines.append("Read spearman(risk,interval) as the decision-relevance check: a real")
    lines.append("negative correlation (higher risk -> shorter interval) means the decision")
    lines.append("layer is actually using the score's own discriminative power, not just")
    lines.append("producing cost/DIDI/catch numbers that happen to look reasonable. Compare")
    lines.append("this against the fixed policy's undefined/zero correlation (no risk used at")
    lines.append("all) and, if re-run at ratio=160 (section 5d), against a correlation that")
    lines.append("would collapse toward the fixed policy's as decisions saturate to 3 months.")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(HERE, "oracle_regret_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
