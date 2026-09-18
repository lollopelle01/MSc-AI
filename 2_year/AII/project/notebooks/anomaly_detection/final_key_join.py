"""
notebooks/anomaly_detection/final_key_join.py

Closes the remaining gap from key_reconciliation.py's 76.2% coverage.

That script matched (RID, VISCODE2) -> VISCODE2_norm by first building a
DXSUM-based lookup, then joining pca_hi_trajectories.csv to it by an EXACT
string match on the raw VISCODE2 code. The 24% miss rate came from that
exact-string-match step: pca_hi_trajectories.csv's own VISCODE2 comes from
the MRI table (UCSFFSX7), and UCSFFSX7 doesn't always use the same raw code
DXSUM uses for what is clinically the same visit.

The fix: skip the intermediate string-matching step entirely and go
straight to what actually identifies "the same visit" -- the exam date.
Match each HI row directly to final.csv by RID + nearest EXAMDATE_DX,
exactly the technique already used everywhere else in both Method A/B and
the team's own approach_pipeline_anomaly.ipynb, just applied here to solve
the KEY problem instead of a feature-matching problem.

Usage:
    python3 final_key_join.py
Outputs (written next to this script):
    - method_b_autoencoder_hi/pca_hi_trajectories_keyed.csv   (overwritten,
      now via direct date match instead of the DXSUM-code intermediate)
    - final_key_join_report.txt
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets")
PCA_HI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "method_b_autoencoder_hi", "pca_hi_trajectories.csv")
WINDOW_DAYS = 180  # same tolerance the team's own pipeline notebook uses


def main():
    final = pd.read_csv(f"{DATA_DIR}/final.csv", low_memory=False)
    final["EXAMDATE_DX"] = pd.to_datetime(final["EXAMDATE_DX"], errors="coerce")
    final_small = final[["RID", "VISCODE2_norm", "EXAMDATE_DX"]].dropna(subset=["EXAMDATE_DX"])

    hi = pd.read_csv(PCA_HI_PATH)
    hi["EXAMDATE"] = pd.to_datetime(hi["EXAMDATE"], errors="coerce")

    lines = []
    lines.append(f"pca_hi_trajectories.csv rows: {len(hi)}  ({hi['RID'].nunique()} patients)")
    lines.append(f"final.csv rows with a diagnosis date: {len(final_small)}  "
                 f"({final_small['RID'].nunique()} patients)")

    rid_overlap = set(hi["RID"]) & set(final_small["RID"])
    rid_hi_only = set(hi["RID"]) - set(final_small["RID"])
    lines.append(f"RIDs in both files: {len(rid_overlap)}")
    lines.append(f"RIDs in pca_hi_trajectories.csv but NOT in final.csv at all: {len(rid_hi_only)} "
                 f"-- these cannot reach 100% no matter how the join is done.")

    # direct nearest-date match, RID + EXAMDATE vs RID + EXAMDATE_DX
    cand = hi.merge(final_small, on="RID", how="inner")
    cand["_gap"] = (cand["EXAMDATE"] - cand["EXAMDATE_DX"]).dt.days.abs()
    cand = cand[cand["_gap"] <= WINDOW_DAYS]
    # for each HI row (RID + its own EXAMDATE), keep the closest final.csv row
    cand = cand.sort_values("_gap").drop_duplicates(["RID", "EXAMDATE"], keep="first")

    keyed = hi.merge(
        cand[["RID", "EXAMDATE", "VISCODE2_norm", "_gap"]].rename(columns={"_gap": "MATCH_GAP_DAYS"}),
        on=["RID", "EXAMDATE"], how="left",
    )
    coverage = keyed["VISCODE2_norm"].notna().mean()
    lines.append("")
    lines.append(f"Direct RID + nearest-EXAMDATE_DX match (tolerance {WINDOW_DAYS} days): "
                 f"{keyed['VISCODE2_norm'].notna().sum()} / {len(keyed)} rows keyed "
                 f"({coverage:.1%})")
    matched = keyed.dropna(subset=["MATCH_GAP_DAYS"])
    lines.append(f"  Exact same-day matches: {(matched['MATCH_GAP_DAYS'] == 0).mean():.1%}")
    lines.append(f"  Match gap distribution (days): median={matched['MATCH_GAP_DAYS'].median():.0f}, "
                 f"95th pct={matched['MATCH_GAP_DAYS'].quantile(0.95):.0f}, "
                 f"max={matched['MATCH_GAP_DAYS'].max():.0f}")

    still_missing = keyed[keyed["VISCODE2_norm"].isna()]
    missing_rid_cause = still_missing["RID"].isin(rid_hi_only).sum()
    missing_other_cause = len(still_missing) - missing_rid_cause
    lines.append("")
    lines.append(f"Still unmatched: {len(still_missing)} rows")
    lines.append(f"  ...because the RID isn't in final.csv at all: {missing_rid_cause}")
    lines.append(f"  ...because RID is in final.csv but no visit within {WINDOW_DAYS} days: {missing_other_cause}")

    max_possible = keyed["RID"].isin(rid_overlap).mean()
    lines.append(f"\nCeiling given final.csv's own cohort: {max_possible:.1%} of HI rows belong to a "
                 f"patient final.csv has at all -- {coverage/max_possible:.1%} of THAT reachable subset "
                 f"is actually keyed within {WINDOW_DAYS} days.")

    keyed.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "method_b_autoencoder_hi", "pca_hi_trajectories_keyed.csv"), index=False)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "final_key_join_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
