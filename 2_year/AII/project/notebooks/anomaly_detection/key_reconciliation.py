"""
notebooks/anomaly_detection/key_reconciliation.py

The actual blocker NOTES.md names for wiring point 1 into the shared
pipeline isn't the feature set -- it's that raw ADNI VISCODE2 codes (e.g.
"sc" at baseline) don't string-match final.csv's own VISCODE2_norm codes
(e.g. "bl" at baseline): "a plain string match on VISCODE2 currently misses
every patient's baseline visit outright."

Rebuilding Method A/B's feature loaders against final.csv directly would
also throw away most of the verified MRI feature set: final.csv only
carries 3 regions (hippocampus, entorhinal, amygdala + ICV), against the
~16 cortical thickness/volume regions load_features.py's MRI_FEATURES uses,
each individually checked against DATADIC. That's a real regression, not a
neutral rewrite, so it's not the right fix.

Instead: build an explicit, DATA-DRIVEN (RID, VISCODE2) -> VISCODE2_norm
lookup table, by matching each final.csv row to the raw DXSUM row for the
same RID whose EXAMDATE is closest to final.csv's own EXAMDATE_DX (the
diagnosis date each final.csv row is keyed to). This is the same
nearest-date-match discipline already used throughout Method A/B, just
applied to solve the KEY problem instead of the FEATURE problem -- it lets
any of Gio's existing outputs (keyed by RID+raw VISCODE2) attach the
correct VISCODE2_norm without touching any feature engineering.

Usage:
    python3 key_reconciliation.py
Outputs (written next to this script):
    - rid_viscode_key_map.csv       (RID, VISCODE2, VISCODE2_norm, MATCH_GAP_DAYS)
    - key_reconciliation_report.txt
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets")
MATCH_WINDOW_DAYS = 30  # this is matching the SAME diagnosis event across two
                         # files, not two different modalities -- should be
                         # near-exact (0-day) for the vast majority of rows


def main():
    final = pd.read_csv(f"{DATA_DIR}/final.csv", low_memory=False)
    final["EXAMDATE_DX"] = pd.to_datetime(final["EXAMDATE_DX"], errors="coerce")
    final_small = final[["RID", "VISCODE2_norm", "EXAMDATE_DX"]].dropna(subset=["EXAMDATE_DX"])

    dxsum = pd.read_csv(f"{DATA_DIR}/DXSUM_12Dec2025.csv")
    dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"], errors="coerce")
    dxsum_small = dxsum[["RID", "VISCODE2", "EXAMDATE"]].dropna(subset=["EXAMDATE"])

    cand = final_small.merge(dxsum_small, on="RID", how="inner")
    cand["_gap"] = (cand["EXAMDATE_DX"] - cand["EXAMDATE"]).dt.days.abs()
    cand = cand[cand["_gap"] <= MATCH_WINDOW_DAYS]
    cand = cand.sort_values("_gap").drop_duplicates(["RID", "VISCODE2_norm"], keep="first")

    key_map = cand[["RID", "VISCODE2", "VISCODE2_norm", "_gap"]].rename(
        columns={"_gap": "MATCH_GAP_DAYS"}
    )
    # if the same (RID, VISCODE2) maps to more than one VISCODE2_norm (shouldn't
    # happen given the window, but check), keep the closest
    key_map = key_map.sort_values("MATCH_GAP_DAYS").drop_duplicates(
        ["RID", "VISCODE2"], keep="first"
    )
    key_map.to_csv("rid_viscode_key_map.csv", index=False)

    lines = []
    lines.append(f"final.csv rows with a diagnosis date: {len(final_small)}")
    lines.append(f"Matched to a raw DXSUM row within {MATCH_WINDOW_DAYS} days: {len(key_map)} "
                 f"({len(key_map) / len(final_small):.1%})")
    lines.append(f"Exact same-day matches: {(key_map['MATCH_GAP_DAYS'] == 0).sum()} "
                 f"({(key_map['MATCH_GAP_DAYS'] == 0).mean():.1%})")
    lines.append(f"Match gap distribution (days): "
                 f"median={key_map['MATCH_GAP_DAYS'].median():.0f}, "
                 f"max={key_map['MATCH_GAP_DAYS'].max():.0f}")
    lines.append("")
    lines.append("Sample of the resulting VISCODE2 -> VISCODE2_norm correspondence "
                 "(most common raw code per normalized code):")
    mapping_mode = (key_map.groupby("VISCODE2_norm")["VISCODE2"]
                     .agg(lambda s: s.value_counts().idxmax()))
    for norm, raw in mapping_mode.sort_index().items():
        n = (key_map["VISCODE2_norm"] == norm).sum()
        lines.append(f"  {norm:10s} <- {raw:10s}  (n={n})")

    unmatched = final_small.merge(
        key_map[["RID", "VISCODE2_norm"]], on=["RID", "VISCODE2_norm"], how="left", indicator=True
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"]
    lines.append("")
    lines.append(f"Unmatched final.csv rows: {len(unmatched)} ({len(unmatched)/len(final_small):.1%}) "
                 f"-- these RIDs/visits have no raw DXSUM row within {MATCH_WINDOW_DAYS} days "
                 f"of final.csv's own diagnosis date (likely dropped/renumbered visits).")

    report = "\n".join(lines)
    print(report)
    with open("key_reconciliation_report.txt", "w") as f:
        f.write(report + "\n")

    # -- now apply the map to the PCA-based HI trajectory output, if present --
    pca_path = "method_b_autoencoder_hi/pca_hi_trajectories.csv"
    if os.path.exists(pca_path):
        pca_hi = pd.read_csv(pca_path)
        keyed = pca_hi.merge(key_map[["RID", "VISCODE2", "VISCODE2_norm"]],
                              on=["RID", "VISCODE2"], how="left")
        coverage = keyed["VISCODE2_norm"].notna().mean()
        keyed.to_csv("method_b_autoencoder_hi/pca_hi_trajectories_keyed.csv", index=False)
        print(f"\nApplied key map to {pca_path}: "
              f"{coverage:.1%} of visits now carry a VISCODE2_norm "
              f"(vs. the ~67% the 180-day nearest-date workaround in "
              f"approach_pipeline_anomaly.ipynb currently reaches).")
        with open("key_reconciliation_report.txt", "a") as f:
            f.write(f"\npca_hi_trajectories_keyed.csv: {coverage:.1%} VISCODE2_norm coverage "
                    f"(exact key match, vs ~67% for the nearest-date workaround)\n")


if __name__ == "__main__":
    main()
