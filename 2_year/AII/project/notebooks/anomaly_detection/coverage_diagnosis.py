"""
notebooks/anomaly_detection/coverage_diagnosis.py

Where does the 65.5% HI coverage on final.csv actually get lost? Breaks it
down step by step through Method B's own loader logic (load_longitudinal.py)
so it's clear which filter is responsible for how much of the gap, before
touching anything.

Usage:
    python3 coverage_diagnosis.py
"""

import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "method_a_pca_density"))
sys.path.append(os.path.join(HERE, "method_b_autoencoder_hi"))
from load_features import DATA_DIR, MRI_FEATURES, _to_num  # noqa: E402

DATASETS_DIR = os.path.join(HERE, "..", "..", "datasets")


def main():
    final = pd.read_csv(f"{DATASETS_DIR}/final.csv", low_memory=False)
    final["EXAMDATE_DX"] = pd.to_datetime(final["EXAMDATE_DX"], errors="coerce")
    final_dated = final.dropna(subset=["EXAMDATE_DX"])
    n_final = len(final_dated)
    n_final_patients = final_dated["RID"].nunique()
    print(f"final.csv rows with a diagnosis date: {n_final} ({n_final_patients} patients)")

    # Step 1: raw UCSFFSX7 rows at all, no QC, per RID -- any MRI ever
    mri_raw = pd.read_csv(f"{DATA_DIR}/UCSFFSX7_12Dec2025.csv", low_memory=False)
    rids_with_any_mri = set(mri_raw["RID"].unique())
    n_step1 = final_dated["RID"].isin(rids_with_any_mri).sum()
    print(f"\nStep 1: final.csv rows whose RID has ANY UCSFFSX7 record at all: "
          f"{n_step1} ({n_step1/n_final:.1%})")

    # Step 2: full MRI_FEATURES present (Method B's QC proxy) at any visit for that RID
    mri_raw = _to_num(mri_raw, MRI_FEATURES)
    mri_qc = mri_raw.dropna(subset=MRI_FEATURES)
    rids_with_qc_mri = set(mri_qc["RID"].unique())
    n_step2 = final_dated["RID"].isin(rids_with_qc_mri).sum()
    print(f"Step 2: ...and RID has >=1 visit with ALL {len(MRI_FEATURES)} MRI features present: "
          f"{n_step2} ({n_step2/n_final:.1%})  [loss from step 1: {n_step1-n_step2} rows]")

    # Step 3: >= 2 such qualifying visits (Method B's MIN_VISITS_PER_PATIENT=2 requirement)
    visits_per_rid = mri_qc.drop_duplicates(["RID", "VISCODE2"]).groupby("RID").size()
    rids_with_2plus = set(visits_per_rid[visits_per_rid >= 2].index)
    n_step3 = final_dated["RID"].isin(rids_with_2plus).sum()
    print(f"Step 3: ...and RID has >=2 such qualifying visits (current MIN_VISITS_PER_PATIENT): "
          f"{n_step3} ({n_step3/n_final:.1%})  [loss from step 2: {n_step2-n_step3} rows]")

    print(f"\nActual measured coverage after the full pipeline join (diagnosis-date match "
          f"included): 65.5% -- so steps 1-3 above (patient/visit eligibility) account for "
          f"most of the gap; the rest is the specific-visit date match within the eligible "
          f"patients.")

    print("\n--- What relaxing MIN_VISITS_PER_PATIENT to 1 would buy ---")
    rids_with_1plus = set(visits_per_rid[visits_per_rid >= 1].index)
    n_relaxed = final_dated["RID"].isin(rids_with_1plus).sum()
    print(f"RID has >=1 qualifying visit (no trajectory requirement): "
          f"{n_relaxed} ({n_relaxed/n_final:.1%})  "
          f"[gain over current >=2 requirement: {n_relaxed-n_step3} rows, "
          f"{(n_relaxed-n_step3)/n_final:.1%} of final.csv]")
    print(f"Patients gained: {len(rids_with_1plus - rids_with_2plus)}")


if __name__ == "__main__":
    main()
