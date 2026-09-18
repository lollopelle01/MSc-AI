"""
notebooks/anomaly_detection/method_b_autoencoder_hi/load_longitudinal.py

Builds a LONGITUDINAL table: every qualifying structural-MRI visit for every
ADNI participant (not just one baseline snapshot like Method A), each tagged
with the diagnosis nearest to that visit. This is what a health-index (HI)
trajectory needs -- a per-visit anomaly score over time, not a single
per-patient number.

Feature set is deliberately the same MRI + demographics "core" set Method A
uses (see method_a_pca_density/load_features.py), so the two methods are
comparable and share the same verified ADNI field codes. Demographics are
still treated as static (one row per RID, attached to every visit).

Cleaning rules mostly follow Method A, with one deliberate difference:
  - UCSFFSX7 is NOT filtered to STATUS == 'complete' here (see
    load_all_mri_visits docstring for why -- that filter is too strict for
    longitudinal data). 3T is still preferred over 1.5T when both exist for
    a visit, and every MRI_FEATURES column must be present.
  - ADNI sentinel codes -1/-4 -> NaN.
  - Each visit's diagnosis is the DXSUM record nearest in time within
    DIAG_MATCH_WINDOW_DAYS (see load_features.MATCH_WINDOW_DAYS for the
    Method A equivalent / rationale).
"""

import os as _os
import sys as _sys

import numpy as np
import pandas as pd

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                    "..", "method_a_pca_density"))
from load_features import (  # noqa: E402
    DATA_DIR, SENTINELS, MRI_FEATURES, DEMOG_FEATURES, _to_num,
)

DIAG_MATCH_WINDOW_DAYS = 180
MIN_VISITS_PER_PATIENT = 2  # need at least 2 points to call it a "trajectory"


def load_all_mri_visits():
    """
    Every qualifying MRI visit (not deduped to one per RID).

    NOTE: unlike Method A's loader, this does NOT filter to STATUS=='complete'.
    That filter is the right call for a *cross-sectional* baseline (Method A
    only needs one good snapshot per patient), but it is far too strict for a
    *longitudinal* trajectory: only 1,065/12,151 UCSFFSX7 rows have
    STATUS=='complete', which left just 36 patients (4 of them CN) with 2+
    qualifying visits -- nowhere near enough to fit a healthy baseline or
    plot a meaningful HI curve.

    Checked directly: of the 12,151 rows, 11,814 already have every MRI
    feature column in MRI_FEATURES populated regardless of STATUS (STATUS
    tracks whether the *full* FreeSurfer QC review was completed, not
    whether these specific volume/thickness fields exist). Using
    "features non-null" instead of "STATUS=='complete'" as the QC proxy for
    this longitudinal table raises the pool to ~2,094 patients with >=2
    visits, ~1,754 with >=3 -- while still being defined by whether the data
    Method B actually consumes is present, not a lower QC bar for its own
    sake. This is a deliberate, documented deviation from Method A -- keep
    it in mind as a limitation (some 'partial' scans may be lower quality
    than 'complete' ones in ways these fields don't capture).
    """
    df = pd.read_csv(f"{DATA_DIR}/UCSFFSX7_12Dec2025.csv", low_memory=False)
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df["_field_rank"] = (df["FIELD_STRENGTH"] == "3T").astype(int)
    df = df.sort_values(["RID", "VISCODE2", "_field_rank"], ascending=[True, True, False])
    df = df.drop_duplicates(["RID", "VISCODE2"], keep="first")
    df = _to_num(df, MRI_FEATURES)
    df = df.dropna(subset=MRI_FEATURES)
    return df[["RID", "VISCODE2", "EXAMDATE", "FIELD_STRENGTH"] + MRI_FEATURES]


def load_all_diagnoses():
    """Every DXSUM row with a valid diagnosis (not deduped to baseline)."""
    df = pd.read_csv(f"{DATA_DIR}/DXSUM_12Dec2025.csv")
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = df[df["DIAGNOSIS"].isin([1, 2, 3])].copy()
    df["DIAGNOSIS_LABEL"] = df["DIAGNOSIS"].map({1: "CN", 2: "MCI", 3: "AD"})
    return df[["RID", "EXAMDATE", "DIAGNOSIS", "DIAGNOSIS_LABEL"]]


def load_demographics_static():
    df = pd.read_csv(f"{DATA_DIR}/PTDEMOG_12Dec2025.csv")
    df["VISDATE"] = pd.to_datetime(df["VISDATE"], errors="coerce")
    df["PTDOBYY"] = pd.to_datetime(df["PTDOBYY"], errors="coerce").dt.year
    df = _to_num(df, ["PTGENDER", "PTEDUCAT", "PTMARRY", "PTHAND"])
    df = df.sort_values("VISDATE").drop_duplicates("RID", keep="first")
    return df[["RID", "PTDOBYY"] + DEMOG_FEATURES]


def _nearest_diagnosis(mri_visits, diagnoses, window_days=DIAG_MATCH_WINDOW_DAYS):
    mri_dates = mri_visits[["RID", "VISCODE2", "EXAMDATE"]].rename(
        columns={"EXAMDATE": "MRI_EXAMDATE"}
    )
    dx_dates = diagnoses.rename(columns={"EXAMDATE": "DX_EXAMDATE"})
    cand = mri_dates.merge(dx_dates, on="RID", how="inner")
    cand["_gap"] = (cand["MRI_EXAMDATE"] - cand["DX_EXAMDATE"]).dt.days.abs()
    cand = cand[cand["_gap"] <= window_days]
    cand = cand.sort_values("_gap").drop_duplicates(["RID", "VISCODE2"], keep="first")
    return cand[["RID", "VISCODE2", "DIAGNOSIS", "DIAGNOSIS_LABEL", "_gap"]].rename(
        columns={"_gap": "DIAG_GAP_DAYS"}
    )


def build_longitudinal_table(min_visits=MIN_VISITS_PER_PATIENT):
    """
    Returns (long_df, feature_cols).

    long_df: one row per (RID, visit), sorted by RID then EXAMDATE, columns:
        RID, VISCODE2, EXAMDATE, VISIT_INDEX (0,1,2... per patient),
        DIAGNOSIS_LABEL (nearest diagnosis within window, may be NaN),
        BASELINE_DIAGNOSIS (this patient's first-visit diagnosis label --
            useful for grouping/coloring trajectories),
        <feature_cols...>

    Only patients with >= min_visits qualifying MRI visits are kept, since a
    single-point "trajectory" isn't useful for an HI curve.
    """
    mri = load_all_mri_visits()
    dx = load_all_diagnoses()
    demog = load_demographics_static()

    diag_match = _nearest_diagnosis(mri, dx)
    df = mri.merge(diag_match, on=["RID", "VISCODE2"], how="left")
    df = df.merge(demog, on="RID", how="left")

    df["AGE"] = df["EXAMDATE"].dt.year - df["PTDOBYY"]
    feature_cols = ["AGE"] + DEMOG_FEATURES + MRI_FEATURES
    df = df.dropna(subset=feature_cols)

    df = df.sort_values(["RID", "EXAMDATE"])
    df["VISIT_INDEX"] = df.groupby("RID").cumcount()

    visit_counts = df.groupby("RID")["VISIT_INDEX"].transform("count")
    df = df[visit_counts >= min_visits].copy()

    baseline_dx = (
        df.sort_values(["RID", "EXAMDATE"])
        .groupby("RID")
        .first()["DIAGNOSIS_LABEL"]
        .rename("BASELINE_DIAGNOSIS")
    )
    df = df.merge(baseline_dx, on="RID", how="left")

    keep = ["RID", "VISCODE2", "EXAMDATE", "VISIT_INDEX",
            "DIAGNOSIS_LABEL", "DIAG_GAP_DAYS", "BASELINE_DIAGNOSIS"] + feature_cols
    return df[keep].reset_index(drop=True), feature_cols


if __name__ == "__main__":
    df, cols = build_longitudinal_table()
    print("Visits:", len(df))
    print("Unique patients:", df["RID"].nunique())
    print("Visits per patient summary:")
    print(df.groupby("RID").size().describe())
    print("\nBaseline diagnosis of included patients:")
    print(df.drop_duplicates("RID")["BASELINE_DIAGNOSIS"].value_counts())
