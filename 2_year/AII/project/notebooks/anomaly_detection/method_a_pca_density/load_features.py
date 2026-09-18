"""
notebooks/anomaly_detection/method_a_pca_density/load_features.py

Builds one cross-sectional multimodal feature row per ADNI participant (RID)
by merging DXSUM (diagnosis), PTDEMOG (demographics), UCSFFSX7 (structural
MRI), UCBERKELEY_AMY_6MM (amyloid PET) and UPENNBIOMK_ROCHE_ELECSYS (CSF).

Each patient's baseline diagnosis (earliest DXSUM row) anchors a target date.
MRI/PET/CSF are then matched to the *visit closest in time* to that date,
within MATCH_WINDOW_DAYS -- not just "each table's independently-earliest
row" as an earlier version of this script did. That earlier approach let a
patient's diagnosis and MRI come from visits years apart for ~35% of the
cohort (median gap was fine at 21 days, but the tail went out to 19 years),
which would silently mislabel what the anomaly score is actually measuring.
Nearest-visit matching with a window fixes that at the cost of some patients
who have no table-B visit close enough to their diagnosis visit.

Cleaning steps applied follow data_evaluation_report.md section 5:
  1. ADNI sentinel codes -1/-4 -> NaN before any statistics.
  2. VISCODE2 used as the canonical visit key (not VISCODE).
  3. UCBERKELEY_AMY_6MM filtered to qc_flag == 2 (QC pass).
  4. UCSFFSX7 filtered to STATUS == 'complete' (not OVERALLQC, ~91% missing);
     when both field strengths exist for a visit, 3T is preferred.
  5. Centiloids preferred over raw SUVR for amyloid burden.
  6. ABETA42/40 ratio only computed when ABETA40 is present (~29% of rows);
     ABETA42, TAU, PTAU alone are the fallback and are always kept.
"""

import os as _os

import numpy as np
import pandas as pd

DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "datasets")

SENTINELS = {-1, -4}

# How close (in days) a table's visit must be to the diagnosis visit to be
# matched to it. 180 days = ~6 months, a reasonable clinical window given
# ADNI's typical 6/12-month visit cadence.
MATCH_WINDOW_DAYS = 180

# Curated, documented "key regions for AD" subset (data_evaluation_report.md /
# datasets_documentation.md) rather than all ~300 raw FreeSurfer/PET columns.
# This keeps missingness manageable and every feature clinically interpretable,
# which matters given the group has no formal medical background.
# NOTE: the field codes listed for these regions in datasets_documentation.md
# turned out to be stale/incorrect when checked against DATADIC_12Dec2025.csv
# (e.g. documented "ST28SA" for left hippocampus is actually unrelated; the
# real code is ST29SV). Every code below was verified directly against
# DATADIC's FLDNAME/TEXT columns before use -- always re-verify ADNI field
# codes against the data dictionary rather than trusting any static doc.
MRI_FEATURES = [
    "ST10CV",              # estimated total intracranial volume (ICV)
    "ST29SV", "ST88SV",    # L/R hippocampus volume
    "ST24CV", "ST83CV",    # L/R entorhinal volume
    "ST37SV", "ST96SV",    # L/R lateral ventricle volume
    "ST12SV", "ST71SV",    # L/R amygdala volume
    "ST24TA", "ST83TA",    # L/R entorhinal thickness
    "ST40TA", "ST99TA",    # L/R middle temporal thickness
    "ST26TA", "ST85TA",    # L/R fusiform thickness
    "ST52TA", "ST111TA",   # L/R precuneus thickness
]

AMYLOID_FEATURES = [
    "CENTILOIDS",
    "CTX_ENTORHINAL_SUVR",
    "CTX_PRECUNEUS_SUVR",
    "CTX_INFERIORPARIETAL_SUVR",
    "CTX_MIDDLETEMPORAL_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR",
    "HIPPOCAMPUS_SUVR",
]

CSF_FEATURES = ["ABETA42", "TAU", "PTAU"]  # ABETA40 / ratio added when available

DEMOG_FEATURES = ["PTGENDER", "PTEDUCAT", "PTMARRY", "PTHAND"]


def _to_num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c].isin(SENTINELS), c] = np.nan
    return df


def nearest_match(dx, other, date_col, window_days=MATCH_WINDOW_DAYS):
    """
    For each RID in `dx` (which must have RID + DX_EXAMDATE), find the row in
    `other` (which must have RID + date_col) whose date_col is closest in
    time to DX_EXAMDATE, keeping it only if the gap is <= window_days.
    Returns dx left-merged with the matched row's other columns (all NaN
    where no visit falls inside the window), plus a `<date_col>_GAP_DAYS`
    column so gap sizes can be audited later.
    """
    cand = dx[["RID", "DX_EXAMDATE"]].merge(other, on="RID", how="inner")
    cand["_gap_days"] = (cand[date_col] - cand["DX_EXAMDATE"]).dt.days.abs()
    cand = cand[cand["_gap_days"] <= window_days]
    cand = cand.sort_values("_gap_days").drop_duplicates("RID", keep="first")
    cand = cand.drop(columns=["DX_EXAMDATE"]).rename(
        columns={"_gap_days": f"{date_col}_GAP_DAYS"}
    )
    return dx.merge(cand, on="RID", how="left")


def load_diagnosis():
    df = pd.read_csv(f"{DATA_DIR}/DXSUM_12Dec2025.csv")
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = df[df["DIAGNOSIS"].isin([1, 2, 3])].copy()
    df = df.sort_values("EXAMDATE").drop_duplicates("RID", keep="first")
    df["DIAGNOSIS_LABEL"] = df["DIAGNOSIS"].map({1: "CN", 2: "MCI", 3: "AD"})
    return df[["RID", "EXAMDATE", "DIAGNOSIS", "DIAGNOSIS_LABEL"]].rename(
        columns={"EXAMDATE": "DX_EXAMDATE"}
    )


def load_demographics():
    # Demographics are near time-invariant (birth year, education, marital
    # status change negligibly across an ADNI follow-up window), so these
    # stay "earliest available row per RID" rather than nearest-date matched.
    df = pd.read_csv(f"{DATA_DIR}/PTDEMOG_12Dec2025.csv")
    df["VISDATE"] = pd.to_datetime(df["VISDATE"], errors="coerce")
    # PTDOBYY is stored as a full date string ("YYYY-01-01"), not a bare year
    df["PTDOBYY"] = pd.to_datetime(df["PTDOBYY"], errors="coerce").dt.year
    df = _to_num(df, ["PTGENDER", "PTEDUCAT", "PTMARRY", "PTHAND"])
    df = df.sort_values("VISDATE").drop_duplicates("RID", keep="first")
    return df[["RID", "VISDATE", "PTDOBYY"] + DEMOG_FEATURES]


def load_mri():
    """Returns ALL qualifying MRI visits per RID (not deduped to one row) so
    build_feature_table can nearest-match each patient's diagnosis date."""
    df = pd.read_csv(f"{DATA_DIR}/UCSFFSX7_12Dec2025.csv", low_memory=False)
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = df[df["STATUS"] == "complete"].copy()
    # prefer 3T when both field strengths exist for the same RID/VISCODE2
    df["_field_rank"] = (df["FIELD_STRENGTH"] == "3T").astype(int)
    df = df.sort_values(["RID", "VISCODE2", "_field_rank"], ascending=[True, True, False])
    df = df.drop_duplicates(["RID", "VISCODE2"], keep="first")
    df = _to_num(df, MRI_FEATURES)
    return df[["RID", "EXAMDATE", "FIELD_STRENGTH"] + MRI_FEATURES].rename(
        columns={"EXAMDATE": "MRI_EXAMDATE"}
    )


def load_amyloid():
    """Returns ALL qc-passing amyloid PET visits per RID (not deduped)."""
    df = pd.read_csv(f"{DATA_DIR}/UCBERKELEY_AMY_6MM_12Dec2025.csv", low_memory=False)
    df["SCANDATE"] = pd.to_datetime(df["SCANDATE"], errors="coerce")
    df = df[df["qc_flag"] == 2].copy()
    df = _to_num(df, AMYLOID_FEATURES)
    return df[["RID", "SCANDATE", "TRACER"] + AMYLOID_FEATURES].rename(
        columns={"SCANDATE": "PET_EXAMDATE"}
    )


def load_csf():
    """Returns ALL CSF visits per RID (not deduped)."""
    df = pd.read_csv(f"{DATA_DIR}/UPENNBIOMK_ROCHE_ELECSYS_12Dec2025.csv")
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = _to_num(df, ["ABETA40"] + CSF_FEATURES)
    df["ABETA_RATIO_42_40"] = df["ABETA42"] / df["ABETA40"]  # NaN where ABETA40 missing
    df["PTAU_ABETA42_RATIO"] = df["PTAU"] / df["ABETA42"]
    keep = ["RID", "EXAMDATE"] + CSF_FEATURES + ["ABETA_RATIO_42_40", "PTAU_ABETA42_RATIO"]
    return df[keep].rename(columns={"EXAMDATE": "CSF_EXAMDATE"})


# "Core" features: demographics + structural MRI. This is the best-covered
# pair (UCSFFSX7 covers 84.3% of diagnosed RIDs per data_evaluation_report.md
# -- though the stricter STATUS=='complete' QC filter, and now the +/-180
# day match window, both cut that further) and yields a usable three-class
# cohort.
#
# "Optional" features: amyloid PET + CSF. These cover only 57%/44% of
# diagnosed RIDs respectively even before date-matching (selection bias --
# PET/CSF sub-studies are optional and invasive). Requiring them shrinks the
# complete-case cohort a lot -- kept as an opt-in tier, not the default.
CORE_FEATURES = ["AGE"] + DEMOG_FEATURES + MRI_FEATURES
OPTIONAL_FEATURES = AMYLOID_FEATURES + ["ABETA42", "TAU", "PTAU", "PTAU_ABETA42_RATIO"]


def build_feature_table(feature_tier="core", window_days=MATCH_WINDOW_DAYS):
    """
    Returns (features_df, feature_cols).

    Each patient's diagnosis visit anchors the match: MRI/PET/CSF are only
    attached if a visit exists within `window_days` of that diagnosis date
    (see nearest_match / MATCH_WINDOW_DAYS docstring above).

    feature_tier="core" (default): demographics + MRI only.
    feature_tier="full": also requires amyloid PET + CSF present within the
        window -- shrinks the cohort substantially, kept for extension only.
    """
    dx = load_diagnosis()
    demog = load_demographics()
    mri = load_mri()
    amy = load_amyloid()
    csf = load_csf()

    df = dx.merge(demog, on="RID", how="left")
    df = nearest_match(df, mri, "MRI_EXAMDATE", window_days)
    df = nearest_match(df, amy, "PET_EXAMDATE", window_days)
    df = nearest_match(df, csf, "CSF_EXAMDATE", window_days)

    # approximate age at diagnosis visit from birth year
    df["AGE"] = df["DX_EXAMDATE"].dt.year - df["PTDOBYY"]

    if feature_tier == "core":
        feature_cols = CORE_FEATURES
    elif feature_tier == "full":
        feature_cols = CORE_FEATURES + OPTIONAL_FEATURES
    else:
        raise ValueError("feature_tier must be 'core' or 'full'")

    df = df.dropna(subset=feature_cols)

    return df.reset_index(drop=True), feature_cols


if __name__ == "__main__":
    df, cols = build_feature_table()
    print("Rows:", len(df))
    print(df["DIAGNOSIS_LABEL"].value_counts())
    print("Feature columns:", len(cols))
    print("MRI_EXAMDATE_GAP_DAYS summary:")
    print(df["MRI_EXAMDATE_GAP_DAYS"].describe())
