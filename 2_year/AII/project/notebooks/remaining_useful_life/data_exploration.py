"""
notebooks/remaining_useful_life/data_exploration.py

Data exploration and visualization for the ADNI cohort used by
rul_model_*.py. Reuses that script's modality loaders (mri / pet / csf, each
with its own EXAMDATE column) and diagnosis-timeline logic so every plot here
is built from the exact same rows the RUL models train on.

Every feature is z-score standardized before it is plotted, so plots are
comparable across features of very different natural scale. The heatmaps
standardize each patient against their own visits (per patient, per
feature) so an individual's trajectory is visible regardless of where they
sit relative to the wider population; the violins standardize against the
full population (mean/std over every measured visit of every patient) since
they compare patients to each other at a single point in time.

Produces, for each of the three modalities (mri, pet, csf):
  1. heatmap_<modality>_top<N>.png
     Feature (y) x visit (x) heatmap concatenating N patients (5 by default)
     picked from that modality's most-visited *converting* patients (same
     converter cohort as rul_model_1.py -- see build_patient_timeline), one
     block per patient, separated by vertical lines -- lets you compare
     several heavily-followed patients' trajectories in one image. Visit
     dates come from that modality's own EXAMDATE column, independent of
     the other modalities' visit schedules. A red dashed line marks each
     patient's conversion (first Dementia diagnosis), placed at the visit
     column closest to that date.
  2. visit_gap_hist_<modality>.png
     Histogram (1-month bins) of the gap between consecutive visits, pooled
     across all patients, to show which follow-up intervals are most common.

And three violin plots, one per point in the converting-patient timeline
(same converter cohort as rul_model_1.py: MCI visits with a later Dementia
diagnosis):
  3. violin_baseline.png    -- first qualifying MCI visit
     violin_conversion.png  -- first Dementia diagnosis
     violin_mid_ad.png      -- midpoint in time between conversion and each
                                patient's last recorded visit
     Each shows one violin per biomarker feature (mri + pet + csf), the
     nearest measured value (within MATCH_WINDOW_DAYS) to that point, across
     all converting patients.

Run: cd leo && python data_exploration.py
Output: PNG files written next to this script.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rul_model_1 import (
     MODALITY_COLS,
    _asof_merge, _path,
    load_mri, load_pet, load_csf,
)

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
HEATMAP_POOL_SIZE = 20            # rank all patients by visit count, keep this many (most to least)
HEATMAP_RANK_INDICES = [0, 1, 2, 3, 4]  # positions within that pool to plot (0 = most visits); edit to pick a different 5
MAX_GAP_MONTHS = 60                # inter-visit gaps beyond this are pooled into one bucket

LOADERS = {"mri": load_mri, "pet": load_pet, "csf": load_csf}
ALL_FEATURE_COLS = [c for cols in MODALITY_COLS.values() for c in cols]

POINT_TITLES = {
    "BASELINE": "Baseline (first qualifying MCI visit)",
    "CONVERSION": "Conversion (first Dementia diagnosis)",
    "MID_AD": "Middle of the Alzheimer's phase (conversion -> last visit midpoint)",
}


# --------------------------------------------------------------------------- #
# Standardization (global stats for the violins; per-patient for heatmaps)    #
# --------------------------------------------------------------------------- #
def compute_global_stats(modality_data):
    """Per-feature (mean, std) over every measured visit of every patient."""
    stats = {}
    for modality, df in modality_data.items():
        for col in MODALITY_COLS[modality]:
            vals = df[col].dropna()
            stats[col] = (vals.mean(), vals.std())
    return stats


def standardize(df, stats, cols):
    df = df.copy()
    for c in cols:
        mu, sd = stats[c]
        df[c] = (df[c] - mu) / (sd if sd and not pd.isna(sd) else 1.0)
    return df


def per_patient_stats(patient, cols):
    """Per-feature (mean, std) over just this patient's own visits."""
    return {c: (patient[c].mean(), patient[c].std()) for c in cols}


# --------------------------------------------------------------------------- #
# 1. Multi-patient, per-modality heatmap (feature x visit, patients          #
#    concatenated along the time axis)                                        #
# --------------------------------------------------------------------------- #
def top_visited_rids(df, eligible_rids, pool_size, rank_indices):
    """RIDs at the given rank positions (0 = most visits) among the pool_size
    most-visited patients, restricted to eligible_rids (converters)."""
    df = df[df["RID"].isin(eligible_rids)]
    pool = df.groupby("RID").size().sort_values(ascending=False).index.tolist()[:pool_size]
    return [pool[i] for i in rank_indices]


def plot_patient_heatmaps(modality_data, conversion_dates, out_dir):
    converter_rids = set(conversion_dates.index)
    for modality, df in modality_data.items():
        cols = MODALITY_COLS[modality]
        rids = top_visited_rids(df, converter_rids, HEATMAP_POOL_SIZE, HEATMAP_RANK_INDICES)

        blocks, dates, boundaries, centers, conv_lines = [], [], [], [], []
        pos = 0
        for rid in rids:
            patient = df[df["RID"] == rid].sort_values("EXAMDATE")
            patient = standardize(patient, per_patient_stats(patient, cols), cols)
            block = patient[cols].to_numpy(dtype=float).T
            blocks.append(block)
            visit_dates = patient["EXAMDATE"]
            dates.extend(visit_dates.dt.strftime("%Y-%m-%d").tolist())
            centers.append(pos + block.shape[1] / 2 - 0.5)

            conv_date = conversion_dates.get(rid)
            if pd.notna(conv_date):
                nearest = (visit_dates - conv_date).abs().to_numpy().argmin()
                conv_lines.append(pos + nearest)

            pos += block.shape[1]
            boundaries.append(pos)

        matrix = np.concatenate(blocks, axis=1)

        fig, ax = plt.subplots(figsize=(max(10, 0.45 * matrix.shape[1]), 1.5 + 0.5 * len(cols)))
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=90, fontsize=6)
        for b in boundaries[:-1]:
            ax.axvline(b - 0.5, color="black", lw=1.3)
        for x in conv_lines:
            ax.axvline(x, color="red", ls="--", lw=1.5,
                       label="conversion (nearest visit to first Dementia diagnosis)")
        for rid, c in zip(rids, centers):
            ax.text(c, 1.02, f"RID {rid}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, fontweight="bold", clip_on=False)
        if conv_lines:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
        ax.set_xlabel("Visit date (each block is one patient, separated by vertical lines)")
        fig.colorbar(im, ax=ax, label="z-score (within patient)")
        fig.suptitle(f"{modality.upper()} -- top-visit patients (features standardized per patient)")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(os.path.join(out_dir, f"heatmap_{modality}_top{len(rids)}.png"), dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Histogram of the gap between consecutive visits, per modality            #
# --------------------------------------------------------------------------- #
def plot_visit_gap_histograms(modality_data, out_dir):
    for modality, df in modality_data.items():
        df = df.sort_values(["RID", "EXAMDATE"])
        gap_days = df.groupby("RID")["EXAMDATE"].diff().dt.days.dropna()
        gap_months = (gap_days / 30.44).round().astype(int)
        gap_months = gap_months[gap_months > 0].clip(upper=MAX_GAP_MONTHS)

        counts = gap_months.value_counts().sort_index()
        labels = [str(m) if m < MAX_GAP_MONTHS else f"{MAX_GAP_MONTHS}+" for m in counts.index]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(counts)), counts.values, color="steelblue")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, rotation=90 if len(counts) > 30 else 0)
        ax.set_xlabel("Gap between consecutive visits (months, rounded)")
        ax.set_ylabel("Number of visit pairs")
        ax.set_title(f"{modality.upper()} -- inter-visit gap distribution (all patients)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"visit_gap_hist_{modality}.png"), dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Converting-patient timeline -> one violin plot per timepoint             #
# --------------------------------------------------------------------------- #
def load_dx():
    """Every DXSUM row with a valid diagnosis (1=CN, 2=MCI, 3=Dementia) and EXAMDATE."""
    dx = pd.read_csv(_path("DXSUM_12Dec2025.csv"), usecols=["RID", "EXAMDATE", "DIAGNOSIS"])
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    return dx[dx["DIAGNOSIS"].isin([1, 2, 3]) & dx["EXAMDATE"].notna()].sort_values(["RID", "EXAMDATE"])


def build_patient_timeline(dx):
    """RID, BASELINE, CONVERSION, LAST_VISIT, MID_AD for converting patients.

    Same converter cohort as rul_model_1.py: MCI visits with a later
    Dementia diagnosis for that RID. BASELINE is the earliest such
    qualifying MCI visit; MID_AD is the time midpoint between CONVERSION
    and the patient's last recorded visit of any diagnosis.
    """
    first_dem = dx[dx["DIAGNOSIS"] == 3].groupby("RID")["EXAMDATE"].min()
    last_visit = dx.groupby("RID")["EXAMDATE"].max()

    mci = dx[dx["DIAGNOSIS"] == 2].copy()
    mci["FIRST_DEM"] = mci["RID"].map(first_dem)
    converted_mci = mci[mci["FIRST_DEM"].notna() & (mci["EXAMDATE"] < mci["FIRST_DEM"])]
    baseline = converted_mci.groupby("RID")["EXAMDATE"].min()

    timeline = pd.DataFrame({"BASELINE": baseline, "CONVERSION": first_dem, "LAST_VISIT": last_visit}).dropna()
    timeline["MID_AD"] = timeline["CONVERSION"] + (timeline["LAST_VISIT"] - timeline["CONVERSION"]) / 2
    return timeline.reset_index()


def attach_features_at(target_dates, modality_data):
    """Nearest measured feature value (within MATCH_WINDOW_DAYS) per RID."""
    point = target_dates.rename("EXAMDATE").reset_index()
    for modality in LOADERS:
        point = _asof_merge(point, modality_data[modality])
    return point


def plot_timepoint_violins(modality_data, stats, dx, out_dir):
    timeline = build_patient_timeline(dx)
    n_patients = len(timeline)

    for point in ("BASELINE", "CONVERSION", "MID_AD"):
        data = attach_features_at(timeline.set_index("RID")[point], modality_data)
        data = standardize(data, stats, ALL_FEATURE_COLS)

        present, series = [], []
        for c in ALL_FEATURE_COLS:
            vals = data[c].dropna().to_numpy()
            if len(vals):
                present.append(c)
                series.append(vals)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.violinplot(series, showmedians=True)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, rotation=45, ha="right")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_ylabel("Standardized value (z-score)")
        ax.set_title(f"{POINT_TITLES[point]} -- across {n_patients} converting patients")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"violin_{point.lower()}.png"), dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    modality_data = {m: loader() for m, loader in LOADERS.items()}
    stats = compute_global_stats(modality_data)
    dx = load_dx()
    conversion_dates = dx[dx["DIAGNOSIS"] == 3].groupby("RID")["EXAMDATE"].min()

    plot_patient_heatmaps(modality_data, conversion_dates, out_dir)
    plot_visit_gap_histograms(modality_data, out_dir)
    plot_timepoint_violins(modality_data, stats, dx, out_dir)

    print(f"plots written to {out_dir}")


if __name__ == "__main__":
    main()
