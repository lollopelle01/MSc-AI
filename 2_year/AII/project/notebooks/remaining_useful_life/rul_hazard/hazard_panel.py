"""
notebooks/remaining_useful_life/rul_hazard/hazard_panel.py

Shared panel-building and survival-curve utilities for the discrete-time
hazard variant of the RUL models next to this file (hazard_model_1/2/3.py),
imported by all three, not run directly.

../rul_model_1.py, ../rul_model_2.py and ../rul_model_3.py regress RUL_YEARS
directly, which only has a well-defined training label for patients who
converted (or, with a horizon cap, non-converters need HORIZON_YEARS worth of
clean follow-up). With HORIZON_YEARS = -1 (the current setting) non-converters
are dropped entirely -- the regressor only ever sees patients who did convert.

This module instead reshapes every MCI visit -- converter or not, any amount
of follow-up -- into one row of a discrete-time hazard panel: EVENT_AT_VISIT
is 1 only at the single visit immediately followed by a clean
MCI -> AD-dementia transition (reusing ../rul_model_1.py's own
_next_ad_date / "clean uninterrupted MCI run" rule unchanged), 0 at every
other MCI visit, converter or not. A classifier fit on this panel outputs,
for any visit, the probability that *this* visit is the one immediately
preceding conversion -- the discrete-time hazard -- and every patient's full
history of visits becomes valid training signal instead of being dropped.

Two covariates the level-only leo models don't need show up here because
intervals between visits aren't fixed length:

    NEXT_GAP_MONTHS      - months from this visit to the patient's actual
                            next recorded visit (of any diagnosis) -- the
                            length of the interval EVENT_AT_VISIT is actually
                            about. This is a legitimate input feature, not a
                            leak: it doesn't reveal whether the event
                            happened, only how long a window the model is
                            being asked to assess risk over -- the same role
                            "exposure time" plays in a Poisson hazard model.
    MCI_DURATION_MONTHS  - months since the start of this patient's current
                            *clean* MCI run (RUN_ID below), replacing
                            PRIOR_DIAGNOSIS (no information here since every
                            base row is already MCI).

RUN_ID identifies one uninterrupted MCI streak: a patient who reverts to
Cognitively Normal and later develops a fresh MCI episode gets two separate
runs, matching the same "clean path" rule rul_model_1.py already applies to
RUL_YEARS -- an eventual AD diagnosis reached only after such a reversion is a
separate episode, not a continuation of the run before it.

Rows whose next visit isn't known at all (a patient's very last recorded
visit overall) are dropped -- not because the patient didn't convert, but
because the interval EVENT_AT_VISIT would refer to has no defined length.
This is the only row-dropping in this module; every other visit, converter or
not, regardless of follow-up length, is kept.

RUL_YEARS_TRUE (true time from this visit to the eventual clean AD diagnosis,
converters only) is carried along purely for validating the RUL derived from
the fitted hazard sequence (see forecast_survival_curves / survival_to_rul
below) against ground truth -- it is never a model input.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rul_model_1 import (  # noqa: E402  (sys.path.append above must run first)
    DEMOG_COLS, MODALITY_COLS,
    _asof_merge, _next_ad_date, _path, _to_num,
    load_demographics, load_mri, load_pet, load_csf,
)

DAYS_PER_MONTH = 30.44
STEP_MONTHS = 6   # assumed regular spacing of future visits when forecasting
MAX_STEPS = 40    # steps * STEP_MONTHS = 20 years of forecast horizon


# --------------------------------------------------------------------------- #
# 1-2. Diagnosis timeline -> hazard panel (one row per MCI visit)             #
# --------------------------------------------------------------------------- #
def build_hazard_base_rows():
    cols = ["RID", "VISCODE2", "EXAMDATE", "DIAGNOSIS", "DXAD", "DXDDUE"]
    dx = pd.read_csv(_path("DXSUM_12Dec2025.csv"), usecols=cols)
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx = dx[dx["DIAGNOSIS"].isin([1, 2, 3]) & dx["EXAMDATE"].notna()]
    dx = _to_num(dx, ["DXAD", "DXDDUE"])
    dx["IS_AD"] = (dx["DIAGNOSIS"] == 3) & ((dx["DXAD"] == 1) | (dx["DXDDUE"] == 1))
    dx = dx.sort_values(["RID", "EXAMDATE"]).reset_index(drop=True)

    # One id per uninterrupted MCI streak: bumps at every non-MCI row and at
    # every patient boundary, unchanged across consecutive MCI rows in between.
    is_mci = dx["DIAGNOSIS"] == 2
    run_break = (~is_mci) | (dx["RID"] != dx["RID"].shift())
    dx["RUN_ID"] = run_break.cumsum()

    next_ad = dx.groupby("RID", group_keys=False).apply(_next_ad_date, include_groups=False)
    dx["FIRST_DEM"] = next_ad.reindex(dx.index)

    # The interval EVENT_AT_VISIT is actually about: months to this patient's
    # own next recorded visit, of any diagnosis (not only the next MCI one).
    same_patient_next = dx["RID"] == dx["RID"].shift(-1)
    next_examdate = dx["EXAMDATE"].shift(-1).where(same_patient_next)
    dx["NEXT_GAP_MONTHS"] = (next_examdate - dx["EXAMDATE"]).dt.days / DAYS_PER_MONTH

    mci = dx[dx["DIAGNOSIS"] == 2].copy()
    mci = mci[mci["NEXT_GAP_MONTHS"].notna()]

    run_start = mci.groupby("RUN_ID")["EXAMDATE"].transform("min")
    mci["MCI_DURATION_MONTHS"] = (mci["EXAMDATE"] - run_start).dt.days / DAYS_PER_MONTH

    mci["CONVERTED"] = mci["FIRST_DEM"].notna()
    mci["RUL_YEARS_TRUE"] = (mci["FIRST_DEM"] - mci["EXAMDATE"]).dt.days / 365.25

    # Within a clean run every visit shares the same FIRST_DEM; the event
    # visit is the one with the latest EXAMDATE among them -- every earlier
    # visit in that same run is a genuine "not yet" (EVENT_AT_VISIT = 0).
    is_event = np.zeros(len(mci), dtype=bool)
    conv_mask = mci["CONVERTED"].to_numpy()
    if conv_mask.any():
        conv = mci[conv_mask]
        last_examdate = conv.groupby(["RID", "FIRST_DEM"])["EXAMDATE"].transform("max")
        is_event[conv_mask] = (conv["EXAMDATE"] == last_examdate).to_numpy()
    mci["EVENT_AT_VISIT"] = is_event

    return mci[["RID", "VISCODE2", "EXAMDATE", "RUN_ID", "NEXT_GAP_MONTHS",
                "MCI_DURATION_MONTHS", "EVENT_AT_VISIT", "CONVERTED", "RUL_YEARS_TRUE"]]


# --------------------------------------------------------------------------- #
# 3-4. Demographics + modalities (loaders reused unchanged from rul_model_1)   #
# --------------------------------------------------------------------------- #
def build_hazard_dataset():
    base = build_hazard_base_rows()

    demog = load_demographics()
    base = base.merge(demog, on="RID", how="left")
    base["AGE"] = base["EXAMDATE"].dt.year - base["BIRTH_YEAR"]

    for loader in (load_mri, load_pet, load_csf):
        base = _asof_merge(base, loader())

    base = base.sort_values(["RID", "EXAMDATE"])
    biomarker_cols = [c for cols in MODALITY_COLS.values() for c in cols]
    for c in biomarker_cols:
        base[c] = base.groupby("RID")[c].ffill()
        base[c] = base.groupby("RID")[c].bfill()

    return base.reset_index(drop=True), biomarker_cols


# --------------------------------------------------------------------------- #
# Calibration: bucketed predicted-vs-observed, same idea as the hazard        #
# notebook's calibration_curve_check (self-contained here, no util import)    #
# --------------------------------------------------------------------------- #
def calibration_table(y_true, y_prob, n_buckets=10):
    order = np.argsort(y_prob)
    y_true = np.asarray(y_true)[order]
    y_prob = np.asarray(y_prob)[order]

    rows = []
    for idx in np.array_split(np.arange(len(y_prob)), n_buckets):
        rows.append({
            "n_rows": len(idx),
            "mean_predicted": y_prob[idx].mean(),
            "observed_fraction": y_true[idx].mean(),
        })
    table = pd.DataFrame(rows)
    table["gap"] = (table["mean_predicted"] - table["observed_fraction"]).abs()
    weighted_gap = (table["gap"] * table["n_rows"]).sum() / table["n_rows"].sum()
    return table, weighted_gap


# --------------------------------------------------------------------------- #
# Hazard sequence -> survival curve -> a single RUL number                    #
# --------------------------------------------------------------------------- #
def forecast_survival_curves(predict_fn, X_raw, feature_cols,
                              gap_col="NEXT_GAP_MONTHS", duration_col="MCI_DURATION_MONTHS",
                              step_months=STEP_MONTHS, max_steps=MAX_STEPS):
    """
    Walks max_steps steps of step_months each, forward from every row of
    X_raw (one row per patient-visit to project forward from). At each step
    every feature is held at its frozen value -- the same persistence
    approximation notebooks/hazard_model/hazard_survival_model.ipynb uses --
    except gap_col, reset to step_months (the assumed regular spacing of
    future visits), and duration_col, advanced by step_months. predict_fn
    takes the raw feature matrix for one step and returns hazard
    probabilities; any imputation/scaling a given model needs must already be
    folded into it.

    Returns S, an (n_rows, max_steps) array; S[:, k-1] = P(still not
    converted after k more steps).
    """
    gap_idx = feature_cols.index(gap_col)
    dur_idx = feature_cols.index(duration_col)
    base_duration = X_raw[:, dur_idx].copy()

    S = np.empty((len(X_raw), max_steps))
    survival = np.ones(len(X_raw))
    for k in range(1, max_steps + 1):
        step_X = X_raw.copy()
        step_X[:, gap_idx] = step_months
        step_X[:, dur_idx] = base_duration + step_months * k
        hazard_k = predict_fn(step_X)
        survival = survival * (1.0 - hazard_k)
        S[:, k - 1] = survival
    return S


def survival_to_rul(S, step_months=STEP_MONTHS, method="median"):
    """
    Collapse an (n_rows, max_steps) survival curve into one RUL number
    (months) per row.

    method="median": months to the first step where S <= 0.5 -- "by this
      point, conversion is more likely than not". Rows whose S never drops
      to 0.5 within max_steps get NaN, reported separately as "no conversion
      predicted within horizon" -- the classification analogue of
      RUL_YEARS = HORIZON_YEARS in rul_model_1.py, read off the model instead
      of assumed at label-build time.
    method="expected": probability-weighted average time to event, using
      only the resolved probability mass (1 - S[:, -1]) as the denominator.
      Also returns that resolved mass so a row where most probability is
      still unresolved by max_steps is visibly less trustworthy rather than
      silently biased low. Returns (rul_months, resolved_mass).
    """
    n_rows, max_steps = S.shape
    steps = np.arange(1, max_steps + 1)

    if method == "median":
        below = S <= 0.5
        first_below = np.where(below.any(axis=1), below.argmax(axis=1) + 1, -1)
        return np.where(first_below > 0, first_below * step_months, np.nan)

    if method == "expected":
        S_prev = np.concatenate([np.ones((n_rows, 1)), S[:, :-1]], axis=1)
        p_exact = S_prev - S
        resolved_mass = p_exact.sum(axis=1)
        weighted = (p_exact * steps[None, :] * step_months).sum(axis=1)
        rul_months = np.where(resolved_mass > 0, weighted / resolved_mass, np.nan)
        return rul_months, resolved_mass

    raise ValueError("method must be 'median' or 'expected'")
