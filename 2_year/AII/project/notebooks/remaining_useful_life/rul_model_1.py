"""
notebooks/remaining_useful_life/rul_model.py

"Remaining Useful Life" (RUL) model for ADNI patients.

RUL here = horizon-capped time from an MCI visit to that patient's conversion
to Dementia due to Alzheimer's Disease specifically (DXAD/DXDDUE == AD, not
just any DIAGNOSIS == 3 -- vascular/other dementia doesn't count):

    RUL_YEARS = min(years_from_visit_to_AD_conversion, HORIZON_YEARS)

An MCI visit only counts as converting if the path forward to that AD
diagnosis is a clean, uninterrupted run of MCI visits -- if the patient's
diagnosis reverts to Cognitively Normal (or moves to a non-AD dementia) before
the AD diagnosis, that MCI visit is treated as not (yet) converted, since the
eventual AD diagnosis is a separate episode, not a continuation of that
decline (see _next_ad_date).

Patients who never convert ("non-converters") are used too: an MCI visit with
at least HORIZON_YEARS of later follow-up and no Dementia diagnosis is an exact
RUL_YEARS = HORIZON_YEARS example. Non-converter visits with shorter follow-up
are ambiguous (censored below the horizon) and dropped.

HORIZON_YEARS = -1 is a sentinel meaning "no horizon": RUL_YEARS is left
uncapped and non-converters are dropped entirely, so training uses only
converter visits with an exact, known time-to-conversion.

One row per qualifying MCI visit. Three biomarker modalities can be switched on
and off freely (see MODALITIES / EXPERIMENTS) to compare their contribution:

    mri  - structural MRI volumes (FreeSurfer, UCSFFSX7)
    pet  - amyloid PET            (UCBERKELEY_AMY_6MM)
    csf  - CSF biomarkers         (UPENNBIOMK_ROCHE_ELECSYS)

Demographics (age, sex, education) are a shared baseline included in every run.

Model: plain regression (RandomForest by default, Ridge optional), evaluated
with patient-grouped 5-fold CV. Metrics: MAE and RMSE, against a mean-predictor
baseline

Field codes for UCSFFSX7 are the ones verified against DATADIC in
notebooks/anomaly_detection/method_a_pca_density/load_features.py (the static docs list stale codes).

Limitation: on this cohort PET and CSF are measured for far fewer visits than
MRI, so their columns are heavily forward/backward-filled or median-filled --
their measured comparison is weaker than MRI's.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*ChainedAssignment.*", category=FutureWarning)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets")

MATCH_WINDOW_DAYS = 180      # how close a scan/sample must be to the visit
HORIZON_YEARS = -1            # RUL is capped here; also the non-converter label
                              # -1: no cap, converters only (see build_base_rows)
HORIZON_LABEL = "no cap, converters only" if HORIZON_YEARS == -1 else f"{HORIZON_YEARS}y"
USE_DEMOGRAPHICS = True      # AGE, PTGENDER, PTEDUCAT shared across every run
MODEL = "rf"                 # "rf" (RandomForestRegressor) | "linear" (Ridge)
N_SPLITS = 5
RANDOM_STATE = 0
PLOT_N_PATIENTS = 15          # patients shown in the true-vs-predicted RUL plot

RUN_ALL = True               # True: loop EXPERIMENTS; False: single MODALITIES run
MODALITIES = ["mri", "pet", "csf"]
EXPERIMENTS = [[], ["mri"], ["pet"], ["csf"], ["mri", "pet", "csf"]]

# The one place modalities are defined: name -> feature columns it contributes.
MODALITY_COLS = {
    "mri": ["HIPPO_ICV", "ENTORHINAL_ICV", "AMYGDALA_ICV"],
    "pet": ["CENTILOIDS", "SUMMARY_SUVR"],
    "csf": ["ABETA42", "TAU", "PTAU"],
}
DEMOG_COLS = ["AGE", "PTGENDER", "PTEDUCAT"]

SENTINELS = {-1, -4}

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _to_num(df, cols):
    """Coerce columns to numeric and turn ADNI sentinel codes into NaN."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c].isin(SENTINELS), c] = np.nan
    return df


def _path(name):
    return os.path.join(DATA_DIR, name)


def _asof_merge(base, mod):
    """Attach `mod` (RID, EXAMDATE, features...) to `base` by nearest date."""
    base = base.sort_values("EXAMDATE")
    mod = mod.sort_values("EXAMDATE").drop_duplicates(["RID", "EXAMDATE"], keep="first")
    return pd.merge_asof(
        base, mod, on="EXAMDATE", by="RID",
        direction="nearest", tolerance=pd.Timedelta(days=MATCH_WINDOW_DAYS),
    )


# --------------------------------------------------------------------------- #
# 1-2. Diagnosis timeline -> base rows + target                                #
# --------------------------------------------------------------------------- #
def _next_ad_date(g):
    """
    Per-patient (chronologically sorted) helper: for every row, the EXAMDATE of
    the next AD-Dementia diagnosis reachable without crossing a CN visit or a
    non-AD-Dementia visit first -- i.e. only a clean, uninterrupted MCI -> AD
    run counts. NaT if the next differing diagnosis isn't AD, or there isn't
    one yet.
    """
    examdate = g["EXAMDATE"].to_numpy()
    is_mci = (g["DIAGNOSIS"] == 2).to_numpy()
    is_ad = g["IS_AD"].to_numpy()

    target = np.full(len(g), np.datetime64("NaT"), dtype=examdate.dtype)
    nxt = np.datetime64("NaT")
    for i in range(len(g) - 1, -1, -1):
        target[i] = nxt
        if not is_mci[i]:
            nxt = examdate[i] if is_ad[i] else np.datetime64("NaT")
    return pd.Series(target, index=g.index)


def build_base_rows():
    cols = ["RID", "VISCODE2", "EXAMDATE", "DIAGNOSIS", "DXAD", "DXDDUE"]
    dx = pd.read_csv(_path("DXSUM_12Dec2025.csv"), usecols=cols)
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx = dx[dx["DIAGNOSIS"].isin([1, 2, 3]) & dx["EXAMDATE"].notna()]
    dx = _to_num(dx, ["DXAD", "DXDDUE"])

    # "Conversion" = Dementia due to Alzheimer's Disease specifically, not any
    # dementia. DXAD covers ADNI1/GO/2, DXDDUE the newer ADNI3 coding.
    dx["IS_AD"] = (dx["DIAGNOSIS"] == 3) & ((dx["DXAD"] == 1) | (dx["DXDDUE"] == 1))
    dx = dx.sort_values(["RID", "EXAMDATE"])

    last_visit = dx.groupby("RID")["EXAMDATE"].max()
    next_ad = dx.groupby("RID", group_keys=False).apply(_next_ad_date, include_groups=False)

    mci = dx[dx["DIAGNOSIS"] == 2].copy()
    mci["FIRST_DEM"] = next_ad.reindex(mci.index)
    mci["LAST_VISIT"] = mci["RID"].map(last_visit)
    mci["CONVERTED"] = mci["FIRST_DEM"].notna()

    conv = mci[mci["CONVERTED"]].copy()
    conv["RUL_YEARS"] = (conv["FIRST_DEM"] - conv["EXAMDATE"]).dt.days / 365.25
    if HORIZON_YEARS == -1:
        base = conv
    else:
        conv["RUL_YEARS"] = conv["RUL_YEARS"].clip(upper=HORIZON_YEARS)

        non = mci[mci["FIRST_DEM"].isna()].copy()
        non["FOLLOW_UP"] = (non["LAST_VISIT"] - non["EXAMDATE"]).dt.days / 365.25
        non = non[non["FOLLOW_UP"] >= HORIZON_YEARS]
        non["RUL_YEARS"] = float(HORIZON_YEARS)

        base = pd.concat([conv, non], ignore_index=True)

    return base[["RID", "VISCODE2", "EXAMDATE", "RUL_YEARS", "CONVERTED"]]


# --------------------------------------------------------------------------- #
# 3. Static demographics                                                       #
# --------------------------------------------------------------------------- #
def load_demographics():
    df = pd.read_csv(_path("PTDEMOG_12Dec2025.csv"), usecols=["RID", "VISDATE", "PTGENDER", "PTDOBYY", "PTEDUCAT"])
    df["VISDATE"] = pd.to_datetime(df["VISDATE"], errors="coerce")
    df["BIRTH_YEAR"] = pd.to_datetime(df["PTDOBYY"], errors="coerce").dt.year
    df = _to_num(df, ["PTGENDER", "PTEDUCAT"])
    df = df.sort_values("VISDATE").drop_duplicates("RID", keep="first")
    return df[["RID", "BIRTH_YEAR", "PTGENDER", "PTEDUCAT"]]


# --------------------------------------------------------------------------- #
# 4. Modality loaders (all qualifying visits, one date column)                 #
# --------------------------------------------------------------------------- #
def load_mri():
    cols = ["RID", "EXAMDATE", "FIELD_STRENGTH", "ST10CV",
            "ST29SV", "ST88SV", "ST24CV", "ST83CV", "ST12SV", "ST71SV"]
    df = pd.read_csv(_path("UCSFFSX7_12Dec2025.csv"), usecols=cols, low_memory=False)
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = _to_num(df, ["ST10CV", "ST29SV", "ST88SV", "ST24CV", "ST83CV", "ST12SV", "ST71SV"])
    df = df[df["EXAMDATE"].notna() & (df["ST10CV"] > 0)]

    # prefer 3T when a visit has both field strengths
    df["_3t"] = (df["FIELD_STRENGTH"] == "3T").astype(int)
    df = df.sort_values(["RID", "EXAMDATE", "_3t"], ascending=[True, True, False])
    df = df.drop_duplicates(["RID", "EXAMDATE"], keep="first")

    icv = df["ST10CV"]
    df["HIPPO_ICV"] = (df["ST29SV"] + df["ST88SV"]) / icv
    df["ENTORHINAL_ICV"] = (df["ST24CV"] + df["ST83CV"]) / icv
    df["AMYGDALA_ICV"] = (df["ST12SV"] + df["ST71SV"]) / icv
    return df[["RID", "EXAMDATE"] + MODALITY_COLS["mri"]]


def load_pet():
    cols = ["RID", "SCANDATE", "qc_flag", "CENTILOIDS", "SUMMARY_SUVR"]
    df = pd.read_csv(_path("UCBERKELEY_AMY_6MM_12Dec2025.csv"), usecols=cols, low_memory=False)
    df = df.rename(columns={"SCANDATE": "EXAMDATE"})
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = _to_num(df, ["CENTILOIDS", "SUMMARY_SUVR"])
    df = df[(df["qc_flag"] == 2) & df["EXAMDATE"].notna()]
    return df[["RID", "EXAMDATE"] + MODALITY_COLS["pet"]]


def load_csf():
    cols = ["RID", "EXAMDATE", "ABETA42", "TAU", "PTAU"]
    df = pd.read_csv(_path("UPENNBIOMK_ROCHE_ELECSYS_12Dec2025.csv"), usecols=cols)
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = _to_num(df, ["ABETA42", "TAU", "PTAU"])
    df = df[df["EXAMDATE"].notna()]
    return df[["RID", "EXAMDATE"] + MODALITY_COLS["csf"]]


# --------------------------------------------------------------------------- #
# Assemble the per-visit table                                                 #
# --------------------------------------------------------------------------- #
def build_dataset():
    base = build_base_rows()

    demog = load_demographics()
    base = base.merge(demog, on="RID", how="left")
    base["AGE"] = base["EXAMDATE"].dt.year - base["BIRTH_YEAR"]

    for loader in (load_mri, load_pet, load_csf):
        base = _asof_merge(base, loader())

    # per-patient forward then backward fill of the matched biomarkers
    base = base.sort_values(["RID", "EXAMDATE"])
    biomarker_cols = [c for cols in MODALITY_COLS.values() for c in cols]
    for c in biomarker_cols:
        base[c] = base.groupby("RID")[c].ffill()
        base[c] = base.groupby("RID")[c].bfill()

    return base.reset_index(drop=True), biomarker_cols


# --------------------------------------------------------------------------- #
# 7. Evaluation                                                                #
# --------------------------------------------------------------------------- #
def make_model():
    if MODEL == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    if MODEL == "linear":
        return Ridge()
    raise ValueError("MODEL must be 'rf' or 'linear'")


def _oof_predict(df, modalities):
    """Grouped-CV out-of-fold predictions; shared by evaluate() and the plot step."""
    feat_cols = (DEMOG_COLS if USE_DEMOGRAPHICS else []) + [
        c for m in modalities for c in MODALITY_COLS[m]
    ]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["RUL_YEARS"].to_numpy(dtype=float)
    groups = df["RID"].to_numpy()

    oof_pred = np.zeros(len(y))
    oof_base = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_te = X[tr].copy(), X[te].copy()

        # fill leftover NaN with training-fold medians
        med = np.nanmedian(X_tr, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X_tr = np.where(np.isnan(X_tr), med, X_tr)
        X_te = np.where(np.isnan(X_te), med, X_te)

        if MODEL == "linear":
            mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
            sd[sd == 0] = 1.0
            X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd

        model = make_model()
        model.fit(X_tr, y[tr])
        oof_pred[te] = model.predict(X_te)
        oof_base[te] = y[tr].mean()

    return feat_cols, y, oof_pred, oof_base


def evaluate(df, modalities):
    feat_cols, y, oof_pred, oof_base = _oof_predict(df, modalities)
    return {
        "modalities": "+".join(modalities) if modalities else "(demographics only)",
        "n_features": len(feat_cols),
        "n_rows": len(y),
        "MAE": mean_absolute_error(y, oof_pred),
        "RMSE": mean_squared_error(y, oof_pred) ** 0.5,
        "MAE_baseline": mean_absolute_error(y, oof_base),
    }


# --------------------------------------------------------------------------- #
# True vs. predicted RUL, a handful of patients concatenated                  #
# --------------------------------------------------------------------------- #
def sample_patient_series(df, y_true, y_pred, n_patients=PLOT_N_PATIENTS, random_state=RANDOM_STATE):
    """
    Pick n_patients patients at random and, for each, return their visits'
    true/predicted RUL in chronological order: [(rid, y_true_i, y_pred_i), ...].

    df, y_true and y_pred must share the same row order (df's positional index).
    """
    rng = np.random.RandomState(random_state)
    rids = df["RID"].unique()
    chosen = rng.choice(rids, size=min(n_patients, len(rids)), replace=False)

    series = []
    for rid in chosen:
        pos = np.flatnonzero((df["RID"] == rid).to_numpy())
        pos = pos[np.argsort(df["EXAMDATE"].to_numpy()[pos])]
        series.append((str(rid), y_true[pos], y_pred[pos]))
    return series


def plot_true_vs_pred_series(patient_series, out_path, title):
    """
    Concatenate each patient's true/predicted RUL sequence along the x axis
    (dashed lines mark patient boundaries) -- a quick visual read of how well
    a model tracks RUL over a handful of patient timelines.
    """
    true_vals, pred_vals, ticks, seps = [], [], [], []
    pos = 0
    for _, y_true_i, y_pred_i in patient_series:
        true_vals.extend(y_true_i.tolist() + [np.nan])
        pred_vals.extend(y_pred_i.tolist() + [np.nan])
        ticks.append(pos + (len(y_true_i) - 1) / 2)
        pos += len(y_true_i) + 1
        seps.append(pos - 1)

    x = np.arange(len(true_vals))
    plt.figure(figsize=(12, 4))
    plt.plot(x, true_vals, marker="o", label="True RUL", color="tab:blue")
    plt.plot(x, pred_vals, marker="o", label="Predicted RUL", color="tab:orange")
    for b in seps[:-1]:
        plt.axvline(b - 0.5, color="grey", linestyle="--", linewidth=0.7)
    plt.xticks(ticks, [rid for rid, _, _ in patient_series])
    plt.xlabel("patient (visits in chronological order, concatenated)")
    plt.ylabel("RUL (years)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    df, _ = build_dataset()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    n_conv = int(df["CONVERTED"].sum())
    print(f"rows: {len(df)}  patients: {df['RID'].nunique()}  "
          f"converter visits: {n_conv}  non-converter visits: {len(df) - n_conv}")
    print(f"model: {MODEL}   horizon: {HORIZON_LABEL}   match window: {MATCH_WINDOW_DAYS}d\n")

    experiments = EXPERIMENTS if RUN_ALL else [MODALITIES]
    results = pd.DataFrame(evaluate(df, m) for m in experiments)

    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(os.path.join(out_dir, "rul_results.csv"), index=False)

    _, y, oof_pred, _ = _oof_predict(df, MODALITIES)
    series = sample_patient_series(df, y, oof_pred)
    plot_true_vs_pred_series(
        series, os.path.join(out_dir, "rul_model_1_pred_vs_true.png"),
        f"Model 1 ({MODEL}, {'+'.join(MODALITIES)}): true vs. predicted RUL",
    )


if __name__ == "__main__":
    main()
