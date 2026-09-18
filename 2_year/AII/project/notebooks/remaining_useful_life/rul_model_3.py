"""
notebooks/remaining_useful_life/rul_model_3.py

Sequence variant of the simple RUL model: a small RNN over a patient's visits.

Models 1 and 2 treat every visit (or visit pair) as an independent row. Model 3
keeps each patient's visits together as an ordered sequence and runs a GRU over
it, predicting RUL at every visit from that visit plus all earlier ones.

Per-visit features (same idea as model 2 -- level *and* rate of change):

    <feat>     = biomarker value at this visit                 (level)
    d_<feat>   = value at this visit - value at previous visit  (change)
    GAP_MONTHS = months since the previous visit
    (first visit of each patient: d_* and GAP_MONTHS = 0)

Target at each timestep: RUL_YEARS at that visit. A unidirectional GRU means the
prediction at visit t only sees visits 1..t.

Dataset shaping (diagnosis timeline, modality matching, forward/backward fill)
is imported unchanged from rul_model_1; only the reshaping into padded
per-patient sequences is new here. Same modality switch: edit MODALITIES /
EXPERIMENTS to compare mri / pet / csf contributions.

Run:  cd leo && python rul_model_3.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

warnings.filterwarnings("ignore", message=".*ChainedAssignment.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from rul_model_1 import (
    build_dataset, MODALITY_COLS, DEMOG_COLS, HORIZON_YEARS, HORIZON_LABEL, MATCH_WINDOW_DAYS,
    PLOT_N_PATIENTS, plot_true_vs_pred_series,
)

PLOT_N_CONVERTERS = 20        # converters shown in the continuous true-time plot

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
MIN_VISITS = 2               # patients need at least this many visits to form a sequence
USE_DEMOGRAPHICS = True      # AGE, PTGENDER, PTEDUCAT as per-timestep context
CELL = "gru"                 # "gru" | "lstm"
HIDDEN_SIZE = 32
EPOCHS = 60
LR = 5e-3
BATCH_SIZE = 64
N_SPLITS = 5
RANDOM_STATE = 0

RUN_ALL = True               # True: loop EXPERIMENTS; False: single MODALITIES run
MODALITIES = ["mri", "pet", "csf"]
EXPERIMENTS = [[], ["mri"], ["pet"], ["csf"], ["mri", "pet", "csf"]]

DAYS_PER_MONTH = 30.44


# --------------------------------------------------------------------------- #
# Reshape per-visit rows -> per-patient sequences                              #
# --------------------------------------------------------------------------- #
def add_temporal_features(visits, biomarker_cols):
    """Add GAP_MONTHS and d_<feat> (change since previous visit) per row."""
    df = visits.sort_values(["RID", "EXAMDATE"]).reset_index(drop=True)
    prev = df.groupby("RID", sort=False).shift(1)

    df["GAP_MONTHS"] = ((df["EXAMDATE"] - prev["EXAMDATE"]).dt.days / DAYS_PER_MONTH).fillna(0.0)
    for c in biomarker_cols:
        df["d_" + c] = (df[c] - prev[c]).fillna(0.0)

    n_visits = df.groupby("RID", sort=False)["EXAMDATE"].transform("size")
    return df[n_visits >= MIN_VISITS].reset_index(drop=True)


def feature_columns(modalities):
    cols = ["GAP_MONTHS"] + (DEMOG_COLS if USE_DEMOGRAPHICS else [])
    mod_feats = [c for m in modalities for c in MODALITY_COLS[m]]
    return cols + mod_feats + ["d_" + c for c in mod_feats]  # levels + changes


def to_padded(df, feat_cols):
    """
    Pack the sequences into fixed-size arrays.

    Returns X (P, T, F), y (P, T), mask (P, T), rid (P,) -- P patients, T the
    longest sequence, F features. mask is 1 on real visits, 0 on padding.
    """
    groups = list(df.groupby("RID", sort=False))
    T = max(len(g) for _, g in groups)
    F = len(feat_cols)

    X = np.zeros((len(groups), T, F), dtype=np.float32)
    y = np.zeros((len(groups), T), dtype=np.float32)
    mask = np.zeros((len(groups), T), dtype=np.float32)
    rid = np.empty(len(groups), dtype=np.int64)

    for i, (r, g) in enumerate(groups):
        L = len(g)
        X[i, :L] = g[feat_cols].to_numpy(np.float32)
        y[i, :L] = g["RUL_YEARS"].to_numpy(np.float32)
        mask[i, :L] = 1.0
        rid[i] = r
    return X, y, mask, rid


# --------------------------------------------------------------------------- #
# Normalisation (fit on training visits only)                                  #
# --------------------------------------------------------------------------- #
def fit_norm(X, mask):
    valid = X.reshape(-1, X.shape[-1])[mask.reshape(-1).astype(bool)]
    med = np.nanmedian(valid, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    filled = np.where(np.isnan(valid), med, valid)
    mu = filled.mean(axis=0)
    sd = filled.std(axis=0)
    sd[sd == 0] = 1.0
    return med, mu, sd


def apply_norm(X, mask, norm):
    med, mu, sd = norm
    X = np.where(np.isnan(X), med, X)
    X = (X - mu) / sd
    return (X * mask[..., None]).astype(np.float32)   # re-zero padding


# --------------------------------------------------------------------------- #
# The RNN                                                                      #
# --------------------------------------------------------------------------- #
class SequenceRUL(nn.Module):
    def __init__(self, n_features, hidden, cell):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)              # (P, T, hidden), causal
        return self.head(out).squeeze(-1)  # (P, T)


def train_predict(X_tr, y_tr, m_tr, X_te):
    torch.manual_seed(RANDOM_STATE)
    model = SequenceRUL(X_tr.shape[-1], HIDDEN_SIZE, CELL)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    Xt, yt, mt = map(torch.from_numpy, (X_tr, y_tr, m_tr))
    for _ in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(Xt)).split(BATCH_SIZE):
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = ((pred - yt[idx]) ** 2 * mt[idx]).sum() / mt[idx].sum()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_te)).numpy()


# --------------------------------------------------------------------------- #
# Evaluation (patient-grouped CV, per-visit metrics)                           #
# --------------------------------------------------------------------------- #
def _oof_predict(df, modalities):
    feat_cols = feature_columns(modalities)
    X, y, mask, rid = to_padded(df, feat_cols)

    oof = np.zeros_like(y)
    base = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y[:, 0], rid):
        norm = fit_norm(X[tr], mask[tr])
        oof[te] = train_predict(apply_norm(X[tr], mask[tr], norm), y[tr], mask[tr],
                                apply_norm(X[te], mask[te], norm))
        base[te] = (y[tr] * mask[tr]).sum() / mask[tr].sum()

    return feat_cols, y, oof, base, mask, rid


def evaluate(df, modalities):
    feat_cols, y, oof, base, mask, rid = _oof_predict(df, modalities)
    m = mask.astype(bool)
    return {
        "model": CELL,
        "modalities": "+".join(modalities) if modalities else "(gap + demographics only)",
        "n_features": len(feat_cols),
        "n_visits": int(m.sum()),
        "MAE": mean_absolute_error(y[m], oof[m]),
        "RMSE": mean_squared_error(y[m], oof[m]) ** 0.5,
        "MAE_baseline": mean_absolute_error(y[m], base[m]),
    }


# --------------------------------------------------------------------------- #
# True vs. predicted RUL, a handful of patients concatenated                  #
# --------------------------------------------------------------------------- #
def sample_patient_series_seq(y, oof, mask, rid, n_patients=PLOT_N_PATIENTS, random_state=RANDOM_STATE):
    """
    Randomly pick n_patients patient sequences from the padded (P, T) arrays.
    Sequences are already in chronological order (see to_padded); mask trims
    off the padding.
    """
    rng = np.random.RandomState(random_state)
    chosen = rng.choice(len(rid), size=min(n_patients, len(rid)), replace=False)

    series = []
    for i in chosen:
        valid = mask[i].astype(bool)
        series.append((str(rid[i]), y[i, valid], oof[i, valid]))
    return series


def top_converter_series(df, y, oof, mask, rid, n_patients=PLOT_N_CONVERTERS):
    """
    Among converters only (CONVERTED == True), pick the n_patients with the
    longest visit sequences. For each, return (rid, elapsed_years, y_true,
    y_pred) where elapsed_years is time since that patient's first visit
    (cumulative GAP_MONTHS) -- a real time axis, not just a visit index.

    `df` must be the same (RID, EXAMDATE)-sorted frame that produced `rid`
    via to_padded, so groupby("RID", sort=False) yields groups in the same
    order as the padded arrays.
    """
    groups = dict(iter(df.groupby("RID", sort=False)))

    candidates = []
    for i, r in enumerate(rid):
        g = groups[r]
        if not bool(g["CONVERTED"].iloc[0]):
            continue
        candidates.append((int(mask[i].sum()), i, r, g))
    candidates.sort(key=lambda t: t[0], reverse=True)

    series = []
    for _, i, r, g in candidates[:n_patients]:
        valid = mask[i].astype(bool)
        elapsed_years = (g["GAP_MONTHS"].cumsum() / 12.0).to_numpy()
        series.append((str(r), elapsed_years, y[i, valid], oof[i, valid]))
    return series


def plot_top_converters_time_series(series, out_path, title, gap_years=1.0):
    """
    Concatenate each selected converter's true/predicted RUL along a real
    time axis (years since that patient's first visit): patients are laid
    out one after another with a small gap so the whole thing reads as one
    continuous line, with dashed verticals marking where one patient's
    history ends and the next begins.
    """
    x_all, true_all, pred_all, ticks, seps = [], [], [], [], []
    offset = 0.0
    for rid, t, y_true_i, y_pred_i in series:
        x = offset + (t - t[0])
        x_all.extend(x.tolist() + [np.nan])
        true_all.extend(y_true_i.tolist() + [np.nan])
        pred_all.extend(y_pred_i.tolist() + [np.nan])
        ticks.append((x[0] + x[-1]) / 2)
        offset = x[-1] + gap_years
        seps.append(offset - gap_years / 2)

    plt.figure(figsize=(16, 4))
    plt.plot(x_all, true_all, marker="o", markersize=3, label="True RUL", color="tab:blue")
    plt.plot(x_all, pred_all, marker="o", markersize=3, label="Predicted RUL", color="tab:orange")
    for b in seps[:-1]:
        plt.axvline(b, color="grey", linestyle="--", linewidth=0.7)
    plt.xticks(ticks, [rid for rid, *_ in series], rotation=90)
    plt.xlabel("elapsed time per patient (years since first visit), concatenated")
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
    visits, biomarker_cols = build_dataset()
    df = add_temporal_features(visits, biomarker_cols)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    seq_len = df.groupby("RID").size()
    print(f"patients: {df['RID'].nunique()}  visits: {len(df)}  "
          f"sequence length: min {seq_len.min()}  median {int(seq_len.median())}  max {seq_len.max()}")
    print(f"cell: {CELL}   hidden: {HIDDEN_SIZE}   epochs: {EPOCHS}   "
          f"horizon: {HORIZON_LABEL}   match window: {MATCH_WINDOW_DAYS}d\n")

    experiments = EXPERIMENTS if RUN_ALL else [MODALITIES]
    results = pd.DataFrame(evaluate(df, m) for m in experiments)

    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(os.path.join(out_dir, "rul_results_3.csv"), index=False)

    _, y, oof, _, mask, rid = _oof_predict(df, MODALITIES)
    series = sample_patient_series_seq(y, oof, mask, rid)
    plot_true_vs_pred_series(
        series, os.path.join(out_dir, "rul_model_3_pred_vs_true.png"),
        f"Model 3 ({CELL}, {'+'.join(MODALITIES)}): true vs. predicted RUL",
    )

    conv_series = top_converter_series(df, y, oof, mask, rid)
    plot_top_converters_time_series(
        conv_series, os.path.join(out_dir, "rul_model_3_top_converters_time.png"),
        f"Model 3 ({CELL}, {'+'.join(MODALITIES)}): true vs. predicted RUL over time -- "
        f"top {len(conv_series)} converters by sequence length",
    )


if __name__ == "__main__":
    main()
