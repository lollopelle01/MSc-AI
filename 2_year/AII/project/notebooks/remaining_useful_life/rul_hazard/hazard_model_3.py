"""
notebooks/remaining_useful_life/rul_hazard/hazard_model_3.py

Classification variant of ../rul_model_3.py: same idea (a GRU/LSTM over each
patient's ordered visits, one prediction per timestep from everything seen so
far), but the target at every timestep is EVENT_AT_VISIT instead of
RUL_YEARS, and sequences are grouped by (RID, RUN_ID) rather than RID alone --
a patient who reverts to CN and later develops a fresh MCI episode gets two
separate sequences, since rate-of-change signal shouldn't be carried across
that gap (see hazard_panel.py's RUN_ID).

Loss is masked binary cross-entropy instead of masked MSE; padding, masking
and GroupKFold are otherwise unchanged from rul_model_3.py. Evaluation is AUC
+ a bucketed calibration gap over every real (non-padding) visit, instead of
MAE/RMSE.

Deriving RUL from this model needs its own forecast loop (forecast_hazard_seq
below), unlike hazard_model_1/2.py's hazard_panel.forecast_survival_curves:
the GRU is stateful, so projecting forward means literally stepping the
recurrent cell forward from the hidden state it reaches at a patient's last
real visit, one hypothetical future visit at a time, re-feeding its own
output at each step -- the same persistence approximation as the other two
hazard models, just applied one RNN step at a time instead of to a flat
feature matrix. Because of this extra cost, derived RUL here is only computed
from each sequence's own most recent real visit, not from every intermediate
one the way hazard_model_1/2.py do.

Run:  cd notebooks/remaining_useful_life/rul_hazard && python hazard_model_3.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

warnings.filterwarnings("ignore", message=".*ChainedAssignment.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rul_model_1 import DEMOG_COLS, MODALITY_COLS, plot_true_vs_pred_series  # noqa: E402
import hazard_panel as hp  # noqa: E402

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
MIN_VISITS = 2
USE_DEMOGRAPHICS = True
CELL = "gru"                 # "gru" | "lstm"
HIDDEN_SIZE = 32
EPOCHS = 60
LR = 5e-3
BATCH_SIZE = 64
N_SPLITS = 5
RANDOM_STATE = 0
RUL_METHOD = "median"
PLOT_N_PATIENTS = 15

RUN_ALL = True
MODALITIES = ["mri", "pet", "csf"]
EXPERIMENTS = [[], ["mri"], ["pet"], ["csf"], ["mri", "pet", "csf"]]

DAYS_PER_MONTH = 30.44


# --------------------------------------------------------------------------- #
# Reshape per-visit rows -> per-(RID, RUN_ID) sequences                        #
# --------------------------------------------------------------------------- #
def add_temporal_features(visits, biomarker_cols):
    """Add SINCE_PREV_GAP_MONTHS and d_<feat> (change since previous visit in
    the same clean MCI run) per row; first visit of each run gets 0 for both."""
    df = visits.sort_values(["RID", "RUN_ID", "EXAMDATE"]).reset_index(drop=True)
    prev = df.groupby(["RID", "RUN_ID"], sort=False).shift(1)

    df["SINCE_PREV_GAP_MONTHS"] = ((df["EXAMDATE"] - prev["EXAMDATE"]).dt.days / DAYS_PER_MONTH).fillna(0.0)
    for c in biomarker_cols:
        df["d_" + c] = (df[c] - prev[c]).fillna(0.0)

    n_visits = df.groupby(["RID", "RUN_ID"], sort=False)["EXAMDATE"].transform("size")
    return df[n_visits >= MIN_VISITS].reset_index(drop=True)


def feature_columns(modalities):
    cols = ["SINCE_PREV_GAP_MONTHS", "NEXT_GAP_MONTHS", "MCI_DURATION_MONTHS"] + \
           (DEMOG_COLS if USE_DEMOGRAPHICS else [])
    mod_feats = [c for m in modalities for c in MODALITY_COLS[m]]
    return cols + mod_feats + ["d_" + c for c in mod_feats]


def to_padded(df, feat_cols):
    """
    Returns X (P, T, F), y (P, T), mask (P, T), rid (P,), seq_key (list of
    (RID, RUN_ID)) -- P sequences, T the longest one, F features.
    """
    groups = list(df.groupby(["RID", "RUN_ID"], sort=False))
    T = max(len(g) for _, g in groups)
    F = len(feat_cols)

    X = np.zeros((len(groups), T, F), dtype=np.float32)
    y = np.zeros((len(groups), T), dtype=np.float32)
    mask = np.zeros((len(groups), T), dtype=np.float32)
    rid = np.empty(len(groups), dtype=np.int64)
    seq_key = []

    for i, (key, g) in enumerate(groups):
        L = len(g)
        X[i, :L] = g[feat_cols].to_numpy(np.float32)
        y[i, :L] = g["EVENT_AT_VISIT"].to_numpy(np.float32)
        mask[i, :L] = 1.0
        rid[i] = key[0]
        seq_key.append(key)
    return X, y, mask, rid, seq_key


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
    return (X * mask[..., None]).astype(np.float32)


# --------------------------------------------------------------------------- #
# The RNN -- outputs logits; sigmoid is applied where needed, not in here      #
# --------------------------------------------------------------------------- #
class SequenceHazard(nn.Module):
    def __init__(self, n_features, hidden, cell):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, h0=None):
        out, h = self.rnn(x, h0)          # (P, T, hidden), causal
        return self.head(out).squeeze(-1), h  # (P, T) logits


def train_model(X_tr, y_tr, m_tr):
    torch.manual_seed(RANDOM_STATE)
    model = SequenceHazard(X_tr.shape[-1], HIDDEN_SIZE, CELL)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    Xt, yt, mt = map(torch.from_numpy, (X_tr, y_tr, m_tr))
    for _ in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(Xt)).split(BATCH_SIZE):
            opt.zero_grad()
            logits, _ = model(Xt[idx])
            loss = (loss_fn(logits, yt[idx]) * mt[idx]).sum() / mt[idx].sum()
            loss.backward()
            opt.step()
    return model


def predict_proba(model, X_te):
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(X_te))
        return torch.sigmoid(logits).numpy()


# --------------------------------------------------------------------------- #
# Evaluation (patient-grouped CV, per-visit metrics)                           #
# --------------------------------------------------------------------------- #
def _oof_predict(df, modalities):
    feat_cols = feature_columns(modalities)
    X, y, mask, rid, seq_key = to_padded(df, feat_cols)

    oof = np.zeros_like(y)
    fold_info = []
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y[:, 0], rid):
        norm = fit_norm(X[tr], mask[tr])
        model = train_model(apply_norm(X[tr], mask[tr], norm), y[tr], mask[tr])
        oof[te] = predict_proba(model, apply_norm(X[te], mask[te], norm))
        fold_info.append((te, model, norm))

    return feat_cols, X, y, oof, mask, rid, seq_key, fold_info


def evaluate(df, modalities):
    feat_cols, X, y, oof, mask, rid, seq_key, fold_info = _oof_predict(df, modalities)
    m = mask.astype(bool)
    _, gap = hp.calibration_table(y[m], oof[m])
    return {
        "model": CELL,
        "modalities": "+".join(modalities) if modalities else "(gap + demographics only)",
        "n_features": len(feat_cols),
        "n_visits": int(m.sum()),
        "event_rate": y[m].mean(),
        "AUC": roc_auc_score(y[m], oof[m]),
        "calibration_gap": gap,
    }


# --------------------------------------------------------------------------- #
# Derived RUL: step the trained recurrent cell forward from its real last     #
# hidden state                                                                 #
# --------------------------------------------------------------------------- #
def forecast_hazard_seq(model, X_raw, mask, norm, feat_cols,
                         step_months=hp.STEP_MONTHS, max_steps=hp.MAX_STEPS):
    """
    Runs each sequence's real (normalised) visits through the RNN, one
    sequence at a time so a shorter sequence's padding never contaminates
    its final hidden state, then steps the cell forward max_steps more times:
    every feature held at its last real value (persistence) except
    NEXT_GAP_MONTHS, reset to step_months, and MCI_DURATION_MONTHS, advanced
    by step_months each step; every d_<feat> is set to 0, consistent with
    freezing the level (no further change). Returns S, (n_seqs, max_steps).
    """
    P, T, F = X_raw.shape
    X_norm = apply_norm(X_raw, mask, norm)
    lengths = mask.sum(axis=1).astype(int)

    gap_idx = feat_cols.index("NEXT_GAP_MONTHS")
    dur_idx = feat_cols.index("MCI_DURATION_MONTHS")
    delta_idx = [k for k, c in enumerate(feat_cols) if c.startswith("d_")]
    med, mu, sd = norm

    model.eval()
    last_raw = np.zeros((P, F), dtype=np.float32)
    if CELL == "gru":
        h = torch.zeros(1, P, HIDDEN_SIZE)
    else:
        h = (torch.zeros(1, P, HIDDEN_SIZE), torch.zeros(1, P, HIDDEN_SIZE))

    with torch.no_grad():
        for i in range(P):
            L = int(lengths[i])
            xi = torch.from_numpy(X_norm[i:i + 1, :L])
            _, hi = model(xi, h0=None)
            last_raw[i] = X_raw[i, L - 1]
            if CELL == "gru":
                h[:, i:i + 1] = hi
            else:
                h[0][:, i:i + 1] = hi[0]
                h[1][:, i:i + 1] = hi[1]

        base_duration = last_raw[:, dur_idx].copy()
        S = np.empty((P, max_steps))
        survival = np.ones(P)
        for k in range(1, max_steps + 1):
            step_raw = last_raw.copy()
            step_raw[:, gap_idx] = step_months
            step_raw[:, dur_idx] = base_duration + step_months * k
            step_raw[:, delta_idx] = 0.0
            step_norm = np.where(np.isnan(step_raw), med, step_raw)
            step_norm = ((step_norm - mu) / sd).astype(np.float32)
            x_step = torch.from_numpy(step_norm[:, None, :])
            logits, h = model(x_step, h0=h)
            hazard_k = torch.sigmoid(logits.squeeze(1)).numpy()
            survival = survival * (1.0 - hazard_k)
            S[:, k - 1] = survival
    return S


def derived_rul_years_last_visit(df, modalities):
    """
    Derived RUL from each sequence's own most recent real visit only (not
    every intermediate visit -- see module docstring for why).
    """
    feat_cols, X, y, oof, mask, rid, seq_key, fold_info = _oof_predict(df, modalities)
    rul_years = np.full(len(seq_key), np.nan)
    for te, model, norm in fold_info:
        S = forecast_hazard_seq(model, X[te], mask[te], norm, feat_cols)
        rul_years[te] = hp.survival_to_rul(S, method=RUL_METHOD) / 12.0
    return seq_key, rul_years


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    visits, biomarker_cols = hp.build_hazard_dataset()
    df = add_temporal_features(visits, biomarker_cols)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    seq_len = df.groupby(["RID", "RUN_ID"]).size()
    print(f"sequences: {len(seq_len)}  visits: {len(df)}  "
          f"sequence length: min {seq_len.min()}  median {int(seq_len.median())}  max {seq_len.max()}")
    print(f"cell: {CELL}   hidden: {HIDDEN_SIZE}   epochs: {EPOCHS}   forecast step: {hp.STEP_MONTHS}mo\n")

    experiments = EXPERIMENTS if RUN_ALL else [MODALITIES]
    results = pd.DataFrame(evaluate(df, m) for m in experiments)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(os.path.join(out_dir, "hazard_results_3.csv"), index=False)

    seq_key, rul_pred_years = derived_rul_years_last_visit(df, MODALITIES)
    last_visit = (df.sort_values("EXAMDATE").groupby(["RID", "RUN_ID"]).tail(1)
                  .set_index(["RID", "RUN_ID"]))
    lookup = last_visit.loc[seq_key]
    conv_mask = lookup["CONVERTED"].to_numpy()

    resolved = ~np.isnan(rul_pred_years) & conv_mask
    print(f"\nDerived RUL ({RUL_METHOD}) vs RUL_YEARS_TRUE, at each sequence's last real "
          f"visit, converter sequences only: resolved {resolved.sum()} / {int(conv_mask.sum())}")
    if resolved.any():
        mae = mean_absolute_error(lookup.loc[resolved, "RUL_YEARS_TRUE"], rul_pred_years[resolved])
        print(f"  MAE (resolved rows): {mae:.3f} years")

    # A single point per converter sequence (its last real visit), not a
    # trajectory -- unlike hazard_model_1/2.py's plots, see module docstring.
    rng = np.random.RandomState(RANDOM_STATE)
    conv_idx = np.flatnonzero(conv_mask)
    chosen = rng.choice(conv_idx, size=min(PLOT_N_PATIENTS, len(conv_idx)), replace=False)
    series = [
        (f"{seq_key[i][0]}/{seq_key[i][1]}",
         np.array([lookup.iloc[i]["RUL_YEARS_TRUE"]]),
         np.array([rul_pred_years[i]]))
        for i in chosen
    ]
    plot_true_vs_pred_series(
        series, os.path.join(out_dir, "hazard_model_3_pred_vs_true.png"),
        f"Hazard model 3 ({CELL}, {'+'.join(MODALITIES)}): true vs. derived RUL "
        f"(last visit of converter sequences)",
    )


if __name__ == "__main__":
    main()
