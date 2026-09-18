"""
notebooks/remaining_useful_life/rul_hazard/hazard_model_1_cost_sensitive.py

Cost-sensitive variant of hazard_model_1.py -- approach 2b ("cost-weighted
retraining") from the calibration discussion in ../RUL_METHODS.md. Same
dataset, same features, same RandomForest/LogisticRegression choice (MODEL),
same patient-grouped 5-fold CV; the only thing that changes is class_weight.

hazard_model_1.py uses class_weight="balanced", which only corrects for the
event/non-event class imbalance (~8% event rate here, sklearn reweights by
inverse frequency, roughly 1:12). That has nothing to do with what a missed
event actually costs downstream. This script instead sets

    class_weight = {0: CHECK_COST, 1: MISSED_CONVERSION_COST}

reusing, unchanged, the same clinical cost assumptions
notebooks/decision_support/temporal_window/1_fixed_policy.ipynb documents in
its "The Cost Model" section:

    CHECK_COST                   = 1.0   (one routine clinical check)
    MISSED_CONVERSION_COST_BASE  = 20.0  (missed conversion, CN patient)
    MCI_OR_WORSE_COST_MULTIPLIER = 2.8   (missed conversion, MCI/Dementia
                                          patient -- costed higher since
                                          there is more ground to lose)

hazard_panel.build_hazard_base_rows keeps DIAGNOSIS == 2 (MCI) rows only, so
every row in this panel already falls in the "MCI or worse" bucket -- the
notebook's per-row stratification collapses to one constant here:
20.0 * 2.8 = 56.0 for every row, not a per-row DIAGNOSIS lookup.

This is Elkan's cost-proportionate weighting heuristic: reweighting the
training loss so that missing an event visit (a false negative) is penalized
exactly as expensive, relative to a false alarm, as the clinical cost model
says it is (56x here), rather than only by how rare it is in the data (12x
under "balanced"). It nudges the fitted decision boundary itself toward
minimizing expected clinical cost.

The trade-off this script exists to make visible, not hide: the resulting
probabilities are no longer calibrated in the statistical sense (see
RUL_METHODS.md's approach-1-vs-approach-2 discussion) -- hazard_panel.
calibration_table's gap is expected to get worse, because the classifier is
now deliberately biased toward over-predicting the costly class. What should
improve instead is the expected clinical cost of the classifier's own
decisions, computed below (expected_cost) the same way a false negative /
false positive would be priced by util.decision_util.ConversionCostModel,
and reported side by side against the class_weight="balanced" baseline for
every modality combination.

sweep_best_threshold below adds the natural follow-up question: retraining
with class_weight isn't the only way to act on the cost model -- approach 2a
(picking a cost-aware decision threshold on an already-trained, still
calibrated classifier) is the cheaper alternative, and needs to be compared
against, not assumed inferior to, class-weighted retraining. Both weightings
are evaluated at a naive threshold=0.5 *and* at their own swept cost-optimal
threshold, so the comparison is fair in both directions: a class_weight
retrain evaluated only at 0.5 is undersold (0.5 stops being a meaningful
cutoff once the classes are reweighted), and the "balanced" baseline
evaluated only at 0.5 is equally undersold, for the same reason absent any
weighting at all -- 0.5 was never chosen with this cost model in mind either.
On this dataset the two weightings' swept optima land within noise of each
other, because the classifiers rank visits almost identically (AUC barely
moves between them) -- reweighting the loss and moving the threshold after
the fact are two different routes to approximating the same Bayes-optimal
decision boundary, and when the underlying ranking is the same, both routes
converge. That makes plain thresholding (2a) the better default here: same
achievable cost, without paying calibration's price -- see RUL_METHODS.md
section 10 for the full comparison.

Run:  cd notebooks/remaining_useful_life/rul_hazard && python hazard_model_1_cost_sensitive.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*ChainedAssignment.*", category=FutureWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rul_model_1 import sample_patient_series, plot_true_vs_pred_series  # noqa: E402
import hazard_panel as hp  # noqa: E402
from hazard_model_1 import feature_columns, _impute_scale, RUL_METHOD  # noqa: E402

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
MODEL = "forest"             # "forest" (RandomForestClassifier) | "logistic" (LogisticRegression)
N_SPLITS = 5
RANDOM_STATE = 0

# Same clinical cost assumptions as notebooks/decision_support/temporal_window
# (1_fixed_policy.ipynb, "The Cost Model"). Every row here is an MCI visit, so
# the notebooks' MCI_OR_WORSE_COST_MULTIPLIER applies to all of them, not a
# fraction picked out by DIAGNOSIS the way the notebooks' stratified helper
# does over a mixed-diagnosis cohort.
CHECK_COST = 1.0
MISSED_CONVERSION_COST_BASE = 20.0
MCI_OR_WORSE_COST_MULTIPLIER = 2.8
MISSED_CONVERSION_COST = MISSED_CONVERSION_COST_BASE * MCI_OR_WORSE_COST_MULTIPLIER  # 56.0

DECISION_THRESHOLD = 0.5     # naive predict_proba cutoff, reported alongside the swept optimum below
THRESHOLD_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 0.999, 300)])

RUN_ALL = True
MODALITIES = ["mri", "pet", "csf"]
EXPERIMENTS = [[], ["mri"], ["pet"], ["csf"], ["mri", "pet", "csf"]]

WEIGHTINGS = ["balanced", "cost_sensitive"]


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
def make_model(weighting):
    if weighting == "balanced":
        class_weight = "balanced"
    elif weighting == "cost_sensitive":
        class_weight = {0: CHECK_COST, 1: MISSED_CONVERSION_COST}
    else:
        raise ValueError("weighting must be 'balanced' or 'cost_sensitive'")

    if MODEL == "forest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                       n_jobs=-1, class_weight=class_weight)
    if MODEL == "logistic":
        return LogisticRegression(penalty="l1", solver="liblinear",
                                   class_weight=class_weight, max_iter=2000)
    raise ValueError("MODEL must be 'forest' or 'logistic'")


def _oof_predict(df, modalities, weighting):
    """Same shape as hazard_model_1._oof_predict, parameterized by weighting."""
    feat_cols = feature_columns(modalities)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["EVENT_AT_VISIT"].to_numpy(dtype=float)
    groups = df["RID"].to_numpy()
    scale = (MODEL == "logistic")

    oof_prob = np.zeros(len(y))
    fold_info = []
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        X_tr, X_te, med, scaler = _impute_scale(X[tr].copy(), X[te].copy(), scale)
        model = make_model(weighting)
        model.fit(X_tr, y[tr])
        oof_prob[te] = model.predict_proba(X_te)[:, 1]

        def predict_fn(step_X, model=model, med=med, scaler=scaler):
            step_X = np.where(np.isnan(step_X), med, step_X)
            if scaler is not None:
                mu, sd = scaler
                step_X = (step_X - mu) / sd
            return model.predict_proba(step_X)[:, 1]

        fold_info.append((te, predict_fn))

    return feat_cols, y, oof_prob, fold_info, X


# --------------------------------------------------------------------------- #
# Expected classification cost -- the metric class_weight is trained against  #
# --------------------------------------------------------------------------- #
def expected_cost(y_true, y_prob, threshold=DECISION_THRESHOLD):
    """
    Mean per-visit clinical cost of the classifier's own decision at
    `threshold`: a missed event (y_true=1, predicted 0) costs
    MISSED_CONVERSION_COST, a false alarm (y_true=0, predicted 1) costs one
    extra CHECK_COST. This is the classification-level analogue of what
    util.decision_util.ConversionCostModel prices for a chosen check
    interval downstream -- not the same computation (no interval_months
    here), but the same false-negative / false-positive price list, scoring
    the over/under-triggering trade-off class_weight is trained against
    directly, in cost units instead of AUC or calibration gap.
    """
    y_pred = (y_prob >= threshold).astype(float)
    false_negatives = (y_true == 1) & (y_pred == 0)
    false_positives = (y_true == 0) & (y_pred == 1)
    total = false_negatives.sum() * MISSED_CONVERSION_COST + false_positives.sum() * CHECK_COST
    return total / len(y_true), int(false_negatives.sum()), int(false_positives.sum())


def sweep_best_threshold(y_true, y_prob, thresholds=THRESHOLD_GRID):
    """
    Approach 2a: the cost-optimal decision threshold for an already-trained
    classifier, found by brute-force search over `thresholds` rather than
    retraining anything. Returns (best_cost, best_threshold, n_fn, n_fp) at
    that threshold -- the number this script's whole point is to compare
    against class-weighted retraining's own expected_cost, not assume one
    beats the other without measuring it.
    """
    best = None
    for t in thresholds:
        cost, fn, fp = expected_cost(y_true, y_prob, threshold=t)
        if best is None or cost < best[0]:
            best = (cost, t, fn, fp)
    return best


def evaluate(df, modalities, weighting):
    feat_cols, y, oof_prob, _, _ = _oof_predict(df, modalities, weighting)
    _, gap = hp.calibration_table(y, oof_prob)
    cost_fixed, fn_fixed, fp_fixed = expected_cost(y, oof_prob, threshold=DECISION_THRESHOLD)
    cost_best, t_best, fn_best, fp_best = sweep_best_threshold(y, oof_prob)
    return {
        "weighting": weighting,
        "modalities": "+".join(modalities) if modalities else "(demographics only)",
        "n_features": len(feat_cols),
        "n_rows": len(y),
        "event_rate": y.mean(),
        "AUC": roc_auc_score(y, oof_prob),
        "calibration_gap": gap,
        "missed_events@0.5": fn_fixed,
        "false_alarms@0.5": fp_fixed,
        "cost@0.5": cost_fixed,
        "best_threshold": t_best,
        "missed_events@best": fn_best,
        "false_alarms@best": fp_best,
        "cost@best": cost_best,
    }


# --------------------------------------------------------------------------- #
# Derived RUL, validated against RUL_YEARS_TRUE on converters                 #
# --------------------------------------------------------------------------- #
def derived_rul_years(df, modalities, weighting):
    feat_cols, _, _, fold_info, X = _oof_predict(df, modalities, weighting)
    rul_months = np.full(len(df), np.nan)
    for te, predict_fn in fold_info:
        S = hp.forecast_survival_curves(predict_fn, X[te], feat_cols)
        result = hp.survival_to_rul(S, method=RUL_METHOD)
        rul_months[te] = result[0] if RUL_METHOD == "expected" else result
    return rul_months / 12.0


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    df, _ = hp.build_hazard_dataset()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"MCI visits: {len(df)}  patients: {df['RID'].nunique()}  "
          f"event visits: {int(df['EVENT_AT_VISIT'].sum())}")
    print(f"model: {MODEL}   check_cost: {CHECK_COST}   "
          f"missed_conversion_cost (MCI/Dementia): {MISSED_CONVERSION_COST}   "
          f"cost ratio: {MISSED_CONVERSION_COST / CHECK_COST:.0f}:1\n")

    experiments = EXPERIMENTS if RUN_ALL else [MODALITIES]
    rows = []
    for weighting in WEIGHTINGS:
        rows += [evaluate(df, m, weighting) for m in experiments]
    results = pd.DataFrame(rows)

    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(os.path.join(out_dir, "hazard_results_1_cost_sensitive.csv"), index=False)

    # The comparison this whole script is for: class-weighted retraining
    # (approach 2b) vs. plain threshold-sweeping on top of it (approach 2a),
    # read off the mri+pet+csf row already computed above -- no retraining.
    full = results[results["modalities"] == "+".join(MODALITIES)]
    print(f"\nApproach 2a vs 2b, {'+'.join(MODALITIES)}:")
    for _, r in full.iterrows():
        print(f"  [{r['weighting']:>14}] @0.5: cost={r['cost@0.5']:.3f} "
              f"(fn={int(r['missed_events@0.5'])}, fp={int(r['false_alarms@0.5'])})   "
              f"@best(t={r['best_threshold']:.4f}): cost={r['cost@best']:.3f} "
              f"(fn={int(r['missed_events@best'])}, fp={int(r['false_alarms@best'])})")

    for weighting in WEIGHTINGS:
        rul_pred_years = derived_rul_years(df, MODALITIES, weighting)
        conv = df[df["CONVERTED"]].copy()
        conv["RUL_PRED_YEARS"] = rul_pred_years[df["CONVERTED"].to_numpy()]
        resolved = conv["RUL_PRED_YEARS"].notna()

        print(f"\n[{weighting}] derived RUL vs RUL_YEARS_TRUE, converter visits only: "
              f"resolved {resolved.sum()} / {len(conv)} ({resolved.mean():.1%})")
        if resolved.any():
            mae = mean_absolute_error(conv.loc[resolved, "RUL_YEARS_TRUE"], conv.loc[resolved, "RUL_PRED_YEARS"])
            print(f"  MAE (resolved rows): {mae:.3f} years")

        conv = conv.reset_index(drop=True)
        series = sample_patient_series(conv, conv["RUL_YEARS_TRUE"].to_numpy(), conv["RUL_PRED_YEARS"].to_numpy())
        plot_true_vs_pred_series(
            series, os.path.join(out_dir, f"hazard_model_1_cost_sensitive_{weighting}_pred_vs_true.png"),
            f"Hazard model 1, {weighting} weighting (mri+pet+csf): true vs. derived RUL (converters only)",
        )


if __name__ == "__main__":
    main()
