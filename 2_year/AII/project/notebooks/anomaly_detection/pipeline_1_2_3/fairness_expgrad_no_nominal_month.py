"""
notebooks/anomaly_detection/pipeline_1_2_3/fairness_expgrad_no_nominal_month.py

Variant of fairness_expgrad_pipeline.py testing the fix proposed but never
attempted in README.md Section 10: the fairness-corrected model (SHAP audit
in fairness_expgrad_no_nominal_month_report.txt) leans on NOMINAL_MONTH -- visit timing, not
a biomarker -- as its #1 driver (0.0166), while SUMMARY_SUVR collapses
~97% (0.0102 -> 0.0003) relative to the plain model. Swapping the
constraint to EqualizedOdds (fairness_expgrad_equalized_odds.py) did not
fix this and gave up most of the fairness gain (DIDI 2.58 vs 1.25).

fairlearn's ExponentiatedGradient has no native "constrain the reduction
but keep the feature for the base predictor" option -- the reduction
reweights/relabels training instances for the SAME base_est.fit(X, y);
it does not select features for the constraint separately from the
predictor. So this script implements the blunter, directly testable
version of the README's proposal: NOMINAL_MONTH is removed from every
tier's feature list entirely (CORE / EXTENDED / *_A and PROTECTED are
otherwise untouched), so neither the reduction nor the base RandomForest
can use it as a lever. This is a real simplification of the README's
more surgical two-model proposal -- reported as such, not as the exact
fix described there.

Replaces the sklearn post-hoc group-shift (fairness_correction.py) with the
in-training fix identified as more principled in
fairness_correction_reduction.py: fairlearn's ExponentiatedGradient
(Agarwal et al. 2018), wrapping the same class-balanced RandomForest used
everywhere else in this project, under a DemographicParity constraint
(eps=0.01 -- the setting that gave a 79% DIDI reduction on the worst tier
in the earlier eps sweep).

Retrains EVERY tier in the soft-fallback cascade this way (not just the
worst one), rebuilds RISK_SCORE from the resulting fair fallback, evaluates
it exactly like fairness_correction.py did (cost / DIDI / catch rate,
before vs after), and then runs the SAME SHAP explainability audit as
explainability_fairness.py -- but on the fair models, not the plain ones --
so the explainability and fairness mechanisms are now the same underlying
model, not two separate stories.

SHAP on a fairlearn mixture: ExponentiatedGradient's fitted model is a
randomized mixture of several RandomForestClassifier instances
(`.predictors_`, indexed the same as `.weights_`), and its predicted
probability is the WEIGHTED AVERAGE of each component's predict_proba.
Shapley values are linear under model averaging (phi(sum w_k f_k) =
sum w_k phi(f_k) for explanations sharing the same feature space), so a
weighted sum of each nonzero-weight component's exact TreeExplainer SHAP
values (and base values) is the exact SHAP decomposition of the mixture's
prediction -- not an approximation of convenience.

Usage:
    python3 fairness_expgrad_pipeline.py
Outputs (written next to this script):
    - fairness_expgrad_no_nominal_month_report.txt
    - shap_summary_extended_anomaly_no_nominal_month.png
    - shap_summary_core_no_nominal_month.png
    - shap_dependence_anomaly_score_no_nominal_month.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, ErrorRate
import pickle

HERE_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ckpt_no_nominal_month")
os.makedirs(HERE_CKPT, exist_ok=True)


def _ckpt_path(name):
    return os.path.join(HERE_CKPT, name + ".pkl")


def load_ckpt(name):
    fp = _ckpt_path(name)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            return pickle.load(f)
    return None


def save_ckpt(name, obj):
    with open(_ckpt_path(name), "wb") as f:
        pickle.dump(obj, f)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.append(REPO_ROOT)
from util import decision_util as du  # noqa: E402
from util import hazard_util as hu  # noqa: E402

DATA_PATH = os.path.join(REPO_ROOT, "datasets", "final.csv")
ANOMALY_KEYED_PATH = os.path.join(REPO_ROOT, "notebooks", "anomaly_detection", "method_b_autoencoder_hi",
                                    "pca_hi_trajectories_keyed.csv")

CORE = ["HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM", "SUMMARY_SUVR", "AGE", "PRIOR_DIAGNOSIS"]  # NOMINAL_MONTH removed -- see docstring
EXTENDED = CORE + ["TAU", "PTAU"]
CORE_A = CORE + ["ANOMALY_SCORE"]
EXTENDED_A = EXTENDED + ["ANOMALY_SCORE"]
PROTECTED = ["PTGENDER", "PTEDUCAT_BUCKET", "PTMARRY"]
EG_EPS = 0.01
EG_MAX_ITER = 30
EG_N_ESTIMATORS = 150
SHAP_SAMPLE_SIZE = 400  # subsample test rows for SHAP -- speed, not signal

CHECK_COST = 1.0
MISSED_CONVERSION_COST = 20.0
REFERENCE_INTERVAL_MONTHS = 12.0
SAFE_INTERVAL_MONTHS = 6


def make_protected(df):
    return {
        "PTGENDER": (1, 2),
        "PTEDUCAT_BUCKET": (0, 1),
        "PTMARRY": tuple(sorted(df["PTMARRY"].dropna().unique())),
    }


def load_panel():
    data = pd.read_csv(DATA_PATH)
    biomarker_cols_to_fill = ["HIPPO_NORM", "ENTORHINAL_NORM", "AMYGDALA_NORM", "SUMMARY_SUVR", "ABETA_RATIO", "TAU", "PTAU"]
    data = du.forward_fill_by_patient(data, biomarker_cols_to_fill, id_col="RID", date_col="EXAMDATE_DX")
    anomaly = pd.read_csv(ANOMALY_KEYED_PATH, usecols=["RID", "VISCODE2_norm", "HI"])
    anomaly = anomaly.dropna(subset=["VISCODE2_norm"]).rename(columns={"HI": "ANOMALY_SCORE"}).drop_duplicates(["RID", "VISCODE2_norm"])
    data = data.merge(anomaly, on=["RID", "VISCODE2_norm"], how="left")
    data = hu.add_nominal_month(data)
    panel = hu.build_hazard_panel(data)
    panel["PTEDUCAT_BUCKET"] = du.bucket_educat(panel["PTEDUCAT"], split_at=16)
    return panel


def at_risk_complete(df, cols):
    return df[df["AT_RISK"]].dropna(subset=cols)


class ExpGradWrapper:
    """Thin wrapper so the fallback cascade can call .predict_proba(X) on an
    ExponentiatedGradient just like it already does on a plain forest, and
    so downstream SHAP code can reach the nonzero-weight component models."""

    def __init__(self, model, cols):
        self.model = model
        self.cols = cols

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.cols)
        return self.model._pmf_predict(X)

    def nonzero_components(self):
        w = self.model.weights_
        nz = w[w > 1e-9]
        return [(self.model.predictors_[i], nz[i]) for i in nz.index]


def fit_expgrad(train_df, cols, protected_cols):
    base_est = RandomForestClassifier(n_estimators=EG_N_ESTIMATORS, max_depth=6, class_weight="balanced", random_state=42)
    # objective=ErrorRate(costs=...) matches the SAME clinical cost asymmetry
    # (missing a conversion costs 20x a false alarm) that class_weight="balanced"
    # approximates for the plain forests elsewhere in this project. Without it,
    # ExponentiatedGradient's default 0/1-loss objective ignores that asymmetry
    # entirely and collapses catch rate (94.9% -> 76-78% in the first pass) --
    # not a fairness/accuracy tradeoff, just an unaligned training objective.
    eg = ExponentiatedGradient(
        estimator=base_est, constraints=DemographicParity(),
        objective=ErrorRate(costs={"fp": 1.0, "fn": MISSED_CONVERSION_COST / CHECK_COST}),
        eps=EG_EPS, max_iter=EG_MAX_ITER,
    )
    eg.fit(train_df[cols], train_df["EVENT_AT_VISIT"], sensitive_features=train_df[protected_cols])
    return ExpGradWrapper(eg, cols)


N_CALIBRATION_PASSES = 8


def fit_group_shifts(df, score_col, protected, n_passes=N_CALIBRATION_PASSES):
    """Same group-mean recalibration as fairness_correction.py, reused here
    to close the cross-tier residual on top of the EG fallback instead of
    on top of a plain one."""
    scores = df[score_col].to_numpy(dtype=float).copy()
    global_mean = scores.mean()
    schedule = []
    for _ in range(n_passes):
        for attr, domain in protected.items():
            for v in domain:
                mask = df[attr].to_numpy() == v
                if mask.sum() == 0:
                    continue
                shift = global_mean - scores[mask].mean()
                scores[mask] += shift
                schedule.append((attr, v, shift))
    return schedule


def apply_group_shifts(df, score_col, schedule):
    scores = df[score_col].to_numpy(dtype=float).copy()
    for attr, v, shift in schedule:
        mask = df[attr].to_numpy() == v
        scores[mask] += shift
    return np.clip(scores, 0.0, 1.0)


def evaluate_policy(df, risk_col, lines, label):
    cmodel = du.ConversionCostModel(
        check_cost=CHECK_COST, missed_conversion_cost=MISSED_CONVERSION_COST,
        safe_interval_months=SAFE_INTERVAL_MONTHS, reference_interval_months=REFERENCE_INTERVAL_MONTHS,
    )
    recommended = du.recommend_interval(
        df[risk_col].values, interval_menu_months=(3, 6, 12),
        check_cost=CHECK_COST, missed_conversion_cost=MISSED_CONVERSION_COST,
        reference_interval_months=REFERENCE_INTERVAL_MONTHS,
    )
    total_cost, _, _ = cmodel.cost(
        rid_ids=df["RID"].values, risk_scores=df[risk_col].values,
        threshold=0.5, interval_months=recommended, return_margin=False,
    )
    didi = du.compute_didi(df, recommended, make_protected(df))
    outcomes = du.compute_diagnosis_worsening(df, id_col="RID", date_col="EXAMDATE_DX", diagnosis_col="DIAGNOSIS")
    outcomes["RECOMMENDED_INTERVAL"] = recommended
    worsened = outcomes["HAS_NEXT_VISIT"] & outcomes["DIAGNOSIS_WORSENED_NEXT"]
    margin = outcomes.loc[worsened, "NEXT_VISIT_GAP_MONTHS"] - outcomes.loc[worsened, "RECOMMENDED_INTERVAL"]
    catch_rate = (margin >= 0).mean() * 100
    lines.append(f"  {label:28s} n={len(df):5d}  cost={total_cost:10.1f}  DIDI={didi:.3f}  catch={catch_rate:5.1f}%")


def weighted_shap(wrapper, X):
    """Exact SHAP for a fairlearn mixture: weighted sum of each nonzero
    component's TreeExplainer SHAP values and base value."""
    total_sv = np.zeros((len(X), X.shape[1]))
    total_base = 0.0
    for predictor, w in wrapper.nonzero_components():
        explainer = shap.TreeExplainer(predictor)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1] if len(np.atleast_1d(base)) > 1 else base[0]
        total_sv += w * sv
        total_base += w * base
    return total_sv, total_base


def global_importance(sv, cols):
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [(cols[i], mean_abs[i]) for i in order]


def main():
    panel = load_panel()
    train_panel, test_panel = du.subject_train_test_split(panel, test_fraction=0.25, random_state=42)
    feature_sets = {"core": CORE, "extended": EXTENDED, "core+anomaly": CORE_A, "extended+anomaly": EXTENDED_A}

    lines = []
    lines.append("=== Fairness + explainability mechanism: ExponentiatedGradient throughout ===")
    lines.append(f"fairlearn.reductions.ExponentiatedGradient(DemographicParity, eps={EG_EPS}, max_iter={EG_MAX_ITER})")
    lines.append("wraps RandomForestClassifier(n_estimators=%d, max_depth=6, class_weight='balanced') -- retrained" % EG_N_ESTIMATORS)
    lines.append("for every tier in the fallback cascade, replacing the plain-forest + post-hoc-shift combo.")
    lines.append("")

    print("Fitting ExponentiatedGradient for each tier...", flush=True)
    fitted = {}
    for tier_name, cols in feature_sets.items():
        train_df = at_risk_complete(train_panel, cols + PROTECTED)
        if len(train_df) < 30:
            continue
        cached = load_ckpt(f"tier_{tier_name}")
        if cached is not None:
            fitted[tier_name] = cached
            print(f"  tier={tier_name} loaded from checkpoint.", flush=True)
            continue
        print(f"  tier={tier_name} n_train={len(train_df)} ...", flush=True)
        fitted[tier_name] = fit_expgrad(train_df, cols, PROTECTED)
        save_ckpt(f"tier_{tier_name}", fitted[tier_name])
        print(f"  tier={tier_name} done, checkpointed.", flush=True)

    # fallback logistic for 'core' tier as last resort (kept plain -- tiny
    # marginal population, not worth another EG fit)
    train_core = at_risk_complete(train_panel, CORE)
    scaler = StandardScaler().fit(train_core[CORE])
    logistic = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, class_weight="balanced", random_state=42)
    logistic.fit(scaler.transform(train_core[CORE]), train_core["EVENT_AT_VISIT"])

    FALLBACK_ORDER = [
        ("extended+anomaly", "expgrad"), ("extended", "expgrad"),
        ("core+anomaly", "expgrad"), ("core", "expgrad"), ("core", "logistic"),
    ]
    panel_fb = panel.copy()
    panel_fb["RISK_SCORE"] = np.nan
    panel_fb["RISK_SCORE_TIER"] = None
    remaining = panel_fb["RISK_SCORE"].isna()
    for tier_name, kind in FALLBACK_ORDER:
        cols = feature_sets.get(tier_name, CORE)
        eligible = remaining & panel_fb[cols].notna().all(axis=1)
        if not eligible.any():
            continue
        X = panel_fb.loc[eligible, cols]
        if kind == "expgrad":
            proba = fitted[tier_name].predict_proba(X)[:, 1]
        else:
            proba = logistic.predict_proba(scaler.transform(X))[:, 1]
        panel_fb.loc[eligible, "RISK_SCORE"] = proba
        panel_fb.loc[eligible, "RISK_SCORE_TIER"] = f"{tier_name}/{kind}"
        remaining = panel_fb["RISK_SCORE"].isna()

    scoreable_fb = panel_fb.dropna(subset=["RISK_SCORE"]).copy()
    scoreable_fb["PTEDUCAT_BUCKET"] = du.bucket_educat(scoreable_fb["PTEDUCAT"], split_at=16)
    coverage = len(scoreable_fb) / len(panel_fb) * 100
    lines.append(f"Coverage of the ExponentiatedGradient fallback: {coverage:.1f}% "
                 f"({len(scoreable_fb)}/{len(panel_fb)} panel rows) -- same cascade structure as before.")
    lines.append("")

    train_fb, test_fb = du.split_by_rid_membership(
        scoreable_fb, set(train_panel["RID"]), set(test_panel["RID"])
    )  # fixed: re-splitting this filtered subset independently leaked most "test"
       # patients from train_panel -- see split_by_rid_membership docstring.

    lines.append("-- Policy evaluation: original plain-forest fallback vs. this all-ExponentiatedGradient fallback --")
    evaluate_policy(train_fb, "RISK_SCORE", lines, "Train, ExpGrad fallback")
    evaluate_policy(test_fb, "RISK_SCORE", lines, "Test, ExpGrad fallback")
    lines.append("(compare to wire_full_pipeline.py's plain fallback: Test cost=10825.8 DIDI=8.570 catch=94.9%,")
    lines.append(" and fairness_correction.py's post-hoc-shifted version: Test cost=10782.1 DIDI=3.153 catch=93.2%)")
    lines.append("")

    # -------------------------------------------------------------
    # Closing the residual gap: each tier satisfies DemographicParity on
    # its OWN data, but didi_breakdown.py already found that which tier a
    # patient lands in is itself correlated with protected attributes
    # (DIDI of tier assignment: 2.923) -- a cross-tier effect no single-tier
    # retrain removes. Layer the cheap post-hoc group-shift from
    # fairness_correction.py on top of the already-fairer EG score to clean
    # up specifically that residual, rather than tightening eps further and
    # paying more catch-rate cost for a problem eps can't reach.
    # -------------------------------------------------------------
    lines.append("-- Closing the residual (cross-tier) gap: post-hoc shift ON TOP of the EG fallback --")
    protected_full = make_protected(train_fb)
    schedule = fit_group_shifts(train_fb, "RISK_SCORE", protected_full)
    train_fb["RISK_SCORE_FAIR"] = apply_group_shifts(train_fb, "RISK_SCORE", schedule)
    test_fb["RISK_SCORE_FAIR"] = apply_group_shifts(test_fb, "RISK_SCORE", schedule)
    evaluate_policy(train_fb, "RISK_SCORE_FAIR", lines, "Train, EG + post-hoc")
    evaluate_policy(test_fb, "RISK_SCORE_FAIR", lines, "Test, EG + post-hoc")
    lines.append("This combination keeps ExponentiatedGradient as the substrate the SHAP audit below")
    lines.append("explains (the retrain that changed the model's actual reasoning, and that ANOMALY_SCORE")
    lines.append("and the proxy checks below describe), and uses the cheap group-shift only for the")
    lines.append("residual cross-tier disparity a single-tier constraint structurally cannot reach --")
    lines.append("the same honest layering explainability_fairness.py's Part 3 flagged as a limitation")
    lines.append("of the shift ALONE, now applied on top of a fairer base instead of a plain one.")
    lines.append("")

    # -------------------------------------------------------------
    # Explainability on the FAIR model itself (not a separate story)
    # -------------------------------------------------------------
    print("Running SHAP on the ExponentiatedGradient mixture (best tier)...", flush=True)
    test_ea = at_risk_complete(test_panel, EXTENDED_A + PROTECTED)
    if len(test_ea) > SHAP_SAMPLE_SIZE:
        test_ea = test_ea.sample(n=SHAP_SAMPLE_SIZE, random_state=42)
    _ckpt_ea = load_ckpt("shap_ea")
    if _ckpt_ea is not None:
        sv_ea, base_ea, test_ea = _ckpt_ea
        print("  shap_ea loaded from checkpoint.", flush=True)
    else:
        sv_ea, base_ea = weighted_shap(fitted["extended+anomaly"], test_ea[EXTENDED_A])
        save_ckpt("shap_ea", (sv_ea, base_ea, test_ea))
    imp_ea = global_importance(sv_ea, EXTENDED_A)
    lines.append("-- SHAP on the fair (ExponentiatedGradient) extended+anomaly model --")
    lines.append("Global mean |SHAP| feature importance (test split), weighted across mixture components:")
    for feat, val in imp_ea:
        marker = "  <-- point 1's own signal" if feat == "ANOMALY_SCORE" else ""
        lines.append(f"    {feat:20s} {val:.4f}{marker}")
    anomaly_rank = [f for f, _ in imp_ea].index("ANOMALY_SCORE") + 1
    lines.append(f"ANOMALY_SCORE rank: {anomaly_rank} of {len(EXTENDED_A)} (fair model)")
    lines.append("")

    plt.figure()
    shap.summary_plot(sv_ea, test_ea[EXTENDED_A], plot_type="bar", show=False)
    plt.title("Global feature importance (fair model) -- extended+anomaly")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "shap_summary_extended_anomaly_no_nominal_month.png"), dpi=150)
    plt.close()

    plt.figure()
    anomaly_idx = EXTENDED_A.index("ANOMALY_SCORE")
    exp_obj = shap.Explanation(values=sv_ea, base_values=np.full(len(sv_ea), base_ea),
                                data=test_ea[EXTENDED_A].values, feature_names=EXTENDED_A)
    shap.plots.scatter(exp_obj[:, anomaly_idx], show=False)
    plt.title("ANOMALY_SCORE: SHAP value vs. feature value (fair model)")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "shap_dependence_anomaly_score_no_nominal_month.png"), dpi=150)
    plt.close()

    print("Running SHAP on the ExponentiatedGradient mixture (worst-DIDI tier)...", flush=True)
    test_core = at_risk_complete(test_panel, CORE + PROTECTED)
    if len(test_core) > SHAP_SAMPLE_SIZE:
        test_core = test_core.sample(n=SHAP_SAMPLE_SIZE, random_state=42)
    _ckpt_core = load_ckpt("shap_core")
    if _ckpt_core is not None:
        sv_core, base_core, test_core = _ckpt_core
        print("  shap_core loaded from checkpoint.", flush=True)
    else:
        sv_core, base_core = weighted_shap(fitted["core"], test_core[CORE])
        save_ckpt("shap_core", (sv_core, base_core, test_core))
    imp_core = global_importance(sv_core, CORE)
    lines.append("-- SHAP on the fair (ExponentiatedGradient) core model (was the worst-DIDI tier) --")
    lines.append("Global mean |SHAP| feature importance (test split):")
    for feat, val in imp_core:
        lines.append(f"    {feat:20s} {val:.4f}")
    lines.append("")

    plt.figure()
    shap.summary_plot(sv_core, test_core[CORE], plot_type="bar", show=False)
    plt.title("Global feature importance (fair model) -- core (was worst DIDI)")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "shap_summary_core_no_nominal_month.png"), dpi=150)
    plt.close()

    test_core_aligned = test_core.reset_index(drop=True)
    lines.append("Proxy check on the FAIR model: correlation of each feature's per-patient SHAP")
    lines.append("value with each protected attribute (|Pearson r|) -- compare to the plain")
    lines.append("model's strongest proxy, AGE vs PTGENDER at r=-0.224:")
    proxy_rows = []
    for j, feat in enumerate(CORE):
        for attr in PROTECTED:
            attr_vals = test_core_aligned[attr].to_numpy(dtype=float)
            shap_col = sv_core[:, j]
            if np.std(attr_vals) == 0 or np.std(shap_col) == 0:
                continue
            r = np.corrcoef(shap_col, attr_vals)[0, 1]
            proxy_rows.append((feat, attr, r))
    proxy_rows.sort(key=lambda x: -abs(x[2]))
    for feat, attr, r in proxy_rows[:5]:
        lines.append(f"    SHAP({feat:16s}) vs {attr:16s}: r = {r:+.3f}")
    lines.append("")
    lines.append("Conclusion: because ExponentiatedGradient is a genuine retrain (a reweighted")
    lines.append("mixture of forests, not a post-hoc shift), this SHAP audit describes the actual")
    lines.append("model now driving RISK_SCORE -- explainability and fairness are the same object")
    lines.append("here, not two separate patches on top of an unchanged classifier.")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(HERE, "fairness_expgrad_no_nominal_month_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
