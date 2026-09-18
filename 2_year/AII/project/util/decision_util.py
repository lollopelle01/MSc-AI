"""
util/decision_util.py

Point 3, Decision Support. Written as a section meant to be merged into the
team's shared util/util.py once all three points are wired together, the
same way every lesson in the course grows one shared util.py rather than
each topic keeping its own file. Until that merge happens this stays a
separate module so Point 3 is independently runnable, per section 6 of
IMPLEMENTATION_PLAN.md.

Everything here is called the same way the course calls its own util
functions, plain functions and small classes, imported once at the top of
a notebook as:

    from util import decision_util as du

Three groups of functions live here, matching the three parallel tracks
described in NOTES.md.

Cost model (track one)
    ConversionCostModel        follows RULCostModel's calling convention
                                from 04 rul exactly, instantiate once with
                                two cost parameters, then call .cost(...)
                                repeatedly on different subject groups, or
                                .cost_from_forecast(...) when pricing a
                                real per horizon forecast instead of one
                                risk score scaled to fit whichever
                                interval is being priced.
    recommend_interval          turns a risk score into a chosen interval
                                by minimizing ConversionCostModel's cost
                                over a small discrete menu of options.
    compute_risk_trajectory     a third method's input, per patient change
                                in RISK_SCORE between consecutive visits,
                                computed from final.csv's own multi visit
                                structure, no dependency on point one or
                                point two.
    recommend_interval_trajectory   the third policy, projects risk forward
                                across a candidate interval using the
                                trajectory above instead of assuming today's
                                risk holds steady, falls back to
                                recommend_interval's own behavior for a
                                patient's first observed visit.
    recommend_interval_forecast   a fourth policy, reads a real per
                                horizon conversion probability straight
                                off hazard_survival_model.ipynb's own
                                forecast columns instead of scaling one
                                snapshot risk by interval length, paired
                                with ConversionCostModel.cost_from_forecast,
                                which prices it without that same scaling,
                                a forecast already priced at its own
                                horizon should not be rescaled a second
                                time.

Fairness audit (track two)
    bucket_educat                turns PTEDUCAT into a small discrete
                                domain, since DIDI sums over a domain
                                per protected attribute.
    compute_didi                 the DIDI formula, kept here once it has
                                stabilized past the notebook's first,
                                fully written out version.

Attribution (track three)
    compute_biomarker_slopes      compute_risk_trajectory's own analogue,
                                one per month slope per biomarker instead
                                of one for RISK_SCORE, the feature set
                                risk_factors_2_decline.ipynb needs.
    fit_lasso_baseline           a thin wrapper around sklearn's Lasso,
                                mirroring 06-at's baseline approach.
    top_lasso_weights            returns the attributes with the largest
                                absolute coefficient, the Lasso analogue
                                of util.plot_lr_weights in 06-at.
    fit_forest_baseline           a small Random Forest, the nonlinear
                                alternative to Lasso, evaluated the same
                                way.
    compute_shap_importance       SHAP on top of the fitted Random Forest,
                                the third attribution method, both a mean
                                absolute importance per feature and the
                                raw per row values for a single patient
                                breakdown. tensorflow is imported inside
                                the Lagrangian classes only, this follows
                                the same convention, shap is imported
                                inside this one function only.

Fairness correction via Lagrangian relaxation (track two, extended)
    CstDIDIRegressor              a keras.Model wrapper following
                                util.CstDIDIModel from 07-ciml exactly,
                                penalizing DIDI with a fixed multiplier
                                lambda chosen by hand.
    LagDualDIDIRegressor          the Lagrangian dual version, following
                                util.LagDualDIDIModel from 07-ciml, where
                                lambda is itself a trainable variable
                                updated via gradient ascent rather than
                                fixed in advance, which the lesson shows
                                gives a better accuracy for the same
                                fairness level. Both need tensorflow,
                                imported only inside these two classes so
                                the rest of this module stays usable
                                without it.

Sparse biomarker enrichment, used before any track below
    forward_fill_by_patient       carries a sparse biomarker forward, then
                                backward, within one patient's own visits,
                                ordered by date, the same technique
                                `rul_model_1.py` already applies to the
                                raw tables, applied here to final.csv
                                directly, per IMPLEMENTATION_PLAN.md
                                section 4's explicit recommendation.

Subject level split, shared by every track
    subject_train_test_split      splits final.csv by RID rather than by
                                row, the same discipline util.partition_by_
                                machine enforces in 04-rul, so no patient's
                                visits appear in both train and test.

Point 3's own risk measure
    placeholder_risk_score        a simple, transparent composite built
                                from what final.csv already has, DIAGNOSIS
                                and AMYLOID_STATUS, read by every track in
                                this file only by column name, so it can
                                be refined independently later without
                                changing any call site.

Standardized comparison logging, shared by every approach_*.ipynb notebook
    append_approach_result        appends one summary row per split to a
                                single shared
                                results/decision_support/approach_comparison.csv,
                                idempotent per
                                (approach, split) pair so rerunning a
                                notebook replaces its own old row instead
                                of accumulating duplicates. comparison.
                                ipynb reads this one file and groups rows
                                by approach_group, since the different
                                approach groups answer different
                                questions and are not meant to be forced
                                into a single ranking.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ============================================================
# Sparse biomarker enrichment, used before any track below
# ============================================================

def forward_fill_by_patient(df, cols, id_col='RID', date_col='EXAMDATE_DX'):
    """
    Carries each column in cols forward, then backward, within every
    id_col group, ordered by date_col. This is the same technique
    `rul_model_1.py` already uses on the raw ADNI tables, its own per
    patient ffill then bfill of the MRI, PET, and CSF columns it loads,
    applied here to final.csv directly instead, and the fix
    IMPLEMENTATION_PLAN.md section 4 explicitly names for point two's
    target column, 'a simple per subject forward fill before building
    point two's feature set should meaningfully increase how many
    subject visit pairs are usable', which applies just as directly
    here, since track three's attribution and fairness Lagrangian
    tracks both drop any row missing ABETA_RATIO, TAU, or PTAU,
    final.csv's three sparsest columns.

    Only carries a value across a patient's own observed visits, a
    patient with no non null reading anywhere in cols stays null, this
    does not invent a value for a patient who was simply never measured.

    Returns a copy of df with cols filled, row order unchanged.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    sort_order = out.sort_values([id_col, date_col]).index
    filled = (
        out.loc[sort_order]
        .groupby(id_col)[cols]
        .transform(lambda s: s.ffill().bfill())
    )
    out.loc[sort_order, cols] = filled
    return out


# ============================================================
# Subject level split, shared by every track
# ============================================================

def subject_train_test_split(df, test_fraction=0.25, random_state=42):
    """
    Splits a table by RID rather than by row, mirroring
    util.partition_by_machine from 04-rul, which splits by machine before
    ever touching a row, since the course material's own running example
    has multiple rows, one per engine cycle, for every machine, the same
    shape final.csv has, multiple rows, one per visit, for every RID.

    A plain row level train_test_split would let the same patient's visits
    land in both train and test, which is the same subject level leakage
    the anomaly detection stage's own per patient AUC check in point one
    already guards against, and section 6 of IMPLEMENTATION_PLAN.md asks
    every stage to guard against the same way, split by RID once, use
    that split everywhere.

    Returns (train_df, test_df), two DataFrames whose RID sets are
    disjoint and together cover every row in df.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df['RID']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def split_by_rid_membership(df, train_rids, test_rids, id_col='RID'):
    """
    Splits df into (train_df, test_df) using an ALREADY-DECIDED RID
    membership, instead of drawing a fresh subject_train_test_split.

    Bug this exists to fix: several pipeline_1_2_3 scripts split a hazard
    panel once (train_panel/test_panel, via subject_train_test_split), fit
    a model on train_panel, then later needed to evaluate that model on a
    FILTERED subset of the panel (rows with a particular feature tier
    present, or a RISK_SCORE already assigned) and called
    subject_train_test_split AGAIN on that subset with the same
    random_state, assuming that would reproduce the same patient split.
    It does not: GroupShuffleSplit's selection is drawn from the sorted
    list of unique groups it is GIVEN, so once the subset's patient count
    differs from the original panel's, the same random_state selects a
    different set of patients. Checked empirically on this project's own
    wired pipeline: ~73-75% of the "test" patients produced this way had
    already been used to fit the hazard estimator -- a real leakage bug,
    not a hypothetical one.

    The fix is to never re-split a filtered subset. Split the panel once,
    keep the two RID sets, and filter every later, more-restricted subset
    by membership in those same two sets.
    """
    train_df = df[df[id_col].isin(train_rids)].reset_index(drop=True)
    test_df = df[df[id_col].isin(test_rids)].reset_index(drop=True)
    return train_df, test_df


# ============================================================
# Cost model, track one
# ============================================================

class ConversionCostModel:
    """
    Point 3's analogue of util.RULCostModel from 04-rul.

    RULCostModel is instantiated as
        util.RULCostModel(maintenance_cost=..., safe_interval=...)
    and called as
        cmodel.cost(machine_ids, predictions, threshold, return_margin=True)
    returning a total cost, a failure count, and a margin.

    ConversionCostModel keeps the same shape. maintenance_cost becomes the
    cost of one routine clinical check, safe_interval becomes the shortest
    check spacing considered clinically sensible, machine_ids becomes an
    array of RID, predictions becomes a risk score in [0, 1] per row, and
    threshold becomes the risk level above which a shorter interval is
    recommended.

    Neither cost value can be estimated from ADNI. They are documented
    assumptions, passed in explicitly, and meant to be varied across a
    small set of settings rather than fixed once, per the open items in
    IMPLEMENTATION_PLAN.md section 7.
    """

    def __init__(self, check_cost, missed_conversion_cost, safe_interval_months=6,
                 reference_interval_months=12.0):
        self.check_cost = check_cost
        self.missed_conversion_cost = missed_conversion_cost
        self.safe_interval_months = safe_interval_months
        self.reference_interval_months = reference_interval_months

    def cost(self, rid_ids, risk_scores, threshold, interval_months=None,
             return_margin=False):
        """
        Computes total cost for a group of subject rows.

        rid_ids            array of RID, one per row, used only to report
                            a per subject breakdown if needed later, the
                            cost itself is computed per row.
        risk_scores         array of risk in [0, 1], one per row, whatever
                            risk measure the caller passes in, the
                            placeholder score here or a more refined one
                            later, this method only ever reads the array
                            it is given.
        threshold           risk level above which a row counts toward
                            over_threshold_count, a clinically readable
                            headline number reported alongside the cost,
                            it does not itself gate the cost calculation.
        interval_months      the chosen check interval per row, in months.
                            If not given, defaults to safe_interval_months
                            everywhere, i.e. the cost of a single fixed
                            policy rather than a per patient one.
        return_margin        if True, also returns risk_scores - threshold
                            as a margin array, following cmodel.cost's own
                            return_margin option in 04-rul.

        The missed conversion term scales risk_scores by
        interval_months / reference_interval_months, the same scaling
        recommend_interval uses to choose a policy, so that a policy chosen
        under one cost model is scored under that same cost model. Scoring
        a chosen policy under a different cost model than the one that
        chose it makes any comparison meaningless, an earlier version of
        this method used a fixed, unscaled missed cost gated only by
        threshold, which made an adaptive policy score worse than a fixed
        one even when it was making the more sensible choice per row, this
        scaling is the fix for that.

        Returns (total_cost, over_threshold_count, margin_or_None), the
        same three item shape cmodel.cost returns in the course material.
        """
        rid_ids = np.asarray(rid_ids)
        risk_scores = np.asarray(risk_scores, dtype=float)
        n = len(risk_scores)

        if interval_months is None:
            interval_months = np.full(n, self.safe_interval_months, dtype=float)
        else:
            interval_months = np.asarray(interval_months, dtype=float)

        over_threshold = risk_scores > threshold
        checks_per_row = 12.0 / np.clip(interval_months, 1e-6, None)
        routine_cost = checks_per_row * self.check_cost
        scaled_risk = risk_scores * (interval_months / self.reference_interval_months)
        missed_cost = scaled_risk * self.missed_conversion_cost

        total_cost = float(np.sum(routine_cost + missed_cost))
        over_threshold_count = int(np.sum(over_threshold))
        margin = (risk_scores - threshold) if return_margin else None

        return total_cost, over_threshold_count, margin

    def cost_from_forecast(self, rid_ids, horizon_probabilities, threshold, interval_months,
                            return_margin=False):
        """
        Point 3's second cost method, for a policy that already has a real
        per horizon conversion probability to price, rather than one
        snapshot risk score meant to stand for every horizon at once.

        cost above treats risk_scores as a probability anchored at
        reference_interval_months, then rescales it by interval_months
        divided by reference_interval_months to approximate risk over
        whichever interval is actually being priced, a reasonable
        approximation when only one number is available. The forecast
        columns hazard_survival_model.ipynb saves remove the need for that
        approximation, each one, CONVERSION_PROB_3M, CONVERSION_PROB_6M,
        and so on, already is the real probability of conversion by that
        specific horizon, following the persistence based forecast the
        course itself covers in 05 pm lesson 6. horizon_probabilities is
        meant to hold exactly that number, already picked out for whichever
        interval_months this call is pricing, applying the same rescaling
        cost above uses would price the same horizon twice over.

        rid_ids                 array of RID, one per row, kept only for a
                                per subject breakdown later, matching cost
                                above.
        horizon_probabilities    array of real, horizon specific conversion
                                probability, one per row, already read for
                                whichever interval_months that row is being
                                priced at.
        threshold                risk level above which a row counts toward
                                over_threshold_count, matching cost above.
        interval_months          the chosen check interval per row, in
                                months, used only for routine check cost
                                here, the missed conversion term already
                                carries its own horizon inside
                                horizon_probabilities.
        return_margin             if True, also returns
                                horizon_probabilities minus threshold as a
                                margin array.

        Returns (total_cost, over_threshold_count, margin_or_None), the
        same three item shape cost above returns.
        """
        rid_ids = np.asarray(rid_ids)
        horizon_probabilities = np.asarray(horizon_probabilities, dtype=float)
        interval_months = np.asarray(interval_months, dtype=float)

        over_threshold = horizon_probabilities > threshold
        checks_per_row = 12.0 / np.clip(interval_months, 1e-6, None)
        routine_cost = checks_per_row * self.check_cost
        missed_cost = horizon_probabilities * self.missed_conversion_cost

        total_cost = float(np.sum(routine_cost + missed_cost))
        over_threshold_count = int(np.sum(over_threshold))
        margin = (horizon_probabilities - threshold) if return_margin else None

        return total_cost, over_threshold_count, margin


def stratified_missed_conversion_cost(df, base_cost, multiplier, diagnosis_col='DIAGNOSIS',
                                       baseline_value=1):
    """
    Builds a per row missed conversion cost that depends on the patient's own
    diagnosis group, instead of charging every patient the same flat number.

    Rows where diagnosis_col equals baseline_value, Cognitively Normal in the
    DIAGNOSIS encoding used throughout this project, get base_cost. Every other
    row, MCI or Dementia, gets base_cost times multiplier.

    This is a documented modeling assumption, not a per patient prediction.
    Missing a conversion in a patient already MCI or worse is assumed to be
    more costly than missing it in a Cognitively Normal patient, since that
    patient is already closer to a clinically significant transition and has
    less safety margin before the next scheduled check. multiplier itself
    cannot be estimated from ADNI, which contains no dollar cost data, it is
    set from health economics literature on Alzheimer's disease severity costs,
    see cost_model_for below for the citation, and deliberately varied across
    a documented range in a sensitivity analysis rather than fixed once, per
    the open items in IMPLEMENTATION_PLAN.md section 7.

    df                the dataframe to build a cost array for.
    base_cost         the missed conversion cost assigned to the baseline
                      diagnosis group.
    multiplier        how many times base_cost every non baseline diagnosis
                      group is charged.
    diagnosis_col     name of the diagnosis column, default DIAGNOSIS.
    baseline_value    the diagnosis code treated as baseline, default 1.

    Returns a numpy array of per row costs, same length as df, in the shape
    ConversionCostModel's own missed_conversion_cost argument expects.
    """
    return np.where(
        df[diagnosis_col].values == baseline_value,
        base_cost,
        base_cost * multiplier,
    )


def cost_model_for(df, check_cost, base_missed_conversion_cost, multiplier,
                    safe_interval_months=6, reference_interval_months=12.0,
                    diagnosis_col='DIAGNOSIS', baseline_value=1):
    """
    Builds a ConversionCostModel scoped to one specific dataframe, with the
    missed conversion cost stratified by diagnosis group through
    stratified_missed_conversion_cost above.

    A cost model built this way cannot be shared across dataframes of
    different length or row order, a bootstrap resample for example, since
    the per row cost array has to line up positionally with whatever rows it
    is applied to. Building a fresh ConversionCostModel right before it is
    used, scoped to that exact dataframe, keeps this alignment correct by
    construction instead of relying on the caller to remember it, the same
    reasoning the notebooks in notebooks/decision_support already documented
    for their own, now retired, local copy of this function.

    The multiplier used by default across this project, 2.8, is the point
    estimate from a medRxiv preprint on Alzheimer's disease care costs by
    severity. That single number was checked against a broader targeted
    literature review, Global Societal Burden of Alzheimer's Disease by
    Severity, published in Neurology and Therapy and covering 81 separate
    studies from 2013 to 2024, which reports that societal cost typically
    rises by at least 50 percent between consecutive severity levels, with
    most studies placing the ratio between mild and severe disease somewhere
    between about 1.4 and 3.6 depending on which severity scale and which
    cost scope the underlying study used, and at least one study in the
    review reporting a fourfold difference. 2.8 sits inside that broader
    picture rather than at either edge of it, but the picture itself is wide
    enough that no single number in it should be presented as the settled
    one. That is why every temporal_window notebook in notebooks/decision_support
    reruns its own policy at 1.5, at 2.8, and at 4.0, the low end, the
    default, and the high end actually seen in this literature, in a
    dedicated sensitivity analysis section, instead of resting its
    conclusions on 2.8 alone.

    df                          the dataframe this cost model will be applied to.
    check_cost                  cost of one scheduled check, same for every patient.
    base_missed_conversion_cost missed conversion cost for the baseline diagnosis group.
    multiplier                  how many times base_missed_conversion_cost every non
                                 baseline diagnosis group is charged.
    safe_interval_months        passed through to ConversionCostModel.
    reference_interval_months   passed through to ConversionCostModel.
    diagnosis_col                name of the diagnosis column, default DIAGNOSIS.
    baseline_value                the diagnosis code treated as baseline, default 1.

    Returns a ConversionCostModel instance scoped to df.
    """
    return ConversionCostModel(
        check_cost=check_cost,
        missed_conversion_cost=stratified_missed_conversion_cost(
            df, base_missed_conversion_cost, multiplier,
            diagnosis_col=diagnosis_col, baseline_value=baseline_value,
        ),
        safe_interval_months=safe_interval_months,
        reference_interval_months=reference_interval_months,
    )


def recommend_interval(risk_scores, interval_menu_months=(3, 6, 12),
                        check_cost=1.0, missed_conversion_cost=20.0,
                        reference_interval_months=12.0):
    """
    Predict then Optimize, as covered in 08-dfl-tf, lesson 1, PFL and DFL.

    Given a risk score per row, chooses the interval from interval_menu_months
    that minimizes expected cost under a simple, explicit model, an interval
    shorter than the risk level warrants wastes routine check cost, an
    interval longer than it warrants risks the missed conversion cost, and
    that missed conversion cost has to scale with the interval itself, a
    conversion missed during a 3 month gap is a smaller failure than the
    same conversion missed during a 12 month gap, since more time passes
    before it is caught either way.

    risk_scores are read as a probability of conversion over
    reference_interval_months, expressed in months, such as 6, 12, or 24 months. The expected missed
    conversion cost for a candidate interval is then risk_scores scaled by
    interval / reference_interval_months, so a longer interval multiplies
    the same underlying risk into a larger expected cost, which is what
    actually lets the optimum shift toward shorter intervals as risk rises.
    An earlier version of this function left the missed conversion term
    constant across the menu, which meant the cheapest routine cost always
    won regardless of risk, this parameter is the fix for that.

    This is the hardcoded threshold policy section 6 of the implementation
    plan asks for as the first cut, before any learned or DFL based policy
    replaces it. It is deliberately simple, a full grid search over a small
    menu, not a proper optimizer, since the menu itself is small and fixed.

    Returns a RECOMMENDED_INTERVAL array, one value per row, matching the
    column name the shared schema in IMPLEMENTATION_PLAN.md section 6 uses.
    """
    risk_scores = np.asarray(risk_scores, dtype=float)
    menu = np.asarray(interval_menu_months, dtype=float)

    expected_cost = np.zeros((len(risk_scores), len(menu)))
    for j, interval in enumerate(menu):
        routine = (12.0 / interval) * check_cost
        scaled_risk = risk_scores * (interval / reference_interval_months)
        expected_cost[:, j] = routine + scaled_risk * missed_conversion_cost

    best_idx = np.argmin(expected_cost, axis=1)
    return menu[best_idx]


def _expanding_linear_slope(elapsed_days, values):
    """
    For a single patient's own visits, already sorted by date, returns
    one slope per row, in units of value change per month, computed by
    an ordinary least squares line through every non missing value up to
    and including that row, against elapsed_days at those same rows.

    elapsed_days   days since that patient's own first visit, one entry
                     per row, already sorted, shared across every column
                     compute_biomarker_slopes loops over so the same
                     regression x axis is reused rather than recomputed
                     per biomarker.
    values         the single column being sloped, RISK_SCORE in
                     compute_risk_trajectory, one biomarker at a time in
                     compute_biomarker_slopes, one entry per row aligned
                     with elapsed_days.

    A row with fewer than two non missing values at or before it, its own
    first observed visit, every earlier reading missing for this
    particular column, or every earlier reading missing date_col itself,
    gets NaN, unchanged from what a plain two point difference already
    left it as, a row whose own date_col is missing can never contribute
    to any slope, neither its own nor a later row's. A row with exactly two such values
    gets the same number a plain two point difference already computed,
    confirmed directly before this helper was added to this module, an
    ordinary least squares line through exactly two points has that same
    slope, so nothing downstream of a row with only two visits changes.
    A row with three or more non missing values at or before it is where
    this helper actually differs from the two point difference it
    replaces, the returned slope now reflects that patient's own fuller
    history up to this row rather than only the single immediately
    preceding visit, following 04 rul, lesson 3's own treatment of a
    trend estimated from more than two readings as a steadier signal than
    one recent pair alone can offer, particularly against the kind of
    single visit to visit noise a biomarker reading can carry.

    Every row is scored using only that row's own visit and the ones
    before it, a later visit is never used to compute an earlier row's
    own slope, the same forward looking discipline this module already
    applied before this helper existed, avoiding the leakage a regression
    across a patient's entire timeline regardless of row would introduce.

    Returns a numpy array the same length as values, one slope per row,
    NaN wherever fewer than two usable readings exist up to that row.
    """
    n = len(values)
    slopes = np.full(n, np.nan)
    values = np.asarray(values, dtype=float)
    elapsed_days = np.asarray(elapsed_days, dtype=float)
    for i in range(n):
        if i < 1:
            continue
        window_days = elapsed_days[:i + 1]
        window_values = values[:i + 1]
        usable = ~np.isnan(window_values) & ~np.isnan(window_days)
        if usable.sum() < 2:
            continue
        usable_days = window_days[usable]
        usable_values = window_values[usable]
        if np.ptp(usable_days) == 0:
            continue
        slope_per_day = np.polyfit(usable_days, usable_values, 1)[0]
        slopes[i] = slope_per_day * 30.44
    return slopes


def compute_risk_trajectory(df, id_col='RID', date_col='EXAMDATE_DX', risk_col='RISK_SCORE'):
    """
    For every row, returns two new columns, RISK_SLOPE, the change in
    risk_col per month, and HAS_TRAJECTORY, True only when RISK_SLOPE
    could be computed at all. With exactly one earlier visit available,
    RISK_SLOPE is the plain two point difference against that visit,
    unchanged from this function's own earlier behavior. With two or
    more earlier visits available, RISK_SLOPE instead comes from an
    ordinary least squares line through every earlier visit and this one,
    _expanding_linear_slope below, a steadier read on direction than the
    single immediately preceding visit alone can offer. A patient's
    first observed visit has no earlier visit at all to compare against,
    RISK_SLOPE is NaN and HAS_TRAJECTORY is False for that row.

    This is what recommend_interval_trajectory below needs, not just a
    patient's current risk but whether it is rising, falling, or flat,
    close in spirit to the health index slope IMPLEMENTATION_PLAN.md
    section 3 already found rising cleanly with severity in the anomaly
    detection stage's own longitudinal method, computed here directly on
    RISK_SCORE rather than on its own HI, since RISK_SCORE is the one column
    every track in this notebook already shares.

    Returns a copy of df with RISK_SLOPE and HAS_TRAJECTORY added, row
    order unchanged, date_col converted to a real datetime in the copy.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    sort_order = out.sort_values([id_col, date_col]).index
    sorted_df = out.loc[sort_order]

    first_date = sorted_df.groupby(id_col)[date_col].transform('min')
    elapsed_days = (sorted_df[date_col] - first_date).dt.days

    risk_slope = np.full(len(sorted_df), np.nan)
    positions_by_id = sorted_df.groupby(id_col).indices
    for patient_id, positions in positions_by_id.items():
        group_elapsed = elapsed_days.iloc[positions].to_numpy()
        group_values = sorted_df[risk_col].iloc[positions].to_numpy()
        risk_slope[positions] = _expanding_linear_slope(group_elapsed, group_values)
    out.loc[sort_order, 'RISK_SLOPE'] = risk_slope
    out['HAS_TRAJECTORY'] = out['RISK_SLOPE'].notna()
    return out


def recommend_interval_trajectory(risk_scores, risk_slopes, interval_menu_months=(3, 6, 12),
                                   check_cost=1.0, missed_conversion_cost=20.0,
                                   reference_interval_months=12.0):
    """
    A third policy for track one, alongside the fixed policy and the
    snapshot only recommend_interval above. Both of those price a
    candidate interval as if today's risk_scores held steady for the
    whole interval. This version instead projects risk forward across
    the candidate interval using risk_slopes, the per month change
    already observed for that patient from compute_risk_trajectory, so
    a patient whose risk is climbing gets a shorter interval than the
    same current level would suggest on its own, and a patient whose
    risk is falling or flat is not penalized for a snapshot taken
    during a temporary spike.

    For each candidate interval, projected_risk = clip(risk_scores +
    risk_slopes * interval, 0, 1), then the same expected cost formula
    recommend_interval uses, routine cost plus projected_risk scaled by
    interval / reference_interval_months times missed_conversion_cost,
    picks the minimizing interval.

    risk_slopes is NaN wherever compute_risk_trajectory found no prior
    visit, treated here as a slope of 0, the same snapshot only
    recommendation recommend_interval already makes, so every row
    still gets a recommendation, coverage of the trajectory term
    itself should be reported separately by the caller since it is
    only real information for rows where HAS_TRAJECTORY is True.

    Returns a RECOMMENDED_INTERVAL array, one value per row, same
    shape and units as recommend_interval's return value, so both can
    feed the same ConversionCostModel.cost call for a fair,
    apples to apples score.
    """
    risk_scores = np.asarray(risk_scores, dtype=float)
    risk_slopes = np.asarray(risk_slopes, dtype=float)
    risk_slopes = np.where(np.isnan(risk_slopes), 0.0, risk_slopes)
    menu = np.asarray(interval_menu_months, dtype=float)

    expected_cost = np.zeros((len(risk_scores), len(menu)))
    for j, interval in enumerate(menu):
        projected_risk = np.clip(risk_scores + risk_slopes * interval, 0.0, 1.0)
        routine = (12.0 / interval) * check_cost
        scaled_risk = projected_risk * (interval / reference_interval_months)
        expected_cost[:, j] = routine + scaled_risk * missed_conversion_cost

    best_idx = np.argmin(expected_cost, axis=1)
    return menu[best_idx]


def recommend_interval_forecast(forecast_df, interval_menu_months=(3, 6, 12),
                                 check_cost=1.0, missed_conversion_cost=20.0,
                                 horizon_col_prefix='CONVERSION_PROB_', horizon_col_suffix='M'):
    """
    A fourth policy for track one, alongside the fixed policy,
    recommend_interval, and recommend_interval_trajectory above. All three
    of those price a candidate interval by scaling one risk number,
    ConversionCostModel's own reference_interval_months trick. This
    version instead reads a genuinely different number per candidate
    interval straight from hazard_survival_model.ipynb's own forecast
    columns, CONVERSION_PROB_3M, CONVERSION_PROB_6M, and so on, one column
    per month value already present in interval_menu_months, following the
    persistence based forecast 05 pm lesson 6 covers, rather than
    approximating every horizon from a single snapshot.

    For each candidate interval, expected cost is routine cost plus that
    interval's own forecast column times missed_conversion_cost, no
    further rescaling, the forecast itself already carries the horizon.
    The cheapest candidate interval is kept per row, the same grid search
    recommend_interval already runs, only over real per horizon numbers
    instead of one number stretched to fit each candidate in turn.

    forecast_df               a dataframe containing one column per month
                              value in interval_menu_months, named
                              CONVERSION_PROB_{month}M, matching
                              forecast_conversion_probabilities's own
                              output in hazard_util.py.
    interval_menu_months       candidate intervals, in months, every value
                              here needs a matching forecast column.
    check_cost                 cost of one scheduled check, same for every
                              patient.
    missed_conversion_cost     scalar or one value per row, matching
                              ConversionCostModel's own
                              missed_conversion_cost.
    horizon_col_prefix          prefix of the forecast column names,
                              default CONVERSION_PROB_.
    horizon_col_suffix          suffix of the forecast column names,
                              default M.

    Returns (recommended_interval, chosen_probability), two arrays, one
    value per row. recommended_interval matches recommend_interval's own
    return value, the same column name and units every other policy in
    this project already writes to RECOMMENDED_INTERVAL. chosen_probability
    is the forecast probability actually used at that row's own chosen
    interval, meant to be passed straight into
    ConversionCostModel.cost_from_forecast as horizon_probabilities, so the
    same number that chose the interval is also the number that prices it.
    """
    menu = np.asarray(interval_menu_months, dtype=float)
    mcc = np.asarray(missed_conversion_cost, dtype=float)

    horizon_probs = np.column_stack([
        forecast_df[f'{horizon_col_prefix}{int(interval)}{horizon_col_suffix}'].to_numpy(dtype=float)
        for interval in menu
    ])
    routine = (12.0 / menu) * check_cost
    if mcc.ndim == 0:
        missed = horizon_probs * mcc
    else:
        missed = horizon_probs * mcc.reshape(-1, 1)
    expected_cost = routine[None, :] + missed

    best_idx = np.argmin(expected_cost, axis=1)
    recommended_interval = menu[best_idx]
    chosen_probability = np.take_along_axis(horizon_probs, best_idx[:, None], axis=1).ravel()
    return recommended_interval, chosen_probability


def placeholder_risk_score(df):
    """
    Point 3's own composite risk measure, built independently from
    DIAGNOSIS and AMYLOID_STATUS, both already clean in final.csv, scaled
    to [0, 1].

    This is not a survival model and should not be read as one, it is a
    deliberately simple, transparent stand in so track one, two, and three
    can be built, run, and reviewed now, rather than a fitted model of its
    own. Every call site reads it only through the RISK_SCORE column it
    fills in, so a more refined risk measure could replace this function's
    body later without any other line in this module needing to change.
    """
    diag_component = df['DIAGNOSIS'].map({1: 0.15, 2: 0.55, 3: 0.85}).fillna(0.35)
    amyloid_component = df['AMYLOID_STATUS'].map({0: 0.0, 1: 0.2}).fillna(0.0)
    risk = (diag_component + amyloid_component).clip(0, 1)
    return risk.to_numpy()


def compute_diagnosis_worsening(df, id_col='RID', date_col='EXAMDATE_DX', diagnosis_col='DIAGNOSIS'):
    """
    The forward looking twin of compute_risk_trajectory above, same sort by
    id_col then date_col, same groupby shift, only looking ahead in time
    instead of back. For every row, finds the same patient's immediately
    following visit and returns three new columns, NEXT_VISIT_GAP_MONTHS,
    the time in months until that visit, DIAGNOSIS_WORSENED_NEXT, True
    when diagnosis_col at that later visit is numerically higher than at
    this one, and HAS_NEXT_VISIT, True only when a later visit genuinely
    exists. A patient's last observed visit has nothing after it to
    compare against, all three columns reflect that, NEXT_VISIT_GAP_MONTHS
    is NaN and the other two are False.

    Where compute_risk_trajectory asks what a patient's risk was doing
    before this visit, this asks what actually happened to them afterward,
    whether their real diagnosis got worse before their next recorded
    visit, and how long that took. DIAGNOSIS in final.csv is already an
    ordered severity code, 1 for cognitively normal, 2 for mild cognitive
    impairment, 3 for dementia, so numerically higher is exactly clinical
    worsening, never a sideways or improving change.

    This is what lets a recommended interval be checked against something
    that is not the placeholder RISK_SCORE above, real diagnostic
    conversion already recorded in final.csv and used nowhere else in
    this project. A policy recommends some interval at a visit, this
    function reports what really happened next, the two only meet once a
    caller lines a row's RECOMMENDED_INTERVAL up against its own
    NEXT_VISIT_GAP_MONTHS and DIAGNOSIS_WORSENED_NEXT, which the interval
    policy notebooks now do.

    Returns a copy of df with NEXT_VISIT_GAP_MONTHS, DIAGNOSIS_WORSENED_NEXT
    and HAS_NEXT_VISIT added, row order unchanged, date_col converted to a
    real datetime in the copy.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    sort_order = out.sort_values([id_col, date_col]).index
    sorted_df = out.loc[sort_order]

    next_diagnosis = sorted_df.groupby(id_col)[diagnosis_col].shift(-1)
    next_date = sorted_df.groupby(id_col)[date_col].shift(-1)
    gap_months = (next_date - sorted_df[date_col]).dt.days / 30.44
    gap_months = gap_months.replace(0, np.nan)

    out.loc[sort_order, 'NEXT_VISIT_GAP_MONTHS'] = gap_months
    out['HAS_NEXT_VISIT'] = out['NEXT_VISIT_GAP_MONTHS'].notna()
    out['DIAGNOSIS_WORSENED_NEXT'] = (next_diagnosis > sorted_df[diagnosis_col]) & gap_months.notna()
    return out


# ============================================================
# Fairness audit, track two
# ============================================================

def bucket_educat(ptEducat, split_at=16):
    """
    Turns PTEDUCAT, years of education, a roughly continuous attribute in
    final.csv, into a two level discrete domain, since DIDI as defined in
    07-ciml sums over a domain per protected attribute rather than treating
    it as continuous. split_at=16 follows the median in final.csv, roughly
    a four year college degree, and is one of the open items in NOTES.md,
    worth trying at a different split too rather than assumed settled.

    Returns an array of 0 (at or below split_at) and 1 (above split_at).
    """
    ptEducat = np.asarray(ptEducat, dtype=float)
    return (ptEducat > split_at).astype(int)


def compute_didi(data, pred, protected):
    """
    The Disparate Impact Discrimination Index, following DIDI_r exactly as
    written out in 07-ciml, lesson 1, Fairness in ML Models.

        DIDI_r = sum over protected attributes j, sum over values v in the
        domain of j, of the absolute difference between the global average
        prediction and the average prediction within the group where
        attribute j equals v.

    data          a DataFrame whose columns include every key in protected.
    pred          an array of predictions or recommendations, one per row
                    of data, aligned by position.
    protected      a dict mapping attribute name to its discrete domain,
                    e.g. {'PTGENDER': (1, 2), 'PTMARRY_BUCKET': (0, 1)}.

    This is kept here once it has stabilized, the notebook shows the same
    function written out in full the first time it is used, following how
    07-ciml writes DIDI_r directly in the notebook rather than only
    importing it, since it is the concept the lesson is actually about.
    """
    pred = np.asarray(pred, dtype=float)
    res, avg = 0.0, float(np.mean(pred))
    for attribute_name, domain in protected.items():
        for value in domain:
            mask = (data[attribute_name].to_numpy() == value)
            if mask.sum() == 0:
                continue
            res += abs(avg - float(np.mean(pred[mask])))
    return res


def biomarker_protected_association(df, biomarker_cols, protected_cols):
    """
    Checks whether each column in biomarker_cols is itself associated
    with each column in protected_cols, run directly on the biomarkers
    used as model inputs, before any model is fit on them, the check
    NOTES.md's own fairness section names as necessary but had not yet
    actually run, removing a protected attribute from a model's inputs
    does not prevent disparate treatment through its correlates, so the
    biomarkers standing in for those correlates are worth checking on
    their own terms.

    For each pair, one biomarker column and one protected column, this
    runs a one way ANOVA, scipy.stats.f_oneway, across the groups
    protected_cols defines, following the same logic compute_didi above
    already uses to walk a protected attribute's own discrete domain,
    row groups given by PTGENDER, or by PTMARRY's own categories, or by
    an already bucketed PTEDUCAT_BUCKET. Eta squared, the between group
    sum of squares divided by the total sum of squares, is reported
    alongside the F test's own p value, since eta squared answers how
    much of that biomarker's own variance the protected attribute
    explains, a question worth asking on its own even where a small
    sample makes the p value alone hard to read, and because eta squared
    stays on one comparable scale whether the protected attribute has
    two groups, PTGENDER, or several, PTMARRY.

    df               a DataFrame whose columns include every entry in
                        biomarker_cols and every entry in protected_cols.
    biomarker_cols    column names to check, typically the same
                        feature_cols or slope_cols one of the risk_factors
                        notebooks already fits a model on.
    protected_cols    column names whose own groups define the comparison,
                        typically PTGENDER, PTEDUCAT_BUCKET, and PTMARRY,
                        matching make_protected's own three attributes in
                        approach_fairness_correction.ipynb.

    Returns a DataFrame with one row per biomarker column times protected
    column pair, columns biomarker, protected, eta_squared, p_value, and
    n_groups, the number of groups that pairing's own ANOVA actually ran
    across, a group with fewer than two members after dropping missing
    rows is left out of that count and out of the test itself, following
    the same zero row skip compute_didi's own loop already uses.
    """
    from scipy import stats

    rows = []
    for protected_col in protected_cols:
        for biomarker_col in biomarker_cols:
            paired = df[[biomarker_col, protected_col]].dropna()
            groups = [
                group[biomarker_col].to_numpy()
                for _, group in paired.groupby(protected_col)
                if len(group) >= 2
            ]
            if len(groups) < 2:
                rows.append({
                    'biomarker': biomarker_col,
                    'protected': protected_col,
                    'eta_squared': np.nan,
                    'p_value': np.nan,
                    'n_groups': len(groups),
                })
                continue

            f_stat, p_value = stats.f_oneway(*groups)

            grand_mean = paired[biomarker_col].mean()
            ss_total = float(((paired[biomarker_col] - grand_mean) ** 2).sum())
            ss_between = float(sum(
                len(g) * (g.mean() - grand_mean) ** 2
                for g in groups
            ))
            eta_squared = ss_between / ss_total if ss_total > 0 else np.nan

            rows.append({
                'biomarker': biomarker_col,
                'protected': protected_col,
                'eta_squared': eta_squared,
                'p_value': float(p_value),
                'n_groups': len(groups),
            })
    return pd.DataFrame(rows)


# ============================================================
# Attribution, track three
# ============================================================

def compute_biomarker_slopes(df, biomarker_cols, id_col='RID', date_col='EXAMDATE_DX'):
    """
    Track three's own analogue of compute_risk_trajectory above, one
    slope per entry in biomarker_cols instead of one slope for
    RISK_SCORE alone.

    For every row, returns, for each column in biomarker_cols, a new
    column named f'{column}_SLOPE', the change in that column per month.
    With exactly one earlier visit carrying that column, the slope is
    the plain two point difference against that visit, unchanged from
    this function's own earlier behavior. With two or more earlier
    visits carrying that column, the slope instead comes from an
    ordinary least squares line through every earlier reading and this
    one, _expanding_linear_slope below, computed independently per
    column since a patient can easily have an earlier reading for one
    biomarker missing while another is present. HAS_SLOPE_HISTORY is
    True whenever at least one of those new slope columns has a usable
    value for that row, a patient's first observed visit has no earlier
    visit at all to compare against, every slope column is NaN and
    HAS_SLOPE_HISTORY is False for that row.

    This is what risk_factors_2_decline.ipynb needs, not how high
    a biomarker currently sits but how fast it is moving, the same
    reframing recommend_interval_trajectory already applies to track
    one's interval choice, applied here to track three's attribution
    question instead, does the speed of decline explain RISK_SCORE
    differently than the raw level already does in
    risk_factors_1_global.ipynb.

    Returns a copy of df with one new column per entry in biomarker_cols
    plus HAS_SLOPE_HISTORY, row order unchanged, date_col converted to a
    real datetime in the copy.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    sort_order = out.sort_values([id_col, date_col]).index
    sorted_df = out.loc[sort_order]

    first_date = sorted_df.groupby(id_col)[date_col].transform('min')
    elapsed_days = (sorted_df[date_col] - first_date).dt.days
    positions_by_id = sorted_df.groupby(id_col).indices

    slope_cols = []
    for col in biomarker_cols:
        slope = np.full(len(sorted_df), np.nan)
        for patient_id, positions in positions_by_id.items():
            group_elapsed = elapsed_days.iloc[positions].to_numpy()
            group_values = sorted_df[col].iloc[positions].to_numpy()
            slope[positions] = _expanding_linear_slope(group_elapsed, group_values)
        slope_col = f'{col}_SLOPE'
        out.loc[sort_order, slope_col] = slope
        slope_cols.append(slope_col)

    out['HAS_SLOPE_HISTORY'] = out[slope_cols].notna().any(axis=1)
    return out


def fit_lasso_baseline(X_train, y_train, alpha=0.01):
    """
    A thin wrapper mirroring 06-at, lesson 2, A Baseline Approach, a
    Logistic Regression or Lasso model as the first, interpretable step
    before reaching for SHAP over a nonlinear model. Standardizes inputs
    first, since Lasso's L1 penalty compares coefficients directly and
    needs attributes on a comparable scale to do that meaningfully.

    Returns (fitted_model, fitted_scaler), both needed at prediction time.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Lasso(alpha=alpha, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def top_lasso_weights(model, feature_names, top_n=15):
    """
    The Lasso analogue of util.plot_lr_weights from 06-at, returning the
    top_n attributes with the largest absolute coefficient rather than
    plotting them directly, so the notebook controls the plot itself.

    Returns a DataFrame with columns feature and weight, sorted by
    absolute weight, descending.
    """
    weights = pd.Series(model.coef_, index=feature_names, name='weight')
    ordered = weights.reindex(weights.abs().sort_values(ascending=False).index)
    return ordered.head(top_n).reset_index().rename(columns={'index': 'feature'})


def scale_excluding_protected(X_train, X_test, feature_cols, protected_cols):
    """
    Standardizes only the continuous feature columns, leaving the columns
    in protected_cols untouched, in their original categorical values.

    This matters specifically for CstDIDIRegressor and LagDualDIDIRegressor
    below, which locate each row's protected group by comparing a column
    of the network's own input tensor against the literal values passed in
    protected_columns, e.g. column PTGENDER equal to 1 or 2. Standardizing
    that column first turns those literal values into arbitrary floats
    like -1.14, silently making every group comparison match zero rows,
    which is not a corner case, it is what happens on every call unless
    this split is made, since a plain StandardScaler over every column,
    including the protected ones, is otherwise the natural first thing to
    reach for.

    fit_lasso_baseline and fit_forest_baseline do not need this since
    their PTGENDER, PTEDUCAT, and PTMARRY are excluded from the input
    entirely, per the note in track two, so this is only needed for the
    Lagrangian correction, where the protected columns are inputs by
    necessity, the constraint needs to read them at training time.

    Returns (X_train_scaled, X_test_scaled), both as float32 arrays in the
    same column order as feature_cols, scaler columns standardized, the
    protected_cols columns copied through unchanged.
    """
    scale_cols = [c for c in feature_cols if c not in protected_cols]
    scaler = StandardScaler()
    scaler.fit(X_train[scale_cols])

    def _apply(X):
        X_out = X[feature_cols].copy()
        X_out[scale_cols] = scaler.transform(X[scale_cols])
        return X_out.to_numpy(dtype='float32')

    return _apply(X_train), _apply(X_test)


def fit_forest_baseline(X_train, y_train, n_estimators=200, max_depth=4, random_state=42):
    """
    A small Random Forest, the nonlinear alternative to the Lasso baseline
    above, following 06-at's own framing that a Logistic Regression or
    Lasso baseline can be followed by a more expressive model once the
    linear one underfits noticeably. A shallow forest is used here rather
    than the XGBoost plus SHAP combination 06-at itself reaches for, since
    the point of this comparison is checking whether a nonlinear model
    fits meaningfully better at all, SHAP is worth adding on top only once
    that check comes back positive, per 06-at, lesson 4, Additive Feature
    Attribution.

    No scaling is applied, tree splits do not depend on feature scale the
    way Lasso's penalty does.

    Returns the fitted model directly, predict is called on it as is.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def compute_shap_importance(model, X, feature_names):
    """
    Mean absolute SHAP value per feature for a fitted tree model, using
    shap.TreeExplainer, the additive attribution method named directly
    in 06-at, lesson 4, Additive Feature Attribution. This is
    attribution's third method, alongside the Lasso weights and the
    Random Forest's own feature_importances_, three vantage points on
    the same question rather than two, Lasso's signed linear
    coefficients, the forest's split based importance, and SHAP's
    additive, per patient attribution.

    fit_forest_baseline's own docstring originally reserved SHAP for
    once the Random Forest clears the Lasso baseline by a real margin,
    added here regardless of that margin since a three way comparison
    with its own pros and cons is now the point, not a gate on one
    metric, the caller is expected to report both R squared numbers
    alongside this so the reader can judge that margin directly.

    shap is imported here, inside this one function, rather than at
    module level, the same convention this module already uses for
    tensorflow inside CstDIDIRegressor and LagDualDIDIRegressor, so
    the rest of this module stays usable without shap installed.

    Returns (mean_abs_shap, shap_values), a DataFrame with columns
    feature and mean_abs_shap sorted descending, and the raw, per row
    per feature SHAP value array, row i of shap_values lines up with
    row i of X, for a single patient breakdown.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    return mean_abs_shap, shap_values


def evaluate_regression(model, X, y_true, scaler=None):
    """
    A small, shared evaluation step for both attribution methods, R
    squared and mean absolute error, following the same two metrics
    util.print_ml_metrics reports across the course material. scaler is
    applied to X first when given, since the Lasso baseline expects scaled
    inputs and the forest baseline does not.

    Returns a dict with keys r2 and mae.
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    X_input = scaler.transform(X) if scaler is not None else X
    y_pred = model.predict(X_input)
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
    }


def fit_logistic_baseline(X_train, y_train, C=1.0):
    """
    A thin wrapper mirroring fit_lasso_baseline above, for a binary
    target instead of a continuous one, following the same 06-at,
    lesson 2, A Baseline Approach choice between a Logistic Regression
    or a Lasso model as the first, interpretable step. Standardizes
    inputs first for the same reason fit_lasso_baseline does, an L1
    penalty compares coefficients directly and needs a comparable
    scale to do that meaningfully.

    Returns (fitted_model, fitted_scaler), both needed at prediction
    time, the same shape fit_lasso_baseline returns.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def evaluate_classification(model, X, y_true, scaler=None):
    """
    The classification analogue of evaluate_regression above, area under
    the ROC curve instead of R squared and mean absolute error, the
    right metric for a genuine yes or no target rather than a continuous
    one. scaler is applied to X first when given, the same convention
    evaluate_regression follows.

    Returns a dict with key auc.
    """
    from sklearn.metrics import roc_auc_score
    X_input = scaler.transform(X) if scaler is not None else X
    y_proba = model.predict_proba(X_input)[:, 1]
    return {'auc': roc_auc_score(y_true, y_proba)}


# ============================================================
# Fairness correction via Lagrangian relaxation, track two extended
# ============================================================
#
# Both classes below follow 07-ciml, lesson 2, Lagrangian Approaches for
# Constraint Injection, which reframes fairness as a constraint on training
# rather than only a number measured after training:
#
#     argmin_theta  E[L(y, f(x, theta))]
#         such that  DIDI(f(x, theta)) <= threshold
#
# turned into an unconstrained problem by folding the constraint into the
# loss as a penalty term:
#
#     argmin_theta  E[L(y, f(x, theta))] + lambda * max(0, DIDI(...) - threshold)
#
# CstDIDIRegressor uses a fixed lambda, chosen by hand, following
# util.CstDIDIModel. LagDualDIDIRegressor instead treats lambda itself as a
# trainable variable, updated by gradient ascent while the network weights
# are updated by gradient descent, following util.LagDualDIDIModel, which
# the lesson shows reaches the same fairness level with noticeably better
# accuracy, since the penalty strength adapts during training instead of
# being guessed once in advance.
#
# Both need a protected attribute to be part of the model's own input
# tensor at the column positions given in `protected`, exactly as the
# lesson does, DIDI cannot be computed without knowing, for each row,
# which protected group it belongs to, even though that group is not part
# of the prediction target itself.

def _make_base_regressor(input_dim, hidden=()):
    """
    A small keras Sequential regressor, the base predictor both Lagrangian
    wrappers below wrap. Kept separate so both classes can be built the
    same way util.build_nn_model is used in 07-ciml, a plain feed forward
    network with a single linear output neuron.
    """
    from tensorflow import keras
    layers = [keras.layers.Input(shape=(input_dim,))]
    for width in hidden:
        layers.append(keras.layers.Dense(width, activation='relu'))
    layers.append(keras.layers.Dense(1, activation='linear'))
    return keras.Sequential(layers)


def _didi_penalty_term(tf, x, y_pred, protected_columns, threshold):
    """
    Shared by both classes below, the DIDI penalty term itself,
    max(0, DIDI(y_pred) - threshold), computed with tensorflow ops so it
    can be differentiated, following the didi computation inside
    util.CstDIDIModel and util.LagDualDIDIModel in 07-ciml exactly, only
    keyed by input column position rather than column name.
    """
    ymean = tf.math.reduce_mean(y_pred)
    didi = 0.0
    for col_idx, domain in protected_columns.items():
        for value in domain:
            mask = (x[:, col_idx] == value)
            group_mean = tf.math.reduce_mean(tf.boolean_mask(y_pred, mask))
            didi = didi + tf.math.abs(ymean - group_mean)
    return tf.math.maximum(0.0, didi - threshold)


class CstDIDIRegressor:
    """
    Follows util.CstDIDIModel from 07-ciml, lesson 2, Lagrangian Approaches
    for Constraint Injection, a fixed multiplier Lagrangian penalty on
    DIDI, wrapped around a small feed forward network.

    protected_columns is a dict mapping a column index in the input array
    to its discrete domain, e.g. {8: (1, 2)} if column 8 is PTGENDER with
    values 1 and 2, the same shape as the protected dict DIDI_r itself
    takes, except keyed by column position rather than column name, since
    inside the network the input is a plain array, not a DataFrame.

    alpha is the fixed Lagrangian multiplier (lambda in the lesson),
    threshold is the DIDI level the constraint tries to stay under.
    Neither can be estimated, both are choices to document and vary, the
    lesson itself picks alpha=5 by guessing and refining from there.

    tensorflow is imported only inside __init__, not at module level, so
    the rest of this module stays usable without it. The real keras.Model
    is built as a locally defined class named without a leading
    underscore, since keras 3 derives an internal name scope from the
    class's own name and rejects a name that starts with one, which the
    original leading underscore naming used here first ran into.
    """

    def __init__(self, input_dim, protected_columns, alpha, threshold, hidden=()):
        import tensorflow as tf
        from tensorflow import keras
        # Fixed seed, otherwise every rerun of this cell reshuffles the
        # network's initial weights and this class's reported dual_lambda,
        # R2 and DIDI drift meaningfully from run to run. Keras 3's own
        # single entry point for seeding python, numpy and tensorflow's
        # generators together, called once per model built.
        keras.utils.set_random_seed(42)

        class KerasCstDIDIModel(keras.Model):
            def __init__(self_inner):
                super().__init__()
                self_inner.base_pred = _make_base_regressor(input_dim, hidden=hidden)
                self_inner.alpha = alpha
                self_inner.threshold = threshold
                self_inner.protected_columns = protected_columns

            def call(self_inner, x):
                return self_inner.base_pred(x)

            def train_step(self_inner, data):
                x, y_true = data
                with tf.GradientTape() as tape:
                    y_pred = self_inner.base_pred(x, training=True)
                    mse = self_inner.compute_loss(x=x, y=y_true, y_pred=y_pred)
                    cst = _didi_penalty_term(tf, x, y_pred, self_inner.protected_columns, self_inner.threshold)
                    loss = mse + self_inner.alpha * cst
                grads = tape.gradient(loss, self_inner.trainable_variables)
                self_inner.optimizer.apply_gradients(zip(grads, self_inner.trainable_variables))
                return {'loss': loss, 'mse': mse, 'didi_penalty': cst}

        self._model = KerasCstDIDIModel()
        self._model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs=2000, verbose=0):
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32').reshape(-1, 1)
        return self._model.fit(X, y, epochs=epochs, batch_size=len(X), verbose=verbose)

    def predict(self, X):
        X = np.asarray(X, dtype='float32')
        return self._model.predict(X, verbose=0).ravel()


class LagDualDIDIRegressor:
    """
    Follows util.LagDualDIDIModel from 07-ciml, lesson 2, the Lagrangian
    dual version of CstDIDIRegressor above. Instead of a fixed alpha
    chosen by hand, alpha is a trainable variable, updated via a gradient
    ascent step on the same loss the network's weights descend on, so the
    penalty strength adapts during training, alpha(0) = 0, growing only as
    much as the DIDI constraint is actually violated.

    The lesson demonstrates this reaches the same fairness level as the
    fixed multiplier version with noticeably better accuracy, since it
    does not force a strong penalty from the very first training step, and
    is the version worth reporting as the main result once both are tried.

    protected_columns and threshold have the same meaning as in
    CstDIDIRegressor. dual_learning_rate controls how fast alpha itself is
    allowed to grow, a smaller value gives gentler, more stable updates,
    matching the lesson's own note that this reduces oscillation compared
    to the classical penalty method of just multiplying alpha by a fixed
    ratio whenever the constraint is violated.
    """

    def __init__(self, input_dim, protected_columns, threshold, hidden=(),
                 dual_learning_rate=0.01):
        import tensorflow as tf
        from tensorflow import keras
        # Same fixed seed as CstDIDIRegressor above, and for the same
        # reason, this class's alpha already starts at a fixed 0.0, but
        # without this the base network's own initial weights still vary
        # run to run, which is what actually caused the dual_lambda drift
        # observed across repeated executions of the fairness section.
        keras.utils.set_random_seed(42)

        class KerasLagDualDIDIModel(keras.Model):
            def __init__(self_inner):
                super().__init__()
                self_inner.base_pred = _make_base_regressor(input_dim, hidden=hidden)
                self_inner.threshold = threshold
                self_inner.protected_columns = protected_columns
                self_inner.alpha = tf.Variable(0.0, name='alpha')
                self_inner.dual_optimizer = keras.optimizers.SGD(learning_rate=dual_learning_rate)

            def call(self_inner, x):
                return self_inner.base_pred(x)

            def _custom_loss(self_inner, x, y_true, sign=1.0):
                y_pred = self_inner.base_pred(x, training=True)
                mse = self_inner.compute_loss(x=x, y=y_true, y_pred=y_pred)
                cst = _didi_penalty_term(tf, x, y_pred, self_inner.protected_columns, self_inner.threshold)
                loss = mse + self_inner.alpha * cst
                return sign * loss, mse, cst

            def train_step(self_inner, data):
                x, y_true = data
                # Gradient descent step, network weights
                with tf.GradientTape() as tape:
                    loss, mse, cst = self_inner._custom_loss(x, y_true, sign=1.0)
                tr_vars = self_inner.trainable_variables
                grads = tape.gradient(loss, tr_vars)
                self_inner.optimizer.apply_gradients(zip(grads, tr_vars))
                # Gradient ascent step, the multiplier alpha itself
                with tf.GradientTape() as tape:
                    loss_neg, mse, cst = self_inner._custom_loss(x, y_true, sign=-1.0)
                alpha_grad = tape.gradient(loss_neg, self_inner.alpha)
                self_inner.dual_optimizer.apply_gradients([(alpha_grad, self_inner.alpha)])
                self_inner.alpha.assign(tf.math.maximum(0.0, self_inner.alpha))
                return {'loss': mse + self_inner.alpha * cst, 'mse': mse,
                        'didi_penalty': cst, 'alpha': self_inner.alpha}

        self._model = KerasLagDualDIDIModel()
        self._model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs=2000, verbose=0):
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32').reshape(-1, 1)
        return self._model.fit(X, y, epochs=epochs, batch_size=len(X), verbose=verbose)

    def predict(self, X):
        X = np.asarray(X, dtype='float32')
        return self._model.predict(X, verbose=0).ravel()

    @property
    def final_alpha(self):
        """The trained value of the Lagrangian multiplier, for reporting."""
        return float(self._model.alpha.numpy())


# ============================================================
# Bootstrap confidence intervals, shared by every approach_*.ipynb
# ============================================================

def bootstrap_ci(df, metric_fn, id_col='RID', n_bootstrap=400, ci=0.95, random_state=42):
    """
    A general purpose patient level bootstrap, attaches an honest
    uncertainty band to any single number this project would otherwise
    report as a bare point estimate from one train or test split.

    metric_fn takes one argument, a DataFrame shaped like df, and returns
    a single float, computed exactly the way the caller already computes
    it on the real split. Passing the same closure a notebook already
    built for its own point estimate means the bootstrap reuses that
    exact computation rather than a second, separately maintained copy
    that could quietly drift from it.

    Resamples id_col values with replacement, n_bootstrap times, a
    patient's own full set of rows moves together, never a single visit
    on its own, since two visits from the same patient are not two
    independent draws, the same reasoning subject_train_test_split
    already applies to the train test split itself. This is the cluster
    bootstrap, patients are the exchangeable unit, not rows.

    A patient drawn twice in a given resample needs care beyond simply
    duplicating their rows. compute_risk_trajectory and
    compute_diagnosis_worsening both group by id_col internally and sort
    by date within a group, so two duplicated copies of the same patient
    sitting under the same id would be merged into one artificial double
    length timeline, a real visit compared against its own duplicate
    instead of against the next real visit. Each draw is therefore given
    its own disambiguated id, patient 42 drawn twice becomes two entirely
    separate patients as far as any grouped computation inside metric_fn
    can tell, each carrying patient 42's real, unmixed visit history
    once. A metric_fn with no notion of patient order at all, an
    attribution model's r2 or mae, is unaffected either way, disambiguation
    costs it nothing and protects every other metric_fn for free.

    Returns (point_estimate, lower, upper, samples), the metric on df
    itself unresampled, the two sided percentile interval at the given
    ci level across the n_bootstrap resamples, and the full array of
    bootstrap estimates, kept for plotting the bootstrap distribution
    itself rather than only its two endpoints.
    """
    rng = np.random.default_rng(random_state)
    ids = df[id_col].unique()
    n_ids = len(ids)
    point_estimate = metric_fn(df)
    row_positions_by_id = df.groupby(id_col).indices

    samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.choice(ids, size=n_ids, replace=True)
        row_positions, draw_ids = [], []
        for draw_index, patient_id in enumerate(chosen):
            positions = row_positions_by_id[patient_id]
            row_positions.append(positions)
            draw_ids.append(np.full(len(positions), f'{patient_id}__draw{draw_index}'))
        row_positions = np.concatenate(row_positions)
        draw_ids = np.concatenate(draw_ids)
        resampled_df = df.iloc[row_positions].reset_index(drop=True).copy()
        resampled_df[id_col] = draw_ids
        samples[b] = metric_fn(resampled_df)

    alpha = (1 - ci) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    return point_estimate, lower, upper, samples


def bootstrap_attribution_weights(df, fit_fn, weight_fn, id_col='RID', n_bootstrap=200, ci=0.95, random_state=42):
    """
    The attribution counterpart to bootstrap_ci above. bootstrap_ci
    reuses an already fitted model, evaluating it fresh on each
    resample, appropriate for a metric like r2 or mae where the fitted
    model itself is exactly what the notebook wants to keep constant
    across resamples. A biomarker weight has no such fixed point to
    reuse, the weight is a property of a fitted model, so this bootstrap
    fits a brand new model inside every resample instead of reusing one
    fit from the real split.

    fit_fn takes one argument, a DataFrame shaped like df, and returns
    whatever a caller's own fit_lasso_baseline, fit_forest_baseline, or
    fit_logistic_baseline call already returns, a bare model or a
    (model, scaler) pair, fit_fn is where that choice is made. weight_fn
    takes two arguments, whatever fit_fn just returned and the same
    resampled DataFrame, and returns a pandas Series of one score per
    biomarker, indexed by feature name, matching what a caller already
    extracts by hand today, model.coef_ for Lasso or the logistic
    weights, model.feature_importances_ for the Random Forest, or
    compute_shap_importance's own mean_abs_shap column for SHAP.
    Passing the resampled DataFrame through to weight_fn as well as
    fit_fn lets SHAP's own weight_fn read the same resample back off as
    its explanation sample, without a second, separately drawn sample
    that would not match what was just fit.

    The same patient level cluster bootstrap bootstrap_ci already uses,
    id_col values are resampled with replacement, a patient's own full
    set of rows moves together, and a patient drawn twice is given two
    disambiguated ids so compute_risk_trajectory and
    compute_diagnosis_worsening, if either sits inside fit_fn or
    weight_fn, cannot merge the duplicate into one artificial timeline.
    See bootstrap_ci's own docstring above for the full reasoning behind
    both of those choices, repeated here only where the two functions
    actually differ.

    Returns a DataFrame with one row per biomarker, columns feature,
    point (the weight from fit_fn and weight_fn applied to df itself,
    unresampled), lower, upper (the two sided percentile interval at
    the given ci level), and n_bootstrap (how many resamples that
    biomarker's own interval is built from, equal to the n_bootstrap
    argument unless a resample happened to leave that particular
    biomarker out of weight_fn's own returned Series entirely).
    """
    rng = np.random.default_rng(random_state)
    ids = df[id_col].unique()
    n_ids = len(ids)
    row_positions_by_id = df.groupby(id_col).indices

    point_fitted = fit_fn(df)
    point_weights = weight_fn(point_fitted, df)

    sampled_weights = []
    for b in range(n_bootstrap):
        chosen = rng.choice(ids, size=n_ids, replace=True)
        row_positions, draw_ids = [], []
        for draw_index, patient_id in enumerate(chosen):
            positions = row_positions_by_id[patient_id]
            row_positions.append(positions)
            draw_ids.append(np.full(len(positions), f'{patient_id}__draw{draw_index}'))
        row_positions = np.concatenate(row_positions)
        draw_ids = np.concatenate(draw_ids)
        resampled_df = df.iloc[row_positions].reset_index(drop=True).copy()
        resampled_df[id_col] = draw_ids

        fitted = fit_fn(resampled_df)
        sampled_weights.append(weight_fn(fitted, resampled_df))

    samples_by_feature = pd.DataFrame(sampled_weights)

    alpha = (1 - ci) / 2
    rows = []
    for feature in point_weights.index:
        feature_samples = samples_by_feature[feature].dropna() if feature in samples_by_feature.columns else pd.Series(dtype=float)
        if len(feature_samples) > 0:
            lower, upper = np.quantile(feature_samples, [alpha, 1 - alpha])
        else:
            lower, upper = np.nan, np.nan
        rows.append({
            'feature': feature,
            'point': point_weights[feature],
            'lower': lower,
            'upper': upper,
            'n_bootstrap': len(feature_samples),
        })
    return pd.DataFrame(rows)


# ============================================================
# Standardized comparison logging, shared by every approach_*.ipynb
# ============================================================

def append_approach_result(results_dir, approach, approach_group, split, n_rows,
                            total_cost=np.nan, over_threshold_count=np.nan,
                            didi=np.nan, catch_rate=np.nan, r2=np.nan, mae=np.nan,
                            auc=np.nan, ci=None, notes=''):
    """
    Standardized result logging for every approach_*.ipynb notebook,
    so comparison.ipynb has a single shared file to read instead of a
    differently shaped one per notebook.

    results_dir is the results/decision_support folder, approach_comparison.csv
    lives directly inside it, created with the right columns the first
    time this is called. approach is the notebook's own short name, e.g.
    "snapshot_adaptive", approach_group is the comparison bucket used by
    comparison.ipynb to group results that answer the same question,
    currently one of "interval_policy", "fairness_correction",
    "attribution", split is "train" or "test".

    catch_rate is the percent of rows with a real DIAGNOSIS worsening
    event before the patient's next recorded visit for which this
    approach's own recommended interval was short enough to have caught
    it in time, from compute_diagnosis_worsening above, applicable only
    to the interval_policy group, left at its float('nan') default by
    every other approach_group, exactly like r2 and mae are.

    ci is an optional dict from bootstrap_ci above, mapping a metric name,
    one of "total_cost", "didi", "catch_rate", "r2", "mae", to its own
    (lower, upper) pair, for example ci={'total_cost': (58000.0, 61000.0)}.
    Every metric gets its own two extra columns, metric_ci_low and
    metric_ci_high, only the metrics actually present in ci are filled
    in, everything else stays at the float('nan') default, the same
    convention the point estimate columns themselves already follow.

    Columns left at the float('nan') default are simply not applicable to
    that approach's kind of result, an interval policy has no r2, an
    attribution method has no total_cost, comparison.ipynb only compares
    columns that are populated within a given approach_group.

    Idempotent per (approach, split) pair, calling this again after
    rerunning a notebook replaces that notebook's own previous row for
    the same split rather than appending a duplicate, so the shared file
    always reflects only the latest execution of each notebook and can be
    safely rerun during development.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'approach_comparison.csv')

    ci = ci or {}
    ci_metrics = ['total_cost', 'didi', 'catch_rate', 'r2', 'mae', 'auc']
    ci_columns = [f'{metric}_ci_{bound}' for metric in ci_metrics for bound in ('low', 'high')]
    ci_values = {}
    for metric in ci_metrics:
        low, high = ci.get(metric, (np.nan, np.nan))
        ci_values[f'{metric}_ci_low'] = low
        ci_values[f'{metric}_ci_high'] = high

    columns = (['approach', 'approach_group', 'split', 'n_rows', 'total_cost',
                'over_threshold_count', 'didi', 'catch_rate', 'r2', 'mae', 'auc']
               + ci_columns + ['notes'])
    new_row = pd.DataFrame([{
        'approach': approach,
        'approach_group': approach_group,
        'split': split,
        'n_rows': n_rows,
        'total_cost': total_cost,
        'over_threshold_count': over_threshold_count,
        'didi': didi,
        'catch_rate': catch_rate,
        'r2': r2,
        'mae': mae,
        'auc': auc,
        **ci_values,
        'notes': notes,
    }], columns=columns)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        keep = ~((existing['approach'] == approach) & (existing['split'] == split))
        existing = existing[keep]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined = combined[columns]
    combined.to_csv(path, index=False)
    return combined


def append_biomarker_ranking(results_dir, notebook, method, biomarker_scores):
    """
    Standardized attribution ranking logging, the counterpart to
    append_approach_result above for a full ranked list of biomarkers
    per method rather than a single scalar metric per split.

    results_dir is the results/decision_support folder, biomarker_rankings.csv
    lives directly inside it, created with the right columns the first
    time this is called. notebook is the calling notebook's own short
    name, e.g. "risk_factors_1_global", method is the attribution method
    that produced this ranking, one of "lasso", "random_forest", "shap",
    or "logistic", biomarker_scores is a dict or pandas Series mapping
    each biomarker's own base name, HIPPO_NORM rather than
    HIPPO_NORM_SLOPE, so a level based and a slope based ranking can be
    compared biomarker by biomarker, to that method's own signed or
    unsigned importance score for it, sign is kept where the method has
    one, Lasso and the logistic weights, rank is computed here from the
    absolute value of each score, largest magnitude first, matching
    top_lasso_weights' own convention above.

    Idempotent per (notebook, method) pair, calling this again after
    rerunning a notebook replaces that notebook's own previous ranking
    for the same method rather than appending a duplicate, the same
    convention append_approach_result already follows.

    Returns the combined DataFrame written to disk.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'biomarker_rankings.csv')

    scores = pd.Series(biomarker_scores)
    order = scores.abs().sort_values(ascending=False).index
    ranked = pd.DataFrame({
        'notebook': notebook,
        'method': method,
        'biomarker': order,
        'rank': range(1, len(order) + 1),
        'score': scores.reindex(order).values,
    })

    if os.path.exists(path):
        existing = pd.read_csv(path)
        keep = ~((existing['notebook'] == notebook) & (existing['method'] == method))
        existing = existing[keep]
        combined = pd.concat([existing, ranked], ignore_index=True)
    else:
        combined = ranked
    combined = combined[['notebook', 'method', 'biomarker', 'rank', 'score']]
    combined.to_csv(path, index=False)
    return combined
def calibration_curve_check(y_true, y_prob, n_buckets=10):
    """
    Checks whether a fitted classifier's own predicted probability can
    be trusted at face value, the calibration question hazard_survival_model.ipynb's
    own 0.791 AUC leaves entirely open, since AUC checks ranking, a
    patient more likely to convert should score higher than one less
    likely to, not whether the probability value itself is correct in
    an absolute sense. NOTES.md names this exactly, "the natural next
    step this project has not yet taken", and evaluative_synthesis.ipynb's
    own closing section repeats it as the single check most likely to
    change how forecast_adaptive's own numbers should be read, since
    that policy compounds this same uncalibrated probability across up
    to eight future steps.

    y_true    the real recorded 0 or 1 outcome, one per row, aligned by
                position with y_prob, DIAGNOSIS_WORSENED_NEXT or
                EVENT_AT_VISIT depending on the caller.
    y_prob    the model's own predicted probability for that same row,
                one float in [0, 1] per row, aligned by position.
    n_buckets  how many roughly equal sized groups to sort predictions
                into, following the same bucket and compare logic
                compute_didi above already applies to a protected
                attribute's own discrete domain, here applied to
                pd.qcut's own quantile buckets instead, equal sized
                groups rather than equal width ones, so a model whose
                own predictions cluster near one end of [0, 1], exactly
                what forecast_adaptive's own near universal shortest
                interval recommendation already hints at, still gets
                buckets with enough rows in each to compare rather than
                several empty ones and one crowded one.

    For each bucket, reports the mean predicted probability against the
    real observed fraction of positives, mean_predicted and
    observed_fraction should sit close to the diagonal, mean_predicted
    equal to observed_fraction, for a model whose own stated probability
    can be trusted, a bucket sitting well above the diagonal means this
    project is systematically overconfident there, a bucket below it
    means the model is underselling its own risk.

    Returns a DataFrame with one row per bucket, columns bucket,
    n_rows, mean_predicted, observed_fraction, and calibration_gap, the
    signed difference mean_predicted minus observed_fraction, plus a
    single summary float, the row count weighted mean absolute value of
    calibration_gap across every bucket, a single number this project
    can track over time the same way it already tracks a point estimate
    everywhere else, low is well calibrated, high is not, on the same
    [0, 1] scale probability itself lives on.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    paired = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).dropna()

    paired['bucket'] = pd.qcut(paired['y_prob'], q=n_buckets, duplicates='drop')

    rows = []
    for bucket, group in paired.groupby('bucket', observed=True):
        mean_predicted = float(group['y_prob'].mean())
        observed_fraction = float(group['y_true'].mean())
        rows.append({
            'bucket': str(bucket),
            'n_rows': len(group),
            'mean_predicted': mean_predicted,
            'observed_fraction': observed_fraction,
            'calibration_gap': mean_predicted - observed_fraction,
        })
    result = pd.DataFrame(rows)

    weighted_mean_abs_gap = float(
        (result['calibration_gap'].abs() * result['n_rows']).sum() / result['n_rows'].sum()
    )
    return result, weighted_mean_abs_gap
