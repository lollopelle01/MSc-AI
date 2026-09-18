"""
util/hazard_util.py

A new module for the discrete time hazard estimator that the team's own
IMPLEMENTATION_PLAN.md describes as point two, conversion risk through a
discrete time hazard survival model, never built by anyone until now.
Point three took this on directly, following a conversation with Pelle
about elaborating every available biomarker into RISK_SCORE through a
real fitted model instead of the two feature placeholder, since nobody
else on the team had claimed this piece yet.

The method follows 05 pm, lesson 6, Survival Analysis via Neural Models,
directly. That lesson's own core idea is what makes this module possible
in a reasonable amount of time rather than a much longer research effort,
a discrete time hazard function reduces to an ordinary binary classifier
once every patient visit is treated as its own training example, labeled
one only at the visit where a real worsening transition first happens and
zero everywhere else, with every visit after a patient's own first
worsening transition dropped from the training set entirely, since that
patient is no longer at risk of a first worsening once it has already
happened. Patients who never worsen contribute every one of their visits
as a zero labeled example, the discrete time equivalent of the lesson's
own censored machines, entities the study simply never observed failing.

Kept separate from decision_util.py rather than added to it, since this
module answers a different, upstream question, what is a patient's real
risk, where decision_util.py's own three tracks all take RISK_SCORE as
already given and ask what to do with it. Functions from decision_util.py
are imported and reused directly wherever the job is identical, the
subject level train test split, the classifier evaluation and bootstrap
confidence interval machinery, rather than kept twice over in two files.

Two functions build the panel every hazard model in this module is fit
on.
    add_nominal_month               turns final.csv's own VISCODE2_norm
                                     visit code, bl, m06, m12, and so on,
                                     into a plain number of months since
                                     baseline, the exact calendar grid
                                     ADNI itself schedules visits on,
                                     avoiding any fuzzy date matching.
    build_hazard_panel               adds PRIOR_DIAGNOSIS, a patient's own
                                     diagnosis as of their previous visit
                                     rather than this one, EVENT_AT_VISIT,
                                     true only at a visit where DIAGNOSIS
                                     is numerically higher than
                                     PRIOR_DIAGNOSIS, and AT_RISK, true for
                                     every visit up to and including a
                                     patient's own first EVENT_AT_VISIT.
                                     PRIOR_DIAGNOSIS is used as a model
                                     input rather than DIAGNOSIS itself,
                                     since DIAGNOSIS at this same visit is
                                     part of how EVENT_AT_VISIT is defined
                                     at that visit, an input a model could
                                     only ever appear to predict perfectly.
                                     Every row keeps a well defined
                                     PRIOR_DIAGNOSIS and EVENT_AT_VISIT
                                     regardless of AT_RISK, so a fitted
                                     model can still be scored on every
                                     row of final.csv afterward, training
                                     only ever uses the AT_RISK subset.

Two functions fit the hazard estimator itself, following the lesson's own
framing that the hazard function is, once the panel above exists, simply
a classifier, and that which classifier family is used is an
implementation choice rather than the core idea.
    fit_hazard_logistic               an L1 penalized Logistic Regression,
                                     the same interpretable first step
                                     06 at, lesson 2 already established
                                     for this project's other attribution
                                     models, with class_weight set to
                                     balanced so that the heavy skew
                                     toward zero labeled, still healthy
                                     visits does not simply teach the
                                     model to always predict the majority
                                     class, the same concern 05 pm itself
                                     raises about censoring skewing a
                                     hazard estimator's label balance,
                                     handled here through the ordinary
                                     scikit learn mechanism built for
                                     exactly this situation rather than a
                                     hand written sample weight formula.
    fit_hazard_forest                 a Random Forest counterpart, the
                                     same nonlinear alternative
                                     risk_factors_1_global.ipynb already
                                     pairs against a linear baseline, also
                                     with class_weight balanced. The
                                     lesson's own reference implementation
                                     uses a small neural network instead,
                                     a Random Forest is used here for
                                     consistency with every other paired
                                     comparison already in this project
                                     and because it supports class
                                     weighting directly, where scikit
                                     learn's own neural network classifier
                                     does not.

One function turns a fitted hazard estimator into the forecast columns
IMPLEMENTATION_PLAN.md and NOTES.md both already describe point two as
owing point three, conversion probability at fixed future horizons.
    forecast_conversion_probabilities   follows the lesson's own
                                     persistence approximation directly,
                                     since a patient's future biomarkers
                                     are not observable at forecast time,
                                     every future step queries the fitted
                                     hazard estimator at that future month
                                     with every other input held at its
                                     last observed value, then multiplies
                                     one minus hazard across every step up
                                     to a horizon to get a survival
                                     probability, and reports one minus
                                     that as the conversion probability.
                                     Steps every step_months apart, three
                                     by default, so a 3, 6, 12, and 24
                                     month set of horizons can all be read
                                     off the same underlying product.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# Panel construction
# ============================================================

def add_nominal_month(df, viscode_col='VISCODE2_norm'):
    """
    Turns final.csv's own VISCODE2_norm visit code into a plain number of
    months since baseline, bl becomes 0, m06 becomes 6, m12 becomes 12,
    and so on through the highest code observed, m234. This is ADNI's own
    designed visit schedule, a fixed calendar grid the study itself
    already commits to, so no fuzzy date matching is needed the way
    notebooks/anomaly_detection/method_a_pca_density/load_features.py needs it to align a
    different raw table's own separate visit coding to a diagnosis date.

    A small number of rows carry a code this function does not recognize,
    nine rows project wide, one written as uns1 for an unscheduled visit
    and eight with no visit code at all, both get NOMINAL_MONTH equal to
    NaN and are naturally dropped later by any step that requires this
    column, rather than guessed at.

    Returns a copy of df with NOMINAL_MONTH added.
    """
    def _parse_one(code):
        if pd.isna(code):
            return np.nan
        code = str(code).strip().lower()
        if code == 'bl':
            return 0.0
        if code.startswith('m') and code[1:].isdigit():
            return float(code[1:])
        return np.nan

    out = df.copy()
    out['NOMINAL_MONTH'] = out[viscode_col].map(_parse_one)
    return out


def build_hazard_panel(df, id_col='RID', date_col='EXAMDATE_DX',
                        diagnosis_col='DIAGNOSIS'):
    """
    Builds the panel every hazard model in this module is fit on, adding
    three columns, PRIOR_DIAGNOSIS, EVENT_AT_VISIT, and AT_RISK.

    Visits are ordered per patient by date_col, the same ordering
    compute_risk_trajectory and compute_diagnosis_worsening already use
    in decision_util.py. PRIOR_DIAGNOSIS is the same patient's diagnosis
    at their immediately preceding visit, or their own diagnosis at this
    visit when no earlier one exists, a patient's very first visit by
    definition. EVENT_AT_VISIT is true only when diagnosis_col at this
    visit is numerically higher than PRIOR_DIAGNOSIS, the same ordered
    severity coding, one for cognitively normal, two for mild cognitive
    impairment, three for dementia, that compute_diagnosis_worsening
    already reads DIAGNOSIS_WORSENED_NEXT from, only compared against the
    visit before this one instead of the visit after it.

    PRIOR_DIAGNOSIS rather than DIAGNOSIS itself is what a hazard model
    should see as its disease stage input, since DIAGNOSIS at this same
    visit is one of the two values EVENT_AT_VISIT at this same visit is
    directly computed from, a model given that column as an input would
    only ever appear to predict the label well, not actually learn
    anything about which patients are truly headed toward a worsening
    transition next.

    AT_RISK is true for every visit up to and including a patient's own
    first EVENT_AT_VISIT, and false for every visit after it, following
    05 pm, lesson 6's own discrete time survival setup directly, once an
    entity's event has happened, it stops contributing further zero
    labeled examples to the training set, a patient already diagnosed
    with dementia two years ago is not a meaningful zero label for
    whether dementia has newly appeared today. A patient who never
    worsens keeps every one of their visits at risk, contributing a zero
    labeled example at each one, the discrete time equivalent of a
    machine that ran its entire observed life without failing.

    Every row keeps a well defined PRIOR_DIAGNOSIS and EVENT_AT_VISIT
    regardless of AT_RISK, only fitting a model should ever filter to
    AT_RISK rows, scoring an already fitted model back onto every row of
    final.csv, including a visit recorded after a patient's own first
    worsening, is still meaningful and still needed downstream, that
    patient still needs a RISK_SCORE feeding their own next interval
    decision.

    Returns a copy of df with PRIOR_DIAGNOSIS, EVENT_AT_VISIT, and AT_RISK
    added, row order unchanged, date_col converted to a real datetime in
    the copy.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    sort_order = out.sort_values([id_col, date_col]).index
    sorted_df = out.loc[sort_order]

    prior_diagnosis = sorted_df.groupby(id_col)[diagnosis_col].shift(1)
    event_at_visit = (sorted_df[diagnosis_col] > prior_diagnosis).fillna(False)
    prior_diagnosis = prior_diagnosis.fillna(sorted_df[diagnosis_col])

    event_int = event_at_visit.astype(int)
    prior_event_count = event_int.groupby(sorted_df[id_col]).cumsum() - event_int
    at_risk = prior_event_count == 0

    out.loc[sort_order, 'PRIOR_DIAGNOSIS'] = prior_diagnosis
    out.loc[sort_order, 'EVENT_AT_VISIT'] = event_at_visit
    out.loc[sort_order, 'AT_RISK'] = at_risk
    out['EVENT_AT_VISIT'] = out['EVENT_AT_VISIT'].astype(bool)
    out['AT_RISK'] = out['AT_RISK'].astype(bool)
    return out


# ============================================================
# Hazard estimators
# ============================================================

def fit_hazard_logistic(X_train, y_train, C=1.0):
    """
    An L1 penalized Logistic Regression hazard estimator, the same
    interpretable first step 06 at, lesson 2 already established for
    this project's other attribution models, mirroring
    decision_util.fit_logistic_baseline's own shape exactly, standardize
    then fit, with one addition, class_weight set to balanced.

    05 pm, lesson 6 itself flags that censoring skews a hazard
    estimator's own label balance toward the zero, still at risk class,
    and corrects it with a hand written sample weight formula sized to
    that lesson's own particular censoring rate. This project's own
    censoring comes from real patients who simply have not worsened yet
    or left the study before worsening, not from an experiment design
    choice with one fixed known rate, so a fixed formula copied from the
    lesson would be an arbitrary, undocumented choice here. class_weight
    equal to balanced reweights every training example by the inverse of
    its own label's frequency automatically, the same correction the
    lesson applies by hand, computed from whatever this panel's real
    label balance turns out to be rather than guessed at.

    Returns (fitted_model, fitted_scaler), the same two item shape every
    other fit_*_baseline function in this project returns.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        penalty='l1', solver='liblinear', C=C,
        class_weight='balanced', random_state=42,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def fit_hazard_forest(X_train, y_train, n_estimators=300, max_depth=6, random_state=42):
    """
    A Random Forest hazard estimator, the nonlinear counterpart to
    fit_hazard_logistic above, the same paired comparison
    risk_factors_1_global.ipynb already runs between a Lasso baseline and
    a Random Forest. 05 pm, lesson 6's own reference implementation uses
    a small neural network instead, a Random Forest is used here for two
    reasons, consistency with every other linear against nonlinear
    comparison already in this project, and because scikit learn's own
    neural network classifier does not accept class_weight or a per row
    sample weight at fit time at all, where RandomForestClassifier
    accepts class_weight directly, the same censoring correction
    fit_hazard_logistic above already applies.

    No scaling is applied, tree splits do not depend on feature scale.

    Returns the fitted model directly, predict_proba is called on it as
    is, with no scaler needed at prediction time either.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight='balanced', random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# Forecasting
# ============================================================

def forecast_conversion_probabilities(model, X, feature_cols, scaler=None,
                                       month_col='NOMINAL_MONTH',
                                       horizons_months=(3, 6, 12, 24),
                                       step_months=3):
    """
    Turns a fitted hazard estimator into the forecast columns
    IMPLEMENTATION_PLAN.md and NOTES.md both already describe point two as
    owing point three, conversion probability at fixed future horizons,
    here 3, 6, 12, and 24 months, a superset of both the exact 6, 12, and
    24 month set those documents name and the 3, 6, and 12 month menu
    recommend_interval already searches over, so every interval policy in
    this project can be evaluated against a genuine forecast at exactly
    its own candidate horizons.

    Follows 05 pm, lesson 6's own persistence approximation directly,
    cell 44 of Survival Analysis via Neural Models. A patient's future
    biomarkers are not observable at forecast time, so every future step
    queries the fitted hazard estimator at that future month with every
    other input held fixed at its last observed value, exactly as the
    lesson's own forecasting section does when it assumes x_t is stable
    for some time rather than attempting to predict it. Survival to a
    given horizon is then the product of one minus hazard across every
    step up to that horizon, and the conversion probability reported is
    one minus that survival, the chance of at least one worsening
    transition happening somewhere along the way.

    Every row's own steps are computed in a single stacked prediction
    call rather than one call per row or per step, all step_months apart
    steps for every row are assembled into one large matrix, scored once,
    then reshaped back, which is materially faster than looping in plain
    python across thousands of rows.

    model               a fitted hazard estimator, from fit_hazard_logistic
                        or fit_hazard_forest above, or any other model
                        exposing predict_proba.
    X                   a DataFrame containing at least feature_cols,
                        including month_col, one row per patient visit to
                        forecast from.
    feature_cols        the exact column order the model was fit on.
    scaler              the fitted scaler fit_hazard_logistic returns, or
                        None for a model fit without one, fit_hazard_forest
                        for example.
    month_col           name of the nominal month column inside
                        feature_cols, the one column this function moves
                        forward at each step.
    horizons_months      the future horizons to report a conversion
                        probability for.
    step_months          spacing between consecutive hazard queries,
                        three months by default, fine enough that every
                        horizon in horizons_months lands exactly on a
                        step.

    Returns a DataFrame indexed like X, with one column per horizon,
    named CONVERSION_PROB_{horizon}M.
    """
    month_idx = feature_cols.index(month_col)
    base = X[feature_cols].to_numpy(dtype=float)
    n = base.shape[0]

    max_horizon = max(horizons_months)
    n_steps = int(round(max_horizon / step_months))

    stacked = np.tile(base, (n_steps, 1))
    for step in range(n_steps):
        start, stop = step * n, (step + 1) * n
        stacked[start:stop, month_idx] = base[:, month_idx] + step_months * (step + 1)

    stacked_input = scaler.transform(stacked) if scaler is not None else stacked
    hazard_stacked = model.predict_proba(stacked_input)[:, 1]
    hazard_by_step = hazard_stacked.reshape(n_steps, n).T

    survival = np.cumprod(1.0 - hazard_by_step, axis=1)

    out = {}
    for horizon in horizons_months:
        step_index = int(round(horizon / step_months)) - 1
        out[f'CONVERSION_PROB_{horizon}M'] = 1.0 - survival[:, step_index]
    return pd.DataFrame(out, index=X.index)
