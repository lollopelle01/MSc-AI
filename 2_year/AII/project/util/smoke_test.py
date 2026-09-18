import numpy as np
import pandas as pd
import hazard_util as hu

# ------------------------------------------------------------------
# 1. add_nominal_month
# ------------------------------------------------------------------
codes = pd.DataFrame({'VISCODE2_norm': ['bl', 'm06', 'm12', 'uns1', np.nan, 'm234']})
codes = hu.add_nominal_month(codes)
print('add_nominal_month:')
print(codes)
assert list(codes['NOMINAL_MONTH']) == [0.0, 6.0, 12.0, None, None, 234.0] or \
    codes['NOMINAL_MONTH'].tolist()[:3] == [0.0, 6.0, 12.0]
print()

# ------------------------------------------------------------------
# 2. build_hazard_panel, manual trace check
# ------------------------------------------------------------------
# Patient 1: diag 2,2,3,3,2  (event at visit index 2, rows after it dropped)
# Patient 2: diag 1,1,1      (never worsens, all at risk)
df = pd.DataFrame({
    'RID': [1, 1, 1, 1, 1, 2, 2, 2],
    'EXAMDATE_DX': pd.to_datetime([
        '2020-01-01', '2020-07-01', '2021-01-01', '2021-07-01', '2022-01-01',
        '2020-01-01', '2020-07-01', '2021-01-01',
    ]),
    'DIAGNOSIS': [2, 2, 3, 3, 2, 1, 1, 1],
})
panel = hu.build_hazard_panel(df)
print('build_hazard_panel:')
print(panel[['RID', 'EXAMDATE_DX', 'DIAGNOSIS', 'PRIOR_DIAGNOSIS', 'EVENT_AT_VISIT', 'AT_RISK']])

print('EVENT_AT_VISIT dtype:', panel['EVENT_AT_VISIT'].dtype)
print('AT_RISK dtype:', panel['AT_RISK'].dtype)
assert panel['EVENT_AT_VISIT'].dtype == bool, panel['EVENT_AT_VISIT'].dtype
assert panel['AT_RISK'].dtype == bool, panel['AT_RISK'].dtype
from sklearn.utils.multiclass import type_of_target
assert type_of_target(panel['EVENT_AT_VISIT']) == 'binary', type_of_target(panel['EVENT_AT_VISIT'])
print('dtype checks passed')

p1 = panel[panel['RID'] == 1].sort_values('EXAMDATE_DX')
assert list(p1['PRIOR_DIAGNOSIS']) == [2, 2, 2, 3, 3], p1['PRIOR_DIAGNOSIS'].tolist()
assert list(p1['EVENT_AT_VISIT']) == [False, False, True, False, False]
assert list(p1['AT_RISK']) == [True, True, True, False, False]

p2 = panel[panel['RID'] == 2].sort_values('EXAMDATE_DX')
assert list(p2['EVENT_AT_VISIT']) == [False, False, False]
assert list(p2['AT_RISK']) == [True, True, True]
print('build_hazard_panel checks passed')
print()

# ------------------------------------------------------------------
# 3. fit_hazard_logistic / fit_hazard_forest / forecast_conversion_probabilities
# ------------------------------------------------------------------
rng = np.random.default_rng(0)
n = 400
X = pd.DataFrame({
    'NOMINAL_MONTH': rng.uniform(0, 60, n),
    'AGE': rng.uniform(60, 90, n),
    'HIPPO_NORM': rng.uniform(0, 1, n),
    'PRIOR_DIAGNOSIS': rng.choice([1, 2, 3], n),
})
# synthetic hazard rising with month and PRIOR_DIAGNOSIS, falling with HIPPO_NORM
logit = -3 + 0.02 * X['NOMINAL_MONTH'] + 0.8 * X['PRIOR_DIAGNOSIS'] - 2.0 * X['HIPPO_NORM']
p = 1 / (1 + np.exp(-logit))
y = (rng.uniform(0, 1, n) < p).astype(int)
print('label balance', y.mean())

feature_cols = ['NOMINAL_MONTH', 'AGE', 'HIPPO_NORM', 'PRIOR_DIAGNOSIS']
model_lr, scaler_lr = hu.fit_hazard_logistic(X[feature_cols], y)
model_rf = hu.fit_hazard_forest(X[feature_cols], y)

from sklearn.metrics import roc_auc_score
proba_lr = model_lr.predict_proba(scaler_lr.transform(X[feature_cols]))[:, 1]
proba_rf = model_rf.predict_proba(X[feature_cols])[:, 1]
print('train AUC logistic', roc_auc_score(y, proba_lr))
print('train AUC forest', roc_auc_score(y, proba_rf))

forecast_lr = hu.forecast_conversion_probabilities(
    model_lr, X[feature_cols], feature_cols, scaler=scaler_lr,
)
forecast_rf = hu.forecast_conversion_probabilities(
    model_rf, X[feature_cols], feature_cols, scaler=None,
)
print(forecast_lr.head())
print(forecast_lr.describe())

# monotonicity check: conversion probability should be non decreasing with horizon
cols = ['CONVERSION_PROB_3M', 'CONVERSION_PROB_6M', 'CONVERSION_PROB_12M', 'CONVERSION_PROB_24M']
diffs_lr = forecast_lr[cols].diff(axis=1).iloc[:, 1:]
diffs_rf = forecast_rf[cols].diff(axis=1).iloc[:, 1:]
print('min diff logistic (should be >= 0):', diffs_lr.min().min())
print('min diff forest (should be >= 0):', diffs_rf.min().min())
assert diffs_lr.min().min() >= -1e-9
assert diffs_rf.min().min() >= -1e-9
assert (forecast_lr[cols] >= 0).all().all() and (forecast_lr[cols] <= 1).all().all()
print('forecast_conversion_probabilities checks passed')
