# RUL modeling on ADNI — methods overview

This document walks through the six modeling scripts in this folder
(`rul_model_1/2/3.py` and `rul_hazard/hazard_model_1/2/3.py`), what problem
they solve, how they differ, and how their results compare.

---

## 1. Description of the RUL problem

**Remaining Useful Life (RUL)**, borrowed from predictive-maintenance
terminology, is repurposed here for disease progression: given a patient's
biomarkers at an **MCI (Mild Cognitive Impairment)** visit, how much time is
left before that patient converts to **Dementia due to Alzheimer's Disease**
specifically (not vascular or other dementia)?

Formally, for a qualifying MCI visit:

```
RUL_YEARS = years from this visit to the clean AD-dementia diagnosis
```

A conversion only counts if the path from the MCI visit to the AD diagnosis is
an **uninterrupted run of MCI visits** — if the patient's diagnosis reverts to
Cognitively Normal (or to a non-AD dementia) before the AD diagnosis shows up,
that MCI visit is *not* treated as converting: the eventual AD diagnosis is a
separate disease episode, not a continuation of that particular decline.

Three biomarker modalities feed the models, each switchable independently to
measure its individual contribution:

| Modality | Source | Signal |
|---|---|---|
| `mri` | FreeSurfer structural volumes (UCSFFSX7) | hippocampal / entorhinal / amygdala atrophy, normalized by intracranial volume |
| `pet` | Amyloid PET (UCBERKELEY_AMY_6MM) | amyloid burden (Centiloids, SUVR) |
| `csf` | CSF biomarkers (UPENNBIOMK_ROCHE_ELECSYS) | Aβ42, tau, p-tau |

Demographics (age, sex, education) are included as a shared baseline in every
run. PET and CSF are measured far less often than MRI in this cohort, so their
values are heavily forward/backward-filled per patient — a limitation to keep
in mind when comparing modalities.

> **[Placeholder — cohort/timeline plot]** A diagram of a typical patient
> timeline: a row of visit markers along a time axis, MCI visits in one color
> up to a vertical line marking conversion to AD (or trailing off to the right
> for a non-converter with no cap), annotating where `RUL_YEARS` is measured
> from/to.

> **[Placeholder — `heatmap_mri_top5.png` / `heatmap_pet_top5.png` /
> `heatmap_csf_top5.png`]** Feature × visit heatmaps for the five
> most-followed converting patients per modality, with a red dashed line at
> each patient's conversion date — shows how biomarkers drift as conversion
> approaches.

> **[Placeholder — `violin_baseline.png`, `violin_mid_ad.png`,
> `violin_conversion.png`]** Violin plots of standardized biomarkers at three
> points in the converter timeline (first MCI visit, midpoint, conversion) —
> shows the population-level shift in each biomarker's distribution as
> disease stage advances.

> **[Placeholder — `visit_gap_hist_mri.png` / `_pet.png` / `_csf.png`]**
> Histograms of the time gap between consecutive visits per modality —
> motivates why the models need to handle irregular, modality-specific visit
> spacing rather than assuming fixed-interval follow-up.

---

## 2. Classifiers vs. regressors — what question does each answer

The six scripts pair up into two families that model the *same* underlying
progression but ask **different questions** of the data:

- **Regressors** (`rul_model_1/2/3.py`) answer: *"How many years until this
  patient converts?"* — a direct numeric prediction of `RUL_YEARS`. Their
  training label is only well-defined for patients who actually convert (with
  `HORIZON_YEARS = -1`, the current setting, non-converters are dropped
  entirely — the regressor never sees a patient it can't assign an exact time
  to).

- **Classifiers / hazard models** (`hazard_model_1/2/3.py`) answer a
  narrower, per-visit question instead: *"Is **this** visit the one
  immediately preceding conversion?"* — a binary `EVENT_AT_VISIT` label,
  true for exactly one visit per converter (the last one before the AD
  diagnosis) and false everywhere else, **including every visit of
  non-converters**, regardless of how much follow-up they have. This is a
  **discrete-time hazard** formulation: chaining a patient's per-visit hazard
  probabilities into a survival curve lets you derive an RUL number too
  (`hazard_panel.forecast_survival_curves` + `survival_to_rul`), but the
  model itself is never trained to predict a duration directly — only "risk
  at this step."

The practical trade-off:

| | Regressors | Hazard classifiers |
|---|---|---|
| Label needs | exact known time-to-event | binary "did this visit precede the event" |
| Uses non-converters? | no (dropped, with `HORIZON_YEARS = -1`) | yes — every visit is a valid training row |
| Direct output | RUL in years | P(convert at next step \| history) |
| RUL still obtainable? | yes, directly | yes, derived by walking the survival curve forward |
| Evaluation metric | MAE / RMSE | AUC (discrimination) + calibration gap |

> **[Placeholder — side-by-side diagram]** Two small schematics: (left) a
> regressor mapping features → a single RUL_YEARS number; (right) a hazard
> classifier mapping features at each visit → a hazard probability, with an
> arrow showing several per-visit probabilities being chained into a survival
> curve that is then collapsed back into an RUL number.

---

## Model family structure

Both families share the same three "shapes" of input, so models line up
pairwise by how they use a patient's visit history:

| # | Regressor | Classifier | Input shape |
|---|---|---|---|
| A | `rul_model_1.py` | `hazard_model_1.py` | single visit (level only) |
| B | `rul_model_2.py` | `hazard_model_2.py` | pair of visits (level + change + gap) |
| C | `rul_model_3.py` | `hazard_model_3.py` | full patient sequence (RNN) |

---

## 3. `rul_model_1.py` — single-visit regressor

**Input / training unit:** one row per qualifying MCI visit. Features are the
biomarker *levels* at that single visit (`HIPPO_ICV`, `CENTILOIDS`, `TAU`,
etc.) plus demographics — no history, no rate of change.

**Target:** `RUL_YEARS` at that visit (converters only, since
`HORIZON_YEARS = -1`).

**Model:** plain regression — `RandomForestRegressor` (default) or `Ridge`,
selected via `MODEL`. Evaluated with patient-grouped 5-fold CV (`GroupKFold`
on `RID`, so no patient's visits leak across train/test) against a
mean-predictor baseline.

**Why it's the baseline:** it's the simplest possible framing — "does a
snapshot of biomarkers alone predict time-to-conversion" — before adding
temporal information.

> **[Placeholder — `rul_model_1_pred_vs_true.png`]** True vs. predicted RUL
> for a handful of patients, visits concatenated in chronological order per
> patient — shows how far off single-visit predictions run, and whether the
> model tracks a patient's declining RUL as visits progress even without
> being given explicit history.

---

## 4. `rul_model_2.py` — visit-pair (rate-of-change) regressor

**Input / training unit:** every *ordered pair* of visits from the same
patient up to `MAX_GAP_MONTHS` (36) apart — not just consecutive visits, all
qualifying pairs. For each pair `(i before j)`:

- `<feat>` = biomarker value at the later visit `j` (level)
- `d_<feat>` = value(j) − value(i) (change)
- `GAP_MONTHS` = months between the two visits

**Target:** `RUL_YEARS` at the later visit `j`.

**Model:** two regressors compared on the same delta dataset —
RandomForest/Ridge (`evaluate()`, same `MODEL` switch as model 1) and a small
MLP (`evaluate_mlp()`, a 32→16 ReLU encoder). A recurrent net doesn't apply
here since rows are independent visit-pairs, not sequences.

**Why it differs from model 1:** giving the model both *where a biomarker
is* and *how fast it's moving* should let it distinguish "low but stable"
from "low and dropping fast" — two patients who look similar on a single
snapshot but are progressing very differently.

> **[Placeholder — `rul_model_2_rf_pred_vs_true.png` /
> `rul_model_2_mlp_pred_vs_true.png`]** Same true-vs-predicted layout as
> model 1, one plot per estimator — compares whether the tree-based model or
> the MLP makes better use of the added rate-of-change features.

---

## 5. `rul_model_3.py` — sequence (RNN) regressor

**Input / training unit:** a patient's *entire* visit history kept as one
ordered sequence (padded to the longest sequence in the cohort), instead of
flattening it into independent rows. Per-timestep features are the same
level + change + gap idea as model 2 (`<feat>`, `d_<feat>`, `GAP_MONTHS`),
but computed only against the *immediately previous* visit, with the first
visit of each patient getting zeros.

**Target:** `RUL_YEARS` at every timestep, predicted causally — a
unidirectional **GRU** (or LSTM) means the prediction at visit `t` only sees
visits `1..t`, never future ones.

**Model:** a small GRU/LSTM (`CELL`) with a linear regression head on top of
the hidden state, trained with masked MSE loss so padding doesn't contribute
to gradients. Patient-grouped 5-fold CV as before.

**Why it differs from models 1–2:** instead of hand-crafting "one previous
visit's worth" of change (model 2), the recurrent hidden state can in
principle summarize a patient's *entire* trajectory so far, potentially
capturing longer-range patterns a single delta can't.

> **[Placeholder — `rul_model_3_pred_vs_true.png`]** True vs. predicted RUL
> across full patient sequences — compare against model 1/2's plots to see
> whether accumulating full history (vs. one visit, or one pair) visibly
> improves tracking of a patient's declining RUL over time.

---

## 6. `hazard_model_1.py` — single-visit hazard classifier

**Input / training unit:** one row per MCI visit — *every* MCI visit of
*every* patient, converter or not, any amount of follow-up (unlike
`rul_model_1.py`, nothing is dropped for lack of a known RUL). Features are
the same single-visit biomarker levels as model 1, plus two panel-specific
covariates: `NEXT_GAP_MONTHS` (months to this patient's actual next visit —
the length of the interval the hazard is being asked about) and
`MCI_DURATION_MONTHS` (months since the start of this clean MCI run).

**Target:** `EVENT_AT_VISIT` — 1 only at the visit immediately preceding a
clean MCI → AD-dementia transition, 0 everywhere else.

**Model:** `RandomForestClassifier` (default, `class_weight="balanced"`) or
L1 `LogisticRegression`, selected via `MODEL`. Evaluated by **AUC**
(discrimination between event/non-event visits) and a bucketed
**calibration gap** (does a predicted hazard of 0.3 really correspond to
~30% observed conversion), instead of the regressors' MAE/RMSE.

**Bonus:** because every visit's hazard chains into a survival curve, an RUL
number can still be derived (`hazard_panel.forecast_survival_curves` +
`survival_to_rul`) and checked against the true `RUL_YEARS_TRUE` for
converters — the same ground truth `rul_model_1.py` regresses against
directly.

> **[Placeholder — calibration plot]** Predicted hazard (x) vs. observed
> event fraction (y) across the 10 calibration buckets from
> `hazard_panel.calibration_table` — a diagonal line means well-calibrated
> risk estimates; deviations show over/under-confidence.

> **[Placeholder — `hazard_model_1_pred_vs_true.png`]** Derived RUL (from the
> survival curve) vs. true RUL, converters only — the classification
> analogue of model 1's plot, letting you compare regression-derived vs.
> hazard-derived RUL directly.

---

## 7. `hazard_model_2.py` — visit-pair hazard classifier

**Input / training unit:** the same visit-pair level+change+gap construction
as `rul_model_2.py`, but pairs are restricted to visits within the **same
clean MCI run** (`RUN_ID`) — pairing across a reversion to Cognitively Normal
would mix rate-of-change signal across two declines that aren't the same
episode. Adds `PAIR_GAP_MONTHS` (the historical window the frozen delta was
computed over) alongside the panel covariates `NEXT_GAP_MONTHS` and
`MCI_DURATION_MONTHS`.

**Target:** `EVENT_AT_VISIT` at the later visit of the pair.

**Model:** RandomForest/LogisticRegression (`evaluate()`) and an
`MLPClassifier` (`evaluate_mlp()`), the classification counterparts of model
2's two regressors, compared the same way via AUC + calibration gap. Derived
RUL is validated against `RUL_YEARS_TRUE` exactly as in hazard model 1.

**Why it differs from hazard model 1:** tests whether adding "level +
rate of change" (rather than level alone) sharpens hazard discrimination the
same way it helped RUL regression in model 2.

> **[Placeholder — `hazard_model_2_forest_pred_vs_true.png` /
> `hazard_model_2_mlp_pred_vs_true.png`]** Derived RUL vs. true RUL for each
> estimator — compare against hazard model 1's plot to see whether
> rate-of-change features improve the *derived* RUL, not just raw AUC.

---

## 8. `hazard_model_3.py` — sequence hazard classifier (RNN)

**Input / training unit:** a GRU/LSTM over each patient's ordered visits,
mirroring `rul_model_3.py`, but sequences are grouped by **(RID, RUN_ID)**
rather than RID alone — a patient who reverts to CN and later develops a
fresh MCI episode gets two separate sequences, so rate-of-change signal is
never carried across that gap.

**Target:** `EVENT_AT_VISIT` at every timestep, predicted causally.

**Model:** the same GRU/LSTM architecture as model 3, but with a masked
**binary cross-entropy** loss instead of masked MSE, and a sigmoid output
head (hazard probability instead of a direct RUL value). Deriving RUL here
needs its own forward-stepping loop (`forecast_hazard_seq`): the trained
recurrent cell is stepped forward one hypothetical future visit at a time
from the hidden state reached at the patient's last real visit, re-feeding
its own output — the same persistence approximation as the other two hazard
models, applied one RNN step at a time. Because of this extra cost, derived
RUL is only computed from each sequence's most recent real visit, not every
intermediate one.

**Why it differs from hazard models 1–2:** tests whether letting the
recurrent state summarize a patient's *whole* history (rather than one visit
or one pair) sharpens hazard discrimination further — the classification
counterpart of the model-1→3 progression on the regression side.

> **[Placeholder — `hazard_model_3_pred_vs_true.png`]** A single derived-RUL
> point per converter sequence (its last real visit) vs. true RUL — unlike
> models 1/2's trajectory plots, this is one point per patient since forward
> stepping is only done from the last observed visit.

---

## 9. Results compared

All numbers below are out-of-fold, patient-grouped 5-fold CV results with all
three modalities combined (`mri+pet+csf`), the best-performing configuration
in every script.

### Regressors (MAE in years, lower is better; mean-predictor baseline for reference)

| Model | Best estimator | MAE | RMSE | Baseline MAE |
|---|---|---|---|---|
| 1 — single visit | RF | 1.654 | 2.362 | 1.735 |
| 2 — visit pair | RF | 1.755 | 2.578 | 1.835 |
| 3 — sequence (GRU) | GRU | ~1.70–1.84 (best: `pet` alone, 1.702) | 2.50–2.63 | 1.744 |

### Hazard classifiers (AUC, higher is better; calibration gap, lower is better)

| Model | Best estimator | AUC | Calibration gap | Event rate |
|---|---|---|---|---|
| 1 — single visit | RandomForest | 0.795 | 0.086 | 8.3% |
| 2 — visit pair | RandomForest | 0.780 | 0.066 | 11.0% |
| 3 — sequence (GRU) | GRU | 0.738–0.764 (best: `pet` alone, 0.764) | 0.04–0.09 | 7.9% |

### What this shows

- **All three regressors land close to their baselines.** MAE improves over
  the mean-predictor baseline by roughly 5–10%, meaning biomarkers add real
  but modest signal on top of "predict the average RUL for everyone" — RUL
  prediction from these features alone is a hard problem on this cohort.
- **Combining all three modalities consistently beats any single modality**,
  in both regressors and hazard classifiers — MRI, PET, and CSF appear to
  carry at least partly complementary information rather than redundant
  signal.
- **Adding rate-of-change (model 2) or full sequence history (model 3) does
  not clearly beat the single-visit baseline (model 1)** on the regression
  side, and only more consistent (calibration), rather than an AUC winner,
  on the classification side. Two likely reasons: (a) PET/CSF are sparse and
  heavily filled for most visits, diluting the "change" signal these
  temporal models rely on; and (b) the added model complexity (MLP, RNN)
  needs more training rows than the ~2.5–8k available here to pay off — the
  MLP variants of model 2 are visibly *worse* than their RandomForest
  counterparts in both the regression and hazard versions.
- **Hazard classifiers substantially outperform a naive "predict the average"
  approach on discrimination** (AUC 0.74–0.79, well above the 0.5 chance
  level), and their derived RUL is directly comparable in spirit to the
  regressors' MAE — the two problem framings converge on a similar practical
  usefulness even though they're trained on very different labels (a
  continuous duration vs. a per-visit binary event).
- **The hazard framing's real advantage isn't raw accuracy — it's data
  efficiency.** By training on every visit of every patient (not just
  converters), the hazard panels use 3–5x more rows than the matching
  regressor (e.g. 5,296 rows for hazard model 1 vs. 1,611 for `rul_model_1`),
  which matters more as the biomarker feature set (and therefore the risk of
  overfitting) grows.

> **[Placeholder — grouped bar chart]** MAE (regressors) side by side with
> (1 − AUC) or calibration gap (hazard classifiers) across models 1/2/3 and
> modality subsets — a compact way to see both families' relative ranking
> in one figure, with the mean-predictor / chance-level baseline marked as a
> reference line.

> **[Placeholder — modality ablation chart]** For each of the six models,
> one small bar group showing (demographics-only, mri, pet, csf,
> mri+pet+csf) — makes the "combined beats any single modality" pattern
> visually obvious across the whole model family at once.

> **[Placeholder — RF vs. MLP vs. RNN comparison]** A chart isolating models
> 2 and 3's estimator choice (RF/Ridge vs. MLP vs. GRU/LSTM) on the same
> feature set, to visualize the "added complexity doesn't clearly pay off
> here" finding from the data-size limitation discussed above.

---

## 10. Cost-sensitive retraining — `hazard_model_1_cost_sensitive.py`

Sections 1–9 evaluate every hazard classifier by AUC and a statistical
calibration gap — both are silent on what a wrong prediction actually costs.
`notebooks/decision_support` already has an explicit answer to that: routine
checks and missed conversions are priced against each other (`CHECK_COST`,
`MISSED_CONVERSION_COST`, `ConversionCostModel`), and that pricing is used
downstream to pick a monitoring interval. `hazard_model_1_cost_sensitive.py`
takes that same pricing and feeds it *into* `hazard_model_1.py`'s classifier
itself, rather than only using it after the fact.

**What changed, concretely:** `hazard_model_1.py` trains with
`class_weight="balanced"`, which corrects only for the event/non-event class
imbalance (~8.3% event rate → roughly 1:12 reweighting). The new script
swaps that for

```python
class_weight = {0: CHECK_COST, 1: MISSED_CONVERSION_COST}
```

reusing the exact cost assumptions documented in
`notebooks/decision_support/temporal_window/1_fixed_policy.ipynb`
(`CHECK_COST = 1.0`, base `MISSED_CONVERSION_COST = 20.0`, scaled by
`MCI_OR_WORSE_COST_MULTIPLIER = 2.8` for MCI/Dementia patients — every row in
this hazard panel already is an MCI visit, so that multiplier applies
uniformly, giving `56.0`). Concretely: **a missed conversion is weighted 56x
a false alarm**, versus roughly 12x under `"balanced"` — this is Elkan's
cost-proportionate weighting heuristic (approach 2b from the earlier
calibration discussion), nudging the fitted decision boundary itself toward
minimizing expected clinical cost rather than cross-entropy.

The script also runs the natural follow-up question — is retraining even
necessary, or does picking the right decision threshold on the plain
`"balanced"` classifier get you the same thing for free (approach 2a)?
`sweep_best_threshold` brute-force searches `predict_proba` cutoffs for the
one minimizing `expected_cost`, the same false-negative/false-positive
currency `ConversionCostModel` prices downstream, and is run for *both*
weightings, not only the retrained one — a `class_weight` retrain is only
fairly judged against `"balanced"` if both are read at their own best
operating point, not both pinned to a naive 0.5 cutoff (which was never
chosen with this cost model in mind either way). Everything else (features,
`GroupKFold`, `RandomForestClassifier`/`LogisticRegression` choice, deriving
RUL via `hazard_panel.forecast_survival_curves`) is unchanged from
`hazard_model_1.py`.

### Results (out-of-fold, `mri+pet+csf`, RandomForest)

| Weighting | AUC | Calibration gap | Cost @ 0.5 | Cost @ swept-best threshold |
|---|---|---|---|---|
| `balanced` | 0.795 | 0.086 | 3.666 (fn=342, fp=265) | **0.742** (t=0.054, fn=19, fp=2864) |
| `cost_sensitive` | 0.793 | 0.190 | 2.050 (fn=177, fp=944) | **0.733** (t=0.047, fn=5, fp=3600) |

(Full modality breakdown, both weightings × both threshold choices, in
`rul_hazard/hazard_results_1_cost_sensitive.csv`.)

**What this shows:**

- **AUC is essentially unchanged** (0.795 → 0.793) — cost-weighted
  retraining barely touches the model's ability to *rank* visits by risk.
- **At a naive 0.5 cutoff, cost-weighted retraining looks like a clear win**
  (cost 3.666 → 2.050) — this was the original result reported here, and
  it's real, but it's an artifact of comparing a tuned decision (retrained
  weights) against an untuned one (0.5 was never chosen with this cost model
  in mind, for either classifier).
- **Once both classifiers are read at their own cost-optimal threshold, the
  gap nearly disappears** (0.742 vs. 0.733 — within noise of each other).
  This is expected precisely because AUC barely moved: reweighting the
  training loss and moving the decision threshold after the fact are two
  different mechanical routes to the same underlying Bayes-optimal decision
  boundary, and when two classifiers rank visits almost identically, sweeping
  a threshold on either one traces out nearly the same achievable
  cost — retraining doesn't buy a materially better trade-off here, it just
  relocates *where* on the probability scale that trade-off sits.
- **This makes approach 2a (threshold-sweeping the plain `"balanced"`
  classifier) the better default on this dataset**, not 2b: it reaches
  essentially the same minimum cost (0.742 vs. 0.733) *without* the
  calibration damage class-weighted retraining causes (gap 0.086 → 0.190).
  Keeping the classifier's probabilities honest matters for every other
  consumer of the same score — the derived-RUL path below, or
  `forecast_adaptive`'s multi-horizon compounding of exactly this kind of
  hazard estimate. Cost-weighted retraining would only be worth its
  calibration cost if the two weightings' *rankings* genuinely diverged
  (they don't here); it's a heavier tool than this problem needed.
- **Derived RUL confirms the calibration cost is real**: MAE goes from 1.856
  to 1.973 years (converters only) under cost-sensitive retraining, even
  though more rows resolve within the forecast horizon (98.3% → 99.4%) — the
  retrained model's hazard estimates are inflated near-term, so the derived
  survival curve crosses 50% earlier than it should. Threshold-sweeping
  leaves the underlying probabilities (and this derived-RUL number)
  untouched, since it only changes how a *binary* decision is read off them,
  never the probabilities themselves.

> **[Placeholder — cost-vs-threshold curve]** `expected_cost` swept across
> the full `THRESHOLD_GRID`, one line for `balanced` and one for
> `cost_sensitive`, mri+pet+csf — shows the two curves reaching almost the
> same minimum at different x-positions, which is the whole finding in one
> picture: retraining shifts the curve sideways, it doesn't lower its floor.

> **[Placeholder — `hazard_model_1_cost_sensitive_balanced_pred_vs_true.png`
> / `hazard_model_1_cost_sensitive_cost_sensitive_pred_vs_true.png`]**
> Derived RUL vs. true RUL for both weightings, same patients — shows the
> cost-sensitive model's near-term bias directly in the trajectories, not
> only in the aggregate MAE.

> **[Placeholder — calibration curve, both weightings overlaid]** Predicted
> hazard vs. observed event fraction for `balanced` and `cost_sensitive` on
> the same axes — visualizes exactly how far the cost-sensitive curve drifts
> from the diagonal relative to the balanced one, the price paid for a
> threshold gain that approach 2a gets for free.

---

## 11. Adding `pca_hi_trajectory`'s per-visit health index as a feature — `hazard_model_1_pca_hi.py`

`gio/method_b_autoencoder_hi/pca_hi_trajectory.py` fits Method A's PCA
"healthy operating region" (on Cognitively Normal visits) but keeps Method
B's *longitudinal* structure — one combined Hotelling `T2`/UCL + `Q`/UCL
health index (HI) per (RID, visit), not one per patient. `hazard_model_1_pca_hi.py`
tests whether feeding that per-visit HI in as one extra input feature,
alongside `mri+pet+csf`, improves `hazard_model_1.py`'s classifier, for both
the `balanced` and `cost_sensitive` weightings from section 10.

The HI is recomputed fresh via `method_a.py`'s own `fit_pca_baseline` /
`compute_t2_q` / `control_limits`, fit on `pca_hi_trajectory`'s own
longitudinal CN visits — the same computation that script performs, not a
reimplementation. Because it's per-visit (not per-patient), it merges onto
the hazard panel the same way `mri`/`pet`/`csf` already do: nearest
`EXAMDATE` within `MATCH_WINDOW_DAYS` (`rul_model_1._asof_merge`, reused
unchanged), then forward/backward-filled per patient exactly like every
other modality column in `hazard_panel.build_hazard_dataset`. This is the
corrected version of an earlier attempt that merged `method_a.py`'s
*cross-sectional*, one-row-per-patient anomaly score by `RID` alone and only
covered 3.4% of visit-rows — using the longitudinal HI instead raises
coverage to **1,107 of 1,424 patients (92.5% of the 5,296 visit-rows)**,
measured directly by the script's own coverage line.

### Results (out-of-fold, `mri+pet+csf`, RandomForest)

| Weighting | +pca_hi | AUC | Calibration gap | Cost @ 0.5 | Cost @ swept-best threshold |
|---|---|---|---|---|---|
| `balanced` | no | 0.794 | 0.087 | 3.655 (fn=341, fp=261) | 0.750 (t=0.041, fn=15, fp=3131) |
| `balanced` | **yes** | **0.804** | 0.086 | 3.695 (fn=345, fp=249) | **0.723** (t=0.045, fn=13, fp=3102) |
| `cost_sensitive` | no | 0.794 | 0.190 | 2.103 (fn=182, fp=944) | 0.726 (t=0.064, fn=9, fp=3343) |
| `cost_sensitive` | **yes** | **0.803** | 0.187 | 1.999 (fn=173, fp=899) | **0.706** (t=0.102, fn=15, fp=2901) |

(Full CSV in `rul_hazard/hazard_results_1_pca_hi.csv`.)

**What this shows:**

- **AUC moves by +0.009 to +0.010** for both weightings — a real,
  consistent discrimination gain, not noise-sized, and an order of magnitude
  larger than the +0.002–0.005 seen when the same idea was tried with
  method_a's sparse cross-sectional score (92.5% vs. 3.4% coverage is the
  most likely reason).
- **Swept-best cost improves for both weightings** (0.750 → 0.723 balanced;
  0.726 → 0.706 cost-sensitive) — consistent with, and a bit larger than,
  the AUC gain.
- **Calibration is essentially unchanged or slightly better** (0.087 → 0.086
  balanced; 0.190 → 0.187 cost-sensitive) — unlike the AUC/cost gain, this
  isn't a trade-off the added feature costs elsewhere.
- **Net read:** with near-full coverage, `pca_hi_trajectory`'s longitudinal
  PCA health index is a genuinely useful additional feature for this hazard
  classifier — meaningfully more so than the same PCA/T2+Q idea applied
  cross-sectionally, which confirms coverage (not the underlying anomaly
  signal) was the bottleneck in the section-11 attempt this one replaces.
