# Pipeline 1 -> 2 -> 3 (experimental)

Wires point 1's anomaly score into point 2's hazard model as an input
feature, then feeds the resulting HAZARD_ESTIMATE into point 3's interval
policy -- a genuine sequential chain, closing the gap identified in
discussion: previously points 1 and 2 both fed point 3 independently
("fan-in"), but point 1 never passed through point 2.

This is a new, separate folder so it does not modify
`notebooks/hazard_model/hazard_survival_model.ipynb` or any
`notebooks/decision_support/` notebook directly. It reuses the exact same
`util/hazard_util.py` / `util/decision_util.py` functions those notebooks
use, so the result is directly comparable and could be folded back into the
team's own notebook later with no logic changes, only which file it lives
in -- worth raising with Pelle before merging anything.

## Run it

```
python3 wire_full_pipeline.py            # base pipeline + soft fallback
python3 didi_breakdown.py                # diagnoses the fallback's fairness gap
python3 fairness_correction.py           # the fix actually used
python3 fairness_correction_tf.py        # in-syllabus alternative, for comparison
python3 explainability_fairness.py       # SHAP explainability + fairness audit
```

## 1. Does point 1's signal actually help?

Adding `ANOMALY_SCORE` (this project's PCA-based health index, joined onto
`final.csv` via the reconciled `RID`+`VISCODE2_norm` key from
`notebooks/anomaly_detection/final_key_join.py`) as one extra feature in the hazard estimator:

| Model | Feature tier | Test AUC without | Test AUC with ANOMALY_SCORE | Delta |
|---|---|---|---|---|
| Logistic | core | 0.743 | 0.747 | +0.004 |
| Random Forest | core | 0.791 | 0.790 | -0.001 |
| Logistic | extended (+CSF) | 0.768 | 0.794 | **+0.025** |
| Random Forest | extended (+CSF) | 0.791 | 0.807 | **+0.016** |

The best single estimator is the Random Forest on the extended tier plus the
anomaly score (test AUC 0.807, up from the existing pipeline's best of
0.791) -- a real, measurable improvement, not just a plausible-sounding idea.

**The honest cost**: this tier needs CSF markers (sparse) and the anomaly
score (65.5% coverage on its own) at the same time, so it only reaches 37.9%
of all panel rows. `explainability_fairness.py`'s Part 1 confirms with SHAP
that this AUC gain is real contribution, not coincidence: ANOMALY_SCORE
receives its own, individually-varying attribution (rank 9 of 10 features by
mean |SHAP|, but non-zero and patient-specific -- see
`shap_dependence_anomaly_score.png` and the two waterfall examples) rather
than just tracking the CSF/imaging features that were already there.

## 1b. Is PCA still the best point-1 signal once wired through the actual pipeline?

Section 1 above and the whole standalone comparison (`compare_three_methods.py`,
`min_visits_gain_all_methods.py`) established PCA as the winner among PCA/GMM/
autoencoder as an isolated anomaly detector. The natural follow-up: does that
ranking survive once the signal is actually wired through `wire_full_pipeline.py`
(point 1 -> point 2 -> point 3), not just measured on its own? `notebooks/anomaly_detection/gmm_key_join.py`
builds a GMM-based `ANOMALY_SCORE` the same way `final_key_join.py` builds PCA's
(99.8% keyed coverage on its own rows), and `wire_full_pipeline_gmm.py` reruns the
identical pipeline with that swapped in -- same tiers, same split, same cost model.

| Metric (test split) | PCA-based ANOMALY_SCORE | GMM-based ANOMALY_SCORE |
|---|---|---|
| Best single tier (extended+anomaly, forest) test AUC | **0.807** | 0.804 |
| core+anomaly, forest test AUC (vs. core alone, 0.791) | 0.790 (-0.001) | 0.784 (**-0.006**) |
| Soft-fallback total cost | **10,825.8** | 10,841.8 |
| Soft-fallback DIDI | 8.570 | 8.563 (marginally lower) |
| Soft-fallback catch rate | **94.9%** | 93.2% |
| Hard-dependency DIDI | 2.590 | 2.672 |

GMM is competitive but not better: it loses a small but consistent amount of
AUC in both anomaly-augmented tiers, costs slightly more, and -- the metric
that matters most given the project's own cost asymmetry -- catches 1.7
points fewer real conversions in the soft-fallback (deployed) configuration.
DIDI is a wash (within noise of PCA's). This is the same "certainty over
coverage" pattern found standalone, now confirmed inside the actual pipeline:
**PCA remains the correct point-1 signal, not just on its own metrics but in
the exact configuration that reaches patients.**

## 2. Soft fallback: recovering coverage

Using the single best tier alone only scores 37.9% of patients. A fallback
cascade (`extended+anomaly` -> `extended` -> `core+anomaly` -> `core/forest`
-> `core/logistic`, each patient scored by the richest tier their own data
supports) recovers coverage to 69.9%, preserving the accuracy gain for
whoever can use it instead of discarding it project-wide. This is the
"fan-in with soft fallback" design -- deliberately not a hard dependency
(which would floor coverage at 37.9% for everyone).

## 3. The fallback's fairness cost, and the fix

The soft fallback initially showed a materially worse test DIDI (8.570)
than any single tier evaluated on its own. `didi_breakdown.py` found two
compounding causes, not one: individual tiers vary a lot in their own DIDI
(1.357 to 7.289, worst on `core/forest` -- the tier serving the
least-complete records), AND which tier a patient lands in is itself mildly
correlated with protected attributes (DIDI of tier assignment: 2.923).

`fairness_correction.py` fixes this with sklearn-only post-hoc group
recalibration: since DIDI is literally the sum of |group mean - global
mean|, shifting each group's scores onto the global mean (fit on train,
applied to test, iterated across the three overlapping protected
attributes) attacks the exact quantity DIDI measures.

**Result**: test DIDI 8.570 -> 3.153 (63% reduction), cost 10,825.8 ->
10,782.1 (flat), catch rate 94.9% -> 93.2% (small dip). This is the
correction actually used.

`fairness_correction_tf.py` is a comparison, not a second fix: it applies
the team's own in-syllabus technique (`LagDualDIDIRegressor`, 07-ciml
lesson 2, reused unmodified from `util/decision_util.py`) to the
worst-offending `core` tier, retraining with the DIDI constraint built in
rather than correcting after the fact. It underperforms badly here --
16% DIDI reduction (8.716 -> 7.313) versus the sklearn fix's 63%, while AUC
collapses from 0.791 to 0.570 -- attributed to the regressor's MSE loss
having no class-imbalance handling for this rare-event target. Kept in the
repo as an honest, evidence-based reason to prefer the simpler fix, not as
something to actually deploy.

## 4. A genuine in-training alternative to the post-hoc fix

`explainability_fairness.py`'s Part 3 was honest that the post-hoc group
shift never touches the model's actual reasoning -- it repairs the group
statistic, not the underlying RandomForest. `fairness_correction_reduction.py`
tries two real retrains instead, both sklearn-ecosystem (no TensorFlow):

- **Reweighing** (Kamiran & Calders 2012): one sample-weight per protected
  group, computed once, fed into a single `RandomForestClassifier.fit()`.
  On this data it did not help -- test DIDI went from 8.716 to 9.056,
  slightly *worse*. A real, reported negative result: the disparity here
  isn't simply a base-rate imbalance reweighing can flatten.

- **ExponentiatedGradient** (Agarwal et al. 2018, via `fairlearn`): wraps
  the same class-balanced RandomForest and iteratively reweights training
  examples across a sequence of cost-sensitive fits so the ensemble
  satisfies a DemographicParity constraint -- DIDI's exact in-training
  analogue. Sweeping its tightness knob (`eps`) traces a real, tunable
  accuracy/fairness tradeoff instead of the TF Lagrangian's one
  catastrophic operating point:

  | eps | Test AUC | Test DIDI (interval) |
  |---|---|---|
  | (baseline, unconstrained) | 0.791 | 8.716 |
  | 0.020 | 0.744 | 6.945 |
  | 0.010 | 0.732 | **1.814** |
  | 0.005 | 0.730 | 5.639 |

  At `eps=0.01`: DIDI drops 79% (8.716 -> 1.814) -- better than the
  post-hoc fix's 63% -- for a real AUC cost (0.791 -> 0.732, about an 8%
  relative drop) instead of none. Unlike the post-hoc shift, this is an
  actual retrain: the resulting model's SHAP decomposition would genuinely
  differ from the baseline's, because the model itself changed. The curve
  is not perfectly monotonic in `eps` (0.005 is worse than 0.010) --
  expected from the reduction's randomized mixture-of-classifiers
  optimization at a capped `max_iter`, not a bug, but worth not
  over-reading any single point.

  **Bottom line for the briefing**: the post-hoc fix remains the more
  practical default (zero accuracy cost, simple to explain), but
  `ExponentiatedGradient` is the honest answer to "is there a way that
  actually changes the model": yes, and it gives a real tradeoff dial
  instead of an all-or-nothing choice -- worth mentioning as the
  more-principled alternative even if not the one shipped.

## 5. Using ExponentiatedGradient as the actual mechanism

The first pass retrained every fallback tier with `ExponentiatedGradient`
under `DemographicParity` and cut test DIDI from 8.570 to 4.914-4.492, but
catch rate collapsed from 94.9% to 76-80%. Raising tree count and
iterations (50/12 -> 150/30) barely moved either number, which ruled out
"not enough compute" as the explanation.

**Root cause, found by inspecting fairlearn's own objective**: by default,
`ExponentiatedGradient` optimizes plain 0/1 misclassification error. It has
no idea that this project's own `ConversionCostModel` treats a missed
conversion as 20x worse than a false alarm -- the same asymmetry
`class_weight="balanced"` was already approximating for every plain forest
elsewhere in this codebase. Without it, the reduction happily trades away
recall on the rare positive class to hit its fairness constraint cheaply,
which is a mismatched-objective bug, not a real fairness/accuracy tradeoff.

**The fix**: pass `objective=ErrorRate(costs={"fp": 1.0, "fn": 20.0})` --
fairlearn's built-in cost-sensitive objective, matching `MISSED_CONVERSION_COST
/ CHECK_COST` exactly -- alongside the same `DemographicParity` constraint.
Result on the test split:

| | Cost | DIDI | Catch rate |
|---|---|---|---|
| Original plain-forest fallback | 10,825.8 | 8.570 | 94.9% |
| EG (unaligned objective, first pass) | 6,973.4 | 4.914 | 78.0% |
| **EG (cost-aligned objective)** | 15,130.5 | **2.098** | **94.9%** |

Catch rate gap fully closed -- 94.9%, an exact match to the original, not
an approximation. DIDI still falls 75.5% (8.570 -> 2.098), better than the
plain post-hoc shift's 3.153. The post-hoc layer from the previous pass
(section 4's group-shift, applied on top) is now redundant here -- test
DIDI is unchanged with or without it (2.098 either way) -- because the
cost-aligned constraint already resolves the cross-tier disparity that
motivated adding it in the first place. Simpler is better: EG alone, with
the aligned objective, is the mechanism actually used.

**Why the +40% cost is a good trade, not a compromise, in this domain**:
total evaluation cost rose from 10,825.8 to 15,130.5 because recovering
full catch rate means recommending shorter, more frequent check-in
intervals across more patients -- literally more monitoring visits. In a
screening pipeline for Alzheimer's conversion, that is the RIGHT direction
to move, not a cost to apologize for. The project's own cost model already
says this: `MISSED_CONVERSION_COST` is set to 20x `CHECK_COST`, meaning the
team building this decided a missed conversion is twenty times worse than
an unnecessary check-in, before fairness ever entered the picture. Spending
~40% more on check-ins to hold catch rate at its original 94.9% *while also*
cutting DIDI by 75% is the pipeline behaving exactly the way its own cost
model says it should: extra monitoring is cheap insurance against the
outcome the whole project exists to prevent. The number worth having ready
is not "why did cost go up" but "cost went up because the model is
checking on more people who need it, fairly."

**DIDI came out lower than the 8.570 baseline in every genuine retrain
tried, not just this one** -- the fairness gain is not a single lucky
setting:

| Approach | Test DIDI | vs. baseline (8.570) |
|---|---|---|
| Post-hoc group-shift alone (section 3) | 3.153 | -63% |
| Reweighing, Kamiran & Calders (section 4) | 9.056 | +6% (worse -- the one exception) |
| EG, isolated core tier, eps=0.01 (section 4 sweep) | 1.814 | -79% |
| EG, full cascade, unaligned objective (first pass) | 4.914 | -43% |
| EG, full cascade, unaligned objective + post-hoc | 1.778 | -79% |
| **EG, full cascade, cost-aligned objective (final)** | **2.098** | **-76%** |

Every technique that genuinely retrains the model landed below baseline;
only Reweighing (a crude pre-processing shortcut, not a constrained
optimization) made DIDI worse. That consistency is itself worth stating in
the briefing -- this isn't one favorable roll of the dice, it's the
predictable result of applying a real fairness constraint to the training
objective, across every configuration that does so properly.

**Explainability on the final model**: ANOMALY_SCORE still receives real,
non-zero SHAP attribution (rank 6 of 10, similar magnitude to the plain
model's rank 9) -- point 1's signal survives this retrain too. The
strongest core-tier proxy is now AMYGDALA_NORM vs PTGENDER (r=+0.122),
smaller than the plain model's AGE vs PTGENDER (r=-0.224) but, as before,
not zero -- DIDI is a group statistic and proxies are feature-level, so a
retrain under one constraint narrows but does not guarantee eliminating
the other.

## 5b. Two honest checks DIDI alone can't answer: calibration, and per-group accuracy

DIDI (used throughout section 5) checks only whether GROUP AVERAGES match.
It says nothing about whether the underlying probabilities are individually
correct, or whether the fairness fix helped some groups' own predictions
while quietly hurting others'. `calibration_and_group_accuracy_check.py`
checks both directly instead of leaving them as caveats.

### Is the hazard model's known calibration gap fixable?

`hazard_survival_model.ipynb` found a weighted mean absolute calibration gap
of 0.242 (AUC ranks patients correctly, but a predicted 0.3 doesn't reliably
mean a real 30% chance). Standard post-hoc sigmoid recalibration
(`CalibratedClassifierCV`, fit on train only, evaluated on test), applied to
the deployed `extended+anomaly/forest` tier:

| | Test AUC | Weighted mean \|calibration gap\| |
|---|---|---|
| Before (plain forest) | 0.807 | 0.208 |
| After (sigmoid-calibrated) | 0.808 | **0.010** |

A 95% reduction in the calibration gap, with AUC essentially unchanged
(ranking ability is preserved, as expected -- recalibration only rescales
the probability axis, it doesn't reorder patients). This is a cheap, clean
fix with no real downside found -- worth adopting as the default going
forward rather than leaving the gap undocumented-but-unfixed.

### Does the fairness fix hurt any single group's own accuracy? Yes, and it's worth stating plainly.

This is a genuinely important finding, not just a confirmation. Computing
AUC and calibration gap SEPARATELY per protected-attribute group, for the
plain fallback RISK_SCORE and the fairness-corrected (ExponentiatedGradient,
cost-aligned) one:

| Group | Plain AUC | Fair AUC | Plain calib. gap | Fair calib. gap |
|---|---|---|---|---|
| PTGENDER=1 | 0.878 | 0.798 | 0.229 | 0.476 |
| PTGENDER=2 | 0.882 | 0.820 | 0.228 | 0.476 |
| PTEDUCAT_BUCKET=0 | 0.890 | 0.803 | 0.233 | 0.471 |
| PTEDUCAT_BUCKET=1 | 0.861 | 0.811 | 0.223 | 0.482 |
| PTMARRY=1 | 0.880 | 0.808 | 0.218 | 0.470 |
| PTMARRY=2 | 0.892 | 0.894 | 0.294 | 0.499 |
| PTMARRY=3 | 0.892 | **0.637** | 0.243 | 0.502 |

Two things stand out, and neither was visible from the DIDI number alone:

1. **The calibration gap roughly DOUBLES for every single group**, not just
   on average -- ExponentiatedGradient's `DemographicParity` constraint
   operates on the fitted mixture's decision distribution, not on
   preserving well-calibrated soft probabilities, and this is the direct,
   previously-unmeasured cost of that. The recalibration fix above has not
   yet been applied to the fair model -- a natural next step.
2. **One subgroup (`PTMARRY=3`, n=226) loses AUC catastrophically**, from
   0.892 to 0.637 -- close to a coin flip for that specific group, even
   though the model's DIDI (a population-level average) improved
   substantially overall. This is exactly the failure mode a
   demographic-parity-only fairness metric cannot see: DIDI fell 76%
   (8.570 -> 2.098) at the same time one subgroup's predictions became
   almost useless.

**This changes the honest framing of the whole fairness mechanism.** The
~40% cost increase documented in section 5 is not the only price paid for
equalizing DIDI -- there is also a real, substantial, and until now
undocumented cost in per-group calibration and, for at least one subgroup,
in ranking accuracy itself. The fairness fix should be described from here
on as "cuts DIDI 76% at a real cost to per-group calibration, including a
near-total accuracy collapse for one subgroup" -- not as a clean win with
only a monitoring-budget cost attached.

## 5c. Calibrating the FAIR model too, and a clean (no-fairness-mechanism) comparison

Section 5b left two things open: the recalibration fix from Part A was
only demonstrated on the plain model, and the ~40% cost figure attributed
to the fairness mechanism was never checked for how much of it was really
a calibration artifact (an uncalibrated `RISK_SCORE` feeds directly into
`recommend_interval`'s cost formula, so a badly-scaled probability
inflates the reported cost regardless of fairness).
`calibrate_fair_and_plain_risk_scores.py` closes both gaps: it Platt-scales
the FAIR (ExponentiatedGradient) cascade the same way Part A calibrated the
plain one (a single held-out calibration split rather than 5-fold CV --
refitting ExponentiatedGradient 5 more times per tier was not worth the
~15 extra minutes for the same well-understood technique), and it also
rebuilds the cascade with NO fairness mechanism at all -- plain
RandomForestClassifier per tier, same fallback order -- calibrated the
same way, for a genuinely apples-to-apples comparison.

*(Figures below come from an 80/20 fit/calibration split of the existing
training set, so the absolute numbers differ slightly from section 5b's
full-train ones -- the pattern is what matters.)*

**Recalibration works on the fair model exactly as cleanly as it did on the
plain one.** Per-group calibration gap collapses from ~0.52 to ~0.01-0.04
for every single group, AUC per group unchanged (+0.000 everywhere):

| Group | Fair calib. gap, before | Fair calib. gap, after | AUC change |
|---|---|---|---|
| PTGENDER=1 | 0.529 | 0.014 | +0.000 |
| PTGENDER=2 | 0.506 | 0.014 | +0.000 |
| PTEDUCAT_BUCKET=0 | 0.514 | 0.018 | +0.000 |
| PTEDUCAT_BUCKET=1 | 0.523 | 0.011 | +0.000 |
| PTMARRY=1 | 0.516 | 0.014 | +0.000 |
| PTMARRY=2 | 0.548 | 0.021 | +0.000 |
| PTMARRY=3 | 0.532 | 0.017 | +0.000 |
| PTMARRY=4 | 0.470 | 0.037 | +0.000 |

**But this does not touch the AUC/accuracy problem.** Calibration only
rescales the probability axis; it cannot fix a model that genuinely
discriminates worse for one subgroup. Comparing the now-BOTH-calibrated
fair and plain cascades directly:

| Group | AUC delta (fair − plain) | Calib. gap delta (fair − plain) |
|---|---|---|
| PTGENDER=1 | −0.061 | −0.008 |
| PTGENDER=2 | −0.074 | −0.004 |
| PTEDUCAT_BUCKET=0 | −0.068 | −0.003 |
| PTEDUCAT_BUCKET=1 | −0.064 | −0.004 |
| PTMARRY=1 | −0.052 | −0.004 |
| PTMARRY=2 | −0.117 | −0.003 |
| PTMARRY=3 | −0.085 | −0.009 |
| PTMARRY=4 | −0.243 | +0.022 |

Once both models are properly calibrated, their per-group calibration gaps
are close (small deltas, fair usually a touch better) -- calibration was
never the fairness mechanism's real problem. The persistent, uncalibratable
cost is accuracy: the fair model's AUC is lower for every group, worst for
`PTMARRY=4` (n=107, −0.243). This confirms section 5b's finding was about
genuine discrimination loss, not a fixable scaling artifact.

**Policy-level comparison, both models calibrated:**

| | AUC | Calib. gap | DIDI | Cost | Catch rate |
|---|---|---|---|---|---|
| Fair (EG), uncalibrated | 0.736 | 0.518 | 2.36 | 16630 | 89.7% |
| Fair (EG), calibrated | 0.736 | 0.012 | **1.28** | 5123 | 65.8% |
| Plain (no fairness), uncalibrated | 0.802 | 0.209 | 8.20 | 11317 | 89.7% |
| Plain (no fairness), calibrated | 0.802 | 0.018 | 2.16 | 4926 | 69.2% |

Two things worth bringing to Monday:

1. **Much of the ~40% cost gap between fair and plain in section 5 was a
   calibration artifact, not a fairness cost.** Once both are calibrated,
   fair costs 5123 vs. plain's 4926 -- about 4% apart, not 40%. An
   uncalibrated `RISK_SCORE` was inflating the *reported* cost of fairness
   because the cost formula treats it as a literal probability; calibrating
   removes that distortion and reveals the real, much smaller cost gap.
   DIDI still favors the fair model clearly (1.28 vs 2.16) even after this
   correction, so the fairness mechanism's core benefit stands.
2. **Calibration is not "free" once it feeds a downstream decision.**
   Section 5b's framing (recalibration only rescales, it "doesn't touch
   which patients get ranked") is true for AUC, but not the whole story:
   `recommend_interval` uses `RISK_SCORE`'s raw magnitude, not just its
   rank, so rescaling it changes which patients get a shorter interval.
   Catch rate drops for BOTH models after calibration (fair: 89.7% ->
   65.8%; plain: 89.7% -> 69.2%) -- correctly-scaled probabilities are, on
   average, lower than the inflated raw ones near the deciding boundary, so
   fewer patients cross the threshold for a short interval. This is a real,
   open trade-off worth naming rather than a pure win: trustworthy cost
   numbers vs. a lower fraction of conversions caught early, under the
   current fixed cost ratio. Re-tuning `MISSED_CONVERSION_COST` /
   `CHECK_COST` after calibration (so the same clinical intent survives the
   rescaling) is the natural next step, not yet done here.

## 5d. Retuning the cost ratio after calibration: works for the plain model, not cleanly for the fair one

Section 5c ended on "retune `MISSED_CONVERSION_COST`/`CHECK_COST` to
recover catch rate" as the natural next step. `retune_cost_ratio_after_calibration.py`
actually does this -- sweeping the ratio from 20 (unchanged) up to 200
against the already-calibrated scores (no refitting: reuses
`calibrated_risk_scores.csv`) -- and the result is more mixed than that
sentence implied.

**Plain (no fairness) model: retuning works cleanly.** Ratio=100 recovers
catch_rate=89.7%, matching the original uncalibrated policy almost exactly,
at cost=10755 (close to the original uncalibrated plain cost of 11317) and
DIDI=7.63 (close to the original uncalibrated plain DIDI of 8.20). In other
words: retuning the ratio on a calibrated score reproduces almost the same
outcome the miscalibrated score was accidentally producing -- but now
because the model was deliberately told missed conversions matter 100x
more than a check, not because its probabilities were silently wrong.

**Fair (EG) model: retuning does not fully work, and it costs the fairness
benefit.** Even at ratio=200 (10x the original), catch rate only reaches
87.7%, short of the 89.7% target. Worse, DIDI rises steadily with the
ratio -- from 1.28 at ratio=20 to 8.65 at ratio=200, similar to or worse
than the plain model's UNCORRECTED baseline (8.570). The fairness gain is
not robust to moving the decision threshold: `ExponentiatedGradient`'s
`DemographicParity` constraint was trained against one specific implicit
operating point, and pushing `recommend_interval`'s effective threshold far
from that point (by raising the ratio) lets the disparity the constraint
was suppressing resurface.

| Ratio | Fair: cost | Fair: DIDI | Fair: catch% | Plain: cost | Plain: DIDI | Plain: catch% |
|---|---|---|---|---|---|---|
| 20 (original) | 5123 | 1.28 | 65.8% | 4926 | 2.16 | 69.2% |
| 100 | 10991 | 5.84 | 85.6% | 10755 | 7.63 | **89.7%** |
| 200 | 15266 | 8.65 | 87.7% (max) | 15306 | 2.69 | 97.3% |

**The honest conclusion for Monday:** "retune the cost ratio" is a clean,
complete fix for the plain model, but not for the fair one -- recovering
catch rate on the fair model means giving up most of the DIDI improvement
that was the whole point of the mechanism, and even then it doesn't fully
get there. This is a real, structural limitation worth naming rather than
assuming away: a fairness constraint fit at one operating point does not
automatically transfer to a different one, so if the clinical priority
(catch rate) and the cost ratio change, the fairness mechanism likely needs
to be RE-FIT against the new ratio, not just have its output re-thresholded.
That refit was not attempted here -- a concrete "what would you do with
more time" item.

## 5e. Refitting the fairness mechanism at the new ratio, instead of re-thresholding it

Section 5d's proposed next step: refit `ExponentiatedGradient` directly at
`ratio=100` (the value that recovered the plain model's original catch
rate) rather than re-thresholding a model fit at `ratio=20`.
`refit_fair_model_at_new_ratio.py` does this (temporarily overriding
`fairness_expgrad_pipeline.MISSED_CONVERSION_COST` before calling the
existing `fit_expgrad`, so this is the same fitting function, a different
cost input, not a reimplementation), then `compare_refit_vs_rethreshold.py`
compares it against the re-thresholded model and the plain model, all
decided at ratio=100:

| | Cost | DIDI | Catch rate | AUC |
|---|---|---|---|---|
| REFIT at ratio=100 | 10881 | **4.12** | 87.0% | 0.740 |
| RE-THRESHOLDED (ratio=20 fit, evaluated at 100 -- section 5d) | 10962 | 5.82 | 85.6% | 0.736 |
| Plain (no fairness), decided at 100 | 10727 | 7.66 | **89.7%** | 0.802 |
| Original fair fit, decided at its own ratio=20 (section 5c) | 5110 | 1.25 | 65.8% | 0.736 |

**Refitting strictly beats re-thresholding**: lower cost, lower DIDI
(better), higher catch rate, and slightly higher AUC -- confirming the
hypothesis that a fairness constraint fit at one operating point
under-performs when evaluated at a different one, and that refitting at
the actual operating point you intend to use is the correct fix, not an
academic nicety. It also meaningfully helped the single worst subgroup
found in section 5c: `PTMARRY=4`'s AUC, which was 0.541 in the original
ratio=20 fit, rises to 0.800 once refit at ratio=100 (full per-group table
in `refit_at_new_ratio_report.txt`).

**It is a genuine improvement, not a full fix.** Even refit at the "right"
ratio, catch rate (87.0%) still falls short of the plain model's 89.7%,
and DIDI (4.12) is higher than the original ratio=20 fit's 1.25 -- raising
the ratio inherently pushes the fair model's behavior closer to the plain
model's (more patients get flagged for shorter intervals regardless of
group), so some of the DIDI gain is structurally, not just accidentally,
traded away as the operating point moves. The honest summary for Monday:
refitting at the new ratio is clearly the right way to change the
mechanism's operating point (better than re-thresholding on every metric,
and it happens to also repair the weakest subgroup's accuracy), but there
is a real, direct trade-off between catch rate and DIDI baked into where
you set the ratio -- not a bug to fix, a design choice to make explicitly
and defend, ideally with the professor's input on which side of it matters
more clinically.

## 6. Explainability + fairness mechanism (plain-model version, for comparison)



`explainability_fairness.py` uses SHAP (`TreeExplainer`, exact for
RandomForest, no sampling approximation) to answer two questions with
evidence rather than assumption:

- **Does ANOMALY_SCORE genuinely contribute, or just correlate?** (Part 1,
  above) -- genuinely contributes; its SHAP attribution varies
  patient-by-patient rather than sitting at zero.
- **Is the `core/forest` tier's fairness problem a proxy effect?** (Part 2)
  -- yes, but a modest, distributed one: `AGE`'s SHAP attribution
  correlates with `PTGENDER` at r=-0.224 (the single strongest proxy found),
  consistent with `didi_breakdown.py`'s finding that this is spread across
  several features and the tier-assignment mechanism itself, not one
  dominant leak.
- **Does the fairness correction actually change the model's reasoning?**
  (Part 3) -- honestly, no: `fairness_correction.py` is a constant
  per-group shift applied to the final score, so it repairs the group-level
  DIDI statistic without touching the underlying RandomForest or its
  per-patient SHAP decomposition. Worth stating plainly if asked in the
  briefing, since it is a real and easily-defended limitation rather than a
  flaw to hide.

## 7. Explaining the OTHER end of the pipeline: the decision itself, not just the score

Sections 5 and 6 explain `RISK_SCORE` -- stage 2's own output, the hazard
model's prediction. They don't explain stage 3: why a given RISK_SCORE
turns into a 3, 6, or 12 month recommendation. That's a real gap, not a
minor one -- SHAP audits the score, but the actual clinical decision is one
layer further downstream.

The reason nothing SHAP-shaped was built for it: `decision_util.recommend_interval`
is not a learned, opaque model. It's a closed-form grid search over a
3-candidate menu, `expected_cost(interval) = routine_cost(interval) +
scaled_risk(interval) * MISSED_CONVERSION_COST`, entirely known in advance.
Approximating it with SHAP would be explaining a formula with a method built
for black boxes -- unnecessary and, for a 3-item menu, strictly less exact
than just showing the formula's own output per candidate.

`decision_mechanism.py` does exactly that: rebuilds the deployed, fair
RISK_SCORE (importing `fairness_expgrad_pipeline.py`'s own functions, so
it's guaranteed to match section 5's mechanism, not a redone
approximation), picks three real test-set patients at the 10th/50th/90th
RISK_SCORE percentile, and plots the exact expected-cost breakdown behind
each one's recommended interval -- routine cost vs. expected
missed-conversion cost, per candidate, with the winning interval
highlighted and its margin over the runner-up reported.

| Example patient | RISK_SCORE | Recommended interval | Margin over runner-up |
|---|---|---|---|
| Low risk (10th pct) | 0.063 | 12 months | 14.0% cheaper |
| Median risk (50th pct) | 0.614 | 3 months | 13.1% cheaper |
| High risk (90th pct) | 0.899 | 3 months | 22.7% cheaper |

The mechanism is visible directly in the numbers: the missed-conversion
cost term scales with `interval / REFERENCE_INTERVAL_MONTHS`, so it grows
fastest for the longest candidate as risk rises -- which is exactly why the
optimal interval flips from 12 months at low risk to 3 months at high risk,
and why the margin widens as risk climbs (the decision gets more clear-cut,
not less). It's the same `MISSED_CONVERSION_COST = 20x CHECK_COST`
asymmetry that motivated section 5's ~40% cost increase, showing up again
at the very last step of the pipeline -- one consistent cost story end to
end, not three independent ones.

## 8. Developing stage 3 further: trajectory-aware and learned policies

`recommend_interval`'s own docstring names two things beyond the hardcoded
grid search as intended next steps: a trajectory-aware version (already
written, never wired in) and, eventually, a learned or DFL-based policy.
`stage3_trajectory_and_learned_policy.py` tries both, reusing
`calibrated_risk_scores.csv` (no refitting needed for either).

**Part A -- `recommend_interval_trajectory`**, which projects risk forward
using each patient's own observed risk slope instead of treating today's
snapshot as constant. 79.6% of rows have a prior visit to compute a slope
from. Adding it changes a real share of decisions (6.9% for the plain
model, 24.4% for the fair one) and recovers a meaningful chunk of the
catch-rate lost to calibration (plain: 69.2% -> 76.7%; fair: 65.8% ->
74.0%) -- but at a real DIDI cost (plain: 2.16 -> 3.31; fair: 1.28 -> 4.27).
The fair model's decisions move much more (24.4% vs 6.9%), likely because
its calibrated scores sit lower and closer to the decision boundary, so a
slope term more often tips them across it. Net read: trajectory-awareness
is a genuine, usable lever for the catch-rate/DIDI trade-off already being
discussed (sections 5d/5e), not free information -- it buys catch rate at
a fairness cost, the same shape of trade-off raising the ratio has.

**Part B -- a learned interval-choice classifier**, trained on a nested
holdout within the test set (test_panel's patients further split
clf_train/clf_test by RID, so the model never sees clf_test patients in
either its features or its bootstrap labels), predicting the interval
directly from the full `extended+anomaly` feature set plus the risk score,
using the current formula's own output as training labels. Result: 99.8%
agreement with the formula on clf_test, and essentially identical
aggregate cost/DIDI/catch-rate. Feature importances show `RISK_SCORE`
dominates (0.556) but is not everything -- `PTAU` (0.103), `TAU` (0.081),
and `AMYGDALA_NORM` (0.072) all carry real weight. **The honest reading**:
those other features matter to the classifier in isolation, but they're
already so strongly summarized by `RISK_SCORE` (which was, after all, fit
on exactly those same features) that using them directly barely changes
any actual decision. This is a clean negative result for "a learned policy
would clearly do more" -- at least with this feature set, it would mostly
reproduce the formula while giving up the formula's key advantage: the
closed-form version can be explained exactly, in three lines of arithmetic
per patient (section 7); a learned classifier could not, without the same
SHAP-style approximation stage 2 already needs.

## 9. Wiring in the other two notebooks/decision_support folders

Confirmed via the project's own `NOTES.md`: Point 3 (Decision Support) is
three parallel tracks reading the same risk column -- the cost model
(`temporal_window/`), the fairness audit, and attribution
(`risk_factors/`). Only the cost model was previously wired to this
pipeline's actual deployed scores (sections 5d/5e/8).
`wire_temporal_window_and_risk_factors.py` wires in the other two pieces.

**Part A completes `temporal_window/`** by adding `1_fixed_policy.ipynb`'s
zero-information baseline (same interval for everyone) alongside the
already-wired snapshot and trajectory policies, all on the actual
deployed calibrated scores:

| | Cost | DIDI | Catch rate |
|---|---|---|---|
| Fixed (no patient info) | 6437 (plain) / 6477 (fair) | 0.000 | 95.2% |
| Snapshot adaptive | 4926 / 5123 | 2.16 / 1.28 | 69.2% / 65.8% |
| Trajectory adaptive | 4969 / 5308 | 3.31 / 4.27 | 76.7% / 74.0% |

Read as a floor-to-ceiling story: the fixed policy is expensive and
catches the most (it over-checks everyone, including low-risk patients),
zero DIDI by construction since nobody is treated differently. Each
successive policy trades some of that catch rate for lower cost by using
more patient information (current level, then direction). This is a clean
way to show the professor what personalization is actually buying, and at
what price, across the full spectrum from none to a lot.
(`4_forecast_adaptive.ipynb` is left out -- it needs multi-horizon
forecast columns this pipeline's cascade doesn't produce; building those
would be a new stage-2 extension, not just wiring in an existing one.)

**Part B wires in `risk_factors/1_global.ipynb`'s attribution method**
(Lasso, Random Forest importance, and SHAP -- all already in
`decision_util.py`, reused not reimplemented) onto the deployed plain and
fair calibrated `RISK_SCORE`, regressing each against the `extended+anomaly`
feature set. **This surfaces a genuinely new, and concerning, finding.**
Forest R2 is high for both (0.816 plain, 0.762 fair -- Lasso's R2 is too
low, 0.18/0.05, for its signed weights to be trustworthy, shown only for
completeness), so the tree-based and SHAP views are the ones to trust:

| Feature | Plain SHAP | Fair SHAP | Change |
|---|---|---|---|
| NOMINAL_MONTH | 0.0097 | **0.0166** | dominant driver for fair, was #3 for plain |
| SUMMARY_SUVR (amyloid PET) | 0.0102 | 0.0003 | **-97%** |
| AMYGDALA_NORM | 0.0108 | 0.0030 | -72% |
| HIPPO_NORM | 0.0051 | 0.0072 | +41% |

Both the Forest importances and SHAP agree on the shape of this: the
plain model's score is driven by a genuine mix of amyloid PET
(`SUMMARY_SUVR`) and structural MRI (`AMYGDALA_NORM`, `HIPPO_NORM`)
biomarkers, with `NOMINAL_MONTH` (time since baseline) a comparable but
not dominant factor. The fair (EG) model's score becomes dominated by
`NOMINAL_MONTH` alone, while `SUMMARY_SUVR`'s contribution nearly
vanishes. **This directly contradicts the earlier, reassuring finding in
section 6** (`explainability_fairness.py`'s Part 3: "the fairness
correction doesn't touch the model's reasoning") -- but that finding was
about the OLD constant per-group shift fix, applied on top of an
unchanged model. The ACTUAL deployed mechanism, `ExponentiatedGradient`,
refits the model itself under a fairness constraint, and this shows it
refits toward relying on visit timing rather than amyloid biomarkers to
satisfy that constraint. This is a real, previously-undocumented
mechanism worth naming plainly: the fairness fix isn't free even in a
sense beyond cost, calibration, or per-group AUC (sections 5-5e) -- it
also appears to be substituting a demographically-correlated but
clinically-weaker signal (how long a patient has been in the study) for
the genuine biomarkers the plain model relies on, likely because
`NOMINAL_MONTH` is an easier lever for satisfying `DemographicParity`
than the actual, harder-to-adjust biomarker relationships. Worth raising
directly with the professor: is optimizing DIDI via a proxy like this an
acceptable mechanism, or does it need to be constrained to leave the
model's genuine clinical reasoning intact?

## 10. Trying to fix the proxy-reliance problem: EqualizedOdds instead of DemographicParity

Section 9's finding (the fair model leans on NOMINAL_MONTH instead of real
biomarkers) suggested a concrete hypothesis: DemographicParity only
requires equal AVERAGE outcomes across groups, by whatever means gets
there cheapest, while EqualizedOdds requires equal true/false positive
rates PER GROUP conditioned on the real label -- a constraint that, in
principle, can't be satisfied by leaning on a feature uncorrelated with
genuine risk, since doing so would hurt real positives and negatives
alike, not just shift a group average.
`fairness_expgrad_equalized_odds.py` tests this directly -- identical base
estimator, cost-aligned objective, fallback cascade, and Platt calibration
to section 5c, only the constraint swapped.

**The hypothesis did not hold up.** EqualizedOdds does not fix the proxy
reliance, and it is a substantially worse fairness mechanism on this data:

| | Cost | DIDI | Catch rate | AUC |
|---|---|---|---|---|
| Fair (EqualizedOdds) | 5086 | 2.58 | 69.2% | 0.759 |
| Fair (DemographicParity), original (5c) | 5110 | **1.25** | 65.8% | 0.736 |
| Plain (no fairness) | 4914 | 2.13 | 69.2% | 0.802 |

EqualizedOdds's DIDI (2.58) is barely better than the plain model's own
(2.13) -- it gives up almost all the fairness benefit DemographicParity
was providing, while still costing real AUC. And the attribution check
shows the NOMINAL_MONTH reliance is essentially unchanged: SHAP importance
0.0146 (EqualizedOdds) vs 0.0166 (DemographicParity) vs 0.0097 (plain) --
both fair variants lean on it far more than the plain model does, and
SUMMARY_SUVR's contribution drops to zero under EqualizedOdds too (0.0000,
even more complete than DemographicParity's 0.0003).

**Honest reading:** the proxy reliance is not really about which fairness
constraint is used. It more likely reflects that NOMINAL_MONTH is simply
the cheapest lever available to ExponentiatedGradient's underlying
reduction (reweighting/resampling by visit timing shifts group-level
statistics cheaply, without needing to touch the harder amyloid/atrophy
relationships) regardless of which fairness criterion the reduction is
asked to satisfy. This is a genuinely open problem, not a solved one -- a
constraint swap was the natural first thing to try, and it's worth being
able to say plainly that it was tried and didn't work, rather than
presenting an untested fix as if it were the answer. A more targeted next
step (not attempted here) would be excluding NOMINAL_MONTH specifically
from what the reduction is allowed to use as a lever, while keeping it
available to the underlying predictor -- a genuinely different mechanism
change than swapping the fairness criterion.

## 11. Are the decisions actually relevant? Oracle regret, not just cost/DIDI/catch

Section 8 and the ratio=160 case in section 5d both showed the same
failure mode from different angles: cost/DIDI/catch-rate numbers can look
reasonable even when a policy has stopped meaningfully using the risk
score (piling decisions onto one interval, or losing the score's own
ranking). `oracle_regret_analysis.py` measures decision relevance
directly, the standard Predict-then-Optimize/DFL way (08-dfl-tf, already
cited by `recommend_interval`'s own docstring):

- **Interval distribution and Spearman correlation** between `RISK_SCORE`
  and `RECOMMENDED_INTERVAL`, to catch degenerate policies directly rather
  than inferring them from aggregate numbers.
- **Regret against an ORACLE** -- `EVENT_AT_VISIT` is the wrong target for
  this (it's whether THIS visit already reflects a past worsening, not
  whether the NEXT visit will, which is what `catch_rate` checks -- an
  error caught when the first attempt produced an oracle with a LOWER
  catch rate than the real models, impossible for genuine perfect
  information). Fixed to use the real forward-looking outcome
  (`compute_diagnosis_worsening`'s `DIAGNOSIS_WORSENED_NEXT`) fed through
  the exact same `recommend_interval`/`ConversionCostModel` formula every
  other policy uses. `regret_pct = (policy_cost - oracle_cost) /
  (naive_cost - oracle_cost) * 100` -- 0% is oracle-optimal, 100% is no
  better than the zero-information fixed policy (section 9).

| Policy | Cost | Regret |
|---|---|---|
| Plain, calibrated | 4926 | 43.3% |
| Fair (DemographicParity), calibrated | 5123 | 50.7% |
| Plain, trajectory-aware | 4969 | 44.9% |
| Fair (DP), trajectory-aware | 5308 | 57.6% |

Every real policy has a strong negative risk/interval correlation (-0.40
to -0.64), confirming they genuinely use the score's ranking, unlike the
fixed policy's undefined (zero-variance) case. But even the best policy
tried is still 43% of the way from oracle-optimal back to doing nothing --
real, quantified headroom remains regardless of which model is used.

**A genuinely important side finding**: the oracle itself -- built from
TRUE future outcomes, no model involved -- has DIDI = 1.488. That is the
level of group disparity genuinely justified by real differences in
conversion rates between groups, not a model artifact. The plain model's
own DIDI (2.16-8.57 across this session's variants) sits well above that
natural baseline -- evidence of real, excess, model-driven unfairness
worth correcting, not just population variation. The fair
(DemographicParity) model's DIDI (1.25-2.10) sits close to or even
slightly below that natural level. So the oracle analysis complicates
"just use the plain model" more than it supports it: the plain model
demonstrably has MORE disparity than the data itself justifies, even
though it also has meaningfully lower regret (decision quality) than the
fair model. Neither model is simply "the right answer" -- the honest
picture is a genuine, now-quantified trade-off between excess disparity
(plain) and decision quality plus an unresolved proxy-reliance problem
(fair), not a case either model clearly wins.

## 12. Further improvements, using techniques the course actually covers

Concrete next steps, each tied to a specific problem found this session,
using material the course teaches rather than reaching outside it:

**Decision-Focused Learning, to directly attack the regret (08-dfl-tf).**
Section 11 found 43-58% regret across every policy tried -- stage 2 is
trained to maximize AUC, a proxy blind to the actual cost structure it
feeds into. DFL trains the predictor directly against
`recommend_interval`'s own expected-cost objective instead of a proxy
loss. `recommend_interval`'s own docstring already names this as the
intended next step, and NOTES.md quotes 08-dfl-tf lesson 2 directly: DFL
is "most useful exactly when the decision has a recourse-like structure...
missing one check does not mean the patient is lost" -- describing this
exact monitoring-interval problem. Real, nontrivial work (a differentiable
cost function, backprop through the decision), not yet attempted.

**Boruta feature selection, to settle the NOMINAL_MONTH question with a
course tool rather than by inspection (06-at lesson 5).** Sections 9-10
diagnosed but didn't resolve the proxy-reliance finding. NOTES.md already
names Boruta ("all-relevant feature selection... to sanity check which
biomarkers the model leans on") as planned but never run. Running it would
answer, rigorously, whether NOMINAL_MONTH is genuinely relevant or only
looks useful to the reduction because of a correlation -- and gives a
principled basis for excluding it as a lever, rather than an ad hoc
removal.

**A post-processing fairness alternative, to sidestep the proxy problem
structurally (07-ciml).** Both mechanisms tried (section 5, section 10:
DemographicParity, EqualizedOdds) are in-training reductions, and both
leaned on the same proxy. Per-group threshold adjustment AFTER training,
without retraining the predictor, can't lean on a proxy feature the same
way, since it never touches the model's inputs -- only the decision
cutoff per group. Worth trying specifically because the first two share a
failure mode that a structurally different technique might not.

**Real multi-horizon forecasts, to unlock the one temporal_window policy
left out (05-pm lesson 6, already used in hazard_survival_model.ipynb's
own forecast columns).** Section 9 excluded `forecast_adaptive` because
this pipeline doesn't produce `CONVERSION_PROB_3M/6M/12M`. Building those
for the actual deployed model (not the historical placeholder-based ones)
would let stage 3 use genuinely different information per candidate
interval instead of one scaled snapshot number -- NOTES.md's own numbers
suggest this changes behavior meaningfully (recommended 3 months for 99.5%
of rows in the original run), worth re-testing against the deployed,
calibrated scores.

Of these, Boruta and the post-processing alternative are the most
tractable to build quickly; DFL is the strongest answer but the most work.

## Files

- `wire_full_pipeline.py` -- builds the sequential 1->2->3 pipeline, the
  best-single-tier result, and the soft fallback cascade.
- `wire_full_pipeline_gmm.py`, `full_pipeline_report_gmm.txt`, `../gmm_key_join.py`, `../method_b_autoencoder_hi/gmm_hi_trajectories_keyed.csv` -- the GMM-based rerun of the same pipeline, section 1b.
- `hazard_auc_with_vs_without_anomaly_score.csv`, `full_pipeline_report.txt`
  -- AUC comparison table and full run output for the above.
- `didi_breakdown.py`, its printed output -- diagnoses the fallback's DIDI
  regression (per-tier DIDI variance + tier-assignment correlation).
- `fairness_correction.py`, `fairness_correction_report.txt` -- the
  sklearn post-hoc recalibration fix actually used.
- `fairness_correction_tf.py`, `fairness_correction_tf_report.txt` -- the
  in-syllabus Lagrangian alternative, kept as a documented comparison.
- `fairness_correction_reduction.py`, `fairness_correction_reduction_report.txt`
  -- Reweighing and fairlearn's ExponentiatedGradient, the genuine in-training
  alternatives described in section 4.
- `fairness_expgrad_pipeline.py`, `fairness_expgrad_report.txt`,
  `shap_summary_extended_anomaly_expgrad.png`, `shap_summary_core_expgrad.png`,
  `shap_dependence_anomaly_score_expgrad.png` -- the ExponentiatedGradient
  fallback actually described as the mechanism used, section 5.
- `calibration_and_group_accuracy_check.py`, `calibration_check_report.txt`, `calibration_before_after.png`, `group_accuracy_report.txt`, `group_accuracy_comparison.png` -- the calibration fix and the per-group accuracy audit, section 5b.
- `calibrate_fair_and_plain_risk_scores.py`, `risk_score_calibration_comparison_report.txt`, `risk_score_calibration_comparison.png` -- calibrating the FAIR model too, and the calibrated fair-vs-no-fairness-mechanism comparison, section 5c.
- `calibrated_risk_scores.csv` -- the four calibrated/uncalibrated risk score columns from section 5c, saved so downstream analysis can reuse them without refitting.
- `retune_cost_ratio_after_calibration.py`, `cost_ratio_retune_report.txt`, `cost_ratio_retune.png` -- re-tuning the cost ratio to recover catch rate post-calibration, section 5d.
- `refit_fair_model_at_new_ratio.py`, `refit_at_ratio100_scores.csv`, `refit_at_new_ratio_report.txt`, `compare_refit_vs_rethreshold.py`, `refit_vs_rethreshold_report.txt` -- refitting the fairness mechanism at the new ratio instead of re-thresholding it, section 5e.
- `stage3_trajectory_and_learned_policy.py`, `stage3_trajectory_and_learned_report.txt` -- trying the trajectory-aware and learned interval-choice policies, section 8.
- `wire_temporal_window_and_risk_factors.py`, `wire_temporal_window_and_risk_factors_report.txt`, `risk_factors_attribution_plain_vs_fair.png` -- wiring in the fixed policy baseline and the risk_factors attribution track onto the deployed scores, section 9.
- `fairness_expgrad_equalized_odds.py`, `fairness_expgrad_equalized_odds_report.txt`, `equalized_odds_attribution_comparison.png` -- testing (and ruling out) EqualizedOdds as a fix for the proxy-reliance problem, section 10.
- `oracle_regret_analysis.py`, `oracle_regret_report.txt` -- measuring decision relevance directly (interval distribution, risk/interval correlation, oracle regret), section 11.
- `decision_mechanism.py`, `decision_mechanism_report.txt`, `decision_mechanism_breakdown.png` -- the stage-3 (decision) explainability mechanism, section 7.
- `explainability_fairness.py`, `explainability_fairness_report.txt`,
  `shap_summary_extended_anomaly.png`, `shap_summary_core.png`,
  `shap_dependence_anomaly_score.png`, `shap_waterfall_anomaly_high.png`,
  `shap_waterfall_anomaly_low.png` -- the explainability + fairness audit
  described in section 6.
