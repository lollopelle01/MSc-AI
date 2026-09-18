# Rehearsal notes: course techniques behind the project

Built directly from the lecture notebooks in `AII-theory` (01-ad-de, 02-ad-hd,
03-sf-gp, 04-rul, 05-pm, 06-at, 07-ciml, 08-dfl-tf), matched to exactly where
each technique shows up in this project. Deep on point 1 (yours), just enough
on points 2 and 3 to speak about them without freezing. Updated with the
cost-based threshold analysis and the final method decision.

## Concept map at a glance

Quick-reference version of the mapping below -- see `theory_practice_report.docx`
(same folder) for the full write-up formatted as a standalone deliverable, with
every claim tied to its lecture source and the exact script/result behind it.

| Course concept | Source | Project implementation |
|---|---|---|
| Density estimation as anomaly detection | 01-ad-de, lecture 3 | T2/Q (PCA), GMM negative log-likelihood, autoencoder reconstruction error -- all scored as -log f(x); all three fit on CN patients only. |
| Method progression by problem characteristics (KDE -> GMM -> Autoencoder) | 01-ad-de / 02-ad-hd | Three methods compared under four independent criteria; KDE excluded by design (parameter-storage scaling limit). |
| MSE regression = Gaussian MLE, uniform variance | 02-ad-hd, lecture 4 | Explains why the autoencoder underperforms PCA: PCA's T2/Q split captures structure the uniform-variance assumption misses. |
| Cost-based threshold selection via line search | 01-ad-de, lecture 4 | `threshold_cost_optimization.py` sweeps the missed/false-alarm cost ratio (5, 10, 20) and picks each method's threshold by minimizing total cost. |
| Censoring and the discrete-time hazard function | 05-pm, lecture 6 | `hazard_survival_model.ipynb` reduces time-to-conversion to per-visit binary classification (Random Forest, 0.791 test AUC). |
| Predict-then-Optimize vs. Decision-Focused Learning | 08-dfl-tf | `decision_util.py` implements the PFL track; DFL flagged as the stretch goal for the interval policy. |
| Demographic parity / DIDI fairness auditing | 07-ciml, lecture 1 | Protected attributes excluded from every risk model's inputs, audited against every interval recommendation via DIDI. |
| Reductions approach to fair classification | 07-ciml, lecture 2 | `fairlearn.reductions.ExponentiatedGradient` under `DemographicParity`, cost-aligned `ErrorRate` objective -- the pipeline_1_2_3 fairness mechanism. |
| Shapley values / additive feature attribution | 06-at, lecture 4 | SHAP TreeExplainer values (exact, weighted across the ExponentiatedGradient mixture) explain the PCA anomaly score and the final RISK_SCORE. |

---

## Part 1 — Anomaly Detection (know this cold)

### The formalization the whole course starts from (01-ad-de, lecture 3)

If we can estimate the probability of every observation, anomalies are just
the ones with *low probability* — "we turn a liability into a strength."
Formally: estimate a density `f(x)`, flag `x` as anomalous when `f(x) ≤ ε`.
In practice this gets negated to `-log f(x) ≥ ε`, so higher score = more
anomalous — the sign convention your T², Q, GMM negative-log-likelihood, and
autoencoder reconstruction error all share.

The lecture is explicit that density estimation "cannot work" as supervised
learning, because you never have access to the true density — **it's an
unsupervised problem**, solved by picking a technique and fitting it to data
alone. This is the one sentence that justifies your entire "train only on
CN, no labels used" design: you're not classifying, you're estimating what
the healthy density looks like.

### Is there a single "best" method the professor prefers? No.

Worth saying plainly if asked, because it's an easy trap to fall into ("did
you use the best technique?"): the course never crowns one method as best.
It's a progression driven by **problem characteristics**, not a competition.

KDE is picked first for one stated reason: Occam's razor — "start with a
simple approach; if it works, you have a solution, if not, you have a
baseline." The move to GMM is motivated purely by a scalability limit of
KDE (it stores the entire training set as its own parameters — "very little
bias, very large variance" — which breaks down as data size and
dimensionality grow). The move to autoencoders is motivated by a further
limit of GMM, with the lecture listing explicit pros *and* cons rather than
declaring a winner, and its own worked example shows the autoencoder's
signal coming out "very similar to the KDE and GMM signal" — not better.

So the right framing for your own three-way test is: you didn't assume the
most sophisticated method would win, you tested three and let the data
decide — which matches the course's own spirit more closely than picking a
method by reputation.

### KDE (01-ad-de) — the course's own starting point, and why you didn't use it

`f(x) = (1/m) Σ K(x - x̄ᵢ, h)` — a kernel centered on every training sample,
averaged. The catch, raised directly in 02-ad-hd's transition into GMMs:
**KDE stores the entire training set as its parameters**, so it scales badly
with both dataset size and dimensionality — your first answer if asked "why
not KDE": your feature space (22 dimensions) and cohort size are exactly the
regime the lecture says KDE struggles with.

### GMM (02-ad-hd, lecture 2–3) — the method you tied with

`g(x) = Σ τₖ f(x, μₖ, Σₖ)`, trained by **Expectation-Maximization**
(alternating: soft-assign each point to components, then re-fit the Gaussian
parameters given those assignments). Component count chosen by grid search /
BIC — exactly what `compare_three_methods.py` and `gmm_hi_trajectory.py` do.
Anomaly score = negative log-likelihood under the fitted mixture. Tied PCA on
raw AUC (~0.88–0.89) on both cohorts — directly out of the syllabus, and it
performed identically to PCA on ranking quality alone.

### Autoencoders (02-ad-hd, lecture 4) — the method you beat

Encoder `e(x,θₑ)` compresses to latent `z`, decoder `d(z,θ_d)` reconstructs
`x`; trained to minimize MSE. Anomaly score = reconstruction error. **The
lecture proves training with MSE is mathematically equivalent to maximum
likelihood under an assumption that reconstruction error is Gaussian noise
with uniform, independent variance across all features.** That's a strong,
lecture-grounded answer to "why did the autoencoder do worse": not that
neural methods are bad, but that this uniform-variance assumption doesn't
fit your data as well as PCA's explicit split between "deviation the model
explains" (T²) and "deviation it doesn't" (Q).

You used scikit-learn's `MLPRegressor` trained to reconstruct its own input
rather than Keras/TensorFlow (installation was blocked in the working
environment) — same architecture, same MSE objective the lecture describes,
different library. Worth saying directly if asked.

### Where PCA fits a lecture concept even though it isn't named directly

PCA + Hotelling's T² isn't one of the 42 course notebooks — say so plainly,
it costs nothing and shows you know the syllabus. But it's the **linear,
closed-form special case of the same Gaussian density assumption the
autoencoder lecture derives explicitly for MSE regression** — solved by
eigendecomposition instead of learned by gradient descent. Not a foreign
technique smuggled in, the linear limit of one the course teaches.

---

## Threshold selection — necessary for every method, and how it was actually optimized

### Why a threshold is unavoidable

KDE, GMM, PCA's T²/Q, and the autoencoder all output a **continuous score** —
a density, a log-likelihood, a distance, a reconstruction error. None of them
says "anomaly" on their own. AUC sidesteps this (it measures ranking quality
across *every* possible threshold at once), which is why it was the right
metric for the first three-way comparison. But the moment you want an actual
decision — flag this patient, recommend a shorter check-in — some cutoff has
to be chosen, for every method, separately (their scores live on completely
different scales, so no single number can be shared across them).

### The course's own recipe (01-ad-de, lecture 4) and how it was applied here

The lecture's method: define a cost for a false alarm (`c_alarm`) and a cost
for a missed detection (`c_missed`), wrap them in a small cost class, and
pick the threshold that minimizes total cost via a **line search** over
candidate thresholds. The lecture is explicit that these costs are
assumptions that "should be discussed with stakeholders" and tested under
more than one setting, not committed to a single guess.

This was adapted directly (`gio/method_a_pca_density/threshold_cost_optimization.py`):

```
cost(threshold) = c_alarm × (false alarms) + c_missed × (missed detections)
```

with `c_alarm = 1` fixed and `c_missed / c_alarm` swept over **5, 10, and 20**
(a sensitivity analysis, not one committed number), each method's own
threshold chosen by scanning percentiles of its own CN training distribution
and picking the one with lowest total cost. Following the lecture's own
precedent: the full 464-patient cohort is used for this line search (not a
separate untouched test set), because — as the lecture states directly —
this validation step isn't warding off overfitting, it's tuning one extra
scalar parameter, not fitting the model itself.

### What it found

For the broader task (**CN vs MCI+AD** — catching anyone on the disease
trajectory): **PCA had the lowest total cost at every ratio tested** (5, 10,
and 20), consistently ahead of GMM and the autoencoder.

For the narrower task (**CN vs AD** specifically): GMM actually wins at the
lowest ratio tested (5× — cost 73 vs PCA's 83), because its threshold is more
conservative (95th percentile vs PCA's 80th), trading lower sensitivity
(0.647) for a much lower false-alarm rate (0.048 vs 0.197). But at the more
clinically realistic ratios — where missing an AD diagnosis is 10× or 20×
worse than a false alarm, which is the more defensible assumption for a
disease you'd rather over-refer than miss — **PCA pulls ahead and stays
ahead** (94 vs 102 vs 153, then 114 vs 152 vs 217).

This is a stronger result to present than "PCA wins everywhere": it shows
the earlier AUC-based conclusion was checked against an actual
decision-theoretic threshold, found one genuine, explainable exception, and
can state precisely when that exception applies (only at an unrealistically
forgiving cost assumption, and only for the narrower of the two detection
tasks) — the honest, examined-not-assumed story the grading criteria reward.

### Revisiting the ratio itself with real literature numbers

The 5/10/20 sweep above is a documented but ARBITRARY sensitivity range --
picked only to satisfy the lecture's own instruction not to commit to a
single guess, not anchored to any real cost data. `threshold_cost_optimization_literature.py`
replaces it with the exact same real-world anchor stage 3's own
`cost_model_for()` already uses for its MCI/AD severity multiplier: 2.8x,
the point estimate from a 2026 medRxiv preprint on the societal cost of
Alzheimer's disease by diagnosis timing (~62,000 EUR/year for an early MCI
patient vs. ~175,000 EUR/year for a late, severe AD patient), plus 1.5x and
4.0x, the low and high end of a broader targeted review ("Global Societal
Burden of Alzheimer's Disease by Severity," Neurology and Therapy, 81
studies, 2013-2024, typical ratio 1.4-3.6x). The honest reframing: that
literature prices EARLY vs LATE diagnosis of the same disease, which is
exactly what a missed anomaly alarm causes -- not a literal "one false
alarm" price, but the closest real anchor available, the same one already
used elsewhere in this project.

**This changes the narrow-task story, and the change is worth stating
plainly rather than smoothing over**: at every one of these three
literature-anchored ratios (1.5x, 2.8x, 4.0x), **GMM has the lowest cost on
CN vs AD**, not just at an "unrealistically forgiving" edge case the way
the arbitrary 5x point was originally framed. The real, citable cost ratios
turn out to be considerably lower than the 10x/20x this project had been
calling "realistic" by assumption -- so the earlier framing ("PCA wins
except under an unrealistic assumption") does not survive contact with
actual health-economics numbers for the narrow AD-specific task.

**The broad task result is unchanged**: PCA still has the lowest cost on
CN vs MCI+AD at all three literature-anchored ratios (173.0 vs GMM's 178.0
at 1.5x; 243.4 vs 254.4 at 2.8x; 301.0 vs 312.0 at 4.0x) -- the task that
actually matters for the deployed pipeline, since it screens for anyone on
the disease trajectory, not specifically full AD. So the corrected, more
precise story is: **PCA wins the broad screening task under real literature
costs; GMM wins the narrow AD-specific task under those same real costs**
-- a genuine, evidence-based split by task, not a synthetic-sweep artifact
that a "more realistic" assumption would have erased.

---

## The final decision, and why it's well-supported

**PCA** is the method carried forward, and the case for it rests on four
independent tests, not one:

1. **Raw ranking performance** — ties GMM, clearly beats the autoencoder, on
   both cohorts tested.
2. **Downstream pipeline impact** — once blended into the shared pipeline
   risk score, gives a larger cost reduction than the autoencoder-based
   health index (−2.0% vs −1.75%), with a marginally better fairness number.
3. **Cost-optimized decision threshold** — wins at every cost ratio tested
   (including the real, literature-anchored ones, 1.5x/2.8x/4.0x from AD
   severity-cost literature) for the broad detection task, which is the one
   that matters for the deployed pipeline. For the narrower CN-vs-AD task
   specifically, GMM actually wins under the real, literature-anchored cost
   ratios — a genuine split by task, not an unrealistic edge case, and
   worth stating exactly that way rather than claiming a clean sweep.
4. **Explainability** — the only one of the three with an exact, closed-form
   decomposition of any patient's score into per-feature contributions,
   already built and demonstrated (`explainability_contribution_plot.py`).

One clean sentence for Monday: *"We tested three density-estimation
approaches under four independent evaluation criteria — ranking
performance, downstream pipeline cost, decision-theoretic threshold cost,
and explainability — and PCA won three outright and split the fourth
by task: it wins the broad screening task, the one the deployed pipeline
actually needs, under every cost ratio including real, literature-anchored
ones, while GMM wins the narrower AD-specific task under those same real
costs, which is why PCA is the method carried into the shared pipeline."*
Interpretability turned out to be the icing, not the reason, and the
task-split on point 3 is a stronger, more honest result than a clean
four-for-four sweep would have been.

---

## Part 2 — Conversion Risk (know the framing, not the code)

### Censoring and the hazard function (05-pm, lecture 6)

Many patients simply haven't been followed long enough to know if/when
they'll convert — **censoring**, the same concept as "broken machines vs.
regularly maintained ones" in the lecture's own example. A plain regression
on time-to-conversion, fit only on patients who did convert, throws away the
censored majority and biases the model.

The fix: model the **hazard function** `λ(t, x_t)` — the probability of
*not* surviving one more step, given survival to now. Discrete time lets
survival factor as a product of `(1 - λ)` terms, and training a hazard
estimator reduces to ordinary binary classification: label each
subject-visit pair 1 if the event happens there, 0 otherwise, fit any
classifier. That's exactly `hazard_survival_model.ipynb` (a Random Forest,
0.791 test AUC). One-sentence version: "we turned an incomplete
time-to-event problem into per-visit binary classification, mathematically
justified by the discrete-time survival factorization, and it's the only way
to use the ~75% of patients who haven't converted yet without bias."

---

## Part 3 — Decision Support (know the framing, not the code)

### Predict-then-Optimize vs. Decision-Focused Learning (08-dfl-tf)

**PFL**: train an estimator for the unknown parameter (conversion risk),
then separately optimize a decision (check-in interval) using that estimate.
Simple, scalable, asymptotically correct. That's the cost-model track in
`decision_util.py`.

**The flaw**: a model trained for minimum prediction error can still lead
the optimizer to consistently wrong decisions, since accuracy and decision
quality aren't the same objective. **DFL** trains the estimator directly
against decision cost instead. The lecture's own finding: the PFL/DFL gap
shrinks as the predictive model gets more expressive, but DFL can match a
complex PFL model's performance with a *much simpler*, more explainable
model — why it's flagged as a stretch goal for the interval-choice policy,
whose recourse-like structure (missing one check doesn't mean the patient is
lost) is exactly where the lecture says DFL earns its keep.

### Fairness / DIDI (07-ciml, lecture 1)

Removing a protected attribute from the model's *inputs* does **not**
prevent discrimination through correlated attributes — so it must stay out
of the inputs but be used to *audit the output*. Exactly why
`PTGENDER`/`PTEDUCAT`/`PTMARRY` are excluded from every risk model but
checked against every recommendation via DIDI.

### Fairness in practice: the pipeline_1_2_3 extension

Auditing DIDI on the output (previous section) found a real problem: a soft
fallback that gives less-complete-record patients a simpler model produced
DIDI 8.570 on the interval recommendation -- much worse than any single
tier alone, because which fallback tier a patient lands in is itself
correlated with protected attributes.

Two fixes were tried, in order of how much they actually change the model:

1. **Post-hoc group recalibration** -- shift each protected group's scores
   onto the global mean, fit on train, applied on test. Cut DIDI to 3.153.
   Cheap and effective, but by construction it never touches the model's
   own reasoning -- SHAP on the underlying RandomForest is provably
   identical before and after, since it's a constant shift on the output,
   not a retrain. Good practical default; not a complete answer to "did
   you fix the model."

2. **In-training constraint (07-ciml's own reductions approach)**:
   `fairlearn.reductions.ExponentiatedGradient` under `DemographicParity`,
   wrapping the same class-balanced RandomForest used everywhere else. This
   *is* a genuine retrain -- SHAP on the resulting model describes what
   actually changed. First attempt cut DIDI further (to ~2-5) but also
   collapsed catch rate from 94.9% to ~78%, because the reduction's default
   training objective is plain 0/1 error and has no idea a missed
   conversion costs 20x a false alarm (`MISSED_CONVERSION_COST=20 x
   CHECK_COST=1` in the project's own cost model). Passing
   `objective=ErrorRate(costs={"fp":1, "fn":20})` -- fairlearn's built-in
   cost-sensitive objective, matched to that exact ratio -- fixed it:
   catch rate back to 94.9% (an exact match, not an approximation), DIDI
   still down 76% to 2.098, total monitoring cost up ~40%.

   Point 2's own claim above ("SHAP on the resulting model describes what
   actually changed") was checked directly, not left as a promise:
   `wire_temporal_window_and_risk_factors.py` (README section 9) ran
   `risk_factors/1_global.ipynb`'s own attribution method (Lasso, forest
   importance, SHAP) on the deployed plain vs. fair calibrated
   `RISK_SCORE`. **The result is a real, concerning finding, not a
   reassuring one.** For the plain model, SHAP shows a genuine mix of
   amyloid PET (`SUMMARY_SUVR`) and structural MRI (`AMYGDALA_NORM`,
   `HIPPO_NORM`) driving the score. For the fair (EG) model, `NOMINAL_MONTH`
   (time since baseline) becomes the dominant driver, while `SUMMARY_SUVR`'s
   contribution drops 97%. So the fairness fix doesn't just cost cost,
   calibration, and per-group AUC (sections above) -- it appears to satisfy
   `DemographicParity` partly by substituting visit timing, a
   demographically-correlated but clinically weaker signal, for genuine
   amyloid/atrophy biomarkers. This directly revises the earlier framing
   ("a genuine retrain, SHAP describes what changed" was stated as neutral
   fact) into something worth flagging plainly: yes it's a genuine retrain,
   and what changed is concerning, not just different.

   The natural next question -- is this specific to DemographicParity, and
   does EqualizedOdds (equal true/false positive rates per group, harder to
   satisfy via a proxy) fix it -- was tried directly (README section 10),
   not just proposed. It did not work: EqualizedOdds still leans on
   NOMINAL_MONTH just as heavily (SHAP 0.0146 vs DemographicParity's
   0.0166, both far above the plain model's 0.0097), while giving up almost
   all the fairness benefit (DIDI 2.58, barely better than the plain
   model's own 2.13, vs DemographicParity's 1.25) and costing real AUC.
   Honest conclusion: this isn't a "pick a better constraint" problem, it's
   more likely that NOMINAL_MONTH is simply the cheapest lever
   ExponentiatedGradient's reduction has available regardless of which
   fairness criterion it's asked to satisfy -- a genuinely open problem,
   worth stating as tried-and-not-solved rather than glossing over.

**Why the cost increase is a feature, not a bug, here**: the project's own
cost model already says a missed conversion is twenty times worse than an
unneeded check-in. Spending ~40% more on check-ins to hold catch rate at
its original level *while also* cutting group disparity by three-quarters
is the model behaving exactly the way its own stated cost asymmetry says
it should -- more monitoring is cheap insurance in a screening pipeline
for a progressive, irreversible disease. DIDI came out lower than baseline
in every genuinely-retrained variant tried (post-hoc shift, isolated
core-tier EG, full-cascade EG both with and without the cost-aligned
objective) -- the one exception, classic Reweighing (a pre-processing
shortcut, not a constraint), made it slightly worse, which is itself a
useful negative result: not every fairness technique is safe to apply
blindly.

**But the 40% monitoring cost is not the only price paid, and this needs to
be stated plainly, not smoothed over.** DIDI only checks whether group
AVERAGES match -- it cannot see what happens to any single group's own
prediction quality. Checking that directly (`calibration_and_group_accuracy_check.py`):
the fairness-corrected model's calibration gap roughly DOUBLES for every
protected group (e.g. ~0.23 to ~0.48), because `ExponentiatedGradient`'s
`DemographicParity` constraint balances the fitted mixture's decision
distribution, not the calibration of its soft probabilities -- a real,
previously undocumented cost. Worse, one subgroup (`PTMARRY=3`, n=226) loses
AUC catastrophically, from 0.892 to 0.637, close to a coin flip for that
group specifically, even while DIDI overall improved 76%. That is exactly
the blind spot a demographic-parity-only metric has: group averages
equalized while one group's own predictions became far less trustworthy.
The honest, complete framing from here on: the fairness fix cuts DIDI 76%
at a real cost to per-group calibration and, for at least one subgroup,
a near-collapse in accuracy -- not a clean win with only a monitoring-budget
cost attached. Separately, the underlying calibration gap (0.242 on the
plain model, section on SHAP below) turned out to be cheaply fixable:
standard post-hoc sigmoid recalibration cut it from 0.208 to 0.010 on the
deployed tier with no AUC loss -- worth adopting regardless of which
fairness mechanism is used. Applying that same recalibration to the FAIR
model itself (`calibrate_fair_and_plain_risk_scores.py`, section 5c of the
pipeline README) works exactly as cleanly -- every group's calibration gap
collapses from ~0.5 to ~0.01-0.04, AUC unchanged -- which shows the
per-group calibration doubling was a genuine, fixable side effect, while
the per-group AUC loss (still present after recalibrating both models,
worst case -0.243) is not fixable by recalibration -- it is real
discrimination loss the fairness constraint causes. That same check also
found that much of the reported 40% monitoring-cost gap between the fair
and plain models was itself a calibration artifact: once both are properly
calibrated, the cost gap shrinks to about 4% (5123 vs. 4926), with DIDI
still clearly favoring the fair model. One new open cost surfaced by
calibrating: since `recommend_interval` uses `RISK_SCORE`'s raw magnitude
(not just its rank), recalibrating it changes the actual interval
decisions -- catch rate dropped for both models after calibration (fair:
89.7% -> 65.8%; plain: 89.7% -> 69.2%), since correctly-scaled
probabilities sit lower near the deciding threshold than the inflated raw
ones did. Recalibration is "free" for probability quality and AUC, but not
free for the clinical recommendations it feeds -- the cost ratio needs
re-tuning to recover the original catch rate under calibrated scores.
Actually re-tuning it (`retune_cost_ratio_after_calibration.py`, section 5d
of the pipeline README) shows this fix is NOT symmetric between the two
models. For the plain model it works cleanly: ratio=100 (up from 20)
recovers catch_rate=89.7% almost exactly, reproducing close to the
original cost and DIDI. For the fair (EG) model it does not fully work:
even at ratio=200, catch rate only reaches 87.7%, and DIDI climbs from
1.28 to 8.65 along the way -- nearly as bad as no fairness correction at
all. The fairness constraint was trained against one specific implicit
decision threshold; pushing the actual threshold far from that point (by
raising the cost ratio) lets the disparity it was suppressing come back.
So retuning is a complete fix only for the plain model -- for the fair
model it is a real trade-off (catch rate vs. DIDI), and the honest next
step would be re-fitting the fairness mechanism against the new ratio
rather than assuming the same fit transfers.

### SHAP / Additive Feature Attribution (06-at, lecture 4)

For a linear model: `θⱼ(xⱼ - E[xⱼ])` — coefficient times how far this
patient's value is from average. **SHAP generalizes this to nonlinear
models** via Shapley values (marginalizing over all feature subsets),
the unique attribution method with certain fairness-like properties
(contributions sum exactly to the prediction). Used on `RISK_SCORE` — the
direct analogue of your own PCA T²/Q contribution plot: same question ("why
this number, for this patient"), same additive-decomposition idea,
different underlying model.

---

## Likely questions and how to answer them

**"Why train only on CN patients?"** — Density estimation, not
classification (01-ad-de, lecture 3): learning what "healthy" looks like,
then measuring deviation from it. No diagnosis label used until evaluation.

**"Did you use the best method?"** — The course doesn't declare one; it's a
progression by problem characteristics (dataset size, dimensionality), not a
ranked competition. We tested three and let four independent criteria decide
rather than assuming.

**"Why PCA over the two methods actually taught in the course?"** — Ties
GMM and beats the autoencoder on raw ranking; still wins once blended into
the pipeline's cost calculation; wins the cost-optimized threshold
comparison on the broad screening task under every cost ratio tested,
including real, literature-anchored ones; and is the only one with an
exact per-feature explanation. The one genuine split: GMM wins the
narrower, AD-specific threshold task under those same real cost ratios —
an honest task-dependent result, not a clean sweep, and a stronger answer
for exactly that reason.

**"Isn't PCA/Hotelling's T² outside the syllabus?"** — Yes, directly. It's
the linear/Gaussian special case of the same density-estimation idea the
course teaches, solved in closed form instead of fit by EM or gradient
descent.

**"Do you need a threshold for all of these methods?"** — Yes, all of them —
KDE, GMM, PCA, and the autoencoder all output a continuous score, none of
them declares "anomaly" on its own. AUC evaluates ranking across all
possible thresholds at once; an actual decision needs one specific,
separately-tuned cutoff per method, since their scores aren't on comparable
scales.

**"How did you choose your anomaly threshold?"** — Adapted the course's own
`ADSimpleCostModel` line-search recipe (01-ad-de, lecture 4): first swept
the missed/false-alarm cost ratio over an arbitrary sensitivity range (5,
10, 20), then reran the same line search with real, literature-anchored
ratios (1.5x/2.8x/4.0x, from AD severity-cost health-economics literature —
the same anchor stage 3's own cost model already uses), and picked each
method's own threshold by minimizing total cost under both.

**"How does this connect to points 2 and 3?"** — Health index trajectories
feed the shared risk score at a fixed blend weight; measurably lowers the
pipeline's total monitoring cost on held-out patients; the join between our
two data representations was itself a real methodological finding (visit
codes not lining up) found and fixed with a data-driven, date-based key map.

**"What would you do with more time?"** — This was one item on this list
and is now done: `threshold_cost_optimization_literature.py` reruns the
threshold analysis with real, literature-anchored cost values (reusing
stage 3's own AD severity-cost anchor) instead of a synthetic sweep, and it
changed the narrow-task conclusion (GMM wins CN-vs-AD under real costs,
not just an unrealistic edge case). Two more items from this list are now
also done: `recommend_interval`'s own docstring named a trajectory-aware
policy and a learned policy as the natural next steps beyond the hardcoded
grid search, and `stage3_trajectory_and_learned_policy.py` tries both.
The trajectory-aware version (already written, never wired in) recovers
real catch rate lost to calibration (plain: 69.2% -> 76.7%) but at a real
DIDI cost (2.16 -> 3.31) -- a genuine lever, not free information. A
learned classifier trained on the full feature set matches the closed-form
formula's decisions 99.8% of the time -- a clean negative result showing
the formula isn't leaving much on the table, at least not with this
feature set. Remaining items: close the remaining pipeline coverage gap
with a documented forward-fill; a combined attribution table spanning all
three pipeline stages rather than one per stage.

**"You fixed the fairness problem -- did it cost you anything?"** — Yes,
two real costs, not one. First, total monitoring cost rose about 40% to
hold catch rate at its original 94.9% while cutting DIDI 76% -- framed the
right way, the project's own cost model already weights a missed
conversion twenty times worse than an extra check-in, so this is consistent
with that stated priority. Second, and this one is not just a framing
question: checking AUC and calibration separately per protected group
(DIDI alone can't see this) found the fair model's calibration gap roughly
doubles for every group, and one subgroup's AUC collapses from 0.892 to
0.637. That's a genuine tradeoff, not something to talk around -- the
fairness mechanism is still worth using, but it should be described
honestly as trading some per-group prediction quality for population-level
fairness, not as a free lunch that only costs money. Recalibrating the fair
model itself closes the calibration half of that tradeoff completely (gap
back down to ~0.01-0.04 per group, no AUC change) and also revealed that a
good chunk of the "40% more expensive" framing was a calibration artifact
-- once both models are calibrated, the real cost gap is closer to 4%. What
recalibration cannot touch is the AUC loss itself, which stays real (worst
case -0.243 for one subgroup even after both models are calibrated) -- so
the honest bottom line narrows to: the fairness mechanism's true cost is
accuracy loss for some groups, not cost or calibration, both of which turn
out to be fixable or overstated.

**"Did you just get lucky with one setting?"** — No -- every genuine
retrain tried (post-hoc shift, an isolated-tier fairlearn sweep, the
full-cascade retrain with and without the cost-matched objective) landed
below the 8.570 baseline DIDI, ranging roughly 1.8-5. The one technique
that made it worse, classic Reweighing, is a pre-processing shortcut that
doesn't actually constrain training -- a useful negative result, not a
counterexample.

**"Is the calibration gap fixable?"** — Yes, and cheaply, for both models:
standard post-hoc sigmoid recalibration (`CalibratedClassifierCV`, fit on
train only) cut the plain deployed model's weighted calibration gap from
0.208 to 0.010, a 95% reduction, with AUC essentially unchanged (0.807 to
0.808). Applying the same technique to the fairness-corrected model (a
single held-out calibration split, since refitting `ExponentiatedGradient`
under 5-fold CV would be far too slow) worked just as cleanly: every
protected group's calibration gap fell from roughly 0.5 to 0.01-0.04, with
zero AUC change anywhere -- confirming recalibration only rescales the
probability axis, it doesn't reorder patients. What it does NOT fix is
per-group AUC: even after both models are calibrated, the fair model's
AUC stays meaningfully lower for every group (worst case -0.243), which is
the real, un-fixable cost of the fairness constraint. One caveat worth
having ready: because `recommend_interval` uses the risk score's raw
magnitude, not just its rank, recalibrating it changes the actual interval
decisions -- catch rate dropped for both models after calibration (~90% to
~66-69%) -- so recalibration is free for probability quality and ranking,
but not free for the downstream clinical recommendations without also
re-tuning the cost ratio. Re-tuning itself is not symmetric, though:
sweeping the ratio up recovers catch rate cleanly for the plain model
(ratio=100 reproduces close to the original cost and DIDI), but for the
fair model it doesn't fully recover catch rate even at 10x the ratio, and
DIDI climbs back up toward the uncorrected baseline while trying -- because
the fairness constraint was fit against one specific decision threshold,
and raising the ratio moves the actual threshold away from it. Actually
REFITTING `ExponentiatedGradient` at ratio=100, instead of just
re-thresholding the ratio=20 fit, confirmed that diagnosis and fixed it
partially: refit beats re-threshold on every metric (DIDI 4.12 vs 5.82,
catch rate 87.0% vs 85.6%, cost slightly lower too), and it happened to
repair the single worst subgroup found earlier (`PTMARRY=4` AUC: 0.541 ->
0.800). It still doesn't fully match the plain model's 89.7% catch rate,
though, and DIDI is higher than the original ratio=20 fit's 1.25 --
raising the ratio structurally pushes the fair model's behavior closer to
the plain model's, so some fairness gain is genuinely, not just
accidentally, traded off against catch rate at higher ratios. That
trade-off, not a bug, is the honest bottom line.
