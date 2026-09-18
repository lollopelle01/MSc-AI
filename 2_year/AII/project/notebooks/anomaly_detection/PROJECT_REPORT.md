# AI in Industry — ADNI Alzheimer's Disease Project
## Project Understanding Report

Prepared ahead of the Monday briefing with Prof. Michele Lombardi.
Team: Giorgio Scavello (Point 1), Pelle (Points 2 & 3), Leo (RUL exploration), University of Bologna.

---

## 1. The project in one paragraph

The team turns the ADNI (Alzheimer's Disease Neuroimaging Initiative) longitudinal dataset into a three-stage clinical decision-support pipeline. Point 1 asks whether a patient visit looks anomalous relative to healthy (cognitively normal) patients, using unsupervised anomaly detection on imaging and biomarker features. Point 2 asks how likely a patient is to convert to a worse diagnosis, framed as a discrete-time survival/hazard problem rather than a naive classification, because most patients are censored (never observed to convert within the study). Point 3 takes that risk estimate and answers a practical question a clinician actually has to act on: how often should this patient be checked, is that recommendation fair across demographic groups, and which biomarkers are driving it. A fourth, experimental extension (`gio/pipeline_1_2_3/`) wires the three stages into one genuine sequential chain instead of three outputs that merely share a target table, and is where this session's calibration, fairness-mechanism, and oracle-regret work lives.

## 2. Data: sources, exploration, and treatment

### 2.1 Raw sources

Six raw ADNI tables anchor the project: `DXSUM` (diagnostic backbone, 15,931 rows / 3,788 unique patients — 1=CN 6,304, 2=MCI 6,580, 3=AD 3,002), `PTDEMOG` (demographics, 6,222 rows / 4,946 patients), `UCBERKELEY_AMY_6MM` (amyloid PET, 4,582 rows / 2,177 patients), `UCSFFSX7` (structural MRI, 12,151 rows / 3,231 patients), `UPENNBIOMK_ROCHE_ELECSYS` (CSF biomarkers, 3,174 rows / 1,660 patients), and `DATADIC` (the ADNI data dictionary). Only **1,333 of 3,788 patients (35%)** have all five modalities at once — a hard ceiling on any "complete case" model.

### 2.2 Data quality issues found and how they were handled

- **Sentinel codes.** ADNI uses `-1`/`-4` as missing-value sentinels in demographic fields (`PTGENDER`, `PTHAND`, `PTMARRY`, `PTEDUCAT`, `PTWORKHS`, …); these must be recoded to `NaN` before any statistic is computed, otherwise they masquerade as real category levels.
- **Visit-code disagreement.** `VISCODE` and `VISCODE2` disagree in **62.3%** of DXSUM rows. `VISCODE2` (normalized as `VISCODE2_norm`) is the canonical key used everywhere downstream.
- **QC flags, not uniform.** Amyloid PET (`qc_flag`) passes 3,919 / fails 597 / borderline 52 / hard-fails 13 (~13% unusable). Structural MRI's `OVERALLQC` is missing for 91% of rows, so `STATUS == 'complete'` is used as the reliable QC proxy instead; a further wrinkle is that 737 rows carry *both* a 3T and a 1.5T acquisition for the same visit (field-strength confound), requiring an explicit tie-break rule (3T preferred).
- **CSF coverage gap.** `ABETA42`/`TAU`/`PTAU` are ~99% populated, but `ABETA40` only 29%, so the Aβ42/40 ratio — the more diagnostically informative quantity — is only computable for a minority of CSF rows; Aβ42 alone is kept as a fallback feature.
- **`HAS_QC_ERROR` ruled out.** Investigated as a candidate blanket filter but populated for only 2,330 / 15,931 rows and always reads 0 — not usable, dropped from consideration (a negative result worth documenting rather than silently discarding).
- **Amyloid tracer pooling.** Three tracers (FBP, FBB, NAV) are pooled by preferring Centiloids over raw SUVR, which is tracer-independent.

### 2.3 The join (`final.csv`)

`exploration/join_datasets.ipynb` merges the five modalities on `RID` + `VISCODE2_norm` into `datasets/final.csv`: **13,827 rows, 3,777 unique patients, 38 columns** (diagnosis/staging, demographics, amyloid PET, structural MRI volumes — hippocampus/entorhinal/amygdala, raw and ICV-normalized — and CSF markers). This closely matches the raw-table patient count (3,788 vs 3,777), confirming the join isn't silently dropping large swaths of patients. **Complete-case multimodal coverage in `final.csv` is ~33%** (1,241/3,777), consistent with the 35% figure computed independently from the raw tables.

### 2.4 What the joined data actually supports (checked, not assumed)

- **Visit depth:** median 3 visits/patient, mean 3.66, max 20; **1,474 patients (39%) appear only once** — no trajectory information for these.
- **Conversion is rare and heavily censored.** Of 2,303 patients with ≥2 diagnosed visits, only 590 (25.6%) ever show a diagnosis *increase*; 74.4% are censored (never observed to worsen within the study window). CN patients convert at 21.2%, MCI at 38.5%.
- **Modality coverage is diagnosis-dependent**, not missing-at-random: amyloid PET coverage is ~42% (CN) vs ~29% (MCI) vs ~22% (Dementia); MRI is more uniform at 73–77% across all three.
- **Preclinical AD is real in this cohort:** of CN patients with known amyloid status, ~1 in 3 (701/2,200) are amyloid-positive — i.e., biomarker-positive before cognitive symptoms, the population Point 1's anomaly detection is specifically trying to catch.

### 2.5 Practical modeling recommendations that came out of this analysis

Recode sentinels to `NaN`; use `VISCODE2` as the canonical key; filter amyloid to `qc_flag==2`; filter MRI on `STATUS=='complete'` (not `OVERALLQC`), preferring 3T when both exist; prefer Centiloids over raw SUVR; expect the Aβ42 vs Aβ42/40 usability gap and keep a fallback feature set; and decide upfront whether the modeling cohort is the ~1,333-patient complete-case set or a per-modality approach with its own missingness handling, since this materially changes sample size and which A/T/N (amyloid/tau/neurodegeneration) components are available per model.

---

## 3. Point 1 — Anomaly Detection

**Question.** Does a patient's visit look anomalous relative to a cognitively-normal reference population, without ever training on a diagnosis label? This targets exactly the preclinical-AD population identified above (biomarker-positive, symptom-negative).

**Methodology.** Two independent methods, both trained only on CN patients and evaluated by how well their anomaly score separates CN from MCI/AD:

- **Method A — PCA + Hotelling's T²/Q (SPE).** Classical statistical-process-control technique: fit PCA on CN baseline visits (14 components, 91.9% variance retained), then score every visit by combined in-subspace (T²) and out-of-subspace (Q/SPE) deviation. Run on 464 patients (269 CN / 161 MCI / 34 AD), cross-sectional (one score per patient, no visit history).
- **Method B — Autoencoder health index.** From course module 02-ad-hd, lesson 4 (Autoencoders for Anomaly Detection). Trained on 3,471 CN visits (823 patients); scored on 9,922 visits (2,093 patients), per-visit, so it naturally supports longitudinal use (this is why it — not Method A — is the one wired into the pipeline extension). A health-index *slope* across a patient's visits rises with true severity: roughly 0.004/yr for CN, ~2× that for MCI, ~8× for AD.

**Results.**

| Method | CN vs MCI AUC | CN vs AD AUC | CN vs (MCI+AD) AUC |
|---|---|---|---|
| A — PCA/Hotelling | 0.696 | 0.890 | 0.730 |
| B — Autoencoder | 0.627 | 0.777 | 0.700 |

Both methods separate CN from AD well and CN from MCI more weakly — expected, since MCI is the harder, earlier signal. Method A's per-patient AUC check corrected for repeated-visit weighting (a subject with more visits shouldn't dominate the metric), a discipline that later carried over into every other stage's subject-level train/test split.

---

## 4. Point 2 — Conversion Risk as Survival Analysis

**Question.** How likely is a patient to convert to a worse diagnosis, and by when? Framed correctly this is a survival-analysis problem, not a regression on the 590 patients who happened to convert — a naive approach would be badly biased by the 74.4% who are censored (never observed to convert, not necessarily never going to).

**Methodology.** Discrete-time hazard model (course module 05-pm, lesson 6, Survival Analysis via Neural Models): reduces to a binary cross-entropy classification of "did this visit already show a worsening from the prior visit" (`EVENT_AT_VISIT`), letting standard classifiers estimate a per-visit hazard. Four candidate estimators were compared — Logistic Regression and Random Forest, each on a "core" feature tier (available for nearly every visit) and an "extended" tier adding CSF markers (available for a smaller subset) — on a held-out, subject-level test split.

**Results.** The Random Forest on the extended tier wins, test **AUC ≈ 0.791**, at a coverage cost: because CSF is measured at only about half of all visits, this estimator reaches a bit over half of all rows; the rest fall back to a simpler Logistic Regression fit on `DIAGNOSIS`/`AMYLOID_STATUS` alone. `forecast_conversion_probabilities` compounds the per-visit hazard forward to give conversion probability forecasts at 3/6/12/24 months, feeding Point 3's most advanced policy.

**An important, explicitly-flagged limitation:** the 0.791 AUC measures *ranking* quality (does a higher-risk patient score higher than a lower-risk one), not *calibration* (is the absolute probability correct). This distinction is not academic — it directly causes a real problem downstream (Section 6.2).

---

## 5. Point 3 — Decision Support

**Question.** Given a risk score, what should actually be *done* — and is that recommendation fair, and explainable? Point 3 is not one model with two add-ons; it is **three parallel tracks** reading the same risk score, answering three independent questions, that can be (and were) built and evaluated separately.

**Track 1 — Cost model (what should be done).** `ConversionCostModel`, modeled directly on the course's `RULCostModel` (04-rul, lesson 4), prices a monitoring interval against a routine-check cost and a missed-conversion cost that scales with the interval's length (missing a conversion during a 12-month gap is worse than during a 3-month gap — both the policy-selection and policy-scoring functions must scale this identically, a bug that is easy to introduce and was checked for). Missed-conversion cost is stratified by diagnosis (MCI/AD priced at 2.8× the CN base rate, anchored to published Alzheimer's cost-of-care literature spanning roughly 1.4×–4×) and swept across that literature range in a documented sensitivity analysis rather than resting on one number. Four interval policies were built, each reading more of a patient's history than the last: **fixed** (same interval for everyone, the necessary bad-but-complete baseline), **snapshot-adaptive** (current risk score only), **trajectory-adaptive** (risk *slope* across a patient's own prior visits), and **forecast-adaptive** (reads Point 2's multi-horizon forecast directly, priced with a structurally different formula, so its total cost is not directly comparable to the other three on the same axis — an honest limitation kept visible rather than folded into one ranking).

**Track 2 — Fairness audit (does it treat people differently).** The Disparate Impact Discrimination Index (DIDI, course module 07-ciml, lesson 1) is computed on the recommended interval / risk score across `PTGENDER`, `PTEDUCAT` (bucketed), and `PTMARRY` — the three protected attributes `final.csv` has clean and complete. Critically, per 07-ciml's own point, a model can still discriminate through *correlates* of a removed attribute even if gender/education are never model inputs, so these attributes are kept out of the model features and used only to audit the output.

**Track 3 — Attribution (why the score is what it is).** Lasso/Logistic baseline + Random Forest, both checked by R²/MAE on a held-out split (not just training fit), then SHAP values (06-at, lesson 4, Additive Feature Attribution) for a clinician-facing, per-patient explanation of which biomarkers drive the score.

**Results (headline numbers).** Fixed policy: cost 6,437 (plain) / 6,477 (fair variant), DIDI = 0 by construction, catch rate 95.2% both. Forecast-adaptive: total cost roughly double snapshot/trajectory-adaptive (38,796 vs ~18,400–18,700 on the test split) for a modestly higher catch rate (91.4% vs 87–88%) — traced directly to Point 2's calibration gap (Section 6.2), not a bug in the cost formula. A first pipeline connection (`approach_pipeline_anomaly.ipynb`) blends Point 1's Method B health index into the risk score at a fixed, documented 80/20 weight and finds a modest (~1.5%) cost reduction with no meaningful DIDI change — a thin but real first answer that upstream signal is worth carrying through the pipeline.

---

## 6. The Pipeline Extension (`gio/pipeline_1_2_3/`) — a genuine 1→2→3 chain

This extension goes one step further than the three points running independently: it wires Point 1's anomaly score into Point 2's hazard model as an input feature, then feeds the resulting hazard estimate into Point 3's interval policy — closing the gap where Points 1 and 2 previously both fed Point 3 independently but Point 1 never passed *through* Point 2. This is also where most of this session's investigative work sits, so it is reported in depth below since it materially changes the "fairness vs. plain model" conclusion the project would otherwise draw.

### 6.1 Does Point 1's signal help Point 2?

| Model | Feature tier | Test AUC without ANOMALY_SCORE | With | Delta |
|---|---|---|---|---|
| Logistic | core | 0.743 | 0.747 | +0.004 |
| Random Forest | core | 0.791 | 0.790 | −0.001 |
| Logistic | extended (+CSF) | 0.768 | 0.794 | **+0.025** |
| Random Forest | extended (+CSF) | 0.791 | 0.807 | **+0.016** |

Yes — modestly, and specifically on the richer feature tier. This is the concrete evidence that chaining the stages, not just running them in parallel, adds value.

### 6.2 Calibration: the fairness/accuracy trade-off was partly an illusion

The uncalibrated fair model (`ExponentiatedGradient` under Demographic Parity) looked much worse than the plain model: AUC 0.736 vs 0.802, **per-group calibration gap 0.518 vs 0.209**, cost 16,630 vs 11,317. Manually Platt-calibrating both models (fit on a held-out calibration split, never touching the test panel — `CalibratedClassifierCV`'s standard 5-fold refit was too expensive against the constrained `ExponentiatedGradient` estimator) collapses the per-group gap to ~0.01–0.04 for *both* models with **zero change in per-group AUC** — proving the AUC gap is genuine discrimination cost, not a scaling artifact, while the calibration gap itself was an artifact all along. Once both are calibrated, the real cost difference shrinks from ~40% to **~4%**, though DIDI still meaningfully favors the fair model.

Calibration is not free, however: it drops catch rate for both models (89.7%→65.8% fair, 89.7%→69.2% plain) because the downstream interval policy uses raw score magnitude, not just rank. Re-thresholding (raising the cost ratio post hoc) fully recovers the plain model's catch rate but not the fair model's. **Refitting** `ExponentiatedGradient` directly at the new cost ratio is strictly better than re-thresholding on every metric and even repairs the worst-performing subgroup's AUC — but still does not close the gap to the plain model. This is a genuine, structural trade-off between catch rate and DIDI, not a bug to be engineered away.

### 6.3 The NOMINAL_MONTH proxy-reliance finding

Wiring the project's own attribution track (Track 3) onto the *actually deployed, calibrated* scores (rather than the earlier placeholder) surfaced a significant problem: under the fair (Demographic-Parity) model, SHAP attribution becomes dominated by `NOMINAL_MONTH` (visit timing) rather than genuine biomarkers — `SUMMARY_SUVR`'s importance collapses by ~97% (0.0102 → 0.0003) while `NOMINAL_MONTH` rises to 0.0166, the top driver. In other words, the demographic-parity constraint is achieving fairness partly by leaning on a *timing* proxy rather than biology — a real explainability red flag for a clinical tool.

**Tested fix that did not work:** swapping `DemographicParity` for `EqualizedOdds` does not solve this — `NOMINAL_MONTH`'s SHAP importance stays essentially unchanged (0.0146 vs 0.0166), and `EqualizedOdds` gives up almost all of the fairness benefit in the process (DIDI 2.579, barely better than the plain model's 2.130, vs. Demographic Parity's 1.253). This negative result is documented as such rather than presented as solved.

### 6.4 Are the decisions actually relevant? The oracle-regret check

To test whether interval recommendations are decision-relevant (not just correlated with risk score for its own sake), an **oracle** policy was built from perfect knowledge of the *forward-looking* outcome the catch-rate metric actually checks (`DIAGNOSIS_WORSENED_NEXT`, from the patient's real next visit) — not from `EVENT_AT_VISIT`, which is backward-looking and was caught and fixed as a genuine methodological bug mid-analysis (an oracle built from the wrong target scored *worse* than the real models, which is logically impossible for perfect information).

Regret (`(policy_cost − oracle_cost) / (naive_cost − oracle_cost) × 100`) results: plain calibrated 43.3%, fair (DP) calibrated 50.7%, plain trajectory-aware 44.9%, fair (DP) trajectory-aware 57.6%. **A key finding that complicates "just use the plain model":** the oracle itself — built from ground truth, no model uncertainty at all — has DIDI = 1.488. That is the level of demographic disparity genuinely justified by real between-group differences in outcomes, not model unfairness. Against that baseline, the plain model's DIDI (2.16–8.57 depending on configuration) represents *real excess* disparity beyond what the data itself justifies, while the fair model's DIDI (1.25–1.28) sits close to the oracle's natural floor. This reframes the trade-off from "fairness costs accuracy" to "the plain model may be carrying disparity the data doesn't actually require."

### 6.5 Honest bottom line for the pipeline extension

Neither "use the plain model" nor "use the fair model" is unambiguously correct given the evidence gathered. The plain model has higher raw AUC/catch rate but carries excess disparity beyond the oracle's natural floor and a genuine per-group accuracy cost that calibration cannot fix. The fair (Demographic Parity) model closes that disparity and matches the oracle's DIDI floor closely, but achieves it partly through a timing proxy rather than biomarkers, and Equalized Odds does not fix that without giving up most of the fairness gain. This is presented to the professor as a real, evidenced trade-off — not resolved, because it should not be artificially resolved.

---

## 7. Explainability and Fairness — summary across the whole project

- **Fairness mechanism used:** `ExponentiatedGradient` (Fairlearn) with swappable constraints (Demographic Parity, Equalized Odds) and a cost-sensitive objective, following 07-ciml's Lagrangian-constraint framing conceptually; a from-scratch Keras Lagrangian implementation (`CstDIDIRegressor`, `LagDualDIDIRegressor`) was also built directly on 07-ciml lesson 2's dual-variable approach for the attribution-track model, with three real implementation bugs found and fixed along the way (Keras 3 naming rule, a `trainable=False` flag that silently blocked gradient ascent on the dual variable, and a scaler that silently broke the fairness constraint by standardizing the protected columns) — documented as debugging lessons, not just results.
- **Explainability method:** SHAP (06-at, lesson 4) on top of a Lasso/Random Forest baseline, checked by held-out R²/MAE, not just training fit; used both to explain individual scores and to diagnose the NOMINAL_MONTH proxy problem above.
- **Central honest finding:** fairness corrections and explainability are not independent add-ons — the DP fairness mechanism's success is entangled with an explainability failure (proxy reliance), which only became visible by running both tracks on the same deployed scores together.

---

## 8. Data → Pipeline schema

The team's agreed shared schema (from `IMPLEMENTATION_PLAN.md`): a single table keyed by `RID` + `VISCODE2_norm`, starting from `final.csv`, gaining `ANOMALY_SCORE`/`HI_SLOPE` from Point 1, `HAZARD_ESTIMATE` + forecast columns from Point 2, and `RECOMMENDED_INTERVAL` + attribution columns from Point 3 — plus a single shared `util.py` module (mirroring the course's own convention of `from util import util`) and one RID-level train/val/test split used consistently across every stage, so a leak in an early stage cannot silently flow into a later one.

---

## 9. Next steps (course-grounded, not yet built)

1. **Decision-Focused Learning** (08-dfl-tf) to train the upstream risk model directly against the deployment cost rather than a proxy loss like cross-entropy — most promising for the recourse-like structure of a multi-interval monitoring schedule (missing one check doesn't mean the patient is lost; a later check can still catch it).
2. **Boruta all-relevant feature selection** (06-at, lesson 5) — already named in the project's own notes but never run — to rigorously test whether `NOMINAL_MONTH`'s outsized influence under fairness constraints reflects genuine relevance or is purely a fairness-training artifact.
3. **Post-processing fairness** (per-group threshold adjustment after training, 07-ciml) rather than in-training reduction — structurally unable to lean on a proxy feature the way `ExponentiatedGradient` can, since it never touches the model's internals.
4. **Calibrate the hazard model itself** (predicted probability vs. observed frequency in bins) before trusting `forecast_adaptive`'s cost figure as a real expected-dollar quantity — the single most concrete, well-scoped next step, since Section 6.2/6.3's problems trace back to this same root cause.
5. **Real multi-horizon forecast columns** on the deployed pipeline (not just the historical hazard model), to fully unlock `forecast_adaptive` once (4) is addressed.

---

*This report draws on `IMPLEMENTATION_PLAN.md`, `data_evaluation_report.md`, `notebooks/decision_support/NOTES.md`, `gio/pipeline_1_2_3/README.md`, and this session's own analysis scripts and results in `gio/pipeline_1_2_3/`.*
