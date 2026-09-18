# ADNI Dataset Evaluation Report — Data Quality & Merge Feasibility

*Generated 2026-08-11 from the six CSVs in `datsets/`*

## 1. Overview

| Table | Shape (rows × cols) | Unique RIDs | Duplicate rows |
|---|---|---|---|
| DATADIC | 34,899 × 13 | — | 1 |
| DXSUM | 15,931 × 41 | 3,788 | 0 |
| PTDEMOG | 6,222 × 84 | 4,946 | 0 |
| UCBERKELEY_AMY_6MM | 4,582 × 344 | 2,177 | 6 |
| UCSFFSX7 | 12,151 × 347 | 3,231 | 0 |
| UPENNBIOMK_ROCHE_ELECSYS | 3,174 × 13 | 1,660 | 0 |

Note: `datasets_documentation.md` lists PTDEMOG as 6,223 rows; the file on disk has 6,222. Trivial, but worth re-checking if the source file was re-exported.

The one duplicate row in DATADIC and six in UCBERKELEY_AMY_6MM are exact full-row duplicates and can be dropped with `drop_duplicates()` with no information loss.

## 2. Missingness

| Table | Columns >50% missing | Columns 100% missing |
|---|---|---|
| DATADIC | 6 / 13 | 0 |
| DXSUM | 18 / 41 | 0 |
| PTDEMOG | 57 / 84 | 0 |
| UCBERKELEY_AMY_6MM | 4 / 344 | 0 |
| UCSFFSX7 | 12 / 347 | 0 |
| UPENNBIOMK_ROCHE_ELECSYS | 2 / 13 | 0 |

PTDEMOG is the most sparse table proportionally (57 of 84 columns over 50% missing), mostly the language-proficiency block (`PTLANGPR[1-6]`, `PTLANGSP[1-6]`, etc.), which only applies to multilingual participants. These are structurally missing, not a data problem — they should be treated as "not applicable" rather than imputed.

UCBERKELEY_AMY_6MM and UCSFFSX7 look clean at the table level (very few columns over 50% missing out of 300+), but the *quality control* fields inside UCSFFSX7 tell a different story — see §3.3.

## 3. Table-by-table quality notes

### 3.1 DXSUM (diagnostic backbone)
- `DIAGNOSIS` is populated for 15,886 / 15,931 rows (45 missing). Distribution: 1=CN 6,304, 2=MCI 6,580, 3=AD 3,002 — a usable three-class target, moderately imbalanced toward MCI.
- `HAS_QC_ERROR` is populated for only 2,330 of 15,931 rows (all 0 — no flagged errors); the other 13,601 are simply unpopulated for this field, so it cannot be relied on as a QC filter across the whole table.
- `VISCODE` vs `VISCODE2` disagree in 62.3% of rows. `VISCODE2` is the field ADNI recommends for cross-phase alignment; use it consistently rather than mixing the two.
- Phase coverage: ADNI2 (5,671) and ADNI1 (3,868) dominate; ADNI4 already has 2,330 records despite being the newest phase.

### 3.2 PTDEMOG (demographics)
- `PTGENDER`, `PTHAND`, `PTMARRY`, `PTEDUCAT`, `PTWORKHS` and 6 other numeric columns contain the ADNI sentinel code **-4** ("unknown/not answered") mixed in with real numeric values — e.g. `PTEDUCAT.min() == -4` pulls the mean/std of years-of-education down if not filtered out first. These sentinels must be recoded to `NaN` before any statistics or modeling, not left as literal values.
- `PTGENDER` itself has 129 true NaN plus 57 sentinel -4 rows (about 3% of the table) with no usable gender value.
- Most rows use `VISCODE = sc` (screening) or `v01`, not a clean single "bl" baseline — baseline extraction logic needs to account for this rather than filtering strictly on `VISCODE=='bl'`.

### 3.3 UCBERKELEY_AMY_6MM (amyloid PET)
- `qc_flag`: 3,919 pass (value 2), 597 fail (-1), 52 borderline (1), 13 hard-fail (-2). ~13% of scans fail QC and should be excluded before amyloid-status analysis.
- Tracer mix: FBP (Florbetapir) 3,593, FBB (Florbetapiben) 921, NAV 68 — three different tracers with different SUVR scaling. Centiloid values are tracer-harmonized and should be preferred over raw SUVR when combining tracers.
- `AMYLOID_STATUS`: 2,433 negative, 2,067 positive, 82 missing — balanced enough for a binary classification target.

### 3.4 UCSFFSX7 (structural MRI)
- `OVERALLQC` is missing for 11,086 of 12,151 rows (91%) — only 1,065 rows have any QC rating at all (448 Pass, 601 Partial, 11 Hippocampus-only, 5 Fail). This is the most significant quality gap in the dataset: the vast majority of MRI-derived volumes have **no recorded QC outcome**, so `OVERALLQC` cannot be used as a blanket filter without discarding ~91% of the table.
- `STATUS` (processing status) is "partial" for 11,086 rows and "complete" for only 1,065 — this tracks almost exactly with the missing `OVERALLQC` rows, suggesting the two fields are linked (QC is only recorded once processing is "complete"). Filtering on `STATUS=='complete'` is the more reliable QC proxy for this table.
- Field strength: 3T (7,671 rows) vs 1.5T (4,480 rows) — a known confound; FreeSurfer volumes are not directly comparable across field strengths and should be harmonized or used as a covariate.
- 737 rows share the same `(RID, VISCODE2)` — not an error: these are dual acquisitions (3T + 1.5T scans) at the same visit, distinguished by `IMAGEUID`/`FIELD_STRENGTH`. When merging to one row per visit, an explicit rule is needed for which scan to keep (e.g., prefer 3T + `STATUS=='complete'`).

### 3.5 UPENNBIOMK_ROCHE_ELECSYS (CSF biomarkers)
- `ABETA42`, `TAU`, `PTAU` are populated for ~99% of rows, but `ABETA40` is populated for only 934 of 3,174 rows (29%). The Aβ42/40 ratio — flagged in the documentation as the most robust amyloid measure — can therefore only be computed for a minority of CSF records; Aβ42 alone or p-tau/Aβ42 will need to be the fallback feature for the rest.
- No duplicate `(RID, VISCODE2)` rows — one record per participant-visit, as expected.

## 4. Cross-table merge feasibility (all keyed on RID)

| Table | RIDs also in DXSUM | Coverage of DXSUM's 3,788 RIDs |
|---|---|---|
| PTDEMOG | 3,787 / 4,946 (76.6%) | 100.0% |
| UCBERKELEY_AMY_6MM | 2,176 / 2,177 (100.0%) | 57.4% |
| UCSFFSX7 | 3,194 / 3,231 (98.9%) | 84.3% |
| UPENNBIOMK_ROCHE_ELECSYS | 1,660 / 1,660 (100.0%) | 43.8% |

- Every DXSUM participant has a PTDEMOG record — demographics can be merged in without loss.
- MRI (UCSFFSX7) covers 84% of diagnosed participants — the best-covered imaging modality.
- Amyloid PET and CSF biomarkers are considerably sparser (57% and 44% of diagnosed participants respectively) — expected, since PET and lumbar puncture are optional, invasive/costly sub-studies, consistent with the selection-bias caveat already noted in the documentation.
- Only **1,333 participants** (35% of the 3,788 with a diagnosis) have records in *all five* clinical/biomarker tables simultaneously. A full A/T/N multimodal model restricted to complete cases would use just over a third of the diagnosed cohort — a meaningful sample-size trade-off against a model that uses each modality separately with modality-specific missingness.
- `VISCODE`/`VISCODE2` disagreement rates of 48–80% across tables (see §3.1) mean that a merge should be done via `VISCODE2` and `EXAMDATE`/`SCANDATE` proximity matching, not a naive join on visit code alone — timepoints from different tables for the "same" visit will not always share an identical visit-code string.

## 5. Practical recommendations before modeling

1. Recode all `-1`/`-4` ADNI sentinel values to `NaN` in PTDEMOG (and check DXSUM/other tables for the same pattern) before computing any statistics.
2. Use `VISCODE2` (not `VISCODE`) as the canonical visit key across tables.
3. Filter UCBERKELEY_AMY_6MM to `qc_flag == 2` before using amyloid measures.
4. Filter UCSFFSX7 to `STATUS == 'complete'` rather than `OVERALLQC`, since the latter is missing for 91% of rows; where both scanner strengths exist for a visit, prefer 3T.
5. Prefer Centiloids over raw SUVR when pooling FBP/FBB/NAV tracers in the amyloid table.
6. Expect an ~99% vs ~29% usability gap between Aβ42 and Aβ42/40 in the CSF table; plan a fallback feature set for records without ABETA40.
7. Decide up front whether the modeling cohort will be the ~1,333 complete-case participants (all 5 tables) or a per-modality approach with separate missingness handling — this materially changes both sample size and which A/T/N components are available.

## 6. Open items / not yet re-verified

- The PTDEMOG row-count discrepancy (6,222 rows here vs. 6,223 in `datasets_documentation.md`) — worth confirming which reflects the current file if reproducibility matters. DATADIC's count matches the documentation exactly.
- The `ydata-profiling` HTML reports in `exploration/reports/` already contain full per-column distributions; this report focuses on the cross-cutting quality and merge issues those single-table reports don't surface.
