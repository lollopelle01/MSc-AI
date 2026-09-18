# Documentation for our datasets from ADNI

# Index

1. [DATADIC_x_.csv](#1-datadic_x_csv) - Data Dictionary
2. [DXSUM_x_.csv](#2-dxsum_x_csv) - Diagnostic Summary
3. [PTDEMOG_x_.csv](#3-ptdemog_x_csv) - Participant Demographics
4. [UCBERKELEY_AMY_6MM_x](#4-ucberkeley_amy_6mm_12dec2025) - Amyloid PET Analysis (UC Berkeley 6mm)
5. [UCSFFSX7_x](#5-ucsffsx7_12dec2025) - FreeSurfer Cross-Sectional MRI Analysis (Version 7.x)
6. [UPENNBIOMK_ROCHE_ELECSYS_x](#6-upennbiomk_roche_elecsys_12dec2025) - CSF Biomarkers (Roche Elecsys)

---

## 1. DATADIC_x_.csv

### 1.1 General Description

The **Data Dictionary** file is the comprehensive reference document for the entire ADNI dataset. It provides detailed metadata for every variable across all ADNI phases and tables, serving as the essential guide for understanding variable definitions, data types, coding schemes, and measurement units.

### 1.2 Attributes Meanings

| Attribute                | Description                                            | Importance Level                                                       |
| ------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------- |
| **PHASE**          | ADNI study phase (ADNI1, ADNIGO, ADNI2, ADNI3, ADNI4)  | Medium - Helps track changes across phases                             |
| **CRFNAME**        | Original Case Report Form name                         | Low - Reference to source documentation                                |
| **TBLNAME**        | Table name where the variable appears                  | High - Maps variables to specific data files                           |
| **FLDNAME**        | Field/Variable name in the table                       | High - The actual column name in CSV files                             |
| **TEXT**           | **Complete textual description of the variable** | **Critical** - Explains what the variable measures               |
| **TYPE**           | Data type (Numeric, Text, Date, etc.)                  | High - Essential for preprocessing                                     |
| **LENGTH**         | Maximum length for text fields                         | Medium - For data validation                                           |
| **DD_CRF_VERSION** | Data Dictionary/CRF version number                     | Medium - Tracks definition changes                                     |
| **CODE**           | **Coding scheme for categorical variables**      | **Critical** - Decodes numeric values (e.g., "1=Male, 2=Female") |
| **UNITS**          | Measurement units for numeric variables                | High - For proper interpretation of scales                             |
| **STATUS**         | Variable status (Active, Inactive, Deprecated)         | Medium - Identifies current vs. legacy variables                       |
| **CODE_CHANGES**   | Notes on coding changes over time                      | Medium - Explains inconsistencies across phases                        |
| **MAPPING_NOTES**  | Notes on variable mapping between ADNI phases          | High - Crucial for longitudinal studies                                |

### 1.3 Important Notes

#### 1.3.1 Role in ML Projects

- **Feature Interpretation**: Use the `TEXT` and `CODE` fields to understand what each variable represents before feature selection.
- **Data Cleaning Guidance**: The `TYPE` and `LENGTH` fields help validate data integrity during preprocessing.
- **Categorical Variable Decoding**: The `CODE` column is essential for converting numeric codes to meaningful labels (e.g., gender, diagnosis codes).
- **Longitudinal Consistency**: `CODE_CHANGES` and `MAPPING_NOTES` help resolve discrepancies when combining data across different ADNI phases.

#### 1.3.2 Practical Usage Tips

1. **Always consult DATADIC first** when encountering an unfamiliar variable
2. **Filter by TBLNAME** to see all variables available in a specific table
3. **Check STATUS** to avoid using deprecated variables
4. **Note PHASE-specific differences** when working with multi-phase data
5. **Save decoded versions** of categorical variables based on CODE descriptions

---

## 2. DXSUM_x_.csv

### 2.1 General Description

The **Diagnostic Summary** file contains longitudinal diagnostic information for all ADNI participants across all study visits. This is the **core clinical outcome file** that tracks disease progression from Cognitively Normal (CN) to Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD) over time. With approximately 16,000 records, it provides rich temporal data for time-series analysis, survival modeling, and progression prediction.

### 2.2 Attributes Meanings

#### 2.2.1 Identification and Temporal Attributes

| Attribute          | Description                              | Importance Level                                           |
| ------------------ | ---------------------------------------- | ---------------------------------------------------------- |
| **PHASE**    | ADNI phase                               | Medium - Diagnostic criteria may vary by phase             |
| **PTID**     | Participant ID (format: XXX_S_XXXXX)     | High - Unique participant identifier                       |
| **RID**      | Research ID (numeric portion of PTID)    | **Critical** - Primary key for merging tables        |
| **VISCODE**  | Visit code (bl, m06, m12, m24, m36, m48) | High - For temporal ordering                               |
| **VISCODE2** | Alternative visit coding                 | Medium - Backup temporal identifier                        |
| **EXAMDATE** | Clinical examination date                | **Critical** - Reference date for temporal alignment |

#### 2.2.2 Primary Diagnostic Variables

| Attribute           | Description                          | Typical Values | Importance Level                                    |
| ------------------- | ------------------------------------ | -------------- | --------------------------------------------------- |
| **DIAGNOSIS** | **Primary clinical diagnosis** | CN, MCI, AD    | **Critical** - Main target for classification |
| **DXNORM**    | Cognitively Normal flag              | 0/1            | High - Alternative binary target                    |
| **DXMCI**     | Mild Cognitive Impairment flag       | 0/1            | High - For MCI subgroup analysis                    |
| **DXAD**      | Alzheimer's Disease flag             | 0/1            | High - For AD conversion prediction                 |

#### 2.2.3 Diagnostic Specifications and Subtypes

| Attribute                 | Description                        | Importance Level                      |
| ------------------------- | ---------------------------------- | ------------------------------------- |
| **DXMDES**          | MCI subtype description            | Medium - For refining MCI subgroups   |
| **DXMPTR1-DXMPTR6** | MCI diagnostic criteria pointers   | Low - Validation metadata             |
| **DXMDUE**          | Primary cause of MCI               | Medium - To exclude non-AD etiologies |
| **DXDSEV**          | Dementia severity rating           | High - For regression targets         |
| **DXAPROB**         | AD probability (probable/possible) | Medium - For probabilistic targets    |
| **DXPARK**          | Parkinsonism flag                  | Medium - Comorbidity exclusion        |
| **DXDEP**           | Depression flag                    | High - Important confounding variable |
| **DXCONFID**        | Diagnostic confidence              | Medium - For weighted learning        |

#### 2.2.4 Quality Control and Metadata

| Attribute              | Description                | Importance Level                      |
| ---------------------- | -------------------------- | ------------------------------------- |
| **SITEID**       | Recruitment site ID        | Medium - For controlling site effects |
| **HAS_QC_ERROR** | Quality control error flag | High - For data filtering             |
| **update_stamp** | Last update timestamp      | Low - Version tracking                |

### 2.3 Important Notes

#### 2.3.1 Role in ML Projects

- **Primary Target Definition**: This table provides all diagnostic labels for supervised learning tasks.
- **Temporal Structure**: Each row represents one participant at one time point, enabling time-series analysis.
- **Progression Modeling**: The sequence of diagnoses (CN→MCI→AD) allows for conversion prediction.
- **Subgroup Identification**: Additional flags enable filtering of specific populations (e.g., amnestic MCI only).

#### 2.3.2 Key Considerations for Time-Series Analysis

1. **Not All Participants Progress Linearly**: Some may remain stable, others may revert (e.g., AD→MCI), which could indicate diagnostic uncertainty.
2. **Variable Visit Intervals**: Participants have different numbers of visits at different time intervals.
3. **Missing Diagnoses**: Some visits may lack diagnostic information.
4. **Diagnostic Confidence**: The `DXCONFID` field can be used to weight training examples or filter uncertain diagnoses.

#### 2.3.3 Recommended Target Definitions for Common ML Tasks

1. **Binary Classification (Conversion Prediction)**:

   - Target = 1 if `DXMCI=1` at time t and `DXAD=1` at time t+Δt (Δt=12, 24, 36 months)
   - Target = 0 if `DXMCI=1` at time t and `DXMCI=1` or `DXNORM=1` at time t+Δt
2. **Multi-class Classification**:

   - Target = `DIAGNOSIS` (CN, MCI, AD) at future time point
3. **Survival Analysis**:

   - Event = First occurrence of `DXAD=1`
   - Time = Months from baseline to event (censored for non-converters)
4. **Regression**:

   - Target = Time to conversion (months)
   - Target = Severity score (`DXDSEV`)

#### 2.3.4 Data Preparation Tips

1. **Align by RID and EXAMDATE** when merging with other tables
2. **Convert VISCODE to numeric months** for consistent temporal features
3. **Check HAS_QC_ERROR** and consider filtering problematic records
4. **Consider site effects** using SITEID as a covariate or for stratified sampling
5. **Account for phase differences** in diagnostic criteria when using multi-phase data

---

## 3. PTDEMOG_x_.csv

### 3.1 General Description

The **Participant Demographics** file contains comprehensive baseline demographic, social, and cultural information for all ADNI participants. This table provides **static covariates** that are essential for controlling confounding factors, understanding population characteristics, and building more accurate predictive models. The data is primarily collected at baseline with minimal changes over time, making it ideal for use as time-invariant features in longitudinal analysis.

### 3.2 Attributes Meanings

#### 3.2.1 Identification and Temporal Attributes

| Attribute          | Description                              | Importance Level                                           |
| ------------------ | ---------------------------------------- | ---------------------------------------------------------- |
| **PHASE**    | ADNI study phase                         | Medium - For tracking phase-specific demographics          |
| **PTID**     | Participant ID (XXX_S_XXXXX)             | High - Participant identifier                              |
| **RID**      | Research ID                              | **Critical** - For merging with other tables         |
| **VISCODE**  | Visit code (typically 'bl' for baseline) | High - Identifies baseline assessment                      |
| **VISCODE2** | Alternative visit coding                 | Medium - Backup temporal identifier                        |
| **VISDATE**  | Visit date                               | **Critical** - Reference date for temporal alignment |

#### 3.2.2 Core Demographic Variables

| Attribute          | Description                         | Typical Values/Importance           | Importance Level                                 |
| ------------------ | ----------------------------------- | ----------------------------------- | ------------------------------------------------ |
| **PTSOURCE** | Data source (self, informant, both) | Medium - Data reliability indicator |                                                  |
| **PTGENDER** | **Participant gender**        | Male, Female                        | **Critical** - Key demographic covariate   |
| **PTDOB**    | Date of birth                       | Date format                         | **Critical** - For calculating precise age |
| **PTDOBYY**  | Year of birth                       | 4-digit year                        | High - Alternative age calculation               |
| **PTHAND**   | Handedness                          | Right, Left, Ambidextrous           | Medium - For neuroimaging studies                |
| **PTMARRY**  | **Marital status**            | Single, Married, etc.               | High - Social support indicator                  |
| **PTEDUCAT** | **Years of education**        | Numeric (0-30)                      | **Critical** - Cognitive reserve proxy     |
| **PTWORKHS** | Work history                        | Yes/No                              | Medium - Socioeconomic indicator                 |
| **PTWORK**   | Current work status                 | Working, Retired, etc.              | Medium - Activity level indicator                |

#### 3.2.3 Cultural and Linguistic Variables

| Attribute          | Description                | Importance Level                                       |
| ------------------ | -------------------------- | ------------------------------------------------------ |
| **PTETHCAT** | **Ethnic category**  | **Critical** - Population stratification control |
| **PTRACCAT** | **Race category**    | **Critical** - Population stratification control |
| **PTTLANG**  | Total languages spoken     | Medium - Cognitive reserve indicator                   |
| **PTPLANG**  | Primary language           | Medium - Cultural/linguistic background                |
| **PTENGSPK** | English speaking ability   | Medium - Test administration validity                  |
| **PTNLANG**  | Number of languages spoken | Medium - Cognitive reserve proxy                       |
| **PTCLANG**  | Current language use       | Medium - Cultural adaptation                           |

#### 3.2.4 Medical History and Onset Variables

| Attribute          | Description                     | Importance Level                    |
| ------------------ | ------------------------------- | ----------------------------------- |
| **PTADBEG**  | Age at AD symptoms onset        | High - Disease duration calculation |
| **PTCOGBEG** | Age at cognitive symptoms onset | High - Disease progression timeline |
| **PTADDX**   | Age at AD diagnosis             | High - Clinical history             |
| **PTNOTRT**  | Not receiving treatment         | Medium - Treatment confounder       |
| **PTRTYR**   | Treatment duration (years)      | Medium - Treatment effect           |

#### 3.2.5 Language Proficiency Details

| Attribute Pattern       | Description                        | Importance Level                           |
| ----------------------- | ---------------------------------- | ------------------------------------------ |
| **PTLANGPR[1-6]** | Proficiency in language [1-6]      | Low-Moderate - Detailed linguistic profile |
| **PTLANGSP[1-6]** | Speaking ability in language [1-6] | Low-Moderate - Communication assessment    |
| **PTLANGRD[1-6]** | Reading ability in language [1-6]  | Low-Moderate - Literacy assessment         |
| **PTLANGWR[1-6]** | Writing ability in language [1-6]  | Low-Moderate - Literacy assessment         |
| **PTLANGUN[1-6]** | Understanding of language [1-6]    | Low-Moderate - Comprehension assessment    |

#### 3.2.6 Quality Control and Metadata

| Attribute              | Description                | Importance Level                      |
| ---------------------- | -------------------------- | ------------------------------------- |
| **SITEID**       | Recruitment site ID        | Medium - For controlling site effects |
| **HAS_QC_ERROR** | Quality control error flag | High - For data filtering             |
| **update_stamp** | Last update timestamp      | Low - Version tracking                |

### 3.3 Important Notes

#### 3.3.1 Role in ML Projects

- **Covariate Control**: Demographic variables (age, gender, education) are critical confounders that must be controlled in Alzheimer's disease prediction models.
- **Population Stratification**: Race and ethnicity variables help account for population genetic differences.
- **Cognitive Reserve Proxy**: Education level (`PTEDUCAT`) is a well-established proxy for cognitive reserve, influencing dementia risk and progression.
- **Data Quality Indicators**: Source variables (`PTSOURCE`) indicate data reliability.

#### 3.3.2 Key Considerations for Feature Engineering

1. **Age Calculation**: Use `PTDOB` with `VISDATE` to calculate precise age at each visit, rather than relying on `PTDOBYY`.
2. **Education Effect**: Consider both linear and non-linear effects of education (thresholds at 12, 16 years).
3. **Missing Data Patterns**: Some variables (e.g., language proficiency details) may have high missingness rates.
4. **Cultural Sensitivity**: Ethnicity and race variables should be used appropriately as covariates, not as biological determinants.

#### 3.3.3 Recommended Feature Engineering Approaches

1. **Age Features**:

   - Chronological age at baseline
   - Age-squared term (for non-linear age effects)
   - Age at symptom onset (`PTADBEG`) when available
2. **Education Features**:

   - Years of education (continuous)
   - Education categories (<12, 12-16, >16 years)
   - Interaction terms with cognitive test scores
3. **Categorical Variables**:

   - One-hot encoding for gender, marital status, race
   - Consider grouping rare categories for race/ethnicity
4. **Composite Scores**:

   - Socioeconomic status proxy from education + work history
   - Multilingualism score from language variables

#### 3.3.4 Data Preparation Tips

1. **Use Baseline Values**: Most demographic variables are time-invariant; use the baseline (`VISCODE='bl'`) values.
2. **Handle Missing Data Strategically**:
   - Impute critical variables (age, gender, education)
   - Consider dropping variables with >30% missingness if not critical
3. **Check Consistency**: Verify that age calculations from `PTDOB` match other age references in the dataset.
4. **Respect Privacy**: When reporting results, follow guidelines for reporting demographic characteristics.
5. **Consider Site Effects**: Use `SITEID` as a random effect or covariate in mixed models.

#### 3.3.5 Important Caveats

1. **Cultural Bias**: Cognitive tests may have cultural biases; language and ethnicity variables help control for this.
2. **Education Quality**: Years of education may not capture education quality across different regions/systems.
3. **Self-report Limitations**: Some variables (e.g., symptom onset) rely on self/informant report and may be imprecise.
4. **Changing Demographics**: Later ADNI phases may have different demographic distributions than earlier phases.

---

## 4. UCBERKELEY_AMY_6MM_12Dec2025

### 4.1 General Description

The **Amyloid PET Analysis (UC Berkeley 6mm)** file contains quantitative amyloid burden measurements from PET imaging across all ADNI phases. This dataset provides **Standardized Uptake Value Ratios (SUVRs)** and **Centiloid values** for multiple brain regions, offering a molecular biomarker of amyloid plaque deposition—a core pathological feature of Alzheimer's disease. Processed with a uniform pipeline (6mm smoothing), this data enables standardized comparison of amyloid load across participants, timepoints, and ADNI phases.

### 4.2 Attributes Meanings

#### 4.2.1 Identification and Temporal Attributes

| Attribute                     | Description                             | Importance Level                                  |
| ----------------------------- | --------------------------------------- | ------------------------------------------------- |
| **LONIUID**             | LONI Image Unique ID                    | Medium - Links to raw imaging data                |
| **PTID**                | Participant ID                          | High - Participant identifier                     |
| **RID**                 | Research ID                             | **Critical** - Primary key for merging      |
| **VISCODE**             | Visit code (bl, m06, m12, etc.)         | High - Temporal ordering                          |
| **VISCODE2**            | Alternative visit coding                | Medium - Backup temporal identifier               |
| **SCANDATE**            | PET scan date                           | **Critical** - Temporal alignment reference |
| **SITEID**              | Imaging site ID                         | Medium - For site-effect adjustment               |
| **PROCESSDATE**         | Processing date                         | Low - Version tracking                            |
| **IMAGE_RESOLUTION**    | Processing resolution (6mm)             | Low - Technical parameter                         |
| **qc_flag**             | Quality control flag                    | **Critical** - Data reliability indicator   |
| **TRACER**              | PET tracer (AV45, PiB, etc.)            | **Critical** - Impacts SUVR interpretation  |
| **TRACER_SUVR_WARNING** | Warning for tracer-specific SUVR issues | Medium - Caution for analysis                     |

#### 4.2.2 Global Amyloid Burden Measures

| Attribute                              | Description                            | Typical Values/Importance         | Importance Level                                     |
| -------------------------------------- | -------------------------------------- | --------------------------------- | ---------------------------------------------------- |
| **AMYLOID_STATUS**               | Binary amyloid positivity              | Positive/Negative                 | **Critical** - Primary categorical measure     |
| **AMYLOID_STATUS_COMPOSITE_REF** | Positivity using composite reference   | Alternative classification        | High - Robust to reference region                    |
| **CENTILOIDS**                   | **Centiloid value**              | Continuous (0-100+), standardized | **Critical** - Tracer-agnostic amyloid measure |
| **SUMMARY_SUVR**                 | Cortical composite SUVR                | Continuous, typically 0.8-2.0     | **Critical** - Primary continuous measure      |
| **SUMMARY_VOLUME**               | Composite region volume                | mm³                              | Low - Anatomical reference                           |
| **WHOLECEREBELLUM_SUVR**         | SUVR normalized to whole cerebellum    | Continuous                        | High - Standard reference region                     |
| **COMPOSITE_REF_SUVR**           | SUVR normalized to composite reference | Continuous                        | High - Alternative normalization                     |

#### 4.2.3 Regional Amyloid Measures

The dataset includes SUVR and volume for ~100 brain regions. Key regions for AD:

| Region Category                 | Example Attributes                                          | Importance Level                              |
| ------------------------------- | ----------------------------------------------------------- | --------------------------------------------- |
| **Early Affected Cortex** | `CTX_ENTORHINAL_SUVR`, `CTX_PRECUNEUS_SUVR`             | **Critical** - Early amyloid deposition |
| **Association Cortex**    | `CTX_INFERIORPARIETAL_SUVR`, `CTX_MIDDLETEMPORAL_SUVR`  | **Critical** - High amyloid in AD       |
| **Frontal Cortex**        | `CTX_FRONTALPOLE_SUVR`, `CTX_ROSTRALMIDDLEFRONTAL_SUVR` | High - Later accumulation                     |
| **Posterior Cingulate**   | `CTX_POSTERIORCINGULATE_SUVR`                             | **Critical** - Default mode network hub |
| **Hippocampus**           | `HIPPOCAMPUS_SUVR`                                        | Medium - More relevant for tau                |
| **Hemisphere-Specific**   | `CTX_LH_*`, `CTX_RH_*` (all regions)                    | Medium - Asymmetry analysis                   |
| **Subcortical**           | `CAUDATE_SUVR`, `PUTAMEN_SUVR`                          | Low - Less affected in AD                     |

*Note: Each region has `_SUVR` (amyloid burden) and `_VOLUME` (region volume) attributes.*

#### 4.2.4 Quality Control and Metadata

| Attribute              | Description           | Importance Level       |
| ---------------------- | --------------------- | ---------------------- |
| **update_stamp** | Last update timestamp | Low - Version tracking |

### 4.3 Important Notes

#### 4.3.1 Role in ML Projects

- **A/T/N Framework**: Represents **A** (Amyloid) component—fundamental for AD biological definition
- **Early Detection**: Amyloid accumulation precedes symptoms by 10-20 years
- **Disease Staging**: Centiloid values enable standardized staging (negative <24, positive ≥24)
- **Treatment Selection**: Essential for anti-amyloid therapy eligibility

#### 4.3.2 Key Considerations

1. **Tracer Differences**: PiB (ADNI1) vs AV45/Florbetapir (later phases) have different kinetics
2. **Reference Region**: Whole cerebellum (standard) vs composite reference
3. **Cutoff Values**: SUVR >1.11 for AV45; Centiloid >24 for positivity
4. **Partial Volume Effects**: Atrophy can artificially lower SUVR

#### 4.3.3 Recommended Feature Engineering

1. **Primary Features**:
   - `CENTILOIDS` (continuous)
   - `AMYLOID_STATUS` (binary)
   - `SUMMARY_SUVR` (continuous alternative)
2. **Regional Patterns**:
   - Posterior-anterior ratio
   - Left-right asymmetry in temporal regions
3. **Composite Scores**:
   - Weighted average of early-affected regions

#### 4.3.4 Data Preparation Tips

1. **Quality Filter**: Exclude rows with `qc_flag` issues
2. **Tracer Consistency**: Consider separate analyses for PiB vs AV45 or use Centiloids
3. **Missing Data**: PET scans are sparse (baseline + every 24 months typically)
4. **Normalization**: SUVR already normalized; no further normalization needed

#### 4.3.5 Important Caveats

1. **Amyloid Plateau**: Levels plateau in dementia stage—limited utility for late-stage progression
2. **Non-AD Amyloid**: Some amyloid positivity in normal aging
3. **Off-target Binding**: White matter binding can affect SUVR
4. **Cost/Access**: PET availability creates selection bias

---

## 5. UCSFFSX7_12Dec2025

### 5.1 General Description

The **FreeSurfer Cross-Sectional MRI Analysis (Version 7.x)** file contains comprehensive structural MRI measures derived from T1-weighted scans using FreeSurfer version 7.x. This dataset provides **volumetric and cortical thickness measurements** for ~150 brain regions, capturing structural neurodegeneration patterns characteristic of Alzheimer's disease. Version 7.x offers improved accuracy and consistency over earlier versions.

### 5.2 Attributes Meanings

#### 5.2.1 Identification and Temporal Attributes

| Attribute                   | Description                    | Importance Level                             |
| --------------------------- | ------------------------------ | -------------------------------------------- |
| **PHASE**             | ADNI phase                     | Medium - Processing differences              |
| **PTID**              | Participant ID                 | High - Participant identifier                |
| **RID**               | Research ID                    | **Critical** - Primary key for merging |
| **VISCODE**           | Visit code                     | High - Temporal ordering                     |
| **VISCODE2**          | Alternative visit coding       | Medium - Backup temporal identifier          |
| **IMAGEUID**          | Image Unique ID                | Medium - Links to raw images                 |
| **FIELD_STRENGTH**    | MRI field strength (1.5T/3T)   | **Critical** - Affects measurements    |
| **EXAMDATE**          | MRI exam date                  | **Critical** - Temporal alignment      |
| **RUNDATE**           | Processing date                | Low - Version tracking                       |
| **STATUS**            | Processing status              | Medium - Success/failure indicator           |
| **FSVER**             | FreeSurfer version (7.x)       | High - Consistency marker                    |
| **OVERALLQC**         | Overall quality control rating | **Critical** - Data reliability        |
| **Regional QC flags** | `TEMPQC`, `FRONTQC`, etc.  | Medium - Regional reliability                |

#### 5.2.2 Global Brain Measures

| Attribute         | Description                                          | Importance Level                             |
| ----------------- | ---------------------------------------------------- | -------------------------------------------- |
| **ST101SV** | **Estimated Total Intracranial Volume (eTIV)** | **Critical** - Normalization reference |
| **ST102CV** | **Total Brain Volume**                         | **Critical** - Global atrophy measure  |
| **ST102SA** | Total cortical surface area                          | Medium - Cortical morphology                 |
| **ST102TA** | Total cortical area                                  | Low - Combined measure                       |
| **ST102TS** | Total cortical surface                               | Low - Combined measure                       |

#### 5.2.3 Key Regional Volumes for AD

| Attribute        | Description                              | Importance Level                          |
| ---------------- | ---------------------------------------- | ----------------------------------------- |
| **ST28SA** | **Left Hippocampus Volume**        | **Critical** - Primary AD biomarker |
| **ST60CV** | **Right Hippocampus Volume**       | **Critical** - Primary AD biomarker |
| **ST13CV** | **Left Entorhinal Cortex Volume**  | **Critical** - Early atrophy region |
| **ST54CV** | **Right Entorhinal Cortex Volume** | **Critical** - Early atrophy region |
| **ST42SV** | **Left Lateral Ventricle Volume**  | High - Indirect atrophy measure           |
| **ST80SV** | **Right Lateral Ventricle Volume** | High - Indirect atrophy measure           |
| **ST4SV**  | Left Amygdala Volume                     | Medium - Emotional processing             |
| **ST49CV** | Right Amygdala Volume                    | Medium - Emotional processing             |

#### 5.2.4 Key Cortical Thickness Measures

| Attribute        | Description                                 | Importance Level                             |
| ---------------- | ------------------------------------------- | -------------------------------------------- |
| **ST13TA** | **Left Entorhinal Cortex Thickness**  | **Critical** - Early cortical thinning |
| **ST54TA** | **Right Entorhinal Cortex Thickness** | **Critical** - Early cortical thinning |
| **ST23TA** | Left Middle Temporal Gyrus Thickness        | High - Language/memory                       |
| **ST83TA** | Right Middle Temporal Gyrus Thickness       | High - Language/memory                       |
| **ST43TA** | Left Fusiform Gyrus Thickness               | High - Visual processing                     |
| **ST94TA** | Right Fusiform Gyrus Thickness              | High - Visual processing                     |
| **ST33TA** | Left Precuneus Thickness                    | High - Default mode network                  |
| **ST84TA** | Right Precuneus Thickness                   | High - Default mode network                  |

*Note: Attributes follow pattern: ST[number][type] where type: CV=Cortical Volume, SA=Subcortical Volume, TA=Thickness Average, SV=Subcortical Volume (alternative).*

#### 5.2.5 Quality Control and Metadata

| Attribute              | Description           | Importance Level       |
| ---------------------- | --------------------- | ---------------------- |
| **update_stamp** | Last update timestamp | Low - Version tracking |

### 5.3 Important Notes

#### 5.3.1 Role in ML Projects

- **N in A/T/N**: Represents **Neurodegeneration** component
- **Disease Progression**: Atrophy rates correlate with cognitive decline
- **Differential Diagnosis**: Patterns distinguish AD from other dementias
- **Multi-region Analysis**: Enables network-based approaches

#### 5.3.2 Key Considerations

1. **Normalization**: Always normalize volumes by eTIV (ST101SV)
2. **Field Strength**: 3T vs 1.5T differences must be accounted for
3. **Version Effects**: FreeSurfer 7.x ≠ earlier versions—do not mix
4. **Quality Control**: OVERALLQC essential for filtering

#### 5.3.3 Recommended Feature Engineering

1. **Normalized Volumes**:
   ```python
   hippoc_vol_norm = (ST28SA + ST60CV) / ST101SV * 1000
   ```
2. **Asymmetry Indices**:
   ```python
   hippoc_asym = (ST60CV - ST28SA) / (ST28SA + ST60CV)
   ```
3. **Composite Scores**:
   - Medial temporal lobe composite
   - Global atrophy score
4. **Cortical Thickness Networks**: PCA on all thickness measures

#### 5.3.4 Data Preparation Tips

1. **Normalization**: Divide all volumes by eTIV × 1000
2. **QC Filter**: Exclude subjects with poor OVERALLQC
3. **Field Strength Adjustment**: Include as covariate or harmonize
4. **Missing Regions**: Some segmentations may fail—impute or exclude

#### 5.3.5 Important Caveats

1. **Cross-sectional Limitations**: Single timepoint; longitudinal processing needed for change measures
2. **Partial Volume Effects**: Misclassification at tissue boundaries
3. **Scanner Effects**: Differences persist despite harmonization
4. **Version Incompatibility**: Do not mix FreeSurfer versions without correction

---

## 6. UPENNBIOMK_ROCHE_ELECSYS_12Dec2025

### 6.1 General Description

The **CSF Biomarkers (Roche Elecsys)** file contains cerebrospinal fluid measurements of core Alzheimer's disease biomarkers using the clinically validated Roche Elecsys platform. This dataset provides **quantitative measures of amyloid-beta peptides (Aβ40, Aβ42), total tau (t-tau), and phosphorylated tau (p-tau)**—the essential components of the A/T/N biological framework. These biomarkers offer direct biochemical evidence of AD pathology with high diagnostic accuracy.

### 6.2 Attributes Meanings

#### 6.2.1 Identification and Temporal Attributes

| Attribute          | Description         | Importance Level                             |
| ------------------ | ------------------- | -------------------------------------------- |
| **PHASE**    | ADNI phase          | Medium - Batch variations                    |
| **PTID**     | Participant ID      | High - Participant identifier                |
| **RID**      | Research ID         | **Critical** - Primary key for merging |
| **VISCODE2** | Visit code          | High - Temporal ordering                     |
| **EXAMDATE** | CSF collection date | **Critical** - Temporal alignment      |
| **BATCH**    | Assay batch number  | **Critical** - Batch effect correction |
| **RUNDATE**  | Assay run date      | Low - Version tracking                       |

#### 6.2.2 Core CSF Biomarkers

| Attribute         | Description                                | Typical Values (pg/mL) | Importance Level                              |
| ----------------- | ------------------------------------------ | ---------------------- | --------------------------------------------- |
| **ABETA40** | Amyloid-beta 40 concentration              | 4000-12000             | High - Reference for ratio                    |
| **ABETA42** | **Amyloid-beta 42 concentration**    | 200-1500               | **Critical** - Decreased in AD          |
| **TAU**     | **Total tau concentration**          | 100-1200               | **Critical** - Neurodegeneration marker |
| **PTAU**    | **Phosphorylated tau concentration** | 15-120                 | **Critical** - Tau pathology marker     |
| **COMMENT** | Assay comments                             | Text annotations       | Medium - Outlier explanations                 |

#### 6.2.3 Derived Ratios (Computed)

| Ratio                       | Formula               | Clinical Cutoff   | Importance                               |
| --------------------------- | --------------------- | ----------------- | ---------------------------------------- |
| **Aβ42/40 Ratio**    | `ABETA42 / ABETA40` | <0.067 = amyloid+ | **Critical** - Most robust measure |
| **p-tau/Aβ42 Ratio** | `PTAU / ABETA42`    | Variable          | High - Combined pathology                |
| **t-tau/Aβ42 Ratio** | `TAU / ABETA42`     | Variable          | Medium - Alternative                     |

#### 6.2.4 Quality Control and Metadata

| Attribute              | Description           | Importance Level       |
| ---------------------- | --------------------- | ---------------------- |
| **update_stamp** | Last update timestamp | Low - Version tracking |

### 6.3 Important Notes

#### 6.3.1 Role in ML Projects

- **Complete A/T/N**: Provides all three components: A (Aβ42/40), T (p-tau), N (t-tau)
- **High Diagnostic Accuracy**: AUC 0.90-0.95 for AD vs controls
- **Early Detection**: Changes precede symptoms by years
- **Clinical Translation**: Used in diagnostic guidelines worldwide

#### 6.3.2 Key Considerations

1. **Batch Effects**: Strong batch effects require correction
2. **Ratio vs Absolute**: Aβ42/40 ratio superior to Aβ42 alone
3. **Pre-analytical Factors**: Sensitive to collection/handling procedures
4. **Platform Specific**: Roche Elecsys values not interchangeable with other platforms

#### 6.3.3 Recommended Feature Engineering

1. **A/T/N Classification**:
   ```python
   A_pos = ABETA42/ABETA40 < 0.067
   T_pos = PTAU > 24
   N_pos = TAU > 300
   ```
2. **Continuous Scores**:
   - Aβ42/40 ratio (continuous)
   - p-tau (continuous)
   - t-tau (continuous)
3. **Composite Biomarker**:
   - Logistic function combining Aβ42/40 and p-tau

#### 6.3.4 Data Preparation Tips

1. **Batch Correction**: Apply ComBat or include batch as covariate
2. **Compute Ratios**: Always calculate Aβ42/40 ratio
3. **Outlier Handling**: Check COMMENT field for assay issues
4. **Missing Data**: CSF often only at baseline—consider as time-invariant

#### 6.3.5 Important Caveats

1. **Invasiveness**: Lumbar puncture limits participation → selection bias
2. **Longitudinal Sparsity**: Rarely repeated (baseline + 2-3 years if at all)
3. **Cutoff Variability**: Population-specific cutoffs may apply
4. **Blood-Brain Barrier**: CSF ≠ peripheral measures

---

## Summary of Multimodal Integration for ML Projects

| Dataset                      | A/T/N Component       | Key Features                           | Temporal Nature               |
| ---------------------------- | --------------------- | -------------------------------------- | ----------------------------- |
| **UCBERKELEY_AMY_6MM** | A (Amyloid)           | Centiloids, SUMMARY_SUVR               | Sparse (0, 24, 48 months)     |
| **UCSFFSX7**           | N (Neurodegeneration) | Hippocampal volume, cortical thickness | Most visits (0, 6, 12, 24...) |
| **UPENNBIOMK**         | A/T/N (All)           | Aβ42/40, p-tau, t-tau                 | Baseline + rare follow-up     |

**Integration Strategy**:

1. **Baseline Features**: Use all biomarkers at baseline
2. **Longitudinal Features**: MRI at each visit, PET/CSF carried forward
3. **Time-invariant**: Genetic (APOE), demographics, baseline biomarkers
4. **Time-varying**: MRI, cognitive scores, diagnosis

---
