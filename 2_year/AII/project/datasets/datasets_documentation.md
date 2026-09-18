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

Shape= 34899 x 13

The **Data Dictionary** file is the comprehensive reference document for the entire ADNI dataset. It provides detailed metadata for every variable across all ADNI phases and tables, serving as the essential guide for understanding variable definitions, data types, coding schemes, and measurement units.

### 1.2 Attributes Meanings

| Attribute                | Description                                            |
| ------------------------ | ------------------------------------------------------ |
| **PHASE**          | ADNI study phase (ADNI1, ADNIGO, ADNI2, ADNI3, ADNI4)  |
| **CRFNAME**        | Original Case Report Form name                         |
| **TBLNAME**        | Table name where the variable appears                  |
| **FLDNAME**        | Field/Variable name in the table                       |
| **TEXT**           | **Complete textual description of the variable** |
| **TYPE**           | Data type (Numeric, Text, Date, etc.)                  |
| **LENGTH**         | Maximum length for text fields                         |
| **DD_CRF_VERSION** | Data Dictionary/CRF version number                     |
| **CODE**           | **Coding scheme for categorical variables**      |
| **UNITS**          | Measurement units for numeric variables                |
| **STATUS**         | Variable status (Active, Inactive, Deprecated)         |
| **CODE_CHANGES**   | Notes on coding changes over time                      |
| **MAPPING_NOTES**  | Notes on variable mapping between ADNI phases          |

---

## 2. DXSUM_x_.csv

### 2.1 General Description

Shape = 15931 x 41

The **Diagnostic Summary** file contains longitudinal diagnostic information for all ADNI participants across all study visits. This is the **core clinical outcome file** that tracks disease progression from Cognitively Normal (CN) to Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD) over time. With approximately 16,000 records, it provides rich temporal data for time-series analysis, survival modeling, and progression prediction.

### 2.2 Attributes Meanings

#### 2.2.1 Identification and Temporal Attributes

| Attribute          | Description                              |
| ------------------ | ---------------------------------------- |
| **PHASE**    | ADNI phase                               |
| **PTID**     | Participant ID (format: XXX_S_XXXXX)     |
| **RID**      | Research ID (numeric portion of PTID)    |
| **VISCODE**  | Visit code (bl, m06, m12, m24, m36, m48) |
| **VISCODE2** | Alternative visit coding                 |
| **EXAMDATE** | Clinical examination date                |

#### 2.2.2 Primary Diagnostic Variables

| Attribute           | Description                          | Typical Values |
| ------------------- | ------------------------------------ | -------------- |
| **DIAGNOSIS** | **Primary clinical diagnosis** | CN, MCI, AD    |
| **DXNORM**    | Cognitively Normal flag              | 0/1            |
| **DXMCI**     | Mild Cognitive Impairment flag       | 0/1            |
| **DXAD**      | Alzheimer's Disease flag             | 0/1            |

#### 2.2.3 Diagnostic Specifications and Subtypes

| Attribute                 | Description                        |
| ------------------------- | ---------------------------------- |
| **DXMDES**          | MCI subtype description            |
| **DXMPTR1-DXMPTR6** | MCI diagnostic criteria pointers   |
| **DXMDUE**          | Primary cause of MCI               |
| **DXDSEV**          | Dementia severity rating           |
| **DXAPROB**         | AD probability (probable/possible) |
| **DXPARK**          | Parkinsonism flag                  |
| **DXDEP**           | Depression flag                    |
| **DXCONFID**        | Diagnostic confidence              |

#### 2.2.4 Quality Control and Metadata

| Attribute              | Description                |
| ---------------------- | -------------------------- |
| **SITEID**       | Recruitment site ID        |
| **HAS_QC_ERROR** | Quality control error flag |
| **update_stamp** | Last update timestamp      |

---

## 3. PTDEMOG_x_.csv

### 3.1 General Description

Shape = 6223 x 84

The **Participant Demographics** file contains comprehensive baseline demographic, social, and cultural information for all ADNI participants. This table provides **static covariates** that are essential for controlling confounding factors, understanding population characteristics, and building more accurate predictive models. The data is primarily collected at baseline with minimal changes over time, making it ideal for use as time-invariant features in longitudinal analysis.

### 3.2 Attributes Meanings

#### 3.2.1 Identification and Temporal Attributes

| Attribute          | Description                              |
| ------------------ | ---------------------------------------- |
| **PHASE**    | ADNI study phase                         |
| **PTID**     | Participant ID (XXX_S_XXXXX)             |
| **RID**      | Research ID                              |
| **VISCODE**  | Visit code (typically 'bl' for baseline) |
| **VISCODE2** | Alternative visit coding                 |
| **VISDATE**  | Visit date                               |

#### 3.2.2 Core Demographic Variables

| Attribute          | Description                         | Typical Values/Importance           |
| ------------------ | ----------------------------------- | ----------------------------------- |
| **PTSOURCE** | Data source (self, informant, both) | Medium - Data reliability indicator |
| **PTGENDER** | **Participant gender**        | Male, Female                        |
| **PTDOB**    | Date of birth                       | Date format                         |
| **PTDOBYY**  | Year of birth                       | 4-digit year                        |
| **PTHAND**   | Handedness                          | Right, Left, Ambidextrous           |
| **PTMARRY**  | **Marital status**            | Single, Married, etc.               |
| **PTEDUCAT** | **Years of education**        | Numeric (0-30)                      |
| **PTWORKHS** | Work history                        | Yes/No                              |
| **PTWORK**   | Current work status                 | Working, Retired, etc.              |

#### 3.2.3 Cultural and Linguistic Variables

| Attribute          | Description                |
| ------------------ | -------------------------- |
| **PTETHCAT** | **Ethnic category**  |
| **PTRACCAT** | **Race category**    |
| **PTTLANG**  | Total languages spoken     |
| **PTPLANG**  | Primary language           |
| **PTENGSPK** | English speaking ability   |
| **PTNLANG**  | Number of languages spoken |
| **PTCLANG**  | Current language use       |

#### 3.2.4 Medical History and Onset Variables

| Attribute          | Description                     |
| ------------------ | ------------------------------- |
| **PTADBEG**  | Age at AD symptoms onset        |
| **PTCOGBEG** | Age at cognitive symptoms onset |
| **PTADDX**   | Age at AD diagnosis             |
| **PTNOTRT**  | Not receiving treatment         |
| **PTRTYR**   | Treatment duration (years)      |

#### 3.2.5 Language Proficiency Details

| Attribute Pattern       | Description                        |
| ----------------------- | ---------------------------------- |
| **PTLANGPR[1-6]** | Proficiency in language [1-6]      |
| **PTLANGSP[1-6]** | Speaking ability in language [1-6] |
| **PTLANGRD[1-6]** | Reading ability in language [1-6]  |
| **PTLANGWR[1-6]** | Writing ability in language [1-6]  |
| **PTLANGUN[1-6]** | Understanding of language [1-6]    |

#### 3.2.6 Quality Control and Metadata

| Attribute              | Description                |
| ---------------------- | -------------------------- |
| **SITEID**       | Recruitment site ID        |
| **HAS_QC_ERROR** | Quality control error flag |
| **update_stamp** | Last update timestamp      |

---

## 4. UCBERKELEY_AMY_6MM_12Dec2025

### 4.1 General Description

Shape = 4582 x 344

The **Amyloid PET Analysis (UC Berkeley 6mm)** file contains quantitative amyloid burden measurements from PET imaging across all ADNI phases. This dataset provides **Standardized Uptake Value Ratios (SUVRs)** and **Centiloid values** for multiple brain regions, offering a molecular biomarker of amyloid plaque deposition—a core pathological feature of Alzheimer's disease. Processed with a uniform pipeline (6mm smoothing), this data enables standardized comparison of amyloid load across participants, timepoints, and ADNI phases.

### 4.2 Attributes Meanings

#### 4.2.1 Identification and Temporal Attributes

| Attribute                     | Description                             |
| ----------------------------- | --------------------------------------- |
| **LONIUID**             | LONI Image Unique ID                    |
| **PTID**                | Participant ID                          |
| **RID**                 | Research ID                             |
| **VISCODE**             | Visit code (bl, m06, m12, etc.)         |
| **VISCODE2**            | Alternative visit coding                |
| **SCANDATE**            | PET scan date                           |
| **SITEID**              | Imaging site ID                         |
| **PROCESSDATE**         | Processing date                         |
| **IMAGE_RESOLUTION**    | Processing resolution (6mm)             |
| **qc_flag**             | Quality control flag                    |
| **TRACER**              | PET tracer (AV45, PiB, etc.)            |
| **TRACER_SUVR_WARNING** | Warning for tracer-specific SUVR issues |

#### 4.2.2 Global Amyloid Burden Measures

| Attribute                              | Description                            | Typical Values/Importance         |
| -------------------------------------- | -------------------------------------- | --------------------------------- |
| **AMYLOID_STATUS**               | Binary amyloid positivity              | Positive/Negative                 |
| **AMYLOID_STATUS_COMPOSITE_REF** | Positivity using composite reference   | Alternative classification        |
| **CENTILOIDS**                   | **Centiloid value**              | Continuous (0-100+), standardized |
| **SUMMARY_SUVR**                 | Cortical composite SUVR                | Continuous, typically 0.8-2.0     |
| **SUMMARY_VOLUME**               | Composite region volume                | mm³                              |
| **WHOLECEREBELLUM_SUVR**         | SUVR normalized to whole cerebellum    | Continuous                        |
| **COMPOSITE_REF_SUVR**           | SUVR normalized to composite reference | Continuous                        |

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

| Attribute              | Description           |
| ---------------------- | --------------------- |
| **update_stamp** | Last update timestamp |

---

## 5. UCSFFSX7_12Dec2025

### 5.1 General Description

Shape = 12151 x 347

The **FreeSurfer Cross-Sectional MRI Analysis (Version 7.x)** file contains comprehensive structural MRI measures derived from T1-weighted scans using FreeSurfer version 7.x. This dataset provides **volumetric and cortical thickness measurements** for ~150 brain regions, capturing structural neurodegeneration patterns characteristic of Alzheimer's disease. Version 7.x offers improved accuracy and consistency over earlier versions.

### 5.2 Attributes Meanings

#### 5.2.1 Identification and Temporal Attributes

| Attribute                   | Description                    |
| --------------------------- | ------------------------------ |
| **PHASE**             | ADNI phase                     |
| **PTID**              | Participant ID                 |
| **RID**               | Research ID                    |
| **VISCODE**           | Visit code                     |
| **VISCODE2**          | Alternative visit coding       |
| **IMAGEUID**          | Image Unique ID                |
| **FIELD_STRENGTH**    | MRI field strength (1.5T/3T)   |
| **EXAMDATE**          | MRI exam date                  |
| **RUNDATE**           | Processing date                |
| **STATUS**            | Processing status              |
| **FSVER**             | FreeSurfer version (7.x)       |
| **OVERALLQC**         | Overall quality control rating |
| **Regional QC flags** | `TEMPQC`, `FRONTQC`, etc.  |

#### 5.2.2 Global Brain Measures

| Attribute         | Description                                          |
| ----------------- | ---------------------------------------------------- |
| **ST101SV** | **Estimated Total Intracranial Volume (eTIV)** |
| **ST102CV** | **Total Brain Volume**                         |
| **ST102SA** | Total cortical surface area                          |
| **ST102TA** | Total cortical area                                  |
| **ST102TS** | Total cortical surface                               |

#### 5.2.3 Key Regional Volumes for AD

| Attribute        | Description                              |
| ---------------- | ---------------------------------------- |
| **ST28SA** | **Left Hippocampus Volume**        |
| **ST60CV** | **Right Hippocampus Volume**       |
| **ST13CV** | **Left Entorhinal Cortex Volume**  |
| **ST54CV** | **Right Entorhinal Cortex Volume** |
| **ST42SV** | **Left Lateral Ventricle Volume**  |
| **ST80SV** | **Right Lateral Ventricle Volume** |
| **ST4SV**  | Left Amygdala Volume                     |
| **ST49CV** | Right Amygdala Volume                    |

#### 5.2.4 Key Cortical Thickness Measures

| Attribute        | Description                                 |
| ---------------- | ------------------------------------------- |
| **ST13TA** | **Left Entorhinal Cortex Thickness**  |
| **ST54TA** | **Right Entorhinal Cortex Thickness** |
| **ST23TA** | Left Middle Temporal Gyrus Thickness        |
| **ST83TA** | Right Middle Temporal Gyrus Thickness       |
| **ST43TA** | Left Fusiform Gyrus Thickness               |
| **ST94TA** | Right Fusiform Gyrus Thickness              |
| **ST33TA** | Left Precuneus Thickness                    |
| **ST84TA** | Right Precuneus Thickness                   |

*Note: Attributes follow pattern: ST[number][type] where type: CV=Cortical Volume, SA=Subcortical Volume, TA=Thickness Average, SV=Subcortical Volume (alternative).*

#### 5.2.5 Quality Control and Metadata

| Attribute              | Description           |
| ---------------------- | --------------------- |
| **update_stamp** | Last update timestamp |

---

## 6. UPENNBIOMK_ROCHE_ELECSYS_12Dec2025

### 6.1 General Description

Shape = 3174 x 13

The **CSF Biomarkers (Roche Elecsys)** file contains cerebrospinal fluid measurements of core Alzheimer's disease biomarkers using the clinically validated Roche Elecsys platform. This dataset provides **quantitative measures of amyloid-beta peptides (Aβ40, Aβ42), total tau (t-tau), and phosphorylated tau (p-tau)**—the essential components of the A/T/N biological framework. These biomarkers offer direct biochemical evidence of AD pathology with high diagnostic accuracy.

### 6.2 Attributes Meanings

#### 6.2.1 Identification and Temporal Attributes

| Attribute          | Description         |
| ------------------ | ------------------- |
| **PHASE**    | ADNI phase          |
| **PTID**     | Participant ID      |
| **RID**      | Research ID         |
| **VISCODE2** | Visit code          |
| **EXAMDATE** | CSF collection date |
| **BATCH**    | Assay batch number  |
| **RUNDATE**  | Assay run date      |

#### 6.2.2 Core CSF Biomarkers

| Attribute         | Description                                | Typical Values (pg/mL) |
| ----------------- | ------------------------------------------ | ---------------------- |
| **ABETA40** | Amyloid-beta 40 concentration              | 4000-12000             |
| **ABETA42** | **Amyloid-beta 42 concentration**    | 200-1500               |
| **TAU**     | **Total tau concentration**          | 100-1200               |
| **PTAU**    | **Phosphorylated tau concentration** | 15-120                 |
| **COMMENT** | Assay comments                             | Text annotations       |

#### 6.2.3 Derived Ratios (Computed)

| Ratio                       | Formula               | Clinical Cutoff   |
| --------------------------- | --------------------- | ----------------- |
| **Aβ42/40 Ratio**    | `ABETA42 / ABETA40` | <0.067 = amyloid+ |
| **p-tau/Aβ42 Ratio** | `PTAU / ABETA42`    | Variable          |
| **t-tau/Aβ42 Ratio** | `TAU / ABETA42`     | Variable          |

#### 6.2.4 Quality Control and Metadata

| Attribute              | Description           |
| ---------------------- | --------------------- |
| **update_stamp** | Last update timestamp |
