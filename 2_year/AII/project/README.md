# aii-adni

A three stage clinical decision support pipeline built on the ADNI (Alzheimer's Disease Neuroimaging Initiative) longitudinal dataset. Developed as part of the AI in Industry course, University of Bologna.

Anomaly detection asks whether a patient visit looks anomalous relative to a cognitively normal reference population, using both a PCA density model and an autoencoder health index. Conversion risk turns that signal plus the raw biomarkers into a discrete time hazard estimate of converting to a worse diagnosis, evaluated as a survival problem rather than a plain regression. Decision support turns that risk estimate into a recommended monitoring interval, audited for fairness across demographic groups and explained through feature attribution. Each stage's output feeds the next, and `notebooks/pipeline/final_pipeline.ipynb` is where all three are checked together on one common patient population.

The full presentation deck in `presentation/` walks through the project end to end, data quality, all three stages, and the pipeline extension that wires them together, with results and honest limitations at each step.

## Layout

```
datasets/           the raw ADNI tables, DATADIC, and the joined final.csv
exploration/        the join notebook and its own data quality report
util/               shared functions, decision_util.py and hazard_util.py
presentation/       the project slide deck plus two standalone methodology slides
notebooks/
  anomaly_detection/     Point 1, two methods (PCA and an autoencoder health index),
                         a pipeline extension with fairness and calibration work,
                         and a walkthrough notebook comparing both
  hazard_model/          Point 2, the discrete time hazard model
  remaining_useful_life/ Leo's exploration of the same conversion question as a
                         regression and hazard classification problem, compared
                         across six model variants
  decision_support/      Point 3, four interval policies, a fairness audit,
                         and an attribution track
  fairness/              a fairness correction approach shared across stages
  pipeline/              notebooks that chain the three stages into one,
                         checked on a common patient population
  archive/               superseded notebooks kept for reference
results/            generated metrics, figures, and standardized comparison tables
```

Each of the three main stages documents itself in more depth close to its own code: `notebooks/anomaly_detection/PROJECT_REPORT.md`, `notebooks/remaining_useful_life/RUL_METHODS.md`, and `notebooks/decision_support/NOTES.md`. `data_evaluation_report.md` covers what the raw ADNI tables and the joined dataset actually support.
