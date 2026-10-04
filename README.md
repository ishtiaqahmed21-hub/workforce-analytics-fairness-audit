# Workforce Analytics & Algorithmic Fairness Audit

An operations analyst needs to understand workforce concentration, compensation patterns and the limits of predictive analytics before recommending action. This project answers those questions with a fictional HR case study and an independent historical income fairness benchmark. It does not model a real employer or recommend individual employment decisions.

## Start here

- [Executive brief](reports/executive_summary.pdf), [measured findings](reports/findings.md), [traceability](reports/traceability.csv).
- [Live local HTML dashboard](dashboard/index.html), [Power BI specification](dashboard/dashboard_specification.md).
- [Methodology](METHODOLOGY.md), [model card](MODEL_CARD.md), [fairness audit](FAIRNESS_AUDIT.md), [data provenance](data/README.md).
- [Machine-readable results](reports/results.json) and [resume bullets](reports/resume_bullets.md).

## Reproduce

Tested with Python 3.14. Install a clean environment, then run from this repository:

```sh
python -m venv .venv
# Activate .venv using your operating system's standard command.
python -m pip install -r requirements.txt
python download_data.py
python run_pipeline.py
python -m pytest -q
python execute_notebooks.py
```

The pipeline rebuilds tables, figures, four PDFs, model/audit cards, dashboard, results and seven notebooks. Notebook execution then validates those notebooks in fresh kernels. Raw sources are hash-checked and excluded from Git. Exact transitive installed package versions are recorded in requirements-lock.txt. Download needs network; subsequent execution works offline. Do not publish raw data by accidentally force-adding ignored files. Code/data licenses are separate.

## Architecture

`data/raw -> src/data_cleaning -> src/workforce_metrics/statistics/modeling/fairness -> data/processed + reports/results.json -> figures + dashboard + reports`

- `src/` contains reusable data, modeling, fairness and statistics functions.
- `config/` records seed, split sizes and threshold assumptions.
- `notebooks/01..07` exposes each analysis stage with executable calculations and interpretation.
- `tests/` protects confusion semantics, undefined subgroup rates, interval boundaries, snapshot counts and training-only preprocessing.
- `reports/` connects actual findings to recommendations; no invented savings.

## Analytical boundaries

SIMULATED: 1,470 fictional IBM records. OBSERVED: historical Adult records and BLS published estimates. DERIVED: metrics retain the source's real/fictional status. ASSUMPTION: train/test fractions, seed, threshold objective and support flag. Adult predicts income class, never attrition; BLS is context, never a matched employer benchmark. Attrition share has no annual period. Income currency is unspecified. Data does not prove discrimination, causation or future real-world model performance.

Logistic regression and random forest are selected only on validation average precision. Thresholds optimize validation F1. Independent test metrics, subgroup sample support, confusion counts, Wilson intervals and calibration accompany findings. No SHAP dependency is needed: held-out permutation importance supplies transparent, repeatable predictive explanations.

## Evidence and next operational step

The run identifies role-level fictional attrition concentration and measures held-out prediction quality. Recommendations are to validate definitions, collect dated cohort records, investigate workload through voluntary feedback and test a measured support pilot. Financial benefits and future attrition reductions require real evidence. See generated findings for exact numerical results.

## License

Original code is MIT. IBM HR derivative data retains ODbL/DbCL, Adult is CC BY 4.0, BLS statistics are public domain. [NOTICE.md](NOTICE.md) contains source attributions and terms. No IBM, Siemens, Deloitte, UCI or BLS endorsement is claimed.
