# Model card

## Purpose and limits

Do not use models for hiring, dismissal, pay setting or employee surveillance. Aggregate support planning requires contemporary representative data, employee consultation and independent governance.

## Data and provenance

Fictional IBM HR data; ODbL 1.0 / DbCL 1.0. Derived educational analysis, not employee decisions.

Adult: 1994 Census-derived sample with restricted inclusion, missing values and historical sex/race categories; unsuitable for contemporary workforce representativeness. IBM data has no real employee provenance or observation period.

## Development protocol

HR uses seeded stratified 60/20/20 training, validation and test partitions. Adult preserves the official 16,281-record test file and divides its 32,561-record training file 80/20 for development. Imputation, scaling and categorical encoding are fitted on training rows only. HR identifiers/constants and Gender are excluded; Adult sex, race and sampling weight are excluded from predictors. Demographics remain available for auditing.

Logistic regression and random forest are compared by validation average precision. The selected model's threshold maximizes validation F1; neither model nor threshold is changed using test outcomes. Final test precision, recall, F1, ROC AUC, average precision, confusion counts and calibration are reported. This cross-sectional exercise does not estimate future exits or business intervention effects.

Excluding sensitive fields does not eliminate proxies or historical bias. Adult has 23 official-test rows identical to original training records (26 predictor-only matches); its official split is preserved and unknown identities may make evaluation optimistic. Group support and Wilson intervals expose sampling uncertainty, but cannot resolve measurement bias. No individual employment decision is supported.

## Model choices

HR: validation average precision (logistic: 0.639, random_forest: 0.620); selected logistic; validation-selected threshold 0.32; validation F1 0.632.

ADULT: validation average precision (logistic: 0.772, random_forest: 0.806); selected random_forest; validation-selected threshold 0.34; validation F1 0.712.

Gradient boosting omitted: two different regularized baselines provide an interpretable comparison. Adding complexity without operational evidence is not justified.

## Measured test results

Held-out HR logistic at validation-selected threshold 0.32: precision 0.500, recall 0.511, F1 0.505, AP 0.572, ROC AUC 0.785; test n=294.

Independent Adult income benchmark test n=16,281; random_forest, threshold 0.34, F1 0.693, AP 0.790. Positive means >$50K historical income.

## Risks and monitoring

Investigate workload and role design through voluntary employee feedback; fictional correlations cannot justify policy changes on their own. In a real organization, validate definitions, establish dated hire/exit histories, and measure retention during an approved pilot.

Do not use models for hiring, dismissal, pay setting or employee surveillance. Aggregate support planning requires contemporary representative data, employee consultation and independent governance.

Track cohort retention, overtime prevalence, exit reasons, prediction calibration, subgroup error rates and missingness after collecting real data. Set thresholds with stakeholder-defined costs, not this educational F1 optimum.

## Sources and data license

IBM HR source and license: https://github.com/IBM/employee-attrition-aif360#dataset-license-details

Original fictional HR listing: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Adult: Becker & Kohavi (1996), DOI 10.24432/C5XW20: https://archive.ics.uci.edu/dataset/2/adult

BLS Employee Tenure in 2026, released September 24 2026: https://www.bls.gov/news.release/tenure.nr0.htm

Data terms: ODbL 1.0 and DbCL 1.0; full license links and alteration methods are in NOTICE.md. Adult: CC BY 4.0. BLS statistics: public domain.
