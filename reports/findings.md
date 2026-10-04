# Findings and management actions

DERIVED FROM SIMULATED DATA: 1,470 HR records include 237 Attrition Yes labels (16.1%); these are not annual turnover rates.

Highest observed fictional role share: Sales Representative, 33/83 (39.8%). Small group support and role composition matter.

Held-out HR logistic at validation-selected threshold 0.32: precision 0.500, recall 0.511, F1 0.505, AP 0.572, ROC AUC 0.785; test n=294.

Independent Adult income benchmark test n=16,281; random_forest, threshold 0.34, F1 0.693, AP 0.790. Positive means >$50K historical income.

Investigate workload and role design through voluntary employee feedback; fictional correlations cannot justify policy changes on their own. In a real organization, validate definitions, establish dated hire/exit histories, and measure retention during an approved pilot.

Do not use models for hiring, dismissal, pay setting or employee surveillance. Aggregate support planning requires contemporary representative data, employee consultation and independent governance.

Track cohort retention, overtime prevalence, exit reasons, prediction calibration, subgroup error rates and missingness after collecting real data. Set thresholds with stakeholder-defined costs, not this educational F1 optimum.
