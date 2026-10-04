# Power BI implementation specification

The supplied HTML is an executable dashboard prototype. It is not a PBIX/TWB file. PNG preview is explicitly a prototype preview, not evidence of Power BI implementation.

## Semantic model

Keep three separate subject areas: HRRecord (fictional), AdultAuditAggregate (historical income), BLSContext (published survey). No relationships between employee or person rows. HR Department, JobRole, Education and Gender dimensions filter HR only; race/sex filter Adult only. Sources: data/processed CSVs and reports/results.json; no employee-level predictions are published.

## Pages

1. Workforce overview: sample records, retained records, attrition label share, role/department distribution, tenure, overtime and education. Always display SIMULATED source badge.
2. Compensation: mean/median, role/department/recorded-gender distributions, sample n, units unspecified, exploratory adjusted association. No red/green discrimination flags.
3. Attrition diagnostics: grouped Yes shares with denominator and intervals; validation/test model performance and confusion counts. Filter sample groups, never infer annual rates.
4. Fairness: separate HR/Adult target selector, subgroup n and positives, selection rate, ratios, TPR/FPR/FNR, Wilson intervals, calibration bins. Blank undefined metrics; warning icon for n<50. Never call historical income classification an actual hiring decision.
5. External context: BLS January 2026 values and source/population footnote, visually separate from fictional sample. No league table or performance gap.

|KPI|Definition/formula|Source|Owner|Refresh|Threshold|
|---|---|---|---|---|---|
|Sample records|COUNTROWS(HR)|IBM CSV|Analytics|Per version|None; not active census|
|Retained records|COUNTIF Attrition=No|IBM CSV|Analytics|Per version|None|
|Attrition label share|Yes records / all selected records|IBM CSV|HR analyst|Per version|No operational RAG|
|Median tenure|MEDIAN YearsAtCompany|IBM CSV|HR analyst|Per version|No cross-population target|
|Mean/median compensation|AVERAGE/MEDIAN MonthlyIncome|IBM CSV|Compensation analyst|Per version|No currency assumption|
|Overtime prevalence|OverTime=Yes / n|IBM CSV|Operations analyst|Per version|No invented hours|
|Selection rate|Predicted positives / n|Held-out audit|Model reviewer|Each run|n<50 support flag|
|Selection ratio|Female selection / Male selection|Held-out audit|Model reviewer|Each run|Undefined when reference zero; no legal cutoff|
|Parity difference|Female rate - Male rate|Held-out audit|Model reviewer|Each run|Diagnostic; no universal acceptable value|
|Equal opportunity difference|Female TPR - Male TPR|Held-out audit|Model reviewer|Each run|Undefined absent actual positives|
|FPR difference|Female FP/(FP+TN) - Male equivalent|Held-out audit|Model reviewer|Each run|Undefined absent negatives|
|FNR difference|Female FN/(FN+TP) - Male equivalent|Held-out audit|Model reviewer|Each run|Undefined absent positives|
|Calibration|Mean p versus positive share in five fixed bins|Calibration CSV|Model reviewer|Each run|Inspect support, no automatic pass/fail|
|BLS tenure|Published median years|BLSContext|Research analyst|Biennial/source update|Context only|

Use bookmarks to explain source status, tooltips with exact denominator, accessible labels and color-independent warnings. Page subtitle records run configuration, source access date and fictional status. A production refresh needs source ownership, privacy review, cohort definitions and approved access; these are outside this educational prototype.
