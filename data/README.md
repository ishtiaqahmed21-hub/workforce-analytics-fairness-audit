# Data contract and selection

|Dataset|Publisher/source|Status|Rows / columns|License|Use|Limits|
|---|---|---|---|---|---|---|
|IBM HR Analytics Employee Attrition & Performance|[IBM-hosted CSV](https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv), [original Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)|Fictional / SIMULATED|1,470 / 35|ODbL + DbCL|HR snapshot, compensation, attrition classification|No dates, actual workforce or documented currency|
|Adult|[UCI official](https://archive.ics.uci.edu/dataset/2/adult)|Historical Census-derived / OBSERVED|48,842 / 15 including target|CC BY 4.0|Independent historical income fairness audit|1994 population; income class is neither salary nor attrition|
|Employee Tenure, January 2026|[BLS official](https://www.bls.gov/news.release/tenure.toc.htm)|Published survey estimates / OBSERVED|Six aggregate tables; project extracts 3 benchmark rows|Public domain|Context only|Different population/time; not matched company benchmark|
|HR Analytics requested Kaggle mirror|[rishikeshkonapure](https://www.kaggle.com/datasets/rishikeshkonapure/hr-analytics-prediction/data)|Not selected|Unverified|Unverified|None|Page title accessible, schema/license unavailable to research interface; do not assume it matches IBM|

Run `python download_data.py`. Downloads require network access; no Kaggle credentials are needed. Hash validation prevents silent source changes. Downloaded raw data is excluded from Git; attribution and reproducible alteration code accompany outputs. See ../NOTICE.md before redistributing derivatives. Sources researched 2026-09-28.

## Schemas and interpretation

IBM CSV fields: Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager.

Age and named tenure fields use years. MonthlyIncome currency is unspecified; rate fields must not be converted into salaries. OverTime is Yes/No, not overtime hours. Attrition is Yes/No, without a dated exposure horizon. Education is ordinal (1 below college, 2 college, 3 bachelor, 4 master, 5 doctor); satisfaction fields are ordinal, not continuous physical measurements. EmployeeNumber is an identifier; EmployeeCount, Over18 and StandardHours are constants excluded from models. Gender is retained for audit and excluded as a direct predictor. Job role, department, education, tenure, overtime, performance and demographics are available for grouped summaries, not causal effects.

Adult fields: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income. Loader converts hyphens to underscores via assigned column names. '?' denotes missing values. Test labels have a terminal dot; first line is a comment. Keep official training/test files distinct. fnlwgt is excluded as a predictor and used for a weighted AP sensitivity calculation. Historical binary sex encoding and coarse race categories limit representation.

See source_manifest.json for hashes. Source variables are OBSERVED within their stated source status; IBM is always fictional. Calculated metrics are DERIVED. Split seed, F1 threshold objective and minimum support flag are ASSUMPTIONS. No simulated corporate savings are presented.
