# Fairness audit

Adult Female-minus-Male diagnostics: selection rate difference: -0.256; tpr difference: -0.105; fpr difference: -0.148; fnr difference: 0.105; selection rate ratio: 0.311

HR Female-minus-Male diagnostics: selection rate difference: 0.029; tpr difference: 0.173; fpr difference: 0.022; fnr difference: -0.173; selection rate ratio: 1.193

Selection rate is predicted-positive share. HR positive is adverse attrition; Adult positive is income classification. Reference labels describe comparisons, not legitimate or privileged groups. TPR equality is equal opportunity; FPR/FNR depend on observed labels, which may encode historical measurement bias. Calibration is agreement with those labels, not social fairness.

Wilson intervals accompany subgroup rates. Group sizes and actual positives are in results.json. Zero denominators produce null; groups below 50 records are flagged. Race categories in Adult are coarse historical encodings; intersectional analyses need additional support. No legal 80% rule conclusion is made.

Metrics change with decision thresholds and prevalence. A single parity measure is not sufficient; calibration and error-rate parity may conflict. Removing sensitive columns leaves proxies (including relationships, occupations and job levels). No discrimination or causal conclusion follows.

hr Female: n=116, actual positives=16, predicted positives=21; TP=10, FP=11, FN=6, TN=89; TPR=0.625, FPR=0.110, Brier=0.0912.

hr Male: n=178, actual positives=31, predicted positives=27; TP=14, FP=13, FN=17, TN=134; TPR=0.452, FPR=0.088, Brier=0.1097.

adult Female: n=5421, actual positives=590, predicted positives=627; TP=400, FP=227, FN=190, TN=4604; TPR=0.678, FPR=0.047, Brier=0.0538.

adult Male: n=10860, actual positives=3256, predicted positives=4034; TP=2549, FP=1485, FN=707, TN=6119; TPR=0.783, FPR=0.195, Brier=0.1210.

Fictional IBM HR data; ODbL 1.0 / DbCL 1.0. Derived educational analysis, not employee decisions.

IBM HR source and license: https://github.com/IBM/employee-attrition-aif360#dataset-license-details
Original fictional HR listing: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Adult: Becker & Kohavi (1996), DOI 10.24432/C5XW20: https://archive.ics.uci.edu/dataset/2/adult
BLS Employee Tenure in 2026, released September 24 2026: https://www.bls.gov/news.release/tenure.nr0.htm
Data terms: ODbL 1.0 and DbCL 1.0; full license links and alteration methods are in NOTICE.md. Adult: CC BY 4.0. BLS statistics: public domain.
