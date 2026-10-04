"""Reproduce the independent workforce, income audit and benchmark tracks."""

from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from src.data_cleaning import load_hr, load_adult, quality
from src.workforce_metrics import workforce_kpis, group_summary
from src.modeling import select_model
from src.evaluation import evaluate
from src.fairness import audit, disparities, calibration
from src.statistics import compensation_tests
from src.artifact_utils import write_report, write_notebook

ROOT = Path(__file__).resolve().parent
NOTICE = "Fictional IBM HR data; ODbL 1.0 / DbCL 1.0. Derived educational analysis, not employee decisions."
SOURCES = [
    (
        "IBM HR source and license",
        "https://github.com/IBM/employee-attrition-aif360#dataset-license-details",
    ),
    (
        "Original fictional HR listing",
        "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
    ),
    (
        "Adult: Becker & Kohavi (1996), DOI 10.24432/C5XW20",
        "https://archive.ics.uci.edu/dataset/2/adult",
    ),
    (
        "BLS Employee Tenure in 2026, released September 24 2026",
        "https://www.bls.gov/news.release/tenure.nr0.htm",
    ),
    ("ODbL 1.0", "https://opendatacommons.org/licenses/odbl/1-0/"),
    ("DbCL 1.0", "https://opendatacommons.org/licenses/dbcl/1-0/"),
]


def save_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
        ),
        encoding="utf-8",
    )


def figure_finish(name, source=NOTICE):
    plt.gcf().text(0.03, 0.015, source, fontsize=7, color="#526879")
    plt.tight_layout(rect=[0, 0.055, 1, 1])
    plt.savefig(ROOT / "figures" / name, dpi=150, bbox_inches="tight")
    plt.close()


def run():
    config = yaml.safe_load((ROOT / "config/project_config.yaml").read_text())
    seed = config["seed"]
    for folder in ["data/processed", "reports", "figures", "dashboard", "notebooks"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    hashes = {
        name: hashlib.sha256((ROOT / "data/raw" / name).read_bytes()).hexdigest()
        for name in ["emp_attrition.csv", "adult.zip"]
    }
    expected = {
        "emp_attrition.csv": "a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659",
        "adult.zip": "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
    }
    if hashes != expected:
        raise ValueError(
            "Raw hashes differ; investigate source version before analysis"
        )
    hr = load_hr(ROOT / "data/raw/emp_attrition.csv")
    adult_train, adult_test = load_adult(ROOT / "data/raw/adult.zip")
    overlap_full = (
        pd.MultiIndex.from_frame(adult_test)
        .isin(pd.MultiIndex.from_frame(adult_train))
        .sum()
    )
    overlap_features = (
        pd.MultiIndex.from_frame(adult_test.drop(columns="income"))
        .isin(pd.MultiIndex.from_frame(adult_train.drop(columns="income")))
        .sum()
    )
    dq = {
        "adult_official_test_rows_matching_train_full": int(overlap_full),
        "adult_official_test_rows_matching_train_predictors": int(overlap_features),
        "hr": quality(hr),
        "adult_train": quality(adult_train),
        "adult_test": quality(adult_test),
        "policy": "Duplicates flagged but retained: Adult lacks unique person IDs. IQR flags are diagnostic, never automatic deletions.",
    }
    kpis = workforce_kpis(hr)
    summaries = {
        g: group_summary(hr, g)
        for g in [
            "Department",
            "JobRole",
            "Gender",
            "OverTime",
            "Education",
            "PerformanceRating",
        ]
    }
    for group, frame in summaries.items():
        frame.to_csv(ROOT / f"data/processed/hr_by_{group}.csv", index=False)
    compensation = compensation_tests(hr)
    trainval_idx, test_idx = train_test_split(
        np.arange(len(hr)),
        test_size=config["hr_test_fraction"],
        random_state=seed,
        stratify=hr.Attrition,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=config["hr_validation_fraction_of_development"],
        random_state=seed,
        stratify=hr.Attrition.iloc[trainval_idx],
    )
    removed = [
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "Gender",
    ]
    x = hr.drop(columns=removed)
    y = hr.Attrition.eq("Yes").astype(int)
    hr_model, hr_selection = select_model(
        x.iloc[train_idx], y.iloc[train_idx], x.iloc[val_idx], y.iloc[val_idx], seed
    )
    hr_prob = hr_model.predict_proba(x.iloc[test_idx])[:, 1]
    hr_test = evaluate(y.iloc[test_idx], hr_prob, hr_selection["threshold"])
    hr_fair = audit(
        y.iloc[test_idx], hr_prob, hr.Gender.iloc[test_idx], hr_selection["threshold"]
    )
    hr_cal = calibration(y.iloc[test_idx], hr_prob, hr.Gender.iloc[test_idx])
    hr_cal.to_csv(ROOT / "data/processed/hr_calibration.csv", index=False)
    importance = permutation_importance(
        hr_model,
        x.iloc[test_idx],
        y.iloc[test_idx],
        n_repeats=5,
        random_state=seed,
        scoring="average_precision",
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": x.columns,
            "mean_ap_decrease": importance.importances_mean,
            "std_ap_decrease": importance.importances_std,
        }
    ).sort_values("mean_ap_decrease", ascending=False)
    importance_frame.to_csv(
        ROOT / "data/processed/hr_permutation_importance.csv", index=False
    )
    adult_x = adult_train.drop(columns=["income", "sex", "race", "fnlwgt"])
    adult_y = adult_train.income.eq(">50K").astype(int)
    a_train, a_val = train_test_split(
        np.arange(len(adult_train)),
        test_size=config["adult_validation_fraction"],
        random_state=seed,
        stratify=adult_y,
    )
    adult_model, adult_selection = select_model(
        adult_x.iloc[a_train],
        adult_y.iloc[a_train],
        adult_x.iloc[a_val],
        adult_y.iloc[a_val],
        seed,
    )
    adult_prob = adult_model.predict_proba(
        adult_test.drop(columns=["income", "sex", "race", "fnlwgt"])
    )[:, 1]
    adult_test_y = adult_test.income.eq(">50K").astype(int)
    adult_metrics = evaluate(adult_test_y, adult_prob, adult_selection["threshold"])
    adult_fair = audit(
        adult_test_y, adult_prob, adult_test.sex, adult_selection["threshold"]
    )
    adult_race = audit(
        adult_test_y, adult_prob, adult_test.race, adult_selection["threshold"]
    )
    adult_cal = calibration(adult_test_y, adult_prob, adult_test.sex)
    adult_cal.to_csv(ROOT / "data/processed/adult_calibration.csv", index=False)
    # Weight-aware sensitivity uses fnlwgt only after fitting, not as a predictor.
    from sklearn.metrics import average_precision_score

    weighted_ap = float(
        average_precision_score(
            adult_test_y, adult_prob, sample_weight=adult_test.fnlwgt
        )
    )
    benchmark = pd.DataFrame(
        {
            "period": ["January 2026"] * 3,
            "group": ["All wage and salary workers", "Men", "Women"],
            "median_years": [4.1, 4.3, 4.0],
            "source_url": ["https://www.bls.gov/news.release/tenure.nr0.htm"] * 3,
            "accessed_date": ["2026-09-28"] * 3,
        }
    )
    benchmark.to_csv(ROOT / "data/processed/bls_context.csv", index=False)
    split_assignment = pd.DataFrame({"row_number": np.arange(len(hr)), "split": ""})
    for name, idx in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        split_assignment.loc[idx, "split"] = name
    split_assignment.to_csv(
        ROOT / "data/processed/hr_split_assignment.csv", index=False
    )
    results = {
        "data_status": {
            "hr": "SIMULATED fictional IBM records",
            "adult": "OBSERVED historical Census-derived records",
            "bls": "OBSERVED published survey estimates; contextual only",
        },
        "source_hashes": hashes,
        "hr_kpis": kpis,
        "quality": dq,
        "compensation": compensation,
        "hr": {
            "split_sizes": {
                "train": len(train_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
            },
            "selection": hr_selection,
            "test": hr_test,
            "gender_audit": hr_fair,
            "disparities_female_minus_male": disparities(hr_fair),
            "top_permutation_features": importance_frame.head(8).to_dict("records"),
        },
        "adult": {
            "split_sizes": {
                "train": len(a_train),
                "validation": len(a_val),
                "official_test": len(adult_test),
            },
            "selection": adult_selection,
            "test": adult_metrics,
            "gender_audit": adult_fair,
            "race_audit": adult_race,
            "disparities_female_minus_male": disparities(adult_fair),
            "fnlwgt_weighted_test_average_precision": weighted_ap,
        },
        "bls_context": benchmark.to_dict("records"),
        "sources": SOURCES,
    }
    save_json(ROOT / "reports/results.json", results)
    save_json(
        ROOT / "data/source_manifest.json",
        {"accessed": "2026-09-28", "hashes": hashes, "sources": SOURCES},
    )
    # Charts are analytic outputs from the saved tables.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#12344a",
            "axes.titleweight": "bold",
        }
    )
    roles = summaries["JobRole"].sort_values("attrition_share")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(roles.JobRole, roles.attrition_share * 100, color="#008f92")
    ax.set(
        xlabel="Attrition Yes share (%) - no annual exposure period",
        title="Where fictional attrition flags concentrate",
    )
    figure_finish("attrition_by_role.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    hr.boxplot(column="MonthlyIncome", by="Gender", ax=ax, grid=False)
    fig.suptitle("")
    ax.set(
        title="Compensation distribution in fictional records",
        ylabel="MonthlyIncome (source units; currency unspecified)",
    )
    figure_finish("compensation.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    imp = importance_frame.head(8).iloc[::-1]
    ax.barh(
        imp.feature, imp.mean_ap_decrease, xerr=imp.std_ap_decrease, color="#008f92"
    )
    ax.set(
        title="Held-out permutation importance",
        xlabel="Mean decrease in average precision; bars show repeat SD",
    )
    figure_finish("permutation_importance.png")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, rows, title in zip(
        axs,
        [hr_fair, adult_fair],
        ["Fictional HR attrition flag", "Adult historical income >$50K"],
    ):
        for j, row in enumerate(rows):
            lo, hi = row["selection_rate_ci95"]
            value = row["selection_rate"]
            ax.errorbar(
                j,
                value,
                yerr=[[value - lo], [hi - value]],
                fmt="o",
                capsize=6,
                color="#008f92",
            )
        ax.set_xticks(range(len(rows)), [f"{r['group']}\nn={r['n']}" for r in rows])
        ax.set(
            title=title, ylabel="Predicted-positive share (Wilson 95% CI)", ylim=(0, 1)
        )
    figure_finish(
        "fairness_selection.png",
        "Separate populations and targets; no real employment selection. Sources: IBM HR (ODbL/DbCL), UCI Adult (CC BY 4.0).",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for group, frame in adult_cal.groupby("group"):
        ax.plot(
            frame.mean_probability, frame.observed_positive_share, "o-", label=group
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive share",
        title="Adult income classifier calibration by recorded sex",
    )
    ax.legend()
    figure_finish(
        "adult_calibration.png",
        "UCI Adult, Becker & Kohavi (1996), CC BY 4.0. Bins may have unequal support; see calibration CSV.",
    )
    build_documents(results, summaries, importance_frame)
    build_notebooks()
    print(
        json.dumps(
            {
                "hr_test": hr_test,
                "adult_test": adult_metrics,
                "hr_top_role": roles.iloc[-1].to_dict(),
            },
            indent=2,
        )
    )
    return results


def build_documents(r, summaries, importance):
    h = r["hr"]
    a = r["adult"]
    k = r["hr_kpis"]
    c = r["compensation"]
    role = summaries["JobRole"].sort_values("attrition_share", ascending=False).iloc[0]
    findings = [
        f"DERIVED FROM SIMULATED DATA: {k['sample_records']:,} HR records include {k['attrition_yes']} Attrition Yes labels ({k['attrition_share']:.1%}); these are not annual turnover rates.",
        f"Highest observed fictional role share: {role.JobRole}, {int(role.attrition_yes)}/{int(role.n)} ({role.attrition_share:.1%}). Small group support and role composition matter.",
        f"Held-out HR {h['selection']['selected_model']} at validation-selected threshold {h['selection']['threshold']:.2f}: precision {h['test']['precision']:.3f}, recall {h['test']['recall']:.3f}, F1 {h['test']['f1']:.3f}, AP {h['test']['average_precision']:.3f}, ROC AUC {h['test']['roc_auc']:.3f}; test n={h['test']['n']}.",
        f"Independent Adult income benchmark test n={a['test']['n']:,}; {a['selection']['selected_model']}, threshold {a['selection']['threshold']:.2f}, F1 {a['test']['f1']:.3f}, AP {a['test']['average_precision']:.3f}. Positive means >$50K historical income.",
    ]
    recommendations = [
        "Investigate workload and role design through voluntary employee feedback; fictional correlations cannot justify policy changes on their own. In a real organization, validate definitions, establish dated hire/exit histories, and measure retention during an approved pilot.",
        "Do not use models for hiring, dismissal, pay setting or employee surveillance. Aggregate support planning requires contemporary representative data, employee consultation and independent governance.",
        "Track cohort retention, overtime prevalence, exit reasons, prediction calibration, subgroup error rates and missingness after collecting real data. Set thresholds with stakeholder-defined costs, not this educational F1 optimum.",
    ]
    source_text = [f"{title}: {url}" for title, url in SOURCES[:4]] + [
        "Data terms: ODbL 1.0 and DbCL 1.0; full license links and alteration methods are in NOTICE.md. Adult: CC BY 4.0. BLS statistics: public domain."
    ]
    clean_protocol = (
        (ROOT / "METHODOLOGY.md").read_text().replace("# Study protocol", "").strip()
    )

    def metric_sentence(values):
        return "; ".join(
            (
                f"{key.replace('_', ' ')}: {value:.3f}"
                if isinstance(value, (int, float))
                else f"{key.replace('_',' ')}: undefined"
            )
            for key, value in values.items()
        )

    def selection_sentence(scope):
        selection = r[scope]["selection"]
        candidates = ", ".join(
            f"{name}: {value:.3f}"
            for name, value in selection["validation_average_precision"].items()
        )
        return f"{scope.upper()}: validation average precision ({candidates}); selected {selection['selected_model']}; validation-selected threshold {selection['threshold']:.2f}; validation F1 {selection['validation_f1']:.3f}."

    fairness_text = [
        f"Adult Female-minus-Male diagnostics: {metric_sentence(a['disparities_female_minus_male'])}",
        f"HR Female-minus-Male diagnostics: {metric_sentence(h['disparities_female_minus_male'])}",
        "Selection rate is predicted-positive share. HR positive is adverse attrition; Adult positive is income classification. Reference labels describe comparisons, not legitimate or privileged groups. TPR equality is equal opportunity; FPR/FNR depend on observed labels, which may encode historical measurement bias. Calibration is agreement with those labels, not social fairness.",
        "Wilson intervals accompany subgroup rates. Group sizes and actual positives are in results.json. Zero denominators produce null; groups below 50 records are flagged. Race categories in Adult are coarse historical encodings; intersectional analyses need additional support. No legal 80% rule conclusion is made.",
        "Metrics change with decision thresholds and prevalence. A single parity measure is not sufficient; calibration and error-rate parity may conflict. Removing sensitive columns leaves proxies (including relationships, occupations and job levels). No discrimination or causal conclusion follows.",
    ]
    for scope in ["hr", "adult"]:
        for row in r[scope]["gender_audit"]:
            fairness_text.append(
                f"{scope} {row['group']}: n={row['n']}, actual positives={row['actual_positive_n']}, predicted positives={row['predicted_positive_n']}; TP={row['tp']}, FP={row['fp']}, FN={row['fn']}, TN={row['tn']}; TPR={row['tpr']:.3f}, FPR={row['fpr']:.3f}, Brier={row['brier']:.4f}."
            )
    model_sections = [
        ("Purpose and limits", recommendations[1]),
        (
            "Data and provenance",
            [
                NOTICE,
                "Adult: 1994 Census-derived sample with restricted inclusion, missing values and historical sex/race categories; unsuitable for contemporary workforce representativeness. IBM data has no real employee provenance or observation period.",
            ],
        ),
        (
            "Development protocol",
            [
                "HR uses seeded stratified 60/20/20 training, validation and test partitions. Adult preserves the official 16,281-record test file and divides its 32,561-record training file 80/20 for development. Imputation, scaling and categorical encoding are fitted on training rows only. HR identifiers/constants and Gender are excluded; Adult sex, race and sampling weight are excluded from predictors. Demographics remain available for auditing.",
                "Logistic regression and random forest are compared by validation average precision. The selected model's threshold maximizes validation F1; neither model nor threshold is changed using test outcomes. Final test precision, recall, F1, ROC AUC, average precision, confusion counts and calibration are reported. This cross-sectional exercise does not estimate future exits or business intervention effects.",
                "Excluding sensitive fields does not eliminate proxies or historical bias. Adult has 23 official-test rows identical to original training records (26 predictor-only matches); its official split is preserved and unknown identities may make evaluation optimistic. Group support and Wilson intervals expose sampling uncertainty, but cannot resolve measurement bias. No individual employment decision is supported.",
            ],
        ),
        (
            "Model choices",
            [
                selection_sentence("hr"),
                selection_sentence("adult"),
                "Gradient boosting omitted: two different regularized baselines provide an interpretable comparison. Adding complexity without operational evidence is not justified.",
            ],
        ),
        ("Measured test results", findings[2:]),
        ("Risks and monitoring", recommendations),
        ("Sources and data license", source_text),
    ]
    executive = [
        ("Decision brief", findings),
        ("Recommended next steps", recommendations),
        (
            "Boundaries",
            [
                "All HR findings derive from fictional records. BLS median tenure 4.1 years in January 2026 is context, not a performance target or like-for-like comparison. No savings, discrimination or future attrition prediction is established."
            ],
        ),
        ("Sources and data license", [NOTICE] + source_text),
    ]
    technical = [
        ("Protocol", clean_protocol),
        (
            "Data validation",
            [
                f"Hashes verified before every run: {r['source_hashes']}",
                f"HR dimensions {k['sample_records']} x 35; missing cells {sum(r['quality']['hr']['missing'].values())}. Exact duplicates {r['quality']['hr']['duplicate_rows']}. IQR flags are diagnostic only.",
                f"Adult original training/test sizes 32561/16281; missing training counts {r['quality']['adult_train']['missing']}. Original test period not claimed distinct in time. Test rows exactly matching original training records: {r['quality']['adult_official_test_rows_matching_train_full']}; predictor-only matches: {r['quality']['adult_official_test_rows_matching_train_predictors']}. Official split preserved; unknown identity and repeated records may make evaluation optimistic.",
            ],
        ),
        ("Results and interpretation", findings),
        (
            "Compensation associations",
            [
                f"Female-minus-Male mean MonthlyIncome difference {c['mean_difference_female_minus_male']:.2f} source units; Cohen d={c['cohens_d']:.3f}, Welch p={c['welch_p']:.4g}; Mann-Whitney p={c['mann_whitney_p']:.4g}, rank-biserial={c['rank_biserial_female_minus_male']:.3f}.",
                f"Adjusted Male-vs-Female log-income coefficient {c['adjusted_log_income_male_vs_female_coefficient']:.4f}; HC3 95% CI {c['adjusted_ci95'][0]:.4f} to {c['adjusted_ci95'][1]:.4f}; p={c['adjusted_p']:.4g}. {c['adjustment']}",
                "Tests are exploratory and not adjusted for multiple comparisons; synthetic records do not establish real pay discrimination. Role adjustment can itself condition on historically biased variables.",
            ],
        ),
        (
            "Explainability",
            [
                f"Largest test permutation AP decreases: {'; '.join(f'{row.feature}: {row.mean_ap_decrease:.3f} (SD {row.std_ap_decrease:.3f})' for row in importance.head(5).itertuples())}",
                "Importance is predictive association, not causal leverage; correlated variables can share or obscure importance. Negative values reflect sampling variability. Test importance is explanatory only and is not fed back to model selection.",
            ],
        ),
        (
            "BLS context",
            [
                "BLS January 2026 medians: overall 4.1, men 4.3, women 4.0 years. Survey wage/salary population differs from fictional undated sample. See separate context CSV."
            ],
        ),
        ("Sources and license", [NOTICE] + source_text),
    ]
    write_report(
        ROOT / "reports/executive_summary.pdf",
        "Workforce Analytics | Executive Brief",
        executive,
        [ROOT / "figures/attrition_by_role.png"],
    )
    write_report(
        ROOT / "reports/technical_report.pdf",
        "Workforce Analytics | Technical Report",
        technical,
        [
            ROOT / "figures/compensation.png",
            ROOT / "figures/permutation_importance.png",
        ],
    )
    write_report(
        ROOT / "reports/fairness_audit.pdf",
        "Algorithmic Fairness | Independent Target Audits",
        [
            ("Interpretation", fairness_text),
            ("Sources and license", [NOTICE] + source_text),
        ],
        [
            ROOT / "figures/fairness_selection.png",
            ROOT / "figures/adult_calibration.png",
        ],
    )
    write_report(
        ROOT / "reports/model_card.pdf",
        "Model Card | Educational Classifiers",
        model_sections,
    )
    (ROOT / "MODEL_CARD.md").write_text(
        "# Model card\n\n"
        + "\n\n".join(
            "## "
            + str(title)
            + "\n\n"
            + ("\n\n".join(text) if isinstance(text, list) else text)
            for title, text in model_sections
        ),
        encoding="utf-8",
    )
    (ROOT / "FAIRNESS_AUDIT.md").write_text(
        "# Fairness audit\n\n"
        + "\n\n".join(fairness_text)
        + "\n\n"
        + NOTICE
        + "\n\n"
        + "\n".join(source_text),
        encoding="utf-8",
    )
    (ROOT / "reports/findings.md").write_text(
        "# Findings and management actions\n\n"
        + "\n\n".join(findings + recommendations),
        encoding="utf-8",
    )
    (ROOT / "reports/traceability.csv").write_text(
        "finding,source,transformation,evidence,recommendation\nF01,IBM HR Attrition and JobRole,group_summary,data/processed/hr_by_JobRole.csv,Investigate workload through a real pilot\nF02,IBM HR income and Gender,compensation_tests,reports/results.json,Validate compensation definitions before further investigation\nF03,Adult official test income and sex,audit,reports/results.json,Monitor multiple error and calibration metrics\n",
        encoding="utf-8",
    )
    (ROOT / "reports/resume_bullets.md").write_text(
        "# Workforce Analytics & Algorithmic Fairness Audit\n\nPython | pandas | scikit-learn | SciPy | statsmodels | Power BI specification\n\n"
        + f'- Built a reproducible workforce analytics pipeline across {k["sample_records"]:,} fictional HR records and an independent {a["test"]["n"]:,}-record Adult test set, with schema validation, source hashes and documented data licenses.\n- Compared logistic regression and random forest with validation-only model/threshold selection, achieving held-out HR F1 {h["test"]["f1"]:.3f} and recall {h["test"]["recall"]:.3f} on synthetic attrition labels.\n- Audited subgroup selection, error rates and calibration with support counts and Wilson intervals, separating historical income classification from fictional workforce decisions.\n',
        encoding="utf-8",
    )
    build_dashboard(r, summaries)


def build_dashboard(r, summaries):
    k = r["hr_kpis"]
    table = summaries["JobRole"].sort_values("attrition_share", ascending=False).copy()
    table["attrition_share"] = table.attrition_share.map(lambda v: f"{v:.1%}")
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{row[col]}</td>"
            for col in ["JobRole", "n", "attrition_yes", "attrition_share"]
        )
        + "</tr>"
        for _, row in table.iterrows()
    )
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Workforce Evidence Studio</title><style>body{font:16px system-ui;background:#eef4f5;color:#163849;margin:0}header{background:#12344a;color:white;padding:32px 5%}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:36px;margin:8px 0}.eyebrow{color:#62d6ca;letter-spacing:3px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.card,section{background:white;padding:22px;border-radius:12px;margin-bottom:22px}.value{font-size:32px;font-weight:700;color:#007f82}button,select{padding:10px;border:1px solid #aac2ca;border-radius:6px;background:white;color:#12344a}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #e3ecef}img{max-width:100%}.note{font-size:14px;color:#526879}footer{padding:24px;background:#12344a;color:white}a{color:inherit}@media(max-width:700px){.cards{grid-template-columns:1fr 1fr}h1{font-size:28px}}</style><header><div class="eyebrow">INDUSTRIAL ENGINEERING + DATA ANALYTICS</div><h1>Workforce Evidence Studio</h1><p>Fictional HR case study / Independent historical income fairness audit</p></header><main><div class="cards">"""
    for label, value in [
        ("Sample records", f'{k["sample_records"]:,}'),
        ("Attrition label share", f'{k["attrition_share"]:.1%}'),
        ("Median tenure", f'{k["median_tenure_years"]:g} years'),
        ("Median income units", f'{k["median_monthly_income_units"]:,.0f}'),
    ]:
        html += f'<div class="card"><div class="note">{label}</div><div class="value">{value}</div></div>'
    html += (
        '</div><section><h2>Role concentration</h2><p>SIMULATED source records; DERIVED shares. No annual attrition period is available.</p><label>Show role <select id="role"><option value="">All roles</option>'
        + "".join(f"<option>{role}</option>" for role in table.JobRole)
        + '</select></label><table><thead><tr><th>Job role</th><th>Records</th><th>Attrition Yes</th><th>Share</th></tr></thead><tbody id="rows">'
        + rows
        + "</tbody></table></section>"
    )
    html += (
        '<section><h2>Compensation and workload investigation</h2><p>Income currency is unspecified. Observed disparity is an association requiring further investigation; it does not establish discrimination.</p><img alt="Income distribution by recorded gender" src="../figures/compensation.png"></section><section><h2>Independent target audits</h2><p>Adult positive means historical income above $50K. HR positive means attrition label. Neither is an actual hiring selection.</p><img alt="Separate subgroup predicted-positive shares with confidence intervals" src="../figures/fairness_selection.png"><p><a href="../FAIRNESS_AUDIT.md">Audit definitions and support counts</a></p></section><section><h2>External context / BLS January 2026</h2><p>Median employer tenure: 4.1 years overall; 4.3 men; 4.0 women. Different population and period; no company performance comparison.</p><a href="https://www.bls.gov/news.release/tenure.nr0.htm">BLS source</a></section></main><footer>'
        + NOTICE
        + '<br>Contains information from <a href="https://github.com/IBM/employee-attrition-aif360">IBM HR</a>, available under <a href="https://opendatacommons.org/licenses/odbl/1-0/">ODbL</a> / <a href="https://opendatacommons.org/licenses/dbcl/1-0/">DbCL</a>. Adult: Becker &amp; Kohavi (1996), CC BY 4.0.</footer><script>document.querySelector("#role").addEventListener("change",e=>{document.querySelectorAll("#rows tr").forEach(r=>r.hidden=!!e.target.value&&r.cells[0].textContent!==e.target.value)})</script></html>'
    )
    (ROOT / "dashboard/index.html").write_text(html, encoding="utf-8")
    from scripts.render_dashboard_preview import main as render_preview

    render_preview()


def build_notebooks():
    common = "import json, pandas as pd\nfrom IPython.display import display, Image\nresults=json.loads((ROOT/'reports/results.json').read_text())"
    notebooks = [
        (
            "01_data_quality",
            "Data quality and provenance",
            "Validate the exact files before interpretation; all HR records are fictional.",
            [
                "from src.data_cleaning import load_hr, load_adult, quality\nhr=load_hr(ROOT/'data/raw/emp_attrition.csv')\ntrain,test=load_adult(ROOT/'data/raw/adult.zip')\ndisplay(pd.DataFrame({'hr':quality(hr)['missing'],'adult_train':quality(train)['missing']}).fillna('not a field'))",
                "display(pd.Series(quality(hr)['dtypes'],name='dtype'))\ndisplay(pd.Series(quality(hr)['iqr_flags'],name='IQR flags'))\nprint('Exact duplicate rows',quality(hr)['duplicate_rows'])\nprint('Source units: income unspecified; tenure years. Outliers retained pending investigation.')",
            ],
        ),
        (
            "02_workforce_eda",
            "Workforce patterns",
            "Sample headcount is not a dated active workforce census; attrition share has no annual denominator.",
            [
                "from src.data_cleaning import load_hr\nfrom src.workforce_metrics import workforce_kpis, group_summary\nhr=load_hr(ROOT/'data/raw/emp_attrition.csv')\ndisplay(pd.Series(workforce_kpis(hr)))\nfor group in ['Department','JobRole','OverTime','Education','PerformanceRating']: display(group_summary(hr,group))",
                "display(Image(filename=str(ROOT/'figures/attrition_by_role.png')))\nprint('Investigate workload in high-share roles; do not infer intervention effects.')",
            ],
        ),
        (
            "03_compensation_analysis",
            "Compensation associations",
            "Use source units, exploratory tests and adjustment; no discrimination claim.",
            [
                "from src.data_cleaning import load_hr\nfrom src.statistics import compensation_tests\nhr=load_hr(ROOT/'data/raw/emp_attrition.csv')\ndisplay(hr.groupby(['Department','Gender']).MonthlyIncome.agg(['count','mean','median','std']))",
                "display(pd.Series(compensation_tests(hr)))\ndisplay(Image(filename=str(ROOT/'figures/compensation.png')))\nprint('Role adjustment does not remove unmeasured confounding; synthetic effects are not real compensation evidence.')",
            ],
        ),
        (
            "04_attrition_analysis",
            "Attrition concentration and support",
            "Investigate association, never call snapshot shares annual turnover.",
            [
                "from src.data_cleaning import load_hr\nfrom src.workforce_metrics import group_summary\nfrom src.fairness import wilson\nhr=load_hr(ROOT/'data/raw/emp_attrition.csv')\nroles=group_summary(hr,'JobRole')\nroles['ci95']=[wilson(int(k),int(n)) for k,n in zip(roles.attrition_yes,roles.n)]\ndisplay(roles.sort_values('attrition_share',ascending=False))",
                "from scipy.stats import chi2_contingency\ncontingency=pd.crosstab(hr.OverTime,hr.Attrition)\nchi2,p,dof,expected=chi2_contingency(contingency)\nprint({'chi_square':chi2,'p':p,'cramers_v':(chi2/len(hr))**.5})\ndisplay(contingency)\nprint('Exploratory fictional association; no causal workload or retention estimate.')",
            ],
        ),
        (
            "05_predictive_model",
            "Leakage-controlled model development",
            "Recompute HR training/validation selection; the pipeline independently evaluates untouched test data.",
            [
                "from src.data_cleaning import load_hr\nfrom src.modeling import select_model\nfrom src.evaluation import evaluate\nhr=load_hr(ROOT/'data/raw/emp_attrition.csv')\nsplit=pd.read_csv(ROOT/'data/processed/hr_split_assignment.csv')\nx=hr.drop(columns=['Attrition','EmployeeNumber','EmployeeCount','Over18','StandardHours','Gender'])\ny=hr.Attrition.eq('Yes').astype(int)\ntrain=split.split.eq('train'); val=split.split.eq('validation'); test=split.split.eq('test')\nassert not (train & val).any() and not (train & test).any()\nmodel,selection=select_model(x[train],y[train],x[val],y[val],42)\ndisplay(selection)",
                "probability=model.predict_proba(x[test])[:,1]\ndisplay(evaluate(y[test],probability,selection['threshold']))\ndisplay(pd.read_csv(ROOT/'data/processed/hr_permutation_importance.csv').head(10))\nprint('False positives: unnecessary contact or stigma. False negatives: missed support. Educational threshold maximizes validation F1, not business benefit.')",
            ],
        ),
        (
            "06_fairness_audit",
            "Fairness, error rates and calibration",
            "Audit separate targets; sensitive-column exclusion does not remove proxies.",
            [
                "from src.fairness import disparities\nfor scope in ['hr','adult']:\n    display(pd.DataFrame(results[scope]['gender_audit']))\n    display(disparities(results[scope]['gender_audit']))\ndisplay(pd.DataFrame(results['adult']['race_audit']))",
                "display(pd.read_csv(ROOT/'data/processed/adult_calibration.csv'))\ndisplay(Image(filename=str(ROOT/'figures/fairness_selection.png')))\nprint('Wilson intervals quantify sampling uncertainty, not historical measurement bias. Undefined rates remain null. No legal discrimination determination.')",
            ],
        ),
        (
            "07_external_benchmarking",
            "External tenure context",
            "BLS January 2026 values are a separate survey population, not a matched employer benchmark.",
            [
                "benchmark=pd.read_csv(ROOT/'data/processed/bls_context.csv')\ndisplay(benchmark)\nprint('OBSERVED published estimates; BLS wage/salary workers, excluding self-employed.')",
                "display(pd.Series(results['hr_kpis']))\nprint('Do not subtract these sample indicators from BLS and rank performance. Fictional undated records, different worker mix and definitions prevent fair comparison.')\nprint('Real deployment needs hire/exit dates, current status, occupation/industry codes, consistent tenure and sampling definitions.')",
            ],
        ),
    ]
    for name, title, narrative, cells in notebooks:
        write_notebook(
            ROOT / f"notebooks/{name}.ipynb", title, narrative, [common] + cells
        )


if __name__ == "__main__":
    run()
