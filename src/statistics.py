"""Exploratory compensation associations in fictional records."""

import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def compensation_tests(frame) -> dict:
    female = frame.loc[frame.Gender == "Female", "MonthlyIncome"].to_numpy()
    male = frame.loc[frame.Gender == "Male", "MonthlyIncome"].to_numpy()
    welch = stats.ttest_ind(female, male, equal_var=False)
    mw = stats.mannwhitneyu(female, male, alternative="two-sided")
    pooled = np.sqrt(
        ((len(female) - 1) * female.var(ddof=1) + (len(male) - 1) * male.var(ddof=1))
        / (len(female) + len(male) - 2)
    )
    model = smf.ols(
        "np.log(MonthlyIncome) ~ C(Gender) + C(JobRole) + Age + TotalWorkingYears",
        data=frame,
    ).fit(cov_type="HC3")
    term = "C(Gender)[T.Male]"
    ci = model.conf_int().loc[term].tolist()
    return {
        "female_n": len(female),
        "male_n": len(male),
        "female_mean": float(female.mean()),
        "male_mean": float(male.mean()),
        "mean_difference_female_minus_male": float(female.mean() - male.mean()),
        "cohens_d": float((female.mean() - male.mean()) / pooled),
        "welch_p": float(welch.pvalue),
        "mann_whitney_p": float(mw.pvalue),
        "rank_biserial_female_minus_male": float(
            2 * mw.statistic / (len(female) * len(male)) - 1
        ),
        "adjusted_log_income_male_vs_female_coefficient": float(model.params[term]),
        "adjusted_ci95": ci,
        "adjusted_p": float(model.pvalues[term]),
        "adjustment": "JobRole + Age + TotalWorkingYears; HC3 robust covariance; exploratory association, not causal",
    }
