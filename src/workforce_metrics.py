"""Snapshot indicators; no implied annual rates or active census."""

import pandas as pd


def workforce_kpis(frame: pd.DataFrame) -> dict:
    return {
        "sample_records": len(frame),
        "attrition_yes": int(frame.Attrition.eq("Yes").sum()),
        "attrition_share": float(frame.Attrition.eq("Yes").mean()),
        "retained_records": int(frame.Attrition.eq("No").sum()),
        "median_tenure_years": float(frame.YearsAtCompany.median()),
        "median_monthly_income_units": float(frame.MonthlyIncome.median()),
    }


def group_summary(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        frame.assign(attrition=frame.Attrition.eq("Yes").astype(int))
        .groupby(group, observed=True)
        .agg(
            n=("attrition", "size"),
            attrition_yes=("attrition", "sum"),
            attrition_share=("attrition", "mean"),
            median_income=("MonthlyIncome", "median"),
            median_tenure=("YearsAtCompany", "median"),
        )
        .reset_index()
    )
