"""Load independent datasets without manufacturing unavailable fields."""

from pathlib import Path
from zipfile import ZipFile
import pandas as pd

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def load_hr(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "Attrition",
        "EmployeeNumber",
        "Gender",
        "MonthlyIncome",
        "Department",
        "JobRole",
        "YearsAtCompany",
        "OverTime",
    }
    if not required.issubset(frame.columns):
        raise ValueError("HR schema mismatch")
    if not frame.Attrition.isin(["Yes", "No"]).all():
        raise ValueError("Unknown attrition labels")
    if frame.EmployeeNumber.duplicated().any():
        raise ValueError("Duplicate employee identifiers")
    if (frame.MonthlyIncome < 0).any() or (frame.YearsAtCompany < 0).any():
        raise ValueError("Negative income or tenure")
    return frame


def load_adult(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    with ZipFile(path) as archive:
        for name in ["adult.data", "adult.test"]:
            with archive.open(name) as stream:
                frame = pd.read_csv(
                    stream,
                    names=ADULT_COLUMNS,
                    skipinitialspace=True,
                    na_values="?",
                    comment="|",
                )
            frame["income"] = frame.income.str.strip().str.rstrip(".")
            if not frame.income.isin(["<=50K", ">50K"]).all():
                raise ValueError("Unknown income labels")
            frames.append(frame)
    return tuple(frames)


def quality(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes("number")
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    outliers = (
        (numeric < q1 - 1.5 * (q3 - q1)) | (numeric > q3 + 1.5 * (q3 - q1))
    ).sum()
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "iqr_flags": outliers.to_dict(),
    }
