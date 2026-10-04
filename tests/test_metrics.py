"""Guard positive-label semantics, undefined denominators and data leakage."""

import numpy as np
import pandas as pd
import pytest
from src.fairness import audit, disparities, wilson
from src.evaluation import evaluate
from src.workforce_metrics import workforce_kpis
from src.modeling import make_pipeline


def test_confusion_semantics():
    result = evaluate([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], 0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["precision"] == 0.5 and result["recall"] == 0.5


def test_zero_denominator_undefined():
    rows = audit([0, 0], [0.1, 0.8], ["Female", "Female"], 0.5)
    assert rows[0]["tpr"] is None
    assert rows[0]["fnr"] is None
    assert rows[0]["tpr_ci95"] == (None, None)
    assert rows[0]["fpr"] == 0.5


def test_wilson_boundary():
    low, high = wilson(0, 10)
    assert abs(low) < 1e-12 and 0 < high < 1
    low, high = wilson(10, 10)
    assert 0 < low < 1 and abs(high - 1) < 1e-12


def test_disparity_direction_and_ratio():
    rows = audit(
        [0, 1, 0, 1], [0.7, 0.9, 0.2, 0.8], ["Female", "Female", "Male", "Male"], 0.5
    )
    result = disparities(rows)
    assert result["selection_rate_difference"] == 0.5
    assert result["selection_rate_ratio"] == 2


def test_preprocessing_fitted_only_on_training():
    x = pd.DataFrame({"income": [1.0, 2.0, 3.0, 4.0], "role": ["a", "b", "a", "b"]})
    model = make_pipeline(x, "logistic", 42).fit(x, [0, 1, 0, 1])
    imputer = (
        model.named_steps["transform"].named_transformers_["num"].named_steps["impute"]
    )
    assert imputer.statistics_[0] == 2.5
    prediction = model.predict_proba(
        pd.DataFrame({"income": [10000.0, np.nan], "role": ["unseen", "a"]})
    )
    assert prediction.shape == (2, 2)
    assert imputer.statistics_[0] == 2.5


def test_snapshot_not_active_headcount():
    frame = pd.DataFrame(
        {
            "Attrition": ["Yes", "No"],
            "YearsAtCompany": [1, 5],
            "MonthlyIncome": [10, 20],
        }
    )
    result = workforce_kpis(frame)
    assert result["sample_records"] == 2
    assert result["retained_records"] == 1
    assert result["attrition_share"] == 0.5


def test_invalid_label_rejected(tmp_path):
    from src.data_cleaning import load_hr

    frame = pd.DataFrame(
        {
            "Attrition": ["Maybe"],
            "EmployeeNumber": [1],
            "Gender": ["Female"],
            "MonthlyIncome": [100],
            "Department": ["Sales"],
            "JobRole": ["Sales"],
            "YearsAtCompany": [1],
            "OverTime": ["No"],
        }
    )
    target = tmp_path / "bad.csv"
    frame.to_csv(target, index=False)
    with pytest.raises(ValueError, match="Unknown attrition"):
        load_hr(target)


def test_duplicate_employee_rejected(tmp_path):
    from src.data_cleaning import load_hr

    frame = pd.DataFrame(
        {
            "Attrition": ["Yes", "No"],
            "EmployeeNumber": [1, 1],
            "Gender": ["Female", "Female"],
            "MonthlyIncome": [100, 200],
            "Department": ["Sales", "Sales"],
            "JobRole": ["Sales", "Sales"],
            "YearsAtCompany": [1, 2],
            "OverTime": ["No", "Yes"],
        }
    )
    target = tmp_path / "bad.csv"
    frame.to_csv(target, index=False)
    with pytest.raises(ValueError, match="Duplicate employee"):
        load_hr(target)
