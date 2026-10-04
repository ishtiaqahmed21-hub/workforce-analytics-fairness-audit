"""Training and validation-only selection; held-out test stays untouched."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score


def make_pipeline(frame: pd.DataFrame, model: str, seed: int) -> Pipeline:
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    numeric = frame.select_dtypes("number").columns.tolist()
    transform = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    estimator = (
        LogisticRegression(max_iter=1500, random_state=seed)
        if model == "logistic"
        else RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=5,
            max_depth=16,
            n_jobs=-1,
            random_state=seed,
        )
    )
    return Pipeline([("transform", transform), ("model", estimator)])


def select_model(train, train_y, validation, validation_y, seed=42):
    fitted = {}
    comparison = {}
    for name in ["logistic", "random_forest"]:
        model = make_pipeline(train, name, seed).fit(train, train_y)
        probability = model.predict_proba(validation)[:, 1]
        comparison[name] = float(average_precision_score(validation_y, probability))
        fitted[name] = model
    winner = max(comparison, key=comparison.get)
    model = fitted[winner]
    probability = model.predict_proba(validation)[:, 1]
    thresholds = np.linspace(0.05, 0.8, 76)
    scores = [
        f1_score(validation_y, probability >= t, zero_division=0) for t in thresholds
    ]
    threshold = float(thresholds[int(np.argmax(scores))])
    return model, {
        "selected_model": winner,
        "validation_average_precision": comparison,
        "threshold": threshold,
        "validation_f1": float(max(scores)),
    }
