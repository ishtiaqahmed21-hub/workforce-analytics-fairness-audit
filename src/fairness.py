"""Support-aware subgroup diagnostics; differences do not prove discrimination."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def rate(k: int, n: int):
    return float(k / n) if n else None


def wilson(k: int, n: int) -> tuple:
    if not n:
        return (None, None)
    z = 1.95996398454
    p = k / n
    middle = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(middle - half), float(middle + half))


def audit(y, probability, group, threshold: float) -> list[dict]:
    y, probability, group = np.asarray(y), np.asarray(probability), np.asarray(group)
    rows = []
    for value in sorted(set(group)):
        keep = group == value
        truth = y[keep]
        prob = probability[keep]
        pred = prob >= threshold
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        n = len(truth)
        positives = int(truth.sum())
        selected = int(pred.sum())
        rows.append(
            {
                "group": str(value),
                "n": n,
                "actual_positive_n": positives,
                "predicted_positive_n": selected,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "selection_rate": rate(selected, n),
                "selection_rate_ci95": wilson(selected, n),
                "tpr": rate(tp, tp + fn),
                "tpr_ci95": wilson(tp, tp + fn),
                "fpr": rate(fp, fp + tn),
                "fpr_ci95": wilson(fp, fp + tn),
                "fnr": rate(fn, fn + tp),
                "fnr_ci95": wilson(fn, fn + tp),
                "brier": float(np.mean((truth - prob) ** 2)),
                "small_group": n < 50,
            }
        )
    return rows


def disparities(rows: list[dict], reference="Male", comparison="Female") -> dict:
    lookup = {r["group"]: r for r in rows}
    a, b = lookup[comparison], lookup[reference]
    result = {
        key
        + "_difference": None if a[key] is None or b[key] is None else a[key] - b[key]
        for key in ["selection_rate", "tpr", "fpr", "fnr"]
    }
    result["selection_rate_ratio"] = (
        a["selection_rate"] / b["selection_rate"] if b["selection_rate"] else None
    )
    return result


def calibration(y, probability, group) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"y": np.asarray(y), "p": probability, "group": np.asarray(group)}
    )
    frame["bin"] = pd.cut(frame.p, np.linspace(0, 1, 6), include_lowest=True)
    return (
        frame.groupby(["group", "bin"], observed=True)
        .agg(
            n=("y", "size"),
            mean_probability=("p", "mean"),
            observed_positive_share=("y", "mean"),
        )
        .reset_index()
    )
