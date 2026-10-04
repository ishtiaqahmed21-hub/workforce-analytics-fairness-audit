"""Classification evaluation with explicit positive-class semantics."""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)


def evaluate(y, probability, threshold: float) -> dict:
    y, probability = np.asarray(y), np.asarray(probability)
    predicted = probability >= threshold
    return {
        "n": len(y),
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None
        ),
        "average_precision": (
            float(average_precision_score(y, probability)) if y.sum() else None
        ),
        "brier": float(brier_score_loss(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }
