"""
Task-specific metrics for model evaluation.
All metrics operate on numpy arrays / lists and return plain floats or dicts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        mean_squared_error, mean_absolute_error, r2_score,
        roc_auc_score, confusion_matrix,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available – some metrics disabled.")


def classification_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    average: str = "weighted",
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not _SKLEARN_AVAILABLE:
        acc = float(np.mean(y_true == y_pred))
        return {"accuracy": acc}
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }


def regression_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not _SKLEARN_AVAILABLE:
        mse = float(np.mean((y_true - y_pred) ** 2))
        return {"mse": mse, "rmse": float(np.sqrt(mse))}
    return {
        "mse":  float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def compute_metrics(
    task_type: str,
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    **kwargs,
) -> dict[str, Any]:
    """
    Unified entry point.

    Parameters
    ----------
    task_type : "classification" | "regression" | "multilabel"
    """
    task_type = task_type.lower()
    if task_type in ("classification", "multilabel"):
        avg = kwargs.get("average", "weighted")
        return classification_metrics(y_true, y_pred, average=avg)
    elif task_type == "regression":
        return regression_metrics(y_true, y_pred)
    else:
        logger.warning("Unknown task_type '%s' – falling back to classification.", task_type)
        return classification_metrics(y_true, y_pred)


def detect_overfitting(
    train_losses: list[float],
    val_losses: list[float],
    threshold: float = 0.10,
) -> dict[str, Any]:
    """
    Detect overfitting by comparing the trend in train vs val loss.

    Returns
    -------
    dict with 'overfitting' (bool), 'gap' (float), 'severity' (str).
    """
    if not train_losses or not val_losses:
        return {"overfitting": False, "gap": 0.0, "severity": "none"}

    gap = val_losses[-1] - train_losses[-1]
    overfitting = gap > threshold

    if gap > 0.3:
        severity = "high"
    elif gap > threshold:
        severity = "moderate"
    else:
        severity = "none"

    return {"overfitting": overfitting, "gap": round(gap, 4), "severity": severity}
