"""Metrics utilities for binary MIL experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def compute_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return threshold maximizing Youden index (TPR - FPR)."""
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    thr = float(thresholds[idx])

    if not np.isfinite(thr):
        return 0.5
    return thr


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    auc = _safe_auc(y_true, y_prob)
    gini = (2.0 * auc - 1.0) if auc is not None else None

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": specificity,
        "f1": float(f1),
        "auc": auc,
        "gini": gini,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

