"""Matplotlib plots for binary MIL training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def _prepare_output(path: Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_roc_curve(
    labels: Sequence[int],
    probabilities: Sequence[float],
    path: Path,
    title: str,
) -> None:
    """Save a ROC curve or an explanatory placeholder for one-class data."""
    output = _prepare_output(path)
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6, 5))
    if np.unique(y_true).size < 2:
        axis.text(
            0.5,
            0.5,
            "ROC no disponible: solo hay una clase",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
    else:
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=axis)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def save_confusion_matrix(
    confusion: Dict[str, int],
    path: Path,
    title: str,
) -> None:
    """Save a binary confusion matrix."""
    output = _prepare_output(path)
    matrix = np.asarray(
        [
            [confusion["tn"], confusion["fp"]],
            [confusion["fn"], confusion["tp"]],
        ],
        dtype=np.int64,
    )
    figure, axis = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["no_cancer", "cancer"],
    )
    display.plot(ax=axis, colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def save_loss_history(history: Sequence[Dict[str, Any]], path: Path) -> None:
    """Save train and validation loss by epoch."""
    output = _prepare_output(path)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    valid_loss = [row["valid_loss"] for row in history]

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, train_loss, marker="o", label="train_loss")
    axis.plot(epochs, valid_loss, marker="o", label="valid_loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("ABMIL + UNI2-h Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def save_metric_history(history: Sequence[Dict[str, Any]], path: Path) -> None:
    """Save validation AUC and F1 by epoch."""
    output = _prepare_output(path)
    epochs = [row["epoch"] for row in history]
    auc_values = [
        np.nan if row.get("valid_auc") is None else row["valid_auc"]
        for row in history
    ]
    f1_values = [row.get("valid_f1", np.nan) for row in history]

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, auc_values, marker="o", label="valid_auc")
    axis.plot(epochs, f1_values, marker="o", label="valid_f1")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("ABMIL + UNI2-h Validation Metrics")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
