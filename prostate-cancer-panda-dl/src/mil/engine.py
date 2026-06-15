"""Reusable training, evaluation and checkpoint utilities for binary MIL."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """Set deterministic random seeds used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pos_weight(
    labels_or_dataset: Sequence[float] | Any,
    configured_value: float | None = None,
) -> float:
    """Return negatives/positives unless an explicit value is configured."""
    if configured_value is not None:
        value = float(configured_value)
        if value <= 0:
            raise ValueError("pos_weight configurado debe ser mayor que cero.")
        return value

    labels = getattr(labels_or_dataset, "labels", labels_or_dataset)
    array = np.asarray(labels, dtype=np.int64)
    positives = int((array == 1).sum())
    negatives = int((array == 0).sum())
    if positives == 0:
        raise ValueError("No hay ejemplos positivos en train; pos_weight no es calculable.")
    if negatives == 0:
        raise ValueError("No hay ejemplos negativos en train; pos_weight no es calculable.")
    return float(negatives / positives)


def _forward_bags(
    model: nn.Module,
    feature_bags: Iterable[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    logits = []
    for features in feature_bags:
        logit, _ = model(features.to(device, non_blocking=device.type == "cuda"))
        logits.append(logit)
    return torch.stack(logits, dim=0).float()


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Any | None = None,
    amp: bool = True,
) -> float:
    """Train ABMIL for one epoch and return mean bag loss."""
    model.train()
    losses: List[float] = []
    amp_enabled = bool(amp and device.type == "cuda")

    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        labels = batch["labels"].to(device, non_blocking=amp_enabled).float()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = _forward_bags(model, batch["features"], device)
            loss = criterion(logits, labels)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        raise RuntimeError("El DataLoader de entrenamiento no produjo batches.")
    return float(np.mean(losses))


def compute_binary_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute robust binary metrics for slide-level predictions."""
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.size == 0:
        raise ValueError("No hay labels para calcular metricas.")
    if y_true.shape != y_prob.shape:
        raise ValueError("labels y probabilities deben tener la misma longitud.")

    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = float(tn / (tn + fp)) if (tn + fp) else None

    auc_roc: float | None
    if np.unique(y_true).size < 2:
        auc_roc = None
    else:
        auc_roc = float(roc_auc_score(y_true, y_prob))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": specificity,
        "f1": float(f1),
        "auc_roc": auc_roc,
        "gini": (2.0 * auc_roc - 1.0) if auc_roc is not None else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def find_youden_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    default_threshold: float = 0.5,
) -> float:
    """Find the validation threshold maximizing Youden's J statistic."""
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if np.unique(y_true).size < 2:
        return float(default_threshold)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_prob)
    index = int(np.argmax(true_positive_rate - false_positive_rate))
    threshold = float(thresholds[index])
    return threshold if np.isfinite(threshold) else float(default_threshold)


def evaluate_binary(
    model: nn.Module,
    loader: Iterable[Dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    amp: bool = True,
) -> Dict[str, Any]:
    """Evaluate ABMIL and return metrics plus slide-level predictions."""
    model.eval()
    losses: List[float] = []
    labels: List[int] = []
    probabilities: List[float] = []
    slide_ids: List[str] = []
    amp_enabled = bool(amp and device.type == "cuda")

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluate", leave=False):
            batch_labels = batch["labels"].to(device, non_blocking=amp_enabled).float()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = _forward_bags(model, batch["features"], device)
                loss = criterion(logits, batch_labels)

            losses.append(float(loss.detach().cpu().item()))
            labels.extend(batch_labels.detach().cpu().numpy().astype(int).tolist())
            probabilities.extend(
                torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
            )
            slide_ids.extend(str(value) for value in batch["slide_ids"])

    if not losses:
        raise RuntimeError("El DataLoader de evaluacion no produjo batches.")

    metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
    metrics.update(
        {
            "loss": float(np.mean(losses)),
            "probabilities": probabilities,
            "labels": labels,
            "slide_ids": slide_ids,
            "predictions": [
                int(probability >= threshold) for probability in probabilities
            ],
        }
    )
    return metrics


def _atomic_torch_save(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    best_epoch: int,
    train_history: List[Dict[str, Any]],
    config: Dict[str, Any],
    seed: int,
    scaler: Any | None = None,
    scheduler: Any | None = None,
    git_info: Dict[str, Any] | None = None,
    early_stopping_counter: int = 0,
) -> None:
    """Persist all state needed to resume after a Colab interruption."""
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "train_history": list(train_history),
        "config": dict(config),
        "random_seed": int(seed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_info or {},
        "early_stopping_counter": int(early_stopping_counter),
    }
    _atomic_torch_save(checkpoint, Path(path))


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Restore model and optional optimizer/AMP/scheduler state."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    required = {"epoch", "model_state_dict", "optimizer_state_dict", "best_metric"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint incompleto. Faltan claves: {missing}")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if (
        scaler is not None
        and checkpoint.get("scaler_state_dict") is not None
    ):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def find_last_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return last_checkpoint.pt when it exists."""
    path = Path(checkpoints_dir) / "last_checkpoint.pt"
    return path if path.is_file() else None


def append_history_csv(
    history: List[Dict[str, Any]],
    path: Path,
) -> None:
    """Atomically rewrite accumulated training history after every epoch."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        pd.DataFrame(history).to_csv(temporary, index=False)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Atomically save JSON metadata and metrics."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
