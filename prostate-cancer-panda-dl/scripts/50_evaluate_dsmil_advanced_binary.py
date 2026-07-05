"""Evaluate DSMIL advanced binary best checkpoint on valid and test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.advanced_embedding_dataset import AdvancedEmbeddingDataset, advanced_bag_collate_fn
from src.mil.dsmil import DSMILBinary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua DSMIL binario advanced con threshold Youden desde valid."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dsmil_virchow2_advanced_train_binary.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--embeddings-root", type=Path, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"No existe la configuracion: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("La configuracion debe ser un objeto YAML.")
    return config


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(config)
    if args.device is not None:
        updated["device"] = args.device
    if args.embeddings_root is not None:
        updated["embeddings_root"] = str(args.embeddings_root)
    if args.max_valid is not None:
        updated["max_valid"] = int(args.max_valid)
    if args.max_test is not None:
        updated["max_test"] = int(args.max_test)
    return updated


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay una GPU CUDA disponible.")
    return device


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_dataset(config: Dict[str, Any], split: str, max_items: int | None) -> AdvancedEmbeddingDataset:
    return AdvancedEmbeddingDataset(
        embeddings_root=Path(config["embeddings_root"]),
        split=split,
        encoder_name=str(config["encoder_name"]),
        model_name=str(config["model_name"]),
        expected_dim=int(config["input_dim"]),
        max_items=max_items,
        validate_on_init=False,
    )


def build_loader(dataset: AdvancedEmbeddingDataset, config: Dict[str, Any], device: torch.device) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size_bags", 1)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=advanced_bag_collate_fn,
    )


def build_model(config: Dict[str, Any]) -> DSMILBinary:
    return DSMILBinary(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )


def forward_batch(
    model: DSMILBinary,
    batch: Dict[str, Any],
    criterion: nn.Module,
    device: torch.device,
    instance_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"].to(device).float()
    logits: List[torch.Tensor] = []
    losses: List[torch.Tensor] = []
    for features, label in zip(batch["features"], labels):
        output = model(features.to(device, non_blocking=device.type == "cuda"))
        bag_logit = output["logit"]
        max_instance_logit = torch.max(output["instance_logits"])
        bag_loss = criterion(bag_logit.view(1), label.view(1))
        instance_loss = criterion(max_instance_logit.view(1), label.view(1))
        loss = bag_loss + float(instance_loss_weight) * instance_loss
        logits.append(bag_logit)
        losses.append(loss)
    return torch.stack(logits).float(), torch.stack(losses).mean()


def evaluate_split(
    model: DSMILBinary,
    loader: Iterable[Dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    instance_loss_weight: float,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    labels: List[int] = []
    probabilities: List[float] = []
    slide_ids: List[str] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluate DSMIL advanced", leave=False):
            logits, loss = forward_batch(model, batch, criterion, device, instance_loss_weight)
            losses.append(float(loss.detach().cpu().item()))
            batch_labels = batch["labels"].detach().cpu().numpy().astype(int).tolist()
            labels.extend(batch_labels)
            probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
            slide_ids.extend(str(value) for value in batch["slide_ids"])
    if not losses:
        raise RuntimeError("El DataLoader de evaluacion no produjo batches.")
    return {
        "loss": float(np.mean(losses)),
        "labels": labels,
        "probabilities": probabilities,
        "slide_ids": slide_ids,
    }


def binary_metrics(
    labels: List[int],
    probabilities: List[float],
    loss: float,
    threshold: float,
) -> Dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
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
    auc_roc = None
    if np.unique(y_true).size >= 2:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    return {
        "threshold": float(threshold),
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
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


def youden_from_valid(labels: List[int], probabilities: List[float]) -> tuple[float, pd.DataFrame]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if np.unique(y_true).size < 2:
        return 0.5, pd.DataFrame(
            [{"threshold": 0.5, "sensitivity": None, "specificity": None, "youden_j": None}]
        )
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    youden_j = tpr + specificity - 1.0
    index = int(np.argmax(youden_j))
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "sensitivity": tpr,
            "specificity": specificity,
            "youden_j": youden_j,
        }
    )
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = 0.5
    return threshold, table


def predictions_frame(result: Dict[str, Any], threshold_youden: float) -> pd.DataFrame:
    rows = []
    for slide_id, label, probability in zip(
        result["slide_ids"],
        result["labels"],
        result["probabilities"],
    ):
        rows.append(
            {
                "slide_id": slide_id,
                "cancer_label": int(label),
                "pred_probability": float(probability),
                "pred_label_threshold_0_5": int(float(probability) >= 0.5),
                "pred_label_threshold_youden": int(float(probability) >= float(threshold_youden)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    device = resolve_device(str(config["device"]))
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(config["checkpoints_dir"]) / "best_model.pt"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(config["output_root"]) / "test_evaluation_best_model"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss()
    instance_loss_weight = float(config.get("instance_loss_weight", 0.3))

    valid_dataset = build_dataset(config, "valid", config.get("max_valid"))
    test_dataset = build_dataset(config, str(args.split), config.get("max_test"))
    valid_loader = build_loader(valid_dataset, config, device)
    test_loader = build_loader(test_dataset, config, device)

    valid_result = evaluate_split(model, valid_loader, criterion, device, instance_loss_weight)
    test_result = evaluate_split(model, test_loader, criterion, device, instance_loss_weight)
    threshold_youden, thresholds_table = youden_from_valid(
        valid_result["labels"],
        valid_result["probabilities"],
    )

    valid_metrics_05 = binary_metrics(
        valid_result["labels"],
        valid_result["probabilities"],
        valid_result["loss"],
        threshold=0.5,
    )
    valid_metrics_youden = binary_metrics(
        valid_result["labels"],
        valid_result["probabilities"],
        valid_result["loss"],
        threshold=threshold_youden,
    )
    test_metrics_05 = binary_metrics(
        test_result["labels"],
        test_result["probabilities"],
        test_result["loss"],
        threshold=0.5,
    )
    test_metrics_youden = binary_metrics(
        test_result["labels"],
        test_result["probabilities"],
        test_result["loss"],
        threshold=threshold_youden,
    )

    predictions_frame(valid_result, threshold_youden).to_csv(
        output_dir / "valid_predictions_best_model.csv",
        index=False,
    )
    predictions_frame(test_result, threshold_youden).to_csv(
        output_dir / "test_predictions_best_model.csv",
        index=False,
    )
    thresholds_table.to_csv(output_dir / "valid_roc_thresholds_youden.csv", index=False)
    save_json(valid_metrics_05, output_dir / "valid_metrics_threshold_0_5.json")
    save_json(valid_metrics_youden, output_dir / "valid_metrics_threshold_youden.json")
    save_json(test_metrics_05, output_dir / "test_metrics_threshold_0_5.json")
    save_json(
        test_metrics_youden,
        output_dir / "test_metrics_threshold_youden_from_valid.json",
    )
    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
            "valid_wsi": len(valid_dataset),
            "evaluated_split": str(args.split),
            "evaluated_wsi": len(test_dataset),
            "threshold_youden_from_valid": float(threshold_youden),
            "valid_metrics_threshold_0_5": valid_metrics_05,
            "valid_metrics_threshold_youden": valid_metrics_youden,
            "test_metrics_threshold_0_5": test_metrics_05,
            "test_metrics_threshold_youden_from_valid": test_metrics_youden,
        },
        output_dir / "evaluation_summary_best_model.json",
    )
    print("[INFO] Evaluacion DSMIL advanced completada")
    print(f"[INFO] Youden threshold valid: {threshold_youden:.6f}")
    print(f"[INFO] Test AUC: {test_metrics_05['auc_roc']}")
    print(f"[INFO] Output dir: {output_dir}")


if __name__ == "__main__":
    main()
