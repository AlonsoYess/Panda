"""Evaluate CLAM + Virchow2 advanced severity 4-class on test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] scikit-learn es requerido para evaluar severity 4-class advanced."
    ) from exc

from src.mil.advanced_severity4_dataset import (
    AdvancedSeverity4Dataset,
    advanced_severity4_bag_collate_fn,
)
from src.mil.clam_multiclass import CLAMMulticlass

NUM_CLASSES = 4
SEVERITY_LABELS = list(range(NUM_CLASSES))
SEVERITY_NAMES = [
    "severity_0_no_cancer",
    "severity_1_low_grade",
    "severity_2_intermediate",
    "severity_3_high_grade",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua CLAM + Virchow2 advanced severity 4-class."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/clam_virchow2_advanced_train_severity4.yaml"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--embeddings-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
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
    if args.output_root is not None:
        output_root = Path(args.output_root)
        updated["output_root"] = str(output_root)
        updated["checkpoints_dir"] = str(output_root / "checkpoints")
        updated["metrics_dir"] = str(output_root / "metrics")
        updated["plots_dir"] = str(output_root / "plots")
        updated["logs_dir"] = str(output_root / "logs")
    if args.batch_size is not None:
        updated["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        updated["num_workers"] = int(args.num_workers)
    if args.max_test is not None:
        updated["max_test"] = int(args.max_test)
    return updated


def resolve_device(requested: str) -> torch.device:
    value = str(requested)
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay GPU CUDA disponible.")
    return device


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_model(config: Dict[str, Any]) -> CLAMMulticlass:
    return CLAMMulticlass(
        input_dim=int(config["input_dim"]),
        num_classes=int(config["num_classes"]),
        hidden_dim=int(config["hidden_dim"]),
        attention_dim=int(config["attention_dim"]),
        dropout=float(config["dropout"]),
    )


def build_loader(
    dataset: AdvancedSeverity4Dataset,
    config: Dict[str, Any],
    device: torch.device,
) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", False)) and device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=advanced_severity4_bag_collate_fn,
    )


def forward_batch(
    model: CLAMMulticlass,
    batch: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = [item.to(device, non_blocking=device.type == "cuda") for item in batch["features"]]
    labels = batch["labels"].to(device, non_blocking=device.type == "cuda").long()
    logits = model(features)["logits"]
    return logits, labels


def evaluate(
    model: CLAMMulticlass,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    slide_ids: List[str] = []
    labels_all: List[int] = []
    predictions_all: List[int] = []
    probabilities_all: List[List[float]] = []
    metadata_all: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluate advanced severity", leave=False):
            logits, labels = forward_batch(model, batch, device)
            loss = criterion(logits.float(), labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            losses.append(float(loss.detach().cpu().item()))
            slide_ids.extend(str(value) for value in batch["slide_ids"])
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            predictions_all.extend(predictions.detach().cpu().numpy().astype(int).tolist())
            probabilities_all.extend(probabilities.detach().cpu().numpy().astype(float).tolist())
            metadata_all.extend(batch["metadata"])
    if not losses:
        raise RuntimeError("El DataLoader de test no produjo batches.")
    return {
        "loss": float(np.mean(losses)),
        "slide_ids": slide_ids,
        "labels": labels_all,
        "predictions": predictions_all,
        "probabilities": probabilities_all,
        "metadata": metadata_all,
    }


def compute_metrics(labels: List[int], predictions: List[int], loss: float) -> Dict[str, Any]:
    matrix = confusion_matrix(labels, predictions, labels=SEVERITY_LABELS)
    report = classification_report(
        labels,
        predictions,
        labels=SEVERITY_LABELS,
        target_names=SEVERITY_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "test_loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "confusion_matrix": matrix.astype(int).tolist(),
        "classification_report": report,
    }


def write_outputs(
    *,
    result: Dict[str, Any],
    metrics: Dict[str, Any],
    output_root: Path,
    split: str,
    checkpoint_path: Path,
) -> None:
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, metrics_dir / "test_metrics.json")

    rows = []
    for slide_id, label, prediction, probabilities, metadata in zip(
        result["slide_ids"],
        result["labels"],
        result["predictions"],
        result["probabilities"],
        result["metadata"],
    ):
        row = {
            "slide_id": slide_id,
            "y_true_severity4": int(label),
            "y_pred_severity4": int(prediction),
            "isup_grade": metadata.get("isup_grade"),
            "cancer_label": metadata.get("cancer_label"),
            "gleason_score": metadata.get("gleason_score"),
        }
        for class_id, probability in enumerate(probabilities):
            row[f"prob_severity_{class_id}"] = float(probability)
        rows.append(row)
    pd.DataFrame(rows).to_csv(metrics_dir / "test_predictions.csv", index=False)

    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in SEVERITY_NAMES],
        columns=[f"pred_{name}" for name in SEVERITY_NAMES],
    ).to_csv(metrics_dir / "confusion_matrix_4x4.csv")
    pd.DataFrame(metrics["classification_report"]).transpose().to_csv(
        metrics_dir / "classification_report.csv"
    )
    save_json(
        {
            "split": split,
            "checkpoint": str(checkpoint_path),
            "output_root": str(output_root),
            "metrics_path": str(metrics_dir / "test_metrics.json"),
            "predictions_path": str(metrics_dir / "test_predictions.csv"),
            "num_wsi": len(result["labels"]),
            "metrics": {
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "qwk": metrics["qwk"],
            },
        },
        metrics_dir / "evaluation_summary.json",
    )


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    device = resolve_device(str(config["device"]))
    output_root = Path(config["output_root"])
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(config["checkpoints_dir"]) / "best_model.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint_path}")

    dataset = AdvancedSeverity4Dataset(
        embeddings_root=Path(config["embeddings_root"]),
        split=args.split,
        max_items=config.get("max_test"),
        validate_on_init=False,
    )
    loader = build_loader(dataset, config, device)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    result = evaluate(model, loader, criterion, device)
    metrics = compute_metrics(result["labels"], result["predictions"], result["loss"])
    write_outputs(
        result=result,
        metrics=metrics,
        output_root=output_root,
        split=args.split,
        checkpoint_path=checkpoint_path,
    )
    print("[INFO] Evaluacion CLAM + Virchow2 advanced severity4 completada")
    print(f"[INFO] WSI evaluadas: {len(result['labels'])}")
    print(f"[INFO] accuracy={metrics['accuracy']:.4f}")
    print(f"[INFO] balanced_accuracy={metrics['balanced_accuracy']:.4f}")
    print(f"[INFO] macro_f1={metrics['macro_f1']:.4f}")
    print(f"[INFO] weighted_f1={metrics['weighted_f1']:.4f}")
    print(f"[INFO] qwk={metrics['qwk']:.4f}")
    print(f"[INFO] metrics_dir={output_root / 'metrics'}")


if __name__ == "__main__":
    main()
