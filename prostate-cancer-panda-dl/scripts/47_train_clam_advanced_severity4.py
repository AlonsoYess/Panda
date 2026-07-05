"""Train CLAM + Virchow2 advanced for severity 4-class classification."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
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
except Exception as exc:  # pragma: no cover - only reached when sklearn is absent.
    raise SystemExit(
        "[ERROR] scikit-learn es requerido para entrenar severity 4-class advanced."
    ) from exc

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - plotting is optional.
    plt = None
    HAS_MATPLOTLIB = False

from src.mil.advanced_severity4_dataset import (
    EXPECTED_EMBEDDING_DIM,
    VALID_SEVERITY_LABELS,
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
REQUIRED_CONFIG_KEYS = {
    "experiment_name",
    "task",
    "label_column",
    "num_classes",
    "input_dim",
    "encoder_name",
    "model_name",
    "embeddings_root",
    "output_root",
    "checkpoints_dir",
    "metrics_dir",
    "plots_dir",
    "logs_dir",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "hidden_dim",
    "attention_dim",
    "dropout",
    "early_stopping_patience",
    "monitor",
    "random_seed",
    "device",
    "num_workers",
    "pin_memory",
    "mixed_precision",
    "save_epoch_checkpoints",
    "resume",
    "max_train",
    "max_valid",
    "max_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena CLAM + Virchow2 advanced severity 4-class."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/clam_virchow2_advanced_train_severity4.yaml"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--embeddings-root", type=Path, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-train-slides", dest="max_train_slides", type=int, default=None)
    parser.add_argument("--max-valid-slides", dest="max_valid_slides", type=int, default=None)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"No existe la configuracion: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("La configuracion debe ser un objeto YAML.")
    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ValueError(f"Faltan claves requeridas en config: {missing}")
    return config


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(config)
    overrides = {
        "device": args.device,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_train": args.max_train_slides if args.max_train_slides is not None else args.max_train,
        "max_valid": args.max_valid_slides if args.max_valid_slides is not None else args.max_valid,
        "resume": args.resume,
    }
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    if args.embeddings_root is not None:
        updated["embeddings_root"] = str(args.embeddings_root)
    if args.output_root is not None:
        output_root = Path(args.output_root)
        updated["output_root"] = str(output_root)
        updated["checkpoints_dir"] = str(output_root / "checkpoints")
        updated["metrics_dir"] = str(output_root / "metrics")
        updated["plots_dir"] = str(output_root / "plots")
        updated["logs_dir"] = str(output_root / "logs")
    return updated


def validate_config(config: Dict[str, Any]) -> None:
    if str(config["label_column"]) != "severity_4_label":
        raise ValueError("label_column debe ser severity_4_label.")
    if int(config["num_classes"]) != NUM_CLASSES:
        raise ValueError("num_classes debe ser 4.")
    if int(config["input_dim"]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("input_dim debe ser 1280 para Virchow2 advanced.")
    if str(config["encoder_name"]).lower() != "virchow2":
        raise ValueError("encoder_name debe ser virchow2.")
    if str(config["model_name"]) != "paige-ai/Virchow2":
        raise ValueError("model_name debe ser paige-ai/Virchow2.")
    if str(config["monitor"]) != "valid_qwk":
        raise ValueError("monitor debe ser valid_qwk.")
    for key in ("epochs", "batch_size", "hidden_dim", "attention_dim"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} debe ser >= 1.")


def resolve_device(requested: str) -> torch.device:
    value = str(requested)
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay GPU CUDA disponible.")
    return device


def create_directories(config: Dict[str, Any]) -> None:
    for key in ("checkpoints_dir", "metrics_dir", "plots_dir", "logs_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def atomic_json_save(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_dataset(config: Dict[str, Any], split: str, max_items: int | None) -> AdvancedSeverity4Dataset:
    return AdvancedSeverity4Dataset(
        embeddings_root=Path(config["embeddings_root"]),
        split=split,
        max_items=max_items,
        validate_on_init=False,
    )


def build_loader(
    dataset: AdvancedSeverity4Dataset,
    config: Dict[str, Any],
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader:
    workers = int(config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config["pin_memory"]) and device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=advanced_severity4_bag_collate_fn,
    )


def build_model(config: Dict[str, Any]) -> CLAMMulticlass:
    return CLAMMulticlass(
        input_dim=int(config["input_dim"]),
        num_classes=int(config["num_classes"]),
        hidden_dim=int(config["hidden_dim"]),
        attention_dim=int(config["attention_dim"]),
        dropout=float(config["dropout"]),
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def label_distribution(labels: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {str(class_id): int(counts.get(class_id, 0)) for class_id in sorted(VALID_SEVERITY_LABELS)}


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    weights = []
    for class_id in SEVERITY_LABELS:
        count = counts.get(class_id, 0)
        if count == 0:
            raise ValueError(f"No hay ejemplos de severity {class_id} en train.")
        weights.append(total / (NUM_CLASSES * count))
    return torch.tensor(weights, dtype=torch.float32)


def forward_batch(
    model: CLAMMulticlass,
    batch: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = [item.to(device, non_blocking=device.type == "cuda") for item in batch["features"]]
    labels = batch["labels"].to(device, non_blocking=device.type == "cuda").long()
    logits = model(features)["logits"]
    return logits, labels


def compute_metrics(
    labels: List[int],
    predictions: List[int],
    loss: float,
) -> Dict[str, Any]:
    report = classification_report(
        labels,
        predictions,
        labels=SEVERITY_LABELS,
        target_names=SEVERITY_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=SEVERITY_LABELS)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "confusion_matrix": matrix.astype(int).tolist(),
        "classification_report": report,
    }


def train_one_epoch(
    model: CLAMMulticlass,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool,
) -> float:
    model.train()
    losses: List[float] = []
    amp_enabled = bool(amp and device.type == "cuda")
    for batch in tqdm(loader, desc="Train CLAM advanced severity", leave=False):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits, labels = forward_batch(model, batch, device)
            loss = criterion(logits.float(), labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Loss no finita durante entrenamiento: {loss}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise RuntimeError("El DataLoader de entrenamiento no produjo batches.")
    return float(np.mean(losses))


def evaluate(
    model: CLAMMulticlass,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp: bool,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    labels_all: List[int] = []
    predictions_all: List[int] = []
    probabilities_all: List[List[float]] = []
    slide_ids: List[str] = []
    amp_enabled = bool(amp and device.type == "cuda")
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Valid CLAM advanced severity", leave=False):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits, labels = forward_batch(model, batch, device)
                loss = criterion(logits.float(), labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Loss no finita durante validacion: {loss}")
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            losses.append(float(loss.detach().cpu().item()))
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            predictions_all.extend(predictions.detach().cpu().numpy().astype(int).tolist())
            probabilities_all.extend(probabilities.detach().cpu().numpy().astype(float).tolist())
            slide_ids.extend(str(value) for value in batch["slide_ids"])
    if not losses:
        raise RuntimeError("El DataLoader de validacion no produjo batches.")
    metrics = compute_metrics(labels_all, predictions_all, float(np.mean(losses)))
    metrics.update(
        {
            "labels": labels_all,
            "predictions": predictions_all,
            "probabilities": probabilities_all,
            "slide_ids": slide_ids,
        }
    )
    return metrics


def save_history(history: List[Dict[str, Any]], metrics_dir: Path) -> None:
    pd.DataFrame(history).to_csv(metrics_dir / "train_history.csv", index=False)
    atomic_json_save({"history": history}, metrics_dir / "train_history.json")


def save_plots(history: List[Dict[str, Any]], plots_dir: Path) -> None:
    if not HAS_MATPLOTLIB or not history:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train_loss")
    axis.plot(epochs, [row["valid_loss"] for row in history], marker="o", label="valid_loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("CLAM Virchow2 Advanced Severity4 Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "training_loss.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, [row["valid_qwk"] for row in history], marker="o", label="valid_qwk")
    axis.plot(epochs, [row["valid_macro_f1"] for row in history], marker="o", label="valid_macro_f1")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric")
    axis.set_title("CLAM Virchow2 Advanced Severity4 Validation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "validation_qwk_f1.png", dpi=200)
    plt.close(figure)


def checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    best_epoch: int,
    config: Dict[str, Any],
    history: List[Dict[str, Any]],
    class_weights: torch.Tensor,
    train_distribution: Dict[str, int],
    valid_distribution: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric_name": "valid_qwk",
        "best_metric_value": float(best_metric),
        "best_epoch": int(best_epoch),
        "config": dict(config),
        "history": list(history),
        "class_weights": class_weights.detach().cpu().numpy().astype(float).tolist(),
        "train_distribution": train_distribution,
        "valid_distribution": valid_distribution,
        "created_at": utc_now(),
    }


def dry_run(config: Dict[str, Any], device: torch.device) -> None:
    train_dataset = build_dataset(config, "train", config.get("max_train"))
    valid_dataset = build_dataset(config, "valid", config.get("max_valid"))
    model = build_model(config).to(device)
    sample = train_dataset[0]
    with torch.inference_mode():
        output = model([sample["features"].to(device)])
    total, trainable = count_parameters(model)
    print("[DRY-RUN] CLAM + Virchow2 advanced severity4")
    print(f"train_wsi={len(train_dataset)} valid_wsi={len(valid_dataset)}")
    print(f"sample_slide_id={sample['slide_id']}")
    print(f"features_shape={tuple(sample['features'].shape)}")
    print(f"label={sample['label']} logits_shape={tuple(output['logits'].shape)}")
    print(f"parameters_total={total} parameters_trainable={trainable}")
    print(f"output_root={config['output_root']}")


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    validate_config(config)
    set_seed(int(config["random_seed"]))
    device = resolve_device(str(config["device"]))

    if args.dry_run:
        dry_run(config, device)
        return

    create_directories(config)
    train_dataset = build_dataset(config, "train", config.get("max_train"))
    valid_dataset = build_dataset(config, "valid", config.get("max_valid"))
    train_labels = train_dataset.load_labels()
    valid_labels = valid_dataset.load_labels()
    train_distribution = label_distribution(train_labels)
    valid_distribution = label_distribution(valid_labels)
    class_weights = compute_class_weights(train_labels).to(device)

    train_loader = build_loader(train_dataset, config, device, shuffle=True)
    valid_loader = build_loader(valid_dataset, config, device, shuffle=False)
    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    amp_enabled = bool(config["mixed_precision"] and device.type == "cuda")

    checkpoints_dir = Path(config["checkpoints_dir"])
    metrics_dir = Path(config["metrics_dir"])
    plots_dir = Path(config["plots_dir"])
    best_model_path = checkpoints_dir / "best_model.pt"
    last_checkpoint_path = checkpoints_dir / "last_checkpoint.pt"

    start_epoch = 1
    best_metric = float("-inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    early_stopping_counter = 0
    if bool(config["resume"]):
        if last_checkpoint_path.is_file():
            checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric = float(checkpoint.get("best_metric_value", best_metric))
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            history = list(checkpoint.get("history", []))
            print(f"[INFO] Resuming training from epoch {start_epoch}")
        else:
            print("[INFO] resume=true, pero no existe last_checkpoint.pt; inicio desde cero.")

    total, trainable = count_parameters(model)
    atomic_json_save(
        {
            "created_at": utc_now(),
            "config": config,
            "device": str(device),
            "train_wsi": len(train_dataset),
            "valid_wsi": len(valid_dataset),
            "train_distribution": train_distribution,
            "valid_distribution": valid_distribution,
            "class_weights": class_weights.detach().cpu().numpy().astype(float).tolist(),
            "parameters_total": total,
            "parameters_trainable": trainable,
        },
        metrics_dir / "run_metadata.json",
    )
    print(f"[INFO] Train WSI: {len(train_dataset)}")
    print(f"[INFO] Valid WSI: {len(valid_dataset)}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Class weights: {class_weights.detach().cpu().numpy().round(6).tolist()}")
    print(f"[INFO] Parameters: {total} total, {trainable} trainable")

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp=amp_enabled,
        )
        valid_metrics = evaluate(
            model,
            valid_loader,
            criterion,
            device,
            amp=amp_enabled,
        )
        current_metric = float(valid_metrics["qwk"])
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            "valid_qwk": valid_metrics["qwk"],
            "is_best": int(is_best),
        }
        history.append(row)
        save_history(history, metrics_dir)
        atomic_json_save(valid_metrics, metrics_dir / "valid_metrics.json")

        payload = checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_metric=best_metric,
            best_epoch=best_epoch,
            config=config,
            history=history,
            class_weights=class_weights,
            train_distribution=train_distribution,
            valid_distribution=valid_distribution,
        )
        atomic_torch_save(payload, last_checkpoint_path)
        if is_best:
            atomic_torch_save(payload, best_model_path)
        if bool(config["save_epoch_checkpoints"]):
            atomic_torch_save(payload, checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt")
        save_plots(history, plots_dir)

        print(
            f"Epoch {epoch:03d}/{config['epochs']} | train_loss={train_loss:.4f} "
            f"| valid_loss={valid_metrics['loss']:.4f} | valid_qwk={valid_metrics['qwk']:.4f} "
            f"| valid_macro_f1={valid_metrics['macro_f1']:.4f} | best={'YES' if is_best else 'NO'}"
        )
        if early_stopping_counter >= int(config["early_stopping_patience"]):
            print("[INFO] Early stopping activado.")
            break


if __name__ == "__main__":
    main()
