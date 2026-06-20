"""Full training for CLAM + Virchow2 ISUP 0-5 multiclass classification.

This script trains the new multiclass experiment only. It does not touch the
existing binary scripts or binary output directories.
"""

from __future__ import annotations

import argparse
import json
import math
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
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
    )
except Exception as exc:  # pragma: no cover - exercised only when sklearn is absent.
    raise SystemExit(
        "[ERROR] scikit-learn es requerido para el entrenamiento completo "
        "multiclase. Instala scikit-learn antes de ejecutar este script."
    ) from exc

from src.mil.clam_multiclass import CLAMMulticlass
from src.mil.virchow2_isup_dataset import (
    DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT,
    EXPECTED_EMBEDDING_DIM,
    VALID_ISUP_GRADES,
    Virchow2ISUPDataset,
    virchow2_isup_bag_collate_fn,
)

DEFAULT_OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/PANDA_PROSTATE/outputs/clam_virchow2_isup_multiclass"
)
NUM_CLASSES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento completo de CLAM + Virchow2 ISUP multiclass."
    )
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=("valid_qwk", "valid_macro_f1", "valid_balanced_accuracy"),
        default="valid_qwk",
    )
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay GPU CUDA disponible.")
    return device


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "output_root": Path(output_root),
        "checkpoints_dir": Path(output_root) / "checkpoints",
        "metrics_dir": Path(output_root) / "metrics",
        "logs_dir": Path(output_root) / "logs",
        "plots_dir": Path(output_root) / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


class Logger:
    """Simple tee logger for console and train_log.txt."""

    def __init__(self, path: Path, append: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self.file = path.open(mode, encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


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


def save_history(history: List[Dict[str, Any]], metrics_dir: Path) -> None:
    pd.DataFrame(history).to_csv(metrics_dir / "training_history.csv", index=False)
    atomic_json_save({"history": history}, metrics_dir / "training_history.json")


def label_distribution(labels: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {str(class_id): int(counts.get(class_id, 0)) for class_id in sorted(VALID_ISUP_GRADES)}


def compute_class_weights(labels: List[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    weights = []
    for class_id in range(num_classes):
        count = counts.get(class_id, 0)
        if count == 0:
            raise ValueError(f"No hay ejemplos de la clase ISUP {class_id} en train.")
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def build_loader(
    dataset: Virchow2ISUPDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(num_workers) > 0,
        collate_fn=virchow2_isup_bag_collate_fn,
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


def train_one_epoch(
    *,
    model: CLAMMulticlass,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None,
    epoch: int,
) -> float:
    model.train()
    losses: list[float] = []
    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"Train epoch {epoch}", leave=False),
        start=1,
    ):
        if max_batches is not None and batch_index > int(max_batches):
            break
        optimizer.zero_grad(set_to_none=True)
        logits, labels = forward_batch(model, batch, device)
        loss = criterion(logits.float(), labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Loss no finita en train epoch={epoch} batch={batch_index}: {loss}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise RuntimeError("No se proceso ningun batch de entrenamiento.")
    return float(sum(losses) / len(losses))


def compute_validation_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "valid_accuracy": float(accuracy_score(labels, predictions)),
        "valid_macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "valid_weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "valid_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "valid_qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
    }


def validate_one_epoch(
    *,
    model: CLAMMulticlass,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    labels_all: list[int] = []
    predictions_all: list[int] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="Valid", leave=False), start=1):
            if max_batches is not None and batch_index > int(max_batches):
                break
            logits, labels = forward_batch(model, batch, device)
            loss = criterion(logits.float(), labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Loss no finita en valid batch={batch_index}: {loss}")
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            losses.append(float(loss.detach().cpu().item()))
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            predictions_all.extend(predictions.detach().cpu().numpy().astype(int).tolist())
    if not losses:
        raise RuntimeError("No se proceso ningun batch de validacion.")
    metrics = compute_validation_metrics(labels_all, predictions_all)
    metrics["valid_loss"] = float(sum(losses) / len(losses))
    return metrics


def build_config(args: argparse.Namespace, paths: dict[str, Path], device: torch.device) -> Dict[str, Any]:
    return {
        "experiment_name": "clam_virchow2_isup_multiclass",
        "task": "isup_grade multiclass 0-5",
        "embeddings_root": str(args.embeddings_root),
        "output_root": str(paths["output_root"]),
        "checkpoints_dir": str(paths["checkpoints_dir"]),
        "metrics_dir": str(paths["metrics_dir"]),
        "logs_dir": str(paths["logs_dir"]),
        "plots_dir": str(paths["plots_dir"]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_workers": int(args.num_workers),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "resume": bool(args.resume),
        "selection_metric": str(args.selection_metric),
        "save_every_epoch": bool(args.save_every_epoch),
        "max_train_batches": args.max_train_batches,
        "max_valid_batches": args.max_valid_batches,
        "device": str(device),
        "input_dim": EXPECTED_EMBEDDING_DIM,
        "num_classes": NUM_CLASSES,
    }


def checkpoint_payload(
    *,
    epoch: int,
    model: CLAMMulticlass,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    best_metric_name: str,
    best_metric_value: float,
    best_epoch: int,
    config: Dict[str, Any],
    class_weights: torch.Tensor,
    train_distribution: Dict[str, int],
    valid_distribution: Dict[str, int],
    history: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "best_epoch": int(best_epoch),
        "config": dict(config),
        "class_weights": class_weights.detach().cpu().tolist(),
        "train_distribution": dict(train_distribution),
        "valid_distribution": dict(valid_distribution),
        "history": list(history),
        "random_seed": int(seed),
        "created_at": utc_now(),
    }


def load_resume_checkpoint(
    *,
    path: Path,
    model: CLAMMulticlass,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def run(args: argparse.Namespace) -> int:
    if int(args.epochs) < 1:
        raise ValueError("--epochs debe ser mayor o igual a 1.")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size debe ser mayor o igual a 1.")

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    paths = ensure_output_dirs(Path(args.output_root))
    config = build_config(args, paths, device)
    last_checkpoint = paths["checkpoints_dir"] / "last_checkpoint.pt"

    if args.resume and not last_checkpoint.is_file():
        raise FileNotFoundError(
            f"Se uso --resume pero no existe last_checkpoint.pt en: {last_checkpoint}"
        )
    if not args.resume and last_checkpoint.is_file():
        raise RuntimeError(
            "Ya existe checkpoints/last_checkpoint.pt. Para evitar sobrescribir avances, "
            "usa --resume o cambia --output-root."
        )

    logger = Logger(paths["logs_dir"] / "train_log.txt", append=bool(args.resume))
    try:
        logger.log("=== CLAM + Virchow2 ISUP multiclass training ===")
        logger.log(f"inicio: {utc_now()}")
        logger.log(f"device: {device}")
        logger.log(f"output_root: {paths['output_root']}")
        logger.log(f"embeddings_root: {args.embeddings_root}")

        train_dataset = Virchow2ISUPDataset(
            embeddings_root=args.embeddings_root,
            split="train",
            validate_on_init=False,
        )
        valid_dataset = Virchow2ISUPDataset(
            embeddings_root=args.embeddings_root,
            split="valid",
            validate_on_init=False,
        )
        logger.log(f"train WSI: {len(train_dataset)}")
        logger.log(f"valid WSI: {len(valid_dataset)}")

        logger.log("Cargando etiquetas para distribuciones y class weights...")
        train_labels = train_dataset.load_labels()
        valid_labels = valid_dataset.load_labels()
        train_distribution = label_distribution(train_labels)
        valid_distribution = label_distribution(valid_labels)
        logger.log(f"train_distribution: {train_distribution}")
        logger.log(f"valid_distribution: {valid_distribution}")

        class_weights = compute_class_weights(train_labels).to(device)
        class_weights_cpu = class_weights.detach().cpu()
        logger.log(
            "class_weights: "
            f"{[round(float(value), 6) for value in class_weights_cpu]}"
        )

        atomic_json_save(config, paths["metrics_dir"] / "config_used.json")
        atomic_json_save(
            {
                "train": train_distribution,
                "valid": valid_distribution,
            },
            paths["metrics_dir"] / "class_distribution.json",
        )
        atomic_json_save(
            {
                str(index): float(value)
                for index, value in enumerate(class_weights_cpu.tolist())
            },
            paths["metrics_dir"] / "class_weights.json",
        )

        train_loader = build_loader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            device=device,
        )
        valid_loader = build_loader(
            valid_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            device=device,
        )

        model = CLAMMulticlass(
            input_dim=EXPECTED_EMBEDDING_DIM,
            num_classes=NUM_CLASSES,
        ).to(device)
        total_params, trainable_params = count_parameters(model)
        logger.log(
            f"modelo: CLAMMulticlass(input_dim={EXPECTED_EMBEDDING_DIM}, "
            f"num_classes={NUM_CLASSES})"
        )
        logger.log(f"parametros: total={total_params:,}, trainable={trainable_params:,}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        start_epoch = 1
        best_metric_name = str(args.selection_metric)
        best_metric_value = float("-inf")
        best_epoch = 0
        history: List[Dict[str, Any]] = []
        stopped_by_early_stopping = False
        epochs_without_improvement = 0

        if args.resume:
            checkpoint = load_resume_checkpoint(
                path=last_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric_name = str(checkpoint.get("best_metric_name", best_metric_name))
            best_metric_value = float(checkpoint.get("best_metric_value", best_metric_value))
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            history = list(checkpoint.get("history", []))
            epochs_without_improvement = max(0, start_epoch - best_epoch - 1)
            logger.log(f"checkpoint cargado: {last_checkpoint}")
            logger.log(f"retomando desde epoch: {start_epoch}")
            logger.log(f"mejor metrica previa: {best_metric_name}={best_metric_value}")
            logger.log(f"mejor epoch previo: {best_epoch}")

        if start_epoch > int(args.epochs):
            logger.log(
                f"No hay epocas pendientes: start_epoch={start_epoch}, epochs={args.epochs}."
            )

        for epoch in range(start_epoch, int(args.epochs) + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                max_batches=args.max_train_batches,
                epoch=epoch,
            )
            valid_metrics = validate_one_epoch(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
                max_batches=args.max_valid_batches,
            )
            selection_value = float(valid_metrics[best_metric_name])
            scheduler.step(selection_value)
            current_lr = float(optimizer.param_groups[0]["lr"])
            is_best = selection_value > best_metric_value

            if is_best:
                best_metric_value = selection_value
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "valid_loss": float(valid_metrics["valid_loss"]),
                "valid_accuracy": float(valid_metrics["valid_accuracy"]),
                "valid_macro_f1": float(valid_metrics["valid_macro_f1"]),
                "valid_weighted_f1": float(valid_metrics["valid_weighted_f1"]),
                "valid_balanced_accuracy": float(valid_metrics["valid_balanced_accuracy"]),
                "valid_qwk": float(valid_metrics["valid_qwk"]),
                "lr": current_lr,
                "is_best": bool(is_best),
            }
            history.append(row)
            save_history(history, paths["metrics_dir"])

            payload = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                best_epoch=best_epoch,
                config=config,
                class_weights=class_weights_cpu,
                train_distribution=train_distribution,
                valid_distribution=valid_distribution,
                history=history,
                seed=int(args.seed),
            )
            atomic_torch_save(payload, paths["checkpoints_dir"] / "last_checkpoint.pt")
            logger.log(f"checkpoint guardado: {paths['checkpoints_dir'] / 'last_checkpoint.pt'}")

            if is_best:
                atomic_torch_save(payload, paths["checkpoints_dir"] / "best_model.pt")
                logger.log(f"best_model.pt actualizado en epoch {epoch}")
            if args.save_every_epoch:
                epoch_path = paths["checkpoints_dir"] / f"epoch_{epoch:03d}.pt"
                atomic_torch_save(payload, epoch_path)
                logger.log(f"checkpoint epoch guardado: {epoch_path}")

            best_text = "YES" if is_best else "NO"
            line = (
                f"Epoch {epoch:02d}/{int(args.epochs)} | "
                f"train_loss={train_loss:.6f} | "
                f"valid_loss={valid_metrics['valid_loss']:.6f} | "
                f"valid_acc={valid_metrics['valid_accuracy']:.6f} | "
                f"valid_macro_f1={valid_metrics['valid_macro_f1']:.6f} | "
                f"valid_weighted_f1={valid_metrics['valid_weighted_f1']:.6f} | "
                f"valid_balanced_acc={valid_metrics['valid_balanced_accuracy']:.6f} | "
                f"valid_qwk={valid_metrics['valid_qwk']:.6f} | "
                f"best={best_text}"
            )
            logger.log(line)

            if epochs_without_improvement >= int(args.patience):
                stopped_by_early_stopping = True
                logger.log(
                    f"early stopping activado en epoch {epoch}; "
                    f"best_epoch={best_epoch}, {best_metric_name}={best_metric_value:.6f}"
                )
                break

        summary = {
            "best_epoch": int(best_epoch),
            "best_metric_name": best_metric_name,
            "best_metric_value": float(best_metric_value),
            "total_epochs_run": len(history),
            "stopped_by_early_stopping": bool(stopped_by_early_stopping),
            "output_root": str(paths["output_root"]),
            "checkpoints_dir": str(paths["checkpoints_dir"]),
            "metrics_dir": str(paths["metrics_dir"]),
            "finished_at": utc_now(),
        }
        atomic_json_save(summary, paths["metrics_dir"] / "training_summary.json")
        logger.log(f"resumen final: {summary}")
        logger.log("Entrenamiento completo finalizado.")
        return 0
    finally:
        logger.close()


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[ERROR] Entrenamiento detenido: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
