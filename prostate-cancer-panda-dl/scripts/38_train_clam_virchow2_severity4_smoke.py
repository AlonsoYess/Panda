"""Short training smoke test for CLAM + Virchow2 severity 4-class.

This script intentionally does not save checkpoints, plots, or official
metrics. It only verifies that a small severity 4-class training loop runs.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.clam_multiclass import CLAMMulticlass
from src.mil.virchow2_severity4_dataset import (
    DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT,
    EXPECTED_EMBEDDING_DIM,
    VALID_SEVERITY_LABELS,
    Virchow2Severity4Dataset,
    virchow2_severity4_bag_collate_fn,
)

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
    )

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

NUM_CLASSES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento corto smoke de CLAM + Virchow2 severity 4-class."
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT,
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=30)
    parser.add_argument("--max-valid-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay GPU CUDA disponible.")
    return device


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def label_distribution(labels: Iterable[int]) -> dict[int, int]:
    counts = Counter(int(label) for label in labels)
    return {
        class_id: int(counts.get(class_id, 0))
        for class_id in sorted(VALID_SEVERITY_LABELS)
    }


def compute_class_weights(labels: List[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    weights = []
    for class_id in range(num_classes):
        count = counts.get(class_id, 0)
        weight = 0.0 if count == 0 else total / (num_classes * count)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def build_loader(
    dataset: Virchow2Severity4Dataset,
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
        collate_fn=virchow2_severity4_bag_collate_fn,
    )


def forward_batch(
    model: CLAMMulticlass,
    batch: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = [item.to(device, non_blocking=device.type == "cuda") for item in batch["features"]]
    labels = batch["labels"].to(device, non_blocking=device.type == "cuda").long()
    output = model(features)
    logits = output["logits"]
    return logits, labels


def train_smoke_epoch(
    *,
    model: CLAMMulticlass,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int,
    epoch: int,
) -> float:
    model.train()
    losses: list[float] = []

    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"Train smoke severity epoch {epoch}", leave=False),
        start=1,
    ):
        if batch_index > int(max_batches):
            break

        optimizer.zero_grad(set_to_none=True)
        logits, labels = forward_batch(model, batch, device)
        loss = criterion(logits.float(), labels)
        require(torch.isfinite(loss).item(), f"Loss no finita en train batch {batch_index}: {loss}")
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        if batch_index == 1 or batch_index % 5 == 0:
            print(f"[TRAIN] epoch={epoch} batch={batch_index} loss={loss_value:.6f}")

    require(losses, "No se proceso ningun batch de entrenamiento.")
    return float(sum(losses) / len(losses))


def quick_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | None]:
    if not labels:
        raise RuntimeError("No hay labels de validacion para calcular metricas.")

    if SKLEARN_AVAILABLE:
        return {
            "valid_accuracy": float(accuracy_score(labels, predictions)),
            "valid_macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "valid_weighted_f1": float(
                f1_score(labels, predictions, average="weighted", zero_division=0)
            ),
            "valid_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "valid_qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        }

    correct = sum(int(y_true == y_pred) for y_true, y_pred in zip(labels, predictions))
    accuracy = correct / max(len(labels), 1)
    print("[WARN] scikit-learn no esta disponible; solo se calculara accuracy.")
    return {
        "valid_accuracy": float(accuracy),
        "valid_macro_f1": None,
        "valid_weighted_f1": None,
        "valid_balanced_accuracy": None,
        "valid_qwk": None,
    }


def validate_smoke(
    *,
    model: CLAMMulticlass,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    labels_all: list[int] = []
    predictions_all: list[int] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(loader, desc="Valid smoke severity", leave=False),
            start=1,
        ):
            if batch_index > int(max_batches):
                break

            logits, labels = forward_batch(model, batch, device)
            loss = criterion(logits.float(), labels)
            require(
                torch.isfinite(loss).item(),
                f"Loss no finita en valid batch {batch_index}: {loss}",
            )
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            losses.append(float(loss.detach().cpu().item()))
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            predictions_all.extend(predictions.detach().cpu().numpy().astype(int).tolist())

    require(losses, "No se proceso ningun batch de validacion.")
    metrics = quick_metrics(labels_all, predictions_all)
    metrics["valid_loss"] = float(sum(losses) / len(losses))
    return metrics


def print_checklist(status: dict[str, bool]) -> bool:
    print("\nCHECKLIST FINAL TRAINING SMOKE SEVERITY 4-CLASS")
    ordered = [
        "Dataset train cargado",
        "Dataset valid cargado",
        "Class weights calculados",
        "Modelo CLAMMulticlass creado con num_classes=4",
        "CrossEntropyLoss configurada",
        "Optimizer configurado",
        "Entrenamiento corto ejecutado",
        "Validacion corta ejecutada",
        "Metricas rapidas calculadas",
        "Listo para entrenamiento completo severity 4-class",
    ]
    all_ok = True
    for item in ordered:
        ok = bool(status.get(item, False))
        all_ok = all_ok and ok
        print(f"[{'OK' if ok else 'ERROR'}] {item}")
    return all_ok


def run(args: argparse.Namespace) -> int:
    status = {
        "Dataset train cargado": False,
        "Dataset valid cargado": False,
        "Class weights calculados": False,
        "Modelo CLAMMulticlass creado con num_classes=4": False,
        "CrossEntropyLoss configurada": False,
        "Optimizer configurado": False,
        "Entrenamiento corto ejecutado": False,
        "Validacion corta ejecutada": False,
        "Metricas rapidas calculadas": False,
        "Listo para entrenamiento completo severity 4-class": False,
    }
    device = resolve_device(args.device)

    print("TRAINING SMOKE: CLAM + Virchow2 severity 4-class")
    print(f"embeddings_root: {args.embeddings_root}")
    print(f"device detectado: {device}")
    print("modo: prueba corta, sin checkpoints, sin resultados oficiales")

    train_dataset = Virchow2Severity4Dataset(
        embeddings_root=args.embeddings_root,
        split="train",
        validate_on_init=False,
    )
    status["Dataset train cargado"] = True
    valid_dataset = Virchow2Severity4Dataset(
        embeddings_root=args.embeddings_root,
        split="valid",
        validate_on_init=False,
    )
    status["Dataset valid cargado"] = True

    print(f"WSI train: {len(train_dataset)}")
    print(f"WSI valid: {len(valid_dataset)}")

    print("\nCargando etiquetas severity para distribucion y class weights...")
    train_labels = train_dataset.load_labels()
    valid_labels = valid_dataset.load_labels()
    print(f"Distribucion severity train: {label_distribution(train_labels)}")
    print(f"Distribucion severity valid: {label_distribution(valid_labels)}")
    class_weights = compute_class_weights(train_labels).to(device)
    status["Class weights calculados"] = True
    print(f"class_weights: {[round(float(value), 6) for value in class_weights.detach().cpu()]}")

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
    status["Modelo CLAMMulticlass creado con num_classes=4"] = True
    print(f"Modelo creado: CLAMMulticlass(input_dim=1280, num_classes=4)")
    print(f"Parametros aproximados: total={total_params:,}, trainable={trainable_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    status["CrossEntropyLoss configurada"] = True
    optimizer = AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    status["Optimizer configurado"] = True
    print(f"Optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay})")

    train_losses: list[float] = []
    for epoch in range(1, int(args.epochs) + 1):
        print(f"\nepoch {epoch}")
        train_loss = train_smoke_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=int(args.max_train_batches),
            epoch=epoch,
        )
        train_losses.append(train_loss)
        print(f"train loss promedio: {train_loss:.6f}")
    status["Entrenamiento corto ejecutado"] = True

    metrics = validate_smoke(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        device=device,
        max_batches=int(args.max_valid_batches),
    )
    status["Validacion corta ejecutada"] = True
    status["Metricas rapidas calculadas"] = True

    print("\nVALIDACION CORTA")
    print(f"valid loss promedio: {metrics['valid_loss']:.6f}")
    print(f"valid accuracy: {metrics['valid_accuracy']}")
    print(f"valid macro-f1: {metrics['valid_macro_f1']}")
    print(f"valid weighted-f1: {metrics['valid_weighted_f1']}")
    print(f"valid balanced accuracy: {metrics['valid_balanced_accuracy']}")
    print(f"valid qwk: {metrics['valid_qwk']}")

    finite_train = all(math.isfinite(value) for value in train_losses)
    finite_valid = math.isfinite(float(metrics["valid_loss"]))
    status["Listo para entrenamiento completo severity 4-class"] = (
        finite_train
        and finite_valid
        and status["Entrenamiento corto ejecutado"]
        and status["Validacion corta ejecutada"]
        and status["Metricas rapidas calculadas"]
    )

    ok = print_checklist(status)
    print("\nComando para ejecutar:")
    print(
        'python scripts/38_train_clam_virchow2_severity4_smoke.py '
        '--embeddings-root "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings" '
        "--epochs 1 --batch-size 4 --max-train-batches 30 --max-valid-batches 20"
    )
    return 0 if ok else 1


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[ERROR] Training smoke severity detenido: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
