"""Smoke test for CLAM + Virchow2 ISUP multiclass data and forward pass.

This script does not train, save checkpoints, or generate final metrics.
It only validates that the multiclass dataset, DataLoader, CLAM forward pass
and CrossEntropyLoss work together.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.clam_multiclass import CLAMMulticlass
from src.mil.virchow2_isup_dataset import (
    DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT,
    EXPECTED_EMBEDDING_DIM,
    VALID_ISUP_GRADES,
    Virchow2ISUPDataset,
    virchow2_isup_bag_collate_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test de CLAM + Virchow2 ISUP multiclass 0-5."
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT,
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--compute-class-weights",
        action="store_true",
        help="Carga labels de train e imprime pesos inversos; no entrena.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay una GPU CUDA disponible.")
    return device


def ok(condition: bool, message: str, errors: list[str]) -> None:
    if condition:
        print(f"[OK] {message}")
    else:
        print(f"[ERROR] {message}")
        errors.append(message)


def summarize_batch(batch: dict[str, Any]) -> None:
    print("\nBATCH TRAIN")
    print(f"- cantidad de WSI en batch: {len(batch['features'])}")
    print(f"- slide_id: {batch['slide_ids']}")
    for index, features in enumerate(batch["features"]):
        print(
            f"- WSI {index}: features.shape={tuple(features.shape)}, "
            f"n_tiles={features.shape[0]}, embedding_dim={features.shape[1]}"
        )
    print(f"- labels: {batch['labels'].tolist()}")
    print(f"- labels dtype: {batch['labels'].dtype}")
    print(f"- labels min: {int(batch['labels'].min().item())}")
    print(f"- labels max: {int(batch['labels'].max().item())}")


def compute_class_weights(dataset: Virchow2ISUPDataset) -> torch.Tensor:
    labels = dataset.load_labels()
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for class_index in sorted(VALID_ISUP_GRADES):
        count = counts.get(class_index, 0)
        weight = 0.0 if count == 0 else total / (len(VALID_ISUP_GRADES) * count)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def first_batch(loader: Iterable[dict[str, Any]]) -> dict[str, Any]:
    try:
        return next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("El DataLoader train no produjo batches.") from exc


def run(args: argparse.Namespace) -> int:
    errors: list[str] = []
    device = resolve_device(args.device)

    print("SMOKE TEST: CLAM + Virchow2 ISUP multiclass")
    print(f"embeddings_root: {args.embeddings_root}")
    print(f"device: {device}")
    print("modo: sin entrenamiento, sin checkpoints")

    try:
        train_dataset = Virchow2ISUPDataset(
            embeddings_root=args.embeddings_root,
            split="train",
            max_items=args.max_train,
            validate_on_init=False,
        )
        ok(True, "Dataset train ISUP creado", errors)
    except Exception as exc:
        print(f"[ERROR] Dataset train ISUP creado: {type(exc).__name__}: {exc}")
        return 1

    try:
        valid_dataset = Virchow2ISUPDataset(
            embeddings_root=args.embeddings_root,
            split="valid",
            max_items=args.max_valid,
            validate_on_init=False,
        )
        ok(True, "Dataset valid ISUP creado", errors)
    except Exception as exc:
        print(f"[ERROR] Dataset valid ISUP creado: {type(exc).__name__}: {exc}")
        return 1

    print(f"- train WSI: {len(train_dataset)}")
    print(f"- valid WSI: {len(valid_dataset)}")

    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=virchow2_isup_bag_collate_fn,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=virchow2_isup_bag_collate_fn,
        )
        _ = valid_loader
        ok(True, "DataLoader train creado", errors)
    except Exception as exc:
        print(f"[ERROR] DataLoader train creado: {type(exc).__name__}: {exc}")
        return 1

    try:
        batch = first_batch(train_loader)
        summarize_batch(batch)
        ok(True, "Batch cargado correctamente", errors)
    except Exception as exc:
        print(f"[ERROR] Batch cargado correctamente: {type(exc).__name__}: {exc}")
        return 1

    labels = batch["labels"]
    labels_ok = (
        labels.dtype == torch.long
        and int(labels.min().item()) >= 0
        and int(labels.max().item()) <= 5
    )
    ok(labels_ok, "Labels torch.long en rango 0-5", errors)

    feature_dims_ok = all(
        isinstance(features, torch.Tensor)
        and features.ndim == 2
        and int(features.shape[1]) == EXPECTED_EMBEDDING_DIM
        for features in batch["features"]
    )
    ok(feature_dims_ok, "Features batch con embedding_dim=1280", errors)

    try:
        model = CLAMMulticlass(
            input_dim=EXPECTED_EMBEDDING_DIM,
            num_classes=6,
            hidden_dim=int(args.hidden_dim),
            attention_dim=int(args.attention_dim),
            dropout=float(args.dropout),
        ).to(device)
        ok(model.num_classes == 6, "CLAMMulticlass instanciado con num_classes=6", errors)
    except Exception as exc:
        print(f"[ERROR] CLAMMulticlass instanciado con num_classes=6: {type(exc).__name__}: {exc}")
        return 1

    try:
        feature_bags = [features.to(device) for features in batch["features"]]
        labels_device = labels.to(device)
        with torch.inference_mode():
            output = model(feature_bags)
        logits = output["logits"]
        ok(True, "Forward pass ejecutado", errors)
    except Exception as exc:
        print(f"[ERROR] Forward pass ejecutado: {type(exc).__name__}: {exc}")
        return 1

    print("\nFORWARD PASS")
    print(f"- logits shape: {tuple(logits.shape)}")
    print(f"- logits dtype: {logits.dtype}")
    logits_shape_ok = tuple(logits.shape) == (len(batch["features"]), 6)
    ok(logits_shape_ok, "Logits shape [batch_size, 6]", errors)

    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    row_sums = probabilities.sum(dim=1)
    softmax_shape_ok = tuple(probabilities.shape) == tuple(logits.shape)
    row_sums_ok = torch.allclose(
        row_sums.detach().cpu(),
        torch.ones_like(row_sums.detach().cpu()),
        atol=1e-5,
    )
    predictions_ok = (
        int(predictions.min().detach().cpu().item()) >= 0
        and int(predictions.max().detach().cpu().item()) <= 5
    )
    print(f"- softmax shape: {tuple(probabilities.shape)}")
    print(f"- suma probabilidades por fila: {row_sums.detach().cpu().tolist()}")
    print(f"- predicciones argmax: {predictions.detach().cpu().tolist()}")
    ok(softmax_shape_ok and row_sums_ok and predictions_ok, "Softmax + argmax funcionando", errors)

    try:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.float(), labels_device.long())
        loss_finite = bool(torch.isfinite(loss).detach().cpu().item())
        print(f"- CrossEntropyLoss: {float(loss.detach().cpu().item()):.6f}")
        ok(loss_finite, "CrossEntropyLoss calculada", errors)
    except Exception as exc:
        print(f"[ERROR] CrossEntropyLoss calculada: {type(exc).__name__}: {exc}")
        return 1

    if args.compute_class_weights:
        print("\nCLASS WEIGHTS TRAIN (solo inspeccion, no entrenamiento)")
        class_weights = compute_class_weights(train_dataset)
        print(f"- class_weights: {class_weights.tolist()}")

    print("\nCHECKLIST FINAL SMOKE TEST")
    final_checks = [
        "Dataset train ISUP creado",
        "Dataset valid ISUP creado",
        "DataLoader train creado",
        "Batch cargado correctamente",
        "Labels torch.long en rango 0-5",
        "CLAMMulticlass instanciado con num_classes=6",
        "Forward pass ejecutado",
        "Logits shape [batch_size, 6]",
        "Softmax + argmax funcionando",
        "CrossEntropyLoss calculada",
        "Listo para crear entrenamiento multiclase",
    ]
    failed = set(errors)
    for message in final_checks:
        if message == "Listo para crear entrenamiento multiclase":
            check_ok = not failed
        else:
            check_ok = message not in failed
        print(f"[{'OK' if check_ok else 'ERROR'}] {message}")

    print("\nComando para ejecutar:")
    print(
        'python scripts/32_smoke_clam_virchow2_isup_multiclass.py '
        '--embeddings-root "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings"'
    )
    return 0 if not errors else 1


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[ERROR] Smoke test detenido: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
