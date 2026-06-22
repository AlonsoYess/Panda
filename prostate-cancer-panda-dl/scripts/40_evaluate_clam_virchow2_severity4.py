"""Final test evaluation for CLAM + Virchow2 severity 4-class.

This script only evaluates the already trained severity model. It does not
train, overwrite checkpoints, or touch previous binary/ISUP experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
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
        precision_score,
        recall_score,
    )
except Exception as exc:  # pragma: no cover - only reached when sklearn is absent.
    raise SystemExit(
        "[ERROR] scikit-learn es requerido para la evaluacion final severity 4-class. "
        "Instala scikit-learn antes de ejecutar este script."
    ) from exc

try:  # matplotlib is optional; CSV/JSON outputs remain the source of truth.
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - depends on local environment.
    plt = None
    HAS_MATPLOTLIB = False

from src.mil.clam_multiclass import CLAMMulticlass
from src.mil.virchow2_severity4_dataset import (
    DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT,
    EXPECTED_EMBEDDING_DIM,
    Virchow2Severity4Dataset,
    virchow2_severity4_bag_collate_fn,
)

DEFAULT_OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/PANDA_PROSTATE/outputs/clam_virchow2_severity4"
)
DEFAULT_ENTREGABLES_DIR = Path(
    "/content/drive/MyDrive/PANDA_PROSTATE/entregables/"
    "clam_virchow2_severity4_resultados"
)
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
        description="Evaluacion final test de CLAM + Virchow2 severity 4-class."
    )
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_VIRCHOW2_EMBEDDINGS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "checkpoints" / "best_model.pt",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay GPU CUDA disponible.")
    return device


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    metrics_dir = Path(output_root) / "metrics"
    entregables_dir = DEFAULT_ENTREGABLES_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    entregables_dir.mkdir(parents=True, exist_ok=True)
    return {
        "metrics_dir": metrics_dir,
        "entregables_dir": entregables_dir,
    }


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"El checkpoint no contiene un diccionario valido: {path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Falta model_state_dict en checkpoint: {path}")
    return checkpoint


def build_loader(
    dataset: Virchow2Severity4Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
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
    logits = model(features)["logits"]
    return logits, labels


def evaluate(
    *,
    model: CLAMMulticlass,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    slide_ids: list[str] = []
    isup_all: list[int] = []
    labels_all: list[int] = []
    preds_all: list[int] = []
    probs_all: list[list[float]] = []
    gleason_scores: list[Any] = []
    cancer_labels: list[Any] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluando test severity", leave=False):
            logits, labels = forward_batch(model, batch, device)
            loss = criterion(logits.float(), labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Loss no finita durante evaluacion: {loss}")

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            metadata = batch["metadata"]

            losses.append(float(loss.detach().cpu().item()))
            slide_ids.extend([str(slide_id) for slide_id in batch["slide_ids"]])
            isup_all.extend(batch["isup_grades"].detach().cpu().numpy().astype(int).tolist())
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            preds_all.extend(predictions.detach().cpu().numpy().astype(int).tolist())
            probs_all.extend(probabilities.detach().cpu().numpy().astype(float).tolist())
            gleason_scores.extend([item.get("gleason_score") for item in metadata])
            cancer_labels.extend([item.get("cancer_label") for item in metadata])

    if not losses:
        raise RuntimeError("No se proceso ningun batch de evaluacion.")

    return {
        "loss": float(sum(losses) / len(losses)),
        "slide_ids": slide_ids,
        "isup_grades": isup_all,
        "labels": labels_all,
        "predictions": preds_all,
        "probabilities": probs_all,
        "gleason_scores": gleason_scores,
        "cancer_labels": cancer_labels,
    }


def severity_metrics(labels: list[int], predictions: list[int], loss: float) -> dict[str, Any]:
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
        "test_loss": float(loss),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "test_macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "test_weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "test_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "test_qwk": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "classification_report": report,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def binary_derived_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    true_binary = [0 if int(label) == 0 else 1 for label in labels]
    pred_binary = [0 if int(label) == 0 else 1 for label in predictions]
    matrix = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel().astype(int).tolist()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    return {
        "binary_accuracy": float(accuracy_score(true_binary, pred_binary)),
        "binary_precision": float(
            precision_score(true_binary, pred_binary, zero_division=0)
        ),
        "binary_recall_sensitivity": float(
            recall_score(true_binary, pred_binary, zero_division=0)
        ),
        "binary_specificity": None if specificity is None else float(specificity),
        "binary_f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def predictions_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, slide_id in enumerate(results["slide_ids"]):
        y_true = int(results["labels"][index])
        y_pred = int(results["predictions"][index])
        probabilities = results["probabilities"][index]
        row = {
            "slide_id": slide_id,
            "y_true_isup": int(results["isup_grades"][index]),
            "y_true_severity": y_true,
            "y_pred_severity": y_pred,
            "prob_severity_0": float(probabilities[0]),
            "prob_severity_1": float(probabilities[1]),
            "prob_severity_2": float(probabilities[2]),
            "prob_severity_3": float(probabilities[3]),
            "y_true_binary": 0 if y_true == 0 else 1,
            "y_pred_binary": 0 if y_pred == 0 else 1,
            "gleason_score": results["gleason_scores"][index],
            "cancer_label": results["cancer_labels"][index],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_confusion_matrix_csv(matrix: np.ndarray, path: Path, labels: list[str]) -> None:
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "true"
    frame.to_csv(path)


def save_binary_confusion_matrix_csv(metrics: Dict[str, Any], path: Path) -> None:
    matrix = metrics["confusion_matrix"]
    frame = pd.DataFrame(
        [
            [matrix["tn"], matrix["fp"]],
            [matrix["fn"], matrix["tp"]],
        ],
        index=["true_0_no_cancer", "true_1_cancer"],
        columns=["pred_0_no_cancer", "pred_1_cancer"],
    )
    frame.index.name = "true"
    frame.to_csv(path)


def save_confusion_matrix_png(
    matrix: np.ndarray,
    path: Path,
    *,
    labels: list[str],
    title: str,
) -> bool:
    if not HAS_MATPLOTLIB:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(title)
    axis.set_xlabel("Prediccion")
    axis.set_ylabel("Etiqueta real")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = int(matrix[row_index, col_index])
            color = "white" if value > threshold else "black"
            axis.text(col_index, row_index, str(value), ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def copy_to_entregables(files: List[Path], entregables_dir: Path) -> None:
    entregables_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        if source.is_file():
            shutil.copy2(source, entregables_dir / source.name)


def print_checklist() -> None:
    print("\nCHECKLIST FINAL EVALUACION TEST SEVERITY 4-CLASS")
    checks = [
        "Checkpoint best_model.pt cargado",
        "Dataset test cargado",
        "Modelo en modo eval",
        "Softmax + argmax aplicado",
        "Metricas severity 4-class calculadas",
        "Matriz de confusion 4x4 generada",
        "Metricas binarias derivadas calculadas",
        "Predicciones guardadas",
        "Resultados guardados en metrics",
        "Resultados copiados/guardados en entregables",
        "Evaluacion final de test completada",
    ]
    for item in checks:
        print(f"[OK] {item}")


def run(args: argparse.Namespace) -> int:
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size debe ser mayor o igual a 1.")
    if str(args.split) != "test":
        print(
            f"[WARN] --split='{args.split}' recibido. El uso oficial esperado para esta "
            "evaluacion final es --split test."
        )

    device = resolve_device(args.device)
    paths = ensure_dirs(Path(args.output_root))
    metrics_dir = paths["metrics_dir"]
    entregables_dir = paths["entregables_dir"]

    checkpoint = load_checkpoint(args.checkpoint, device)
    best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch"))
    best_metric_name = checkpoint.get("best_metric_name", "valid_qwk")
    best_metric_value = checkpoint.get("best_metric_value")

    model = CLAMMulticlass(input_dim=EXPECTED_EMBEDDING_DIM, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = Virchow2Severity4Dataset(
        embeddings_root=args.embeddings_root,
        split=args.split,
        validate_on_init=False,
    )
    if str(args.split) == "test" and len(dataset) != 1077:
        print(
            f"[WARN] Se esperaban 1077 WSI en test, pero se detectaron {len(dataset)}. "
            "Continuo la evaluacion con los archivos disponibles."
        )

    loader = build_loader(
        dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    results = evaluate(model=model, loader=loader, criterion=criterion, device=device)

    labels = results["labels"]
    predictions = results["predictions"]
    severity = severity_metrics(labels, predictions, results["loss"])
    binary = binary_derived_metrics(labels, predictions)
    severity_matrix = np.asarray(severity["confusion_matrix"], dtype=int)
    binary_matrix = np.array(
        [
            [binary["confusion_matrix"]["tn"], binary["confusion_matrix"]["fp"]],
            [binary["confusion_matrix"]["fn"], binary["confusion_matrix"]["tp"]],
        ],
        dtype=int,
    )

    predictions_df = predictions_dataframe(results)
    report_df = pd.DataFrame(severity["classification_report"]).transpose()
    metrics_csv_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in severity.items()
                if key not in {"classification_report", "confusion_matrix"}
            }
        ]
    )

    paths_to_copy: list[Path] = []
    severity_json_path = metrics_dir / "test_metrics_severity4.json"
    severity_csv_path = metrics_dir / "test_metrics_severity4.csv"
    predictions_path = metrics_dir / "test_predictions_severity4.csv"
    severity_matrix_path = metrics_dir / "test_confusion_matrix_severity4_4x4.csv"
    report_path = metrics_dir / "test_classification_report_severity4.csv"
    binary_json_path = metrics_dir / "test_metrics_binary_derived.json"
    binary_matrix_path = metrics_dir / "test_confusion_matrix_binary_derived.csv"
    severity_png_path = metrics_dir / "test_confusion_matrix_severity4_4x4.png"
    binary_png_path = metrics_dir / "test_confusion_matrix_binary_derived.png"

    save_json(
        {
            **severity,
            "checkpoint": str(args.checkpoint),
            "best_epoch": best_epoch,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "split": str(args.split),
            "n_wsi": len(dataset),
            "class_mapping": {
                "0": "no_cancer",
                "1": "low_grade",
                "2": "intermediate",
                "3": "high_grade",
            },
        },
        severity_json_path,
    )
    metrics_csv_df.to_csv(severity_csv_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    save_confusion_matrix_csv(
        severity_matrix,
        severity_matrix_path,
        labels=SEVERITY_NAMES,
    )
    report_df.to_csv(report_path)
    save_json(binary, binary_json_path)
    save_binary_confusion_matrix_csv(binary, binary_matrix_path)

    paths_to_copy.extend(
        [
            severity_json_path,
            predictions_path,
            severity_matrix_path,
            report_path,
            binary_json_path,
            binary_matrix_path,
        ]
    )

    if save_confusion_matrix_png(
        severity_matrix,
        severity_png_path,
        labels=["No cancer", "Low", "Intermediate", "High"],
        title="CLAM + Virchow2 severity 4-class - Test",
    ):
        paths_to_copy.append(severity_png_path)
    if save_confusion_matrix_png(
        binary_matrix,
        binary_png_path,
        labels=["No cancer", "Cancer"],
        title="CLAM + Virchow2 severity derivado binario - Test",
    ):
        paths_to_copy.append(binary_png_path)

    copy_to_entregables(paths_to_copy, entregables_dir)

    print("\n=== Evaluacion final CLAM + Virchow2 severity 4-class ===")
    print(f"checkpoint cargado: {args.checkpoint}")
    print(f"best_epoch checkpoint: {best_epoch}")
    print(f"best {best_metric_name} checkpoint: {best_metric_value}")
    print(f"cantidad WSI {args.split}: {len(dataset)}")
    print(f"test_loss: {severity['test_loss']:.6f}")
    print(f"test_accuracy: {severity['test_accuracy']:.6f}")
    print(f"test_macro_f1: {severity['test_macro_f1']:.6f}")
    print(f"test_weighted_f1: {severity['test_weighted_f1']:.6f}")
    print(f"test_balanced_accuracy: {severity['test_balanced_accuracy']:.6f}")
    print(f"test_qwk: {severity['test_qwk']:.6f}")
    print("\nMatriz de confusion severity 4x4:")
    print(severity_matrix)
    print("\nMetricas binarias derivadas:")
    for key, value in binary.items():
        print(f"{key}: {value}")

    print("\nArchivos guardados en metrics:")
    for path in [
        severity_json_path,
        severity_csv_path,
        predictions_path,
        severity_matrix_path,
        report_path,
        binary_json_path,
        binary_matrix_path,
    ]:
        print(f"- {path}")
    if severity_png_path.is_file():
        print(f"- {severity_png_path}")
    if binary_png_path.is_file():
        print(f"- {binary_png_path}")
    print(f"\nEntregables: {entregables_dir}")
    print_checklist()
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[ERROR] Evaluacion severity detenida: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
