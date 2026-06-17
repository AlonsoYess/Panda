"""Final validation and test evaluation for TransMIL + Virchow2."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.engine import compute_binary_metrics, find_youden_threshold, save_json, set_seed
from src.mil.plots import save_confusion_matrix, save_roc_curve
from src.mil.transmil import TransMILBinary
from src.mil.virchow2_dataset import (
    EXPECTED_EMBEDDING_DIM,
    EXPECTED_ENCODER_FAMILY,
    EXPECTED_ENCODER_NAME,
    Virchow2EmbeddingDataset,
    virchow2_bag_collate_fn,
)
from src.utils.provenance import (
    get_cuda_info,
    get_git_info,
    get_software_versions,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua best_model.pt de TransMIL + Virchow2 en valid y test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/transmil_virchow2_train_binary.yaml"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
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
        raise ValueError("La configuracion TransMIL + Virchow2 debe ser un objeto YAML.")
    return config


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(config)
    if args.device is not None:
        updated["device"] = args.device
    if args.output_root is not None:
        output_root = Path(args.output_root)
        updated["output_root"] = str(output_root)
        updated["checkpoints_dir"] = str(output_root / "checkpoints")
        updated["metrics_dir"] = str(output_root / "metrics")
        updated["plots_dir"] = str(output_root / "plots")
        updated["logs_dir"] = str(output_root / "logs")
        updated["entregables_dir"] = str(
            output_root.parent / "entregables" / "transmil_virchow2_binary_resultados"
        )
    if args.max_valid is not None:
        updated["max_valid"] = args.max_valid
    if args.max_test is not None:
        updated["max_test"] = args.max_test
    return updated


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay una GPU CUDA disponible.")
    return device


def validate_config(config: Dict[str, Any]) -> None:
    if int(config["input_dim"]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("input_dim debe ser 1280 para evaluar TransMIL + Virchow2.")
    if config["encoder_name"] != EXPECTED_ENCODER_NAME:
        raise ValueError(f"encoder_name debe ser {EXPECTED_ENCODER_NAME}.")
    if config["encoder_family"] != EXPECTED_ENCODER_FAMILY:
        raise ValueError("encoder_family debe ser Virchow2.")


def build_model(config: Dict[str, Any]) -> TransMILBinary:
    validate_config(config)
    return TransMILBinary(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_tiles=int(config["max_tiles"]),
    )


def build_loader(
    dataset: Virchow2EmbeddingDataset,
    config: Dict[str, Any],
    device: torch.device,
) -> DataLoader:
    workers = int(config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config["pin_memory"]) and device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=virchow2_bag_collate_fn,
    )


def evaluate_transmil(
    model: TransMILBinary,
    loader: Iterable[Dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    mixed_precision: bool,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    labels: List[int] = []
    probabilities: List[float] = []
    slide_ids: List[str] = []
    amp_enabled = bool(mixed_precision and device.type == "cuda")

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluate TransMIL + Virchow2", leave=False):
            batch_labels = batch["labels"].to(device).float()
            logits: List[torch.Tensor] = []
            batch_losses: List[torch.Tensor] = []
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                for features, label in zip(batch["features"], batch_labels):
                    output = model(features.to(device, non_blocking=device.type == "cuda"))
                    logit = output["logit"]
                    logits.append(logit)
                    batch_losses.append(criterion(logit.view(1), label.view(1)))

            logits_tensor = torch.stack(logits).float()
            loss = torch.stack(batch_losses).mean()
            losses.append(float(loss.detach().cpu().item()))
            labels.extend(batch_labels.detach().cpu().numpy().astype(int).tolist())
            probabilities.extend(
                torch.sigmoid(logits_tensor).detach().cpu().numpy().astype(float).tolist()
            )
            slide_ids.extend(str(value) for value in batch["slide_ids"])

    if not losses:
        raise RuntimeError("El DataLoader de evaluacion TransMIL no produjo batches.")

    metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
    metrics.update(
        {
            "loss": float(sum(losses) / len(losses)),
            "labels": labels,
            "probabilities": probabilities,
            "slide_ids": slide_ids,
        }
    )
    return metrics


def prediction_frame(
    result: Dict[str, Any],
    threshold_default: float,
    threshold_youden: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slide_id": result["slide_ids"],
            "cancer_label": result["labels"],
            "pred_probability": result["probabilities"],
            "pred_label_threshold_0_5": [
                int(value >= threshold_default) for value in result["probabilities"]
            ],
            "pred_label_threshold_youden": [
                int(value >= threshold_youden) for value in result["probabilities"]
            ],
        }
    )


def attention_frame(
    *,
    model: TransMILBinary,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    mixed_precision: bool,
) -> pd.DataFrame:
    """Return one row per tile used by TransMIL with attention scores."""
    model.eval()
    rows = []
    amp_enabled = bool(mixed_precision and device.type == "cuda")

    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device).float()
            for features, label, slide_id, metadata in zip(
                batch["features"],
                labels,
                batch["slide_ids"],
                batch["metadata"],
            ):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = model(features.to(device, non_blocking=device.type == "cuda"))
                logit = output["logit"]
                attention = output["attention"]
                probability = float(torch.sigmoid(logit).detach().cpu().item())
                predicted = int(probability >= threshold)
                tile_ids = list(metadata.get("tile_ids", []))
                tile_paths = list(metadata.get("tile_paths", []))
                attention_values = attention.detach().cpu().numpy().astype(float).tolist()
                for index, score in enumerate(attention_values):
                    rows.append(
                        {
                            "slide_id": str(slide_id),
                            "tile_id": (
                                tile_ids[index]
                                if index < len(tile_ids)
                                else f"{slide_id}_tile_{index}"
                            ),
                            "tile_path": tile_paths[index] if index < len(tile_paths) else "",
                            "attention_score": float(score),
                            "cancer_label": int(label.detach().cpu().item()),
                            "pred_prob": probability,
                            "pred_label": predicted,
                        }
                    )
    return pd.DataFrame(rows)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def create_deliverables(
    *,
    config: Dict[str, Any],
    checkpoint_path: Path,
    valid_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    threshold_payload: Dict[str, Any],
) -> Path:
    """Create a compact deliverable folder for reporting results."""
    output_root = Path(config["output_root"])
    deliverable_dir = Path(
        config.get(
            "entregables_dir",
            output_root.parent.parent / "entregables" / "transmil_virchow2_binary_resultados",
        )
    )
    metrics_dir = Path(config["metrics_dir"])
    plots_dir = Path(config["plots_dir"])

    metricas_dir = deliverable_dir / "metricas"
    graficas_dir = deliverable_dir / "graficas"
    modelo_dir = deliverable_dir / "modelo"
    regiones_dir = deliverable_dir / "regiones_relevantes"
    for directory in (metricas_dir, graficas_dir, modelo_dir, regiones_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for filename in (
        "best_threshold.json",
        "valid_metrics.json",
        "test_metrics.json",
        "valid_predictions.csv",
        "test_predictions.csv",
        "train_history.csv",
        "run_metadata.json",
        "evaluation_run_metadata.json",
    ):
        _copy_if_exists(metrics_dir / filename, metricas_dir / filename)
    _copy_if_exists(
        metrics_dir / "test_attention_scores.csv",
        regiones_dir / "test_attention_scores.csv",
    )
    for filename in (
        "roc_valid.png",
        "roc_test.png",
        "confusion_matrix_valid.png",
        "confusion_matrix_test.png",
        "training_loss.png",
        "validation_auc_f1.png",
    ):
        _copy_if_exists(plots_dir / filename, graficas_dir / filename)
    _copy_if_exists(checkpoint_path, modelo_dir / "best_model.pt")

    summary = {
        "experiment_name": config.get("experiment_name"),
        "encoder_family": config.get("encoder_family"),
        "input_dim": int(config["input_dim"]),
        "checkpoint": str(checkpoint_path),
        "threshold": threshold_payload,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "metrics_dir": str(metrics_dir),
        "plots_dir": str(plots_dir),
        "deliverable_dir": str(deliverable_dir),
    }
    save_json(summary, deliverable_dir / "resumen_resultados_transmil_virchow2.json")
    return deliverable_dir


def run(args: argparse.Namespace) -> int:
    config = apply_overrides(load_config(args.config), args)
    validate_config(config)
    set_seed(int(config["random_seed"]))
    device = resolve_device(str(config["device"]))
    metrics_dir = Path(config["metrics_dir"])
    plots_dir = Path(config["plots_dir"])
    logs_dir = Path(config.get("logs_dir", Path(config["output_root"]) / "logs"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(config["checkpoints_dir"]) / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No existe best_model.pt: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = dict(checkpoint.get("config", config))
    checkpoint_config.update(
        {
            "embeddings_root": config["embeddings_root"],
            "output_root": config["output_root"],
            "checkpoints_dir": config["checkpoints_dir"],
            "metrics_dir": config["metrics_dir"],
            "plots_dir": config["plots_dir"],
            "logs_dir": config.get("logs_dir"),
            "device": config["device"],
            "max_valid": config.get("max_valid"),
            "max_test": config.get("max_test"),
        }
    )

    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss()

    valid_dataset = Virchow2EmbeddingDataset(
        Path(config["embeddings_root"]),
        split="valid",
        max_items=config.get("max_valid"),
        validate_on_init=False,
    )
    test_dataset = Virchow2EmbeddingDataset(
        Path(config["embeddings_root"]),
        split="test",
        max_items=config.get("max_test"),
        validate_on_init=False,
    )
    valid_loader = build_loader(valid_dataset, config, device)
    test_loader = build_loader(test_dataset, config, device)
    threshold_default = float(config["threshold_default"])

    evaluation_metadata = {
        "experiment_name": config.get("experiment_name"),
        "task": config.get("task"),
        "started_at": utc_now_iso(),
        "phase": "evaluation",
        "checkpoint": str(checkpoint_path),
        "config": dict(config),
        "valid_wsi": len(valid_dataset),
        "test_wsi": len(test_dataset),
        "device": str(device),
        "software_versions": get_software_versions(),
        "cuda": get_cuda_info(),
        "git": get_git_info(PROJECT_ROOT),
    }
    save_json(evaluation_metadata, metrics_dir / "evaluation_run_metadata.json")
    run_metadata_path = metrics_dir / "run_metadata.json"
    if not run_metadata_path.exists():
        save_json(evaluation_metadata, run_metadata_path)

    valid_result = evaluate_transmil(
        model,
        valid_loader,
        criterion,
        device,
        threshold=threshold_default,
        mixed_precision=bool(config["mixed_precision"]),
    )
    threshold_youden = find_youden_threshold(
        valid_result["labels"],
        valid_result["probabilities"],
        default_threshold=threshold_default,
    )
    valid_default_metrics = compute_binary_metrics(
        valid_result["labels"],
        valid_result["probabilities"],
        threshold=threshold_default,
    )
    valid_youden_metrics = compute_binary_metrics(
        valid_result["labels"],
        valid_result["probabilities"],
        threshold=threshold_youden,
    )
    threshold_payload = {
        "threshold_default": threshold_default,
        "threshold_youden": threshold_youden,
        "sensitivity_valid_at_threshold": valid_youden_metrics["sensitivity"],
        "specificity_valid_at_threshold": valid_youden_metrics["specificity"],
        "f1_valid_at_threshold": valid_youden_metrics["f1"],
        "auc_valid": valid_youden_metrics["auc_roc"],
        "criterio": "Youden",
    }
    save_json(threshold_payload, metrics_dir / "best_threshold.json")

    test_result = evaluate_transmil(
        model,
        test_loader,
        criterion,
        device,
        threshold=threshold_default,
        mixed_precision=bool(config["mixed_precision"]),
    )
    test_default_metrics = compute_binary_metrics(
        test_result["labels"],
        test_result["probabilities"],
        threshold=threshold_default,
    )
    test_youden_metrics = compute_binary_metrics(
        test_result["labels"],
        test_result["probabilities"],
        threshold=threshold_youden,
    )

    valid_payload = {
        "checkpoint": str(checkpoint_path),
        "loss": valid_result["loss"],
        "threshold_default": valid_default_metrics,
        "threshold_youden": valid_youden_metrics,
    }
    test_payload = {
        "checkpoint": str(checkpoint_path),
        "loss": test_result["loss"],
        "threshold_default": test_default_metrics,
        "threshold_youden_from_validation": test_youden_metrics,
    }
    save_json(valid_payload, metrics_dir / "valid_metrics.json")
    save_json(test_payload, metrics_dir / "test_metrics.json")
    prediction_frame(
        valid_result,
        threshold_default,
        threshold_youden,
    ).to_csv(metrics_dir / "valid_predictions.csv", index=False)
    prediction_frame(
        test_result,
        threshold_default,
        threshold_youden,
    ).to_csv(metrics_dir / "test_predictions.csv", index=False)

    attention_scores = attention_frame(
        model=model,
        loader=test_loader,
        device=device,
        threshold=threshold_youden,
        mixed_precision=bool(config["mixed_precision"]),
    )
    attention_scores.to_csv(metrics_dir / "test_attention_scores.csv", index=False)

    save_confusion_matrix(
        valid_youden_metrics["confusion_matrix"],
        plots_dir / "confusion_matrix_valid.png",
        f"TransMIL + Virchow2 Valid Confusion Matrix (Youden={threshold_youden:.4f})",
    )
    save_confusion_matrix(
        test_youden_metrics["confusion_matrix"],
        plots_dir / "confusion_matrix_test.png",
        f"TransMIL + Virchow2 Test Confusion Matrix (Valid Youden={threshold_youden:.4f})",
    )
    save_roc_curve(
        valid_result["labels"],
        valid_result["probabilities"],
        plots_dir / "roc_valid.png",
        "TransMIL + Virchow2 Validation ROC",
    )
    save_roc_curve(
        test_result["labels"],
        test_result["probabilities"],
        plots_dir / "roc_test.png",
        "TransMIL + Virchow2 Test ROC",
    )

    deliverable_dir = create_deliverables(
        config=config,
        checkpoint_path=checkpoint_path,
        valid_metrics=valid_payload,
        test_metrics=test_payload,
        threshold_payload=threshold_payload,
    )

    print("[INFO] Evaluacion final completada")
    print(f"[INFO] Valid AUC: {valid_youden_metrics['auc_roc']}")
    print(f"[INFO] Youden threshold: {threshold_youden:.6f}")
    print(f"[INFO] Test AUC: {test_youden_metrics['auc_roc']}")
    print(f"[INFO] Metrics: {metrics_dir}")
    print(f"[INFO] Plots: {plots_dir}")
    print(f"[INFO] Entregables: {deliverable_dir}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
