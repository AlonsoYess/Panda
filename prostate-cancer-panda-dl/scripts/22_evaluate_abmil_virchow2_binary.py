"""Final validation and test evaluation for ABMIL + Virchow2."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.abmil import ABMIL
from src.mil.engine import (
    compute_binary_metrics,
    evaluate_binary,
    find_youden_threshold,
    save_json,
    set_seed,
)
from src.mil.plots import save_confusion_matrix, save_roc_curve
from src.mil.virchow2_dataset import (
    EXPECTED_EMBEDDING_DIM,
    EXPECTED_ENCODER_FAMILY,
    EXPECTED_ENCODER_NAME,
    Virchow2EmbeddingDataset,
    virchow2_bag_collate_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua best_model.pt de ABMIL + Virchow2 en valid y test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/abmil_virchow2_train_binary.yaml"),
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
        raise ValueError("La configuracion debe ser un objeto YAML.")
    return config


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay una GPU CUDA disponible.")
    return device


def validate_config(config: Dict[str, Any]) -> None:
    if int(config["input_dim"]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("input_dim debe ser 1280 para evaluar Virchow2.")
    if config["encoder_name"] != EXPECTED_ENCODER_NAME:
        raise ValueError(f"encoder_name debe ser {EXPECTED_ENCODER_NAME}.")
    if config["encoder_family"] != EXPECTED_ENCODER_FAMILY:
        raise ValueError("encoder_family debe ser Virchow2.")


def build_loader(
    dataset: Virchow2EmbeddingDataset,
    config: Dict[str, Any],
    device: torch.device,
) -> DataLoader:
    workers = int(config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size_bags"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=virchow2_bag_collate_fn,
    )


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
    model: ABMIL,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    amp: bool,
) -> pd.DataFrame:
    """Return one row per tile with ABMIL attention scores for test bags."""
    model.eval()
    rows = []
    amp_enabled = bool(amp and device.type == "cuda")

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
                    logit, attention = model(
                        features.to(device, non_blocking=device.type == "cuda")
                    )
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
            output_root.parent.parent / "entregables" / "abmil_virchow2_binary_resultados",
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
    save_json(summary, deliverable_dir / "resumen_resultados_abmil_virchow2.json")
    return deliverable_dir


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.output_root is not None:
        output_root = Path(args.output_root)
        config["output_root"] = str(output_root)
        config["checkpoints_dir"] = str(output_root / "checkpoints")
        config["metrics_dir"] = str(output_root / "metrics")
        config["plots_dir"] = str(output_root / "plots")
        config["logs_dir"] = str(output_root / "logs")
        config["entregables_dir"] = str(
            output_root.parent / "entregables" / "abmil_virchow2_binary_resultados"
        )
    if args.max_valid is not None:
        config["max_valid"] = args.max_valid
    if args.max_test is not None:
        config["max_test"] = args.max_test
    validate_config(config)

    set_seed(int(config["seed"]))
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
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_config = checkpoint.get("config", {})
    input_dim = int(checkpoint_config.get("input_dim", config["input_dim"]))
    if input_dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(f"input_dim invalido en checkpoint Virchow2: {input_dim}")
    encoder_family = checkpoint_config.get("encoder_family", config["encoder_family"])
    if encoder_family != EXPECTED_ENCODER_FAMILY:
        raise ValueError(f"El checkpoint no corresponde a Virchow2: {encoder_family}")

    model = ABMIL(
        input_dim=input_dim,
        hidden_dim=int(checkpoint_config.get("hidden_dim", config["hidden_dim"])),
        dropout=float(checkpoint_config.get("dropout", config["dropout"])),
        attention_dim=int(
            checkpoint_config.get("attention_dim", config["attention_dim"])
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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

    pos_weight = float(checkpoint_config.get("resolved_pos_weight", 1.0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    threshold_default = float(config["threshold_default"])

    valid_result = evaluate_binary(
        model,
        valid_loader,
        criterion,
        device,
        threshold=threshold_default,
        amp=bool(config["amp"]),
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

    test_result = evaluate_binary(
        model,
        test_loader,
        criterion,
        device,
        threshold=threshold_default,
        amp=bool(config["amp"]),
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
        amp=bool(config["amp"]),
    )
    attention_scores.to_csv(metrics_dir / "test_attention_scores.csv", index=False)

    save_confusion_matrix(
        valid_youden_metrics["confusion_matrix"],
        plots_dir / "confusion_matrix_valid.png",
        f"ABMIL + Virchow2 Valid Confusion Matrix (Youden={threshold_youden:.4f})",
    )
    save_confusion_matrix(
        test_youden_metrics["confusion_matrix"],
        plots_dir / "confusion_matrix_test.png",
        f"ABMIL + Virchow2 Test Confusion Matrix (Valid Youden={threshold_youden:.4f})",
    )
    save_roc_curve(
        valid_result["labels"],
        valid_result["probabilities"],
        plots_dir / "roc_valid.png",
        "ABMIL + Virchow2 Validation ROC",
    )
    save_roc_curve(
        test_result["labels"],
        test_result["probabilities"],
        plots_dir / "roc_test.png",
        "ABMIL + Virchow2 Test ROC",
    )

    print("[INFO] Evaluacion final completada")
    print(f"[INFO] Valid AUC: {valid_youden_metrics['auc_roc']}")
    print(f"[INFO] Youden threshold: {threshold_youden:.6f}")
    print(f"[INFO] Test AUC: {test_youden_metrics['auc_roc']}")
    print(f"[INFO] Metrics: {metrics_dir}")
    print(f"[INFO] Plots: {plots_dir}")
    deliverable_dir = create_deliverables(
        config=config,
        checkpoint_path=checkpoint_path,
        valid_metrics=valid_payload,
        test_metrics=test_payload,
        threshold_payload=threshold_payload,
    )
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
