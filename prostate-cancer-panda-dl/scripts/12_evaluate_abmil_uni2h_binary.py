"""Final validation and test evaluation for ABMIL + UNI2-h."""

from __future__ import annotations

import argparse
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
from src.mil.uni2h_dataset import (
    EXPECTED_EMBEDDING_DIM,
    UNI2HEmbeddingDataset,
    uni2h_bag_collate_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua best_model.pt de ABMIL + UNI2-h en valid y test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/abmil_uni2h_train_binary.yaml"),
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


def build_loader(
    dataset: UNI2HEmbeddingDataset,
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
        collate_fn=uni2h_bag_collate_fn,
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


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if int(config["input_dim"]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("input_dim debe ser 1536 para evaluar UNI2-h.")
    if args.device is not None:
        config["device"] = args.device
    if args.output_root is not None:
        output_root = Path(args.output_root)
        config["output_root"] = str(output_root)
        config["checkpoints_dir"] = str(output_root / "checkpoints")
        config["metrics_dir"] = str(output_root / "metrics")
        config["plots_dir"] = str(output_root / "plots")
    if args.max_valid is not None:
        config["max_valid"] = args.max_valid
    if args.max_test is not None:
        config["max_test"] = args.max_test

    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    metrics_dir = Path(config["metrics_dir"])
    plots_dir = Path(config["plots_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    if input_dim == 1024:
        raise ValueError("El checkpoint corresponde a UNI clasico 1024-D.")
    if input_dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(f"input_dim invalido en checkpoint: {input_dim}")

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

    valid_dataset = UNI2HEmbeddingDataset(
        Path(config["embeddings_root"]),
        split="valid",
        max_items=config.get("max_valid"),
    )
    test_dataset = UNI2HEmbeddingDataset(
        Path(config["embeddings_root"]),
        split="test",
        max_items=config.get("max_test"),
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

    save_confusion_matrix(
        valid_youden_metrics["confusion_matrix"],
        plots_dir / "confusion_matrix_valid.png",
        f"Valid Confusion Matrix (Youden={threshold_youden:.4f})",
    )
    save_confusion_matrix(
        test_youden_metrics["confusion_matrix"],
        plots_dir / "confusion_matrix_test.png",
        f"Test Confusion Matrix (Valid Youden={threshold_youden:.4f})",
    )
    save_roc_curve(
        valid_result["labels"],
        valid_result["probabilities"],
        plots_dir / "roc_valid.png",
        "ABMIL + UNI2-h Validation ROC",
    )
    save_roc_curve(
        test_result["labels"],
        test_result["probabilities"],
        plots_dir / "roc_test.png",
        "ABMIL + UNI2-h Test ROC",
    )

    print("[INFO] Evaluacion final completada")
    print(f"[INFO] Valid AUC: {valid_youden_metrics['auc_roc']}")
    print(f"[INFO] Youden threshold: {threshold_youden:.6f}")
    print(f"[INFO] Test AUC: {test_youden_metrics['auc_roc']}")
    print(f"[INFO] Metrics: {metrics_dir}")
    print(f"[INFO] Plots: {plots_dir}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
