"""Final validation and test evaluation for CLAM + UNI2-h."""

from __future__ import annotations

import argparse
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

from src.mil.clam import CLAMBinary
from src.mil.engine import compute_binary_metrics, find_youden_threshold, save_json, set_seed
from src.mil.plots import save_confusion_matrix, save_roc_curve
from src.mil.uni2h_dataset import (
    EXPECTED_EMBEDDING_DIM,
    UNI2HEmbeddingDataset,
    uni2h_bag_collate_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua best_model.pt de CLAM + UNI2-h en valid y test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/clam_uni2h_train_binary.yaml"),
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
        raise ValueError("La configuracion CLAM debe ser un objeto YAML.")
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


def build_model(config: Dict[str, Any]) -> CLAMBinary:
    if int(config["input_dim"]) == 1024:
        raise ValueError("input_dim=1024 corresponde a UNI clasico y esta rechazado.")
    if int(config["input_dim"]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("input_dim debe ser 1536 para CLAM + UNI2-h.")
    return CLAMBinary(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        attention_dim=int(config["attention_dim"]),
        dropout=float(config["dropout"]),
        k_sample=int(config["k_sample"]),
        instance_loss_weight=float(config["instance_loss_weight"]),
    )


def build_loader(
    dataset: UNI2HEmbeddingDataset,
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
        collate_fn=uni2h_bag_collate_fn,
    )


def evaluate_clam(
    model: CLAMBinary,
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
        for batch in tqdm(loader, desc="Evaluate CLAM", leave=False):
            batch_labels = batch["labels"].to(device).float()
            logits: List[torch.Tensor] = []
            batch_losses: List[torch.Tensor] = []
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                for features, label in zip(batch["features"], batch_labels):
                    output = model(features.to(device), return_instance_loss=False)
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
        raise RuntimeError("El DataLoader de evaluacion CLAM no produjo batches.")

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


def run(args: argparse.Namespace) -> int:
    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["random_seed"]))
    device = resolve_device(str(config["device"]))
    metrics_dir = Path(config["metrics_dir"])
    plots_dir = Path(config["plots_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

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
            "device": config["device"],
            "max_valid": config.get("max_valid"),
            "max_test": config.get("max_test"),
        }
    )

    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss()

    valid_dataset = UNI2HEmbeddingDataset(
        Path(config["embeddings_root"]),
        split="valid",
        max_items=config.get("max_valid"),
        validate_on_init=False,
    )
    test_dataset = UNI2HEmbeddingDataset(
        Path(config["embeddings_root"]),
        split="test",
        max_items=config.get("max_test"),
        validate_on_init=False,
    )
    valid_loader = build_loader(valid_dataset, config, device)
    test_loader = build_loader(test_dataset, config, device)
    threshold_default = float(config["threshold_default"])

    valid_result = evaluate_clam(
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

    test_result = evaluate_clam(
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

    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "loss": valid_result["loss"],
            "threshold_default": valid_default_metrics,
            "threshold_youden": valid_youden_metrics,
        },
        metrics_dir / "valid_metrics.json",
    )
    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "loss": test_result["loss"],
            "threshold_default": test_default_metrics,
            "threshold_youden_from_validation": test_youden_metrics,
        },
        metrics_dir / "test_metrics.json",
    )
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
        f"CLAM Valid Confusion Matrix (Youden={threshold_youden:.4f})",
    )
    save_confusion_matrix(
        test_youden_metrics["confusion_matrix"],
        plots_dir / "confusion_matrix_test.png",
        f"CLAM Test Confusion Matrix (Valid Youden={threshold_youden:.4f})",
    )
    save_roc_curve(
        valid_result["labels"],
        valid_result["probabilities"],
        plots_dir / "roc_valid.png",
        "CLAM + UNI2-h Validation ROC",
    )
    save_roc_curve(
        test_result["labels"],
        test_result["probabilities"],
        plots_dir / "roc_test.png",
        "CLAM + UNI2-h Test ROC",
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
