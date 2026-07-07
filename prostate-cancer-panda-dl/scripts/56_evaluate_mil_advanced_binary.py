"""Evaluate advanced binary MIL checkpoints on train/valid/test splits.

This script evaluates ABMIL, CLAM, DSMIL, DTFD-MIL and ACMIL checkpoints
using advanced WSI-level embedding bags.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.advanced_embedding_dataset import advanced_bag_collate_fn


MODEL_TO_SCRIPT = {
    "abmil": PROJECT_ROOT / "scripts/44_train_abmil_advanced_binary.py",
    "clam": PROJECT_ROOT / "scripts/45_train_clam_advanced_binary.py",
    "dsmil": PROJECT_ROOT / "scripts/49_train_dsmil_advanced_binary.py",
    "dtfdmil": PROJECT_ROOT / "scripts/51_train_dtfdmil_advanced_binary.py",
    "acmil": PROJECT_ROOT / "scripts/53_train_acmil_advanced_binary.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an advanced binary MIL checkpoint on a selected split."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_TO_SCRIPT.keys()),
        help="MIL architecture to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to checkpoint, usually checkpoints/best_model.pt.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config. If omitted, checkpoint['config'] is used.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold. If omitted, config['threshold_default'] is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where metrics and predictions will be saved.",
    )
    parser.add_argument(
        "--metrics-filename",
        type=str,
        default=None,
        help="Metrics JSON filename. Default: <split>_metrics.json.",
    )
    parser.add_argument(
        "--predictions-filename",
        type=str,
        default=None,
        help="Predictions CSV filename. Default: <split>_predictions.csv.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for evaluation.",
    )
    parser.add_argument(
        "--amp-mode",
        type=str,
        default="config",
        choices=["config", "on", "off"],
        help="Autocast mode during evaluation.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional limit for quick checks.",
    )
    return parser.parse_args()


def safe_torch_load(path: Path, map_location: str | torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def load_training_module(model_name: str) -> Any:
    script_path = MODEL_TO_SCRIPT[model_name]
    if not script_path.is_file():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    module_name = f"train_{model_name}_advanced_binary"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_config(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    if args.config is not None:
        config = load_yaml(args.config)
    else:
        if "config" not in checkpoint:
            raise KeyError("Checkpoint does not contain config and --config was not provided.")
        config = dict(checkpoint["config"])

    if args.threshold is not None:
        config["threshold_default"] = float(args.threshold)

    if args.max_items is not None:
        split_key = f"max_{args.split}"
        config[split_key] = int(args.max_items)

    return config


def get_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError("No model state dict found in checkpoint.")



def _config_kwargs_for_signature(cls: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(cls)
    kwargs: Dict[str, Any] = {}

    aliases = {
        "input_dim": ["input_dim", "embedding_dim", "in_dim"],
        "hidden_dim": ["hidden_dim"],
        "attention_dim": ["attention_dim", "attn_dim"],
        "dropout": ["dropout", "dropout_rate"],
        "k_sample": ["k_sample"],
        "instance_loss_weight": ["instance_loss_weight"],
        "num_branches": ["num_branches", "n_branches"],
        "n_branches": ["n_branches", "num_branches"],
        "num_pseudo_bags": ["num_pseudo_bags", "n_pseudo_bags"],
        "num_subbags": ["num_subbags", "num_pseudo_bags"],
        "distilled_instances": ["distilled_instances", "num_distilled_instances"],
    }

    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue

        candidate_keys = aliases.get(param_name, [param_name])

        for key in candidate_keys:
            if key in config and config[key] is not None:
                kwargs[param_name] = config[key]
                break

    return kwargs


def build_model_robust(training_module: Any, model_name: str, config: Dict[str, Any]) -> nn.Module:
    if hasattr(training_module, "build_model"):
        return training_module.build_model(config)

    candidate_classes = []

    for attr_name, attr_value in vars(training_module).items():
        if not inspect.isclass(attr_value):
            continue

        try:
            if not issubclass(attr_value, nn.Module):
                continue
        except TypeError:
            continue

        if attr_value is nn.Module:
            continue

        candidate_classes.append((attr_name, attr_value))

    if not candidate_classes:
        raise AttributeError(
            f"No build_model() found and no nn.Module classes detected in {training_module}."
        )

    model_name_lower = model_name.lower()

    priority = []
    for attr_name, cls in candidate_classes:
        name_lower = attr_name.lower()
        score = 0

        if model_name_lower in name_lower:
            score += 100
        if "binary" in name_lower:
            score += 20
        if "mil" in name_lower:
            score += 10
        if "attention" in name_lower and model_name_lower == "abmil":
            score += 30
        if "clam" in name_lower and model_name_lower == "clam":
            score += 30
        if "dsmil" in name_lower and model_name_lower == "dsmil":
            score += 30
        if "dtfd" in name_lower and model_name_lower == "dtfdmil":
            score += 30
        if "acmil" in name_lower and model_name_lower == "acmil":
            score += 30

        priority.append((score, attr_name, cls))

    priority.sort(reverse=True, key=lambda x: x[0])

    errors = []

    for score, attr_name, cls in priority:
        kwargs = _config_kwargs_for_signature(cls, config)

        try:
            model = cls(**kwargs)
            print(f"[INFO] Model built with class {attr_name} and kwargs={kwargs}")
            return model
        except Exception as exc:
            errors.append((attr_name, kwargs, repr(exc)))

    message = "Could not instantiate model. Tried:\n"
    for attr_name, kwargs, error in errors:
        message += f"- {attr_name} kwargs={kwargs} error={error}\n"

    raise RuntimeError(message)



def extract_logit(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("logit", "bag_logit", "logits", "y_logit"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value.view(-1)[0]
        raise KeyError(f"No logit key found. Output keys: {list(output.keys())}")

    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value.view(-1)[0]

    if isinstance(output, torch.Tensor):
        return output.view(-1)[0]

    raise TypeError(f"Unsupported model output type: {type(output)}")


def forward_model(model: nn.Module, features: torch.Tensor, label: torch.Tensor) -> Any:
    try:
        return model(features)
    except TypeError:
        try:
            return model(features, label=label, return_instance_loss=False)
        except TypeError:
            return model(features, label=label)


def compute_metrics(
    labels: List[int],
    probabilities: List[float],
    predictions: List[int],
    losses: List[float],
    tile_counts: List[int],
) -> Dict[str, Any]:
    labels_np = np.asarray(labels, dtype=int)
    probs_np = np.asarray(probabilities, dtype=float)
    preds_np = np.asarray(predictions, dtype=int)

    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()

    auc = roc_auc_score(labels_np, probs_np)
    accuracy = accuracy_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(labels_np, preds_np, zero_division=0)

    return {
        "n_samples": int(len(labels)),
        "n_negative": int((labels_np == 0).sum()),
        "n_positive": int((labels_np == 1).sum()),
        "predicted_negative": int((preds_np == 0).sum()),
        "predicted_positive": int((preds_np == 1).sum()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "auc_roc": float(auc),
        "gini": float(2 * auc - 1),
        "log_loss": float(log_loss(labels_np, probs_np, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels_np, probs_np)),
        "loss": float(np.mean(losses)) if losses else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tile_count_min": int(np.min(tile_counts)) if tile_counts else None,
        "tile_count_mean": float(np.mean(tile_counts)) if tile_counts else None,
        "tile_count_max": int(np.max(tile_counts)) if tile_counts else None,
    }


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    config = resolve_config(args, checkpoint)
    config["device"] = str(device)
    config["resume"] = False

    threshold = float(config.get("threshold_default", 0.5))

    training_module = load_training_module(args.model)

    split_key = f"max_{args.split}"
    max_items = config.get(split_key)

    dataset = training_module.build_dataset(config, args.split, max_items)
    if hasattr(dataset, "load_labels"):
        dataset.load_labels()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=advanced_bag_collate_fn,
    )

    model = build_model_robust(training_module, args.model, config).to(device)
    state_dict = get_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        print(f"[WARN] missing keys: {missing}")
        print(f"[WARN] unexpected keys: {unexpected}")

    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    if args.amp_mode == "on":
        amp_enabled = device.type == "cuda"
    elif args.amp_mode == "off":
        amp_enabled = False
    else:
        amp_enabled = bool(
            (config.get("amp", False) or config.get("mixed_precision", False))
            and device.type == "cuda"
        )

    labels: List[int] = []
    probabilities: List[float] = []
    predictions: List[int] = []
    slide_ids: List[str] = []
    losses: List[float] = []
    tile_counts: List[int] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluate {args.model} {args.split}"):
            batch_labels = batch["labels"].to(device).float()
            batch_slide_ids = batch["slide_ids"]

            for features, label, slide_id in zip(batch["features"], batch_labels, batch_slide_ids):
                features = features.to(device, non_blocking=device.type == "cuda")

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = forward_model(model, features, label)
                    logit = extract_logit(output)
                    loss = criterion(logit.view(1), label.view(1))

                probability = torch.sigmoid(logit).detach().float().cpu().item()
                prediction = int(probability >= threshold)

                labels.append(int(label.detach().cpu().item()))
                probabilities.append(float(probability))
                predictions.append(prediction)
                slide_ids.append(str(slide_id))
                losses.append(float(loss.detach().cpu().item()))
                tile_counts.append(int(features.shape[0]))

    metrics = compute_metrics(labels, probabilities, predictions, losses, tile_counts)
    metrics.update(
        {
            "model": args.model,
            "encoder": config.get("encoder_name"),
            "model_name": config.get("model_name"),
            "experiment_name": config.get("experiment_name"),
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "config_path": str(args.config) if args.config is not None else None,
            "best_epoch": checkpoint.get("best_epoch"),
            "best_valid_metric": checkpoint.get("best_metric"),
            "threshold": threshold,
            "device": str(device),
            "amp_mode": args.amp_mode,
            "amp_enabled": bool(amp_enabled),
        }
    )

    predictions_df = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "label": labels,
            "probability": probabilities,
            "prediction": predictions,
            "tile_count": tile_counts,
        }
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(config.get("metrics_dir", Path(config["output_root"]) / "metrics"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_filename = args.metrics_filename or f"{args.split}_metrics.json"
    predictions_filename = args.predictions_filename or f"{args.split}_predictions.csv"

    metrics_path = output_dir / metrics_filename
    predictions_path = output_dir / predictions_filename

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    predictions_df.to_csv(predictions_path, index=False)

    print("=" * 100)
    print(f"{args.model.upper()} | {config.get('encoder_name')} | {args.split.upper()} | threshold={threshold}")
    print("=" * 100)

    for key in [
        "n_samples",
        "n_negative",
        "n_positive",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "auc_roc",
        "gini",
        "log_loss",
        "brier_score",
        "tn",
        "fp",
        "fn",
        "tp",
        "tile_count_min",
        "tile_count_mean",
        "tile_count_max",
        "amp_enabled",
    ]:
        print(f"{key}: {metrics.get(key)}")

    print("\nSaved:")
    print(metrics_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
