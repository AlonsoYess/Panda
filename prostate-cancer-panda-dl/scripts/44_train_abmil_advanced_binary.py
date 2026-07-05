"""Train ABMIL binary MIL on advanced Virchow2 or UNI2-h embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.abmil import ABMIL
from src.mil.advanced_embedding_dataset import (
    AdvancedEmbeddingDataset,
    advanced_bag_collate_fn,
    expected_dim_for_encoder,
    normalize_encoder_name,
)
from src.mil.engine import (
    append_history_csv,
    build_pos_weight,
    evaluate_binary,
    find_last_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
    train_one_epoch,
)
from src.mil.plots import save_loss_history, save_metric_history
from src.utils.provenance import (
    get_cuda_info,
    get_git_info,
    get_software_versions,
    utc_now_iso,
)

REQUIRED_CONFIG_KEYS = {
    "experiment_name",
    "task",
    "label_key",
    "label_column",
    "input_dim",
    "encoder_name",
    "model_name",
    "embeddings_root",
    "output_root",
    "checkpoints_dir",
    "metrics_dir",
    "plots_dir",
    "logs_dir",
    "seed",
    "device",
    "epochs",
    "batch_size_bags",
    "learning_rate",
    "weight_decay",
    "dropout",
    "hidden_dim",
    "attention_dim",
    "num_workers",
    "amp",
    "early_stopping_patience",
    "monitor_metric",
    "save_every_epoch",
    "resume",
    "threshold_default",
    "max_train",
    "max_valid",
    "max_test",
    "pos_weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena ABMIL binario con embeddings advanced Virchow2/UNI2-h."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/abmil_virchow2_advanced_train_binary.yaml"),
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
        "resume": args.resume,
        "max_train": args.max_train_slides if args.max_train_slides is not None else args.max_train,
        "max_valid": args.max_valid_slides if args.max_valid_slides is not None else args.max_valid,
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
    encoder_name = normalize_encoder_name(str(config["encoder_name"]))
    expected_dim = expected_dim_for_encoder(encoder_name)
    if int(config["input_dim"]) != expected_dim:
        raise ValueError(
            f"input_dim debe ser {expected_dim} para encoder={encoder_name}."
        )
    if str(config["label_column"]) != "cancer_label":
        raise ValueError("label_column debe ser cancer_label.")
    for key in ("epochs", "batch_size_bags", "hidden_dim", "attention_dim"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} debe ser mayor o igual a 1.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate debe ser mayor que cero.")
    if int(config["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience debe ser mayor o igual a 1.")
    if str(config["monitor_metric"]) not in {"valid_auc", "valid_f1"}:
        raise ValueError("monitor_metric debe ser valid_auc o valid_f1.")


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay una GPU CUDA disponible.")
    return device


def create_directories(config: Dict[str, Any]) -> None:
    for key in ("checkpoints_dir", "metrics_dir", "plots_dir", "logs_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def build_loader(
    dataset: AdvancedEmbeddingDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=advanced_bag_collate_fn,
    )


def build_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def build_dataset(config: Dict[str, Any], split: str, max_items: int | None) -> AdvancedEmbeddingDataset:
    return AdvancedEmbeddingDataset(
        embeddings_root=Path(config["embeddings_root"]),
        split=split,
        encoder_name=str(config["encoder_name"]),
        model_name=str(config["model_name"]),
        expected_dim=int(config["input_dim"]),
        max_items=max_items,
        validate_on_init=False,
    )


def dry_run(config: Dict[str, Any], device: torch.device) -> None:
    train_dataset = build_dataset(config, "train", config.get("max_train"))
    valid_dataset = build_dataset(config, "valid", config.get("max_valid"))
    model = ABMIL(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        attention_dim=int(config["attention_dim"]),
    ).to(device)
    sample = train_dataset[0]
    features = sample["features"].to(device)
    with torch.inference_mode():
        logit, attention = model(features)
    total, trainable = count_parameters(model)
    print("[DRY-RUN] ABMIL advanced binary")
    print(f"encoder={config['encoder_name']} model_name={config['model_name']}")
    print(f"train_wsi={len(train_dataset)} valid_wsi={len(valid_dataset)}")
    print(f"sample_slide_id={sample['slide_id']}")
    print(f"features_shape={tuple(sample['features'].shape)}")
    print(f"logit_shape={tuple(logit.shape)} attention_shape={tuple(attention.shape)}")
    print(f"parameters_total={total} parameters_trainable={trainable}")
    print(f"output_root={config['output_root']}")


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    validate_config(config)
    device = resolve_device(str(config["device"]))
    set_seed(int(config["seed"]))

    if args.dry_run:
        dry_run(config, device)
        return

    create_directories(config)
    train_dataset = build_dataset(config, "train", config.get("max_train"))
    valid_dataset = build_dataset(config, "valid", config.get("max_valid"))
    train_dataset.load_labels()
    valid_dataset.load_labels()
    train_loader = build_loader(
        train_dataset,
        batch_size=int(config["batch_size_bags"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        device=device,
    )
    valid_loader = build_loader(
        valid_dataset,
        batch_size=int(config["batch_size_bags"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        device=device,
    )

    model = ABMIL(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        attention_dim=int(config["attention_dim"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    scaler = build_grad_scaler(amp_enabled)
    pos_weight_value = build_pos_weight(train_dataset, config.get("pos_weight"))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )

    checkpoints_dir = Path(config["checkpoints_dir"])
    metrics_dir = Path(config["metrics_dir"])
    plots_dir = Path(config["plots_dir"])
    last_checkpoint = find_last_checkpoint(checkpoints_dir)
    start_epoch = 1
    best_metric = float("-inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    early_stopping_counter = 0
    if bool(config["resume"]) and last_checkpoint is not None:
        checkpoint = load_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        history = list(checkpoint.get("train_history", []))
        early_stopping_counter = int(checkpoint.get("early_stopping_counter", 0))
        print(f"[INFO] Resuming training from epoch {start_epoch}")

    total, trainable = count_parameters(model)
    save_json(
        {
            "created_at": utc_now_iso(),
            "config": config,
            "device": str(device),
            "train_wsi": len(train_dataset),
            "valid_wsi": len(valid_dataset),
            "parameters_total": total,
            "parameters_trainable": trainable,
            "software_versions": get_software_versions(),
            "cuda": get_cuda_info(),
            "git": get_git_info(PROJECT_ROOT),
        },
        metrics_dir / "run_metadata.json",
    )
    print(f"[INFO] Train WSI: {len(train_dataset)}")
    print(f"[INFO] Valid WSI: {len(valid_dataset)}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Parameters: {total} total, {trainable} trainable")

    monitor_metric = str(config["monitor_metric"])
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            amp=amp_enabled,
        )
        valid_metrics = evaluate_binary(
            model,
            valid_loader,
            criterion,
            device,
            threshold=float(config["threshold_default"]),
            amp=amp_enabled,
        )
        current_metric = (
            valid_metrics["auc_roc"]
            if monitor_metric == "valid_auc"
            else valid_metrics["f1"]
        )
        current_metric = float("-inf") if current_metric is None else float(current_metric)
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
            "valid_auc": valid_metrics["auc_roc"],
            "valid_f1": valid_metrics["f1"],
            "accuracy": valid_metrics["accuracy"],
            "precision": valid_metrics["precision"],
            "recall": valid_metrics["recall"],
            "specificity": valid_metrics["specificity"],
            "f1": valid_metrics["f1"],
            "auc": valid_metrics["auc_roc"],
            "gini": valid_metrics["gini"],
            "is_best": int(is_best),
        }
        history.append(row)
        append_history_csv(history, metrics_dir / "train_history.csv")
        save_json(valid_metrics, metrics_dir / "valid_metrics.json")
        save_checkpoint(
            checkpoints_dir / "last_checkpoint.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_metric=best_metric,
            best_epoch=best_epoch,
            train_history=history,
            config=config,
            seed=int(config["seed"]),
            git_info=get_git_info(PROJECT_ROOT),
            early_stopping_counter=early_stopping_counter,
        )
        if is_best:
            save_checkpoint(
                checkpoints_dir / "best_model.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                train_history=history,
                config=config,
                seed=int(config["seed"]),
                git_info=get_git_info(PROJECT_ROOT),
                early_stopping_counter=early_stopping_counter,
            )
        if bool(config["save_every_epoch"]):
            save_checkpoint(
                checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                train_history=history,
                config=config,
                seed=int(config["seed"]),
                git_info=get_git_info(PROJECT_ROOT),
                early_stopping_counter=early_stopping_counter,
            )

        save_loss_history(history, plots_dir / "training_loss.png", title_prefix="ABMIL Advanced")
        save_metric_history(history, plots_dir / "validation_auc_f1.png", title_prefix="ABMIL Advanced")
        print(
            f"Epoch {epoch:03d}/{config['epochs']} | train_loss={train_loss:.4f} "
            f"| valid_loss={valid_metrics['loss']:.4f} | valid_auc={valid_metrics['auc_roc']} "
            f"| valid_f1={valid_metrics['f1']:.4f} | best={'YES' if is_best else 'NO'}"
        )
        if early_stopping_counter >= int(config["early_stopping_patience"]):
            print("[INFO] Early stopping activado.")
            break


if __name__ == "__main__":
    main()
