"""Train resumable CLAM on precomputed UNI2-h embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.clam import CLAMBinary
from src.mil.engine import (
    append_history_csv,
    compute_binary_metrics,
    find_last_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
)
from src.mil.plots import save_loss_history, save_metric_history
from src.mil.uni2h_dataset import (
    EXPECTED_EMBEDDING_DIM,
    UNI2HEmbeddingDataset,
    uni2h_bag_collate_fn,
)
from src.utils.provenance import (
    get_cuda_info,
    get_git_info,
    get_software_versions,
    utc_now_iso,
)

REQUIRED_CONFIG_KEYS = {
    "experiment_name",
    "task",
    "label_column",
    "input_dim",
    "encoder_name",
    "encoder_family",
    "embeddings_root",
    "output_root",
    "checkpoints_dir",
    "metrics_dir",
    "plots_dir",
    "hidden_dim",
    "attention_dim",
    "dropout",
    "k_sample",
    "instance_loss_weight",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "early_stopping_patience",
    "monitor",
    "random_seed",
    "threshold_default",
    "threshold_selection",
    "num_workers",
    "pin_memory",
    "save_epoch_checkpoints",
    "resume",
    "mixed_precision",
    "device",
    "max_train",
    "max_valid",
    "max_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena CLAM binario con embeddings UNI2-h 1536-D."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/clam_uni2h_train_binary.yaml"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
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
        raise ValueError("La configuracion CLAM debe ser un objeto YAML.")
    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ValueError(f"Faltan claves requeridas en config CLAM: {missing}")
    return config


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(config)
    overrides = {
        "device": args.device,
        "epochs": args.epochs,
        "max_train": args.max_train,
        "max_valid": args.max_valid,
        "resume": args.resume,
    }
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


def validate_config(config: Dict[str, Any]) -> None:
    if int(config["input_dim"]) == 1024:
        raise ValueError("input_dim=1024 corresponde a UNI clasico y esta rechazado.")
    if int(config["input_dim"]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("input_dim debe ser 1536 para CLAM + UNI2-h.")
    if config["encoder_name"] != "MahmoodLab/UNI2-h":
        raise ValueError("encoder_name debe ser MahmoodLab/UNI2-h.")
    if config["encoder_family"] != "UNI2-h":
        raise ValueError("encoder_family debe ser UNI2-h.")
    for key in ("hidden_dim", "attention_dim", "epochs", "batch_size"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} debe ser mayor o igual a 1.")
    if int(config["k_sample"]) < 0:
        raise ValueError("k_sample debe ser mayor o igual a 0.")
    if float(config["instance_loss_weight"]) < 0:
        raise ValueError("instance_loss_weight debe ser mayor o igual a 0.")
    if config["monitor"] != "valid_auc":
        raise ValueError("monitor debe ser valid_auc para este experimento.")
    if str(config["threshold_selection"]).lower() != "youden":
        raise ValueError("threshold_selection debe ser Youden.")


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero no hay una GPU CUDA disponible.")
    return device


def create_directories(config: Dict[str, Any]) -> None:
    for key in ("checkpoints_dir", "metrics_dir", "plots_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def build_model(config: Dict[str, Any]) -> CLAMBinary:
    return CLAMBinary(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        attention_dim=int(config["attention_dim"]),
        dropout=float(config["dropout"]),
        k_sample=int(config["k_sample"]),
        instance_loss_weight=float(config["instance_loss_weight"]),
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def build_loader(
    dataset: UNI2HEmbeddingDataset,
    config: Dict[str, Any],
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    workers = int(config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config["pin_memory"]) and device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=uni2h_bag_collate_fn,
    )


def build_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _forward_batch(
    model: CLAMBinary,
    batch: Dict[str, Any],
    criterion: nn.Module,
    device: torch.device,
    return_instance_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits: List[torch.Tensor] = []
    total_losses: List[torch.Tensor] = []
    bag_losses: List[torch.Tensor] = []
    instance_losses: List[torch.Tensor] = []
    labels = batch["labels"].to(device).float()
    instance_weight = float(model.instance_loss_weight)

    for features, label in zip(batch["features"], labels):
        output = model(
            features.to(device, non_blocking=device.type == "cuda"),
            label=label,
            return_instance_loss=return_instance_loss,
        )
        logit = output["logit"]
        bag_loss = criterion(logit.view(1), label.view(1))
        instance_loss = output.get("instance_loss")
        if instance_loss is None:
            instance_loss = torch.zeros((), dtype=bag_loss.dtype, device=device)
        total_loss = bag_loss + instance_weight * instance_loss
        logits.append(logit)
        total_losses.append(total_loss)
        bag_losses.append(bag_loss)
        instance_losses.append(instance_loss)

    return (
        torch.stack(logits).float(),
        torch.stack(total_losses).mean(),
        torch.stack(bag_losses).mean(),
        torch.stack(instance_losses).mean(),
    )


def train_one_epoch_clam(
    model: CLAMBinary,
    loader: Iterable[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Any | None,
    amp: bool,
) -> Dict[str, float]:
    model.train()
    total_losses: List[float] = []
    bag_losses: List[float] = []
    instance_losses: List[float] = []
    amp_enabled = bool(amp and device.type == "cuda")

    for batch in tqdm(loader, desc="Train CLAM", leave=False):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            _, total_loss, bag_loss, instance_loss = _forward_batch(
                model,
                batch,
                criterion,
                device,
                return_instance_loss=True,
            )

        if scaler is not None and amp_enabled:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        total_losses.append(float(total_loss.detach().cpu().item()))
        bag_losses.append(float(bag_loss.detach().cpu().item()))
        instance_losses.append(float(instance_loss.detach().cpu().item()))

    if not total_losses:
        raise RuntimeError("El DataLoader de entrenamiento CLAM no produjo batches.")
    return {
        "train_loss": float(np.mean(total_losses)),
        "train_bag_loss": float(np.mean(bag_losses)),
        "train_instance_loss": float(np.mean(instance_losses)),
    }


def evaluate_clam(
    model: CLAMBinary,
    loader: Iterable[Dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    amp: bool,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    labels: List[int] = []
    probabilities: List[float] = []
    slide_ids: List[str] = []
    amp_enabled = bool(amp and device.type == "cuda")

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluate CLAM", leave=False):
            batch_labels = batch["labels"].to(device).float()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits, loss, _, _ = _forward_batch(
                    model,
                    batch,
                    criterion,
                    device,
                    return_instance_loss=False,
                )
            losses.append(float(loss.detach().cpu().item()))
            labels.extend(batch_labels.detach().cpu().numpy().astype(int).tolist())
            probabilities.extend(
                torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
            )
            slide_ids.extend(str(value) for value in batch["slide_ids"])

    if not losses:
        raise RuntimeError("El DataLoader de validacion CLAM no produjo batches.")
    metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
    metrics.update(
        {
            "loss": float(np.mean(losses)),
            "labels": labels,
            "probabilities": probabilities,
            "slide_ids": slide_ids,
        }
    )
    return metrics


def run(args: argparse.Namespace) -> int:
    config = apply_overrides(load_config(args.config), args)
    validate_config(config)
    set_seed(int(config["random_seed"]))
    device = resolve_device(str(config["device"]))

    train_dataset = UNI2HEmbeddingDataset(
        Path(config["embeddings_root"]),
        split="train",
        max_items=config.get("max_train"),
    )
    valid_dataset = UNI2HEmbeddingDataset(
        Path(config["embeddings_root"]),
        split="valid",
        max_items=config.get("max_valid"),
    )
    model = build_model(config).to(device)
    total_parameters, trainable_parameters = count_parameters(model)

    print(f"[INFO] Train WSI: {len(train_dataset)}")
    print(f"[INFO] Valid WSI: {len(valid_dataset)}")
    print(f"[INFO] Device: {device}")
    print(
        f"[INFO] Parameters: {total_parameters:,} total, "
        f"{trainable_parameters:,} trainable"
    )

    if args.dry_run:
        sample = train_dataset[0]
        label = torch.tensor(sample["label"], dtype=torch.float32, device=device)
        with torch.inference_mode():
            output = model(
                sample["features"].to(device),
                label=label,
                return_instance_loss=True,
            )
        print("[INFO] Dry-run CLAM completado; no se escribieron checkpoints.")
        print(f"[INFO] slide_id: {sample['slide_id']}")
        print(f"[INFO] features shape: {tuple(sample['features'].shape)}")
        print(f"[INFO] logit shape: {tuple(output['logit'].shape)}")
        print(f"[INFO] attention shape: {tuple(output['attention'].shape)}")
        print(f"[INFO] instance_loss: {output['instance_loss']}")
        return 0

    create_directories(config)
    train_loader = build_loader(train_dataset, config, device, shuffle=True)
    valid_loader = build_loader(valid_dataset, config, device, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    amp_enabled = bool(config["mixed_precision"] and device.type == "cuda")
    scaler = build_grad_scaler(amp_enabled)
    scheduler = None

    checkpoints_dir = Path(config["checkpoints_dir"])
    metrics_dir = Path(config["metrics_dir"])
    plots_dir = Path(config["plots_dir"])
    history_path = metrics_dir / "train_history.csv"
    git_info = get_git_info(PROJECT_ROOT)

    start_epoch = 1
    best_metric = float("-inf")
    best_epoch = 0
    early_stopping_counter = 0
    history: list[Dict[str, Any]] = []
    resumed_from: str | None = None

    if bool(config["resume"]):
        last_checkpoint = find_last_checkpoint(checkpoints_dir)
        if last_checkpoint is None:
            print("[INFO] Resume solicitado, pero no existe checkpoint. Iniciando desde cero.")
        else:
            checkpoint = load_checkpoint(
                last_checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                device=device,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric = float(checkpoint.get("best_metric", best_metric))
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            early_stopping_counter = int(checkpoint.get("early_stopping_counter", 0))
            history = list(checkpoint.get("train_history", []))
            resumed_from = str(last_checkpoint)
            print(f"[INFO] Resuming training from epoch {start_epoch}")
    else:
        print("[INFO] Resume desactivado. Iniciando entrenamiento CLAM desde cero.")

    runtime_config = dict(config)
    run_metadata = {
        "experiment_name": config["experiment_name"],
        "task": config["task"],
        "started_at": utc_now_iso(),
        "config": runtime_config,
        "train_wsi": len(train_dataset),
        "valid_wsi": len(valid_dataset),
        "parameters_total": total_parameters,
        "parameters_trainable": trainable_parameters,
        "random_seed": int(config["random_seed"]),
        "device": str(device),
        "software_versions": get_software_versions(),
        "cuda": get_cuda_info(),
        "git": git_info,
        "resumed_from": resumed_from,
        "start_epoch": start_epoch,
    }
    save_json(run_metadata, metrics_dir / "run_metadata.json")

    final_epoch = int(config["epochs"])
    if start_epoch > final_epoch:
        print(
            f"[INFO] El checkpoint ya alcanzo epoch {start_epoch - 1}; "
            f"epochs configurado={final_epoch}. No hay epocas pendientes."
        )
        return 0

    threshold = float(config["threshold_default"])
    patience = int(config["early_stopping_patience"])

    for epoch in range(start_epoch, final_epoch + 1):
        train_result = train_one_epoch_clam(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            amp=bool(config["mixed_precision"]),
        )
        valid_result = evaluate_clam(
            model,
            valid_loader,
            criterion,
            device,
            threshold=threshold,
            amp=bool(config["mixed_precision"]),
        )
        valid_auc = valid_result["auc_roc"]
        score = float(valid_auc) if valid_auc is not None else float(valid_result["f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_result["train_loss"],
            "train_bag_loss": train_result["train_bag_loss"],
            "train_instance_loss": train_result["train_instance_loss"],
            "valid_loss": valid_result["loss"],
            "valid_accuracy": valid_result["accuracy"],
            "valid_precision": valid_result["precision"],
            "valid_recall": valid_result["recall"],
            "valid_specificity": valid_result["specificity"],
            "valid_f1": valid_result["f1"],
            "valid_auc": valid_auc,
            "valid_gini": valid_result["gini"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "monitor_score": score,
        }
        history.append(row)
        append_history_csv(history, history_path)
        save_loss_history(
            history,
            plots_dir / "training_loss.png",
            title_prefix="CLAM + UNI2-h",
        )
        save_metric_history(
            history,
            plots_dir / "validation_auc_f1.png",
            title_prefix="CLAM + UNI2-h",
        )

        improved = score > best_metric
        if improved:
            best_metric = score
            best_epoch = epoch
            early_stopping_counter = 0
            save_checkpoint(
                checkpoints_dir / "best_model.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                train_history=history,
                config=runtime_config,
                seed=int(config["random_seed"]),
                git_info=git_info,
                early_stopping_counter=early_stopping_counter,
            )
        else:
            early_stopping_counter += 1

        if bool(config["save_epoch_checkpoints"]):
            save_checkpoint(
                checkpoints_dir / f"checkpoint_epoch_{epoch:02d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_metric=best_metric,
                best_epoch=best_epoch,
                train_history=history,
                config=runtime_config,
                seed=int(config["random_seed"]),
                git_info=git_info,
                early_stopping_counter=early_stopping_counter,
            )

        save_checkpoint(
            checkpoints_dir / "last_checkpoint.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            best_metric=best_metric,
            best_epoch=best_epoch,
            train_history=history,
            config=runtime_config,
            seed=int(config["random_seed"]),
            git_info=git_info,
            early_stopping_counter=early_stopping_counter,
        )

        print(
            f"[INFO] epoch={epoch} "
            f"train_loss={train_result['train_loss']:.6f} "
            f"valid_loss={valid_result['loss']:.6f} "
            f"valid_auc={valid_auc} "
            f"valid_f1={valid_result['f1']:.6f} "
            f"valid_recall={valid_result['recall']:.6f} "
            f"valid_specificity={valid_result['specificity']}"
        )

        if early_stopping_counter >= patience:
            print(
                f"[INFO] Early stopping en epoch {epoch}; "
                f"best_epoch={best_epoch}, best_metric={best_metric:.6f}."
            )
            break

    print(f"[INFO] Entrenamiento CLAM finalizado. Best epoch: {best_epoch}")
    print(f"[INFO] Best model: {checkpoints_dir / 'best_model.pt'}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
