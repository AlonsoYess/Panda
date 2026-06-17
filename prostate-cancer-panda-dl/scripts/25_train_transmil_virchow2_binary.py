"""Train resumable TransMIL on precomputed Virchow2 embeddings."""

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

REQUIRED_CONFIG_KEYS = {
    "experiment_name",
    "task",
    "label_key",
    "label_column",
    "input_dim",
    "encoder_name",
    "encoder_family",
    "embeddings_root",
    "output_root",
    "checkpoints_dir",
    "metrics_dir",
    "plots_dir",
    "logs_dir",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "dim_feedforward",
    "dropout",
    "max_tiles",
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
        description="Entrena TransMIL binario con embeddings Virchow2 1280-D."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/transmil_virchow2_train_binary.yaml"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
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
        raise ValueError("La configuracion TransMIL + Virchow2 debe ser un objeto YAML.")
    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise ValueError(f"Faltan claves requeridas en config TransMIL + Virchow2: {missing}")
    return config


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(config)
    overrides = {
        "device": args.device,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_train": args.max_train_slides if args.max_train_slides is not None else args.max_train,
        "max_valid": args.max_valid_slides if args.max_valid_slides is not None else args.max_valid,
        "max_test": args.max_test,
        "resume": args.resume,
    }
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
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
    return updated


def validate_config(config: Dict[str, Any]) -> None:
    if int(config["input_dim"]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("input_dim debe ser 1280 para TransMIL + Virchow2.")
    if config["encoder_name"] != EXPECTED_ENCODER_NAME:
        raise ValueError(f"encoder_name debe ser {EXPECTED_ENCODER_NAME}.")
    if config["encoder_family"] != EXPECTED_ENCODER_FAMILY:
        raise ValueError("encoder_family debe ser Virchow2.")
    for key in (
        "hidden_dim",
        "num_layers",
        "num_heads",
        "dim_feedforward",
        "max_tiles",
        "epochs",
        "batch_size",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} debe ser mayor o igual a 1.")
    if int(config["hidden_dim"]) % int(config["num_heads"]) != 0:
        raise ValueError("hidden_dim debe ser divisible por num_heads.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate debe ser mayor que cero.")
    if int(config["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience debe ser mayor o igual a 1.")
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
    for key in ("checkpoints_dir", "metrics_dir", "plots_dir", "logs_dir"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def count_split_embeddings(
    embeddings_root: Path,
    split: str,
    max_items: int | None = None,
) -> int:
    """Count embedding files without loading tensors."""
    split_dir = Path(embeddings_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de embeddings '{split}': {split_dir}"
        )
    files = sorted(split_dir.glob("*.pt"))
    if max_items is not None:
        files = files[: int(max_items)]
    return len(files)


def build_model(config: Dict[str, Any]) -> TransMILBinary:
    return TransMILBinary(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_tiles=int(config["max_tiles"]),
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def build_loader(
    dataset: Virchow2EmbeddingDataset,
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
        collate_fn=virchow2_bag_collate_fn,
    )


def build_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _forward_batch(
    model: TransMILBinary,
    batch: Dict[str, Any],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: List[torch.Tensor] = []
    losses: List[torch.Tensor] = []
    labels = batch["labels"].to(device).float()

    for features, label in zip(batch["features"], labels):
        output = model(features.to(device, non_blocking=device.type == "cuda"))
        logit = output["logit"]
        loss = criterion(logit.view(1), label.view(1))
        logits.append(logit)
        losses.append(loss)

    return torch.stack(logits).float(), torch.stack(losses).mean()


def train_one_epoch_transmil(
    model: TransMILBinary,
    loader: Iterable[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Any | None,
    amp: bool,
) -> float:
    model.train()
    losses: List[float] = []
    amp_enabled = bool(amp and device.type == "cuda")

    for batch in tqdm(loader, desc="Train TransMIL + Virchow2", leave=False):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            _, loss = _forward_batch(model, batch, criterion, device)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        raise RuntimeError("El DataLoader de entrenamiento TransMIL no produjo batches.")
    return float(np.mean(losses))


def evaluate_transmil(
    model: TransMILBinary,
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
        for batch in tqdm(loader, desc="Evaluate TransMIL + Virchow2", leave=False):
            batch_labels = batch["labels"].to(device).float()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits, loss = _forward_batch(model, batch, criterion, device)

            losses.append(float(loss.detach().cpu().item()))
            labels.extend(batch_labels.detach().cpu().numpy().astype(int).tolist())
            probabilities.extend(
                torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
            )
            slide_ids.extend(str(value) for value in batch["slide_ids"])

    if not losses:
        raise RuntimeError("El DataLoader de validacion TransMIL no produjo batches.")
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

    train_dataset = Virchow2EmbeddingDataset(
        Path(config["embeddings_root"]),
        split="train",
        max_items=config.get("max_train"),
        validate_on_init=False,
    )
    valid_dataset = Virchow2EmbeddingDataset(
        Path(config["embeddings_root"]),
        split="valid",
        max_items=config.get("max_valid"),
        validate_on_init=False,
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
        test_count = count_split_embeddings(
            Path(config["embeddings_root"]),
            split="test",
            max_items=config.get("max_test"),
        )
        sample = train_dataset[0]
        with torch.inference_mode():
            output = model(sample["features"].to(device))
        print("[INFO] Dry-run TransMIL + Virchow2 completado; no se escribieron checkpoints.")
        print(f"[INFO] Test WSI: {test_count}")
        print(f"[INFO] Sample slide_id: {sample['slide_id']}")
        print(f"[INFO] Features shape: {tuple(sample['features'].shape)}")
        print(f"[INFO] Embedding dim: {sample['metadata']['embedding_dim']}")
        print(f"[INFO] Encoder family: {sample['metadata']['encoder_family']}")
        print(f"[INFO] Logit shape: {tuple(output['logit'].shape)}")
        print(f"[INFO] Attention shape: {tuple(output['attention'].shape)}")
        print(f"[INFO] Output dir esperado: {config['output_root']}")
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
        print("[INFO] Resume desactivado. Iniciando entrenamiento TransMIL + Virchow2 desde cero.")

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
        train_loss = train_one_epoch_transmil(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            amp=bool(config["mixed_precision"]),
        )
        valid_result = evaluate_transmil(
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
            "train_loss": train_loss,
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
            title_prefix="TransMIL + Virchow2",
        )
        save_metric_history(
            history,
            plots_dir / "validation_auc_f1.png",
            title_prefix="TransMIL + Virchow2",
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
            f"train_loss={train_loss:.6f} "
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

    print(f"[INFO] Entrenamiento TransMIL + Virchow2 finalizado. Best epoch: {best_epoch}")
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
