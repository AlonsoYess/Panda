"""Train resumable ABMIL on precomputed Virchow2 embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.abmil import ABMIL
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena ABMIL binario con embeddings Virchow2 1280-D."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/abmil_virchow2_train_binary.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-train-slides", dest="max_train_slides", type=int, default=None)
    parser.add_argument("--max-valid-slides", dest="max_valid_slides", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
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
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "device": args.device,
        "resume": args.resume,
        "max_train": args.max_train_slides if args.max_train_slides is not None else args.max_train,
        "max_valid": args.max_valid_slides if args.max_valid_slides is not None else args.max_valid,
        "max_test": args.max_test,
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
    return updated


def validate_config(config: Dict[str, Any]) -> None:
    if int(config["input_dim"]) != EXPECTED_EMBEDDING_DIM:
        raise ValueError("input_dim debe ser 1280 para ABMIL + Virchow2.")
    if config["encoder_name"] != EXPECTED_ENCODER_NAME:
        raise ValueError(f"encoder_name debe ser {EXPECTED_ENCODER_NAME}.")
    if config["encoder_family"] != EXPECTED_ENCODER_FAMILY:
        raise ValueError("encoder_family debe ser Virchow2.")
    for key in ("epochs", "batch_size_bags", "hidden_dim", "attention_dim"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} debe ser mayor o igual a 1.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate debe ser mayor que cero.")
    if int(config["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience debe ser mayor o igual a 1.")
    if config["monitor_metric"] not in {"valid_auc", "valid_f1"}:
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
    dataset: Virchow2EmbeddingDataset,
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
        collate_fn=virchow2_bag_collate_fn,
    )


def build_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_model(config: Dict[str, Any]) -> ABMIL:
    return ABMIL(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        attention_dim=int(config["attention_dim"]),
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


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


def monitor_score(
    valid_result: Dict[str, Any],
    monitor_metric: str,
) -> tuple[float, str]:
    if monitor_metric == "valid_f1":
        return float(valid_result["f1"]), "valid_f1"
    if valid_result["auc_roc"] is not None:
        return float(valid_result["auc_roc"]), "valid_auc"
    return float(valid_result["f1"]), "valid_f1_fallback_auc_unavailable"


def run(args: argparse.Namespace) -> int:
    config = apply_overrides(load_config(args.config), args)
    validate_config(config)
    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))

    train_dataset = Virchow2EmbeddingDataset(
        embeddings_root=Path(config["embeddings_root"]),
        split="train",
        max_items=config.get("max_train"),
        validate_on_init=False,
    )
    valid_dataset = Virchow2EmbeddingDataset(
        embeddings_root=Path(config["embeddings_root"]),
        split="valid",
        max_items=config.get("max_valid"),
        validate_on_init=False,
    )
    model = build_model(config).to(device)
    total_parameters, trainable_parameters = count_parameters(model)

    print(f"[INFO] Train WSI: {len(train_dataset)}")
    print(f"[INFO] Valid WSI: {len(valid_dataset)}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Parameters: {total_parameters:,} total, {trainable_parameters:,} trainable")

    if args.dry_run:
        test_count = count_split_embeddings(
            Path(config["embeddings_root"]),
            split="test",
            max_items=config.get("max_test"),
        )
        sample = train_dataset[0]
        with torch.inference_mode():
            logit, attention = model(sample["features"].to(device))
        print("[INFO] Dry-run ABMIL + Virchow2 completado; no se entreno ni se modificaron checkpoints.")
        print(f"[INFO] Test WSI: {test_count}")
        print(f"[INFO] Sample slide_id: {sample['slide_id']}")
        print(f"[INFO] Features shape: {tuple(sample['features'].shape)}")
        print(f"[INFO] Embedding dim: {sample['metadata']['embedding_dim']}")
        print(f"[INFO] Encoder family: {sample['metadata']['encoder_family']}")
        print(f"[INFO] Model logit shape: {tuple(logit.shape)}")
        print(f"[INFO] Attention shape: {tuple(attention.shape)}")
        print(f"[INFO] Output dir esperado: {config['output_root']}")
        return 0

    create_directories(config)
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

    resolved_pos_weight = build_pos_weight(
        train_dataset.labels,
        configured_value=config.get("pos_weight"),
    )
    runtime_config = dict(config)
    runtime_config["resolved_pos_weight"] = resolved_pos_weight
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [resolved_pos_weight],
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    amp_enabled = bool(config["amp"] and device.type == "cuda")
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
        print("[INFO] Resume desactivado. Iniciando entrenamiento desde cero.")

    run_metadata = {
        "experiment_name": config["experiment_name"],
        "task": config["task"],
        "started_at": utc_now_iso(),
        "config": runtime_config,
        "train_wsi": len(train_dataset),
        "valid_wsi": len(valid_dataset),
        "train_class_counts": {
            "0": int(np.sum(np.asarray(train_dataset.labels) == 0)),
            "1": int(np.sum(np.asarray(train_dataset.labels) == 1)),
        },
        "valid_class_counts": {
            "0": int(np.sum(np.asarray(valid_dataset.labels) == 0)),
            "1": int(np.sum(np.asarray(valid_dataset.labels) == 1)),
        },
        "parameters_total": total_parameters,
        "parameters_trainable": trainable_parameters,
        "resolved_pos_weight": resolved_pos_weight,
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

    patience = int(config["early_stopping_patience"])
    threshold = float(config["threshold_default"])

    for epoch in range(start_epoch, final_epoch + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            amp=bool(config["amp"]),
        )
        valid_result = evaluate_binary(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            amp=bool(config["amp"]),
        )
        score, monitor_used = monitor_score(valid_result, str(config["monitor_metric"]))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_result["loss"],
            "valid_accuracy": valid_result["accuracy"],
            "valid_precision": valid_result["precision"],
            "valid_recall": valid_result["recall"],
            "valid_specificity": valid_result["specificity"],
            "valid_f1": valid_result["f1"],
            "valid_auc": valid_result["auc_roc"],
            "valid_gini": valid_result["gini"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "monitor_used": monitor_used,
            "monitor_score": score,
        }
        history.append(row)
        append_history_csv(history, history_path)
        save_loss_history(
            history,
            plots_dir / "training_loss.png",
            title_prefix="ABMIL + Virchow2",
        )
        save_metric_history(
            history,
            plots_dir / "validation_auc_f1.png",
            title_prefix="ABMIL + Virchow2",
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
                seed=int(config["seed"]),
                git_info=git_info,
                early_stopping_counter=early_stopping_counter,
            )
        else:
            early_stopping_counter += 1

        if bool(config["save_every_epoch"]):
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
                seed=int(config["seed"]),
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
            seed=int(config["seed"]),
            git_info=git_info,
            early_stopping_counter=early_stopping_counter,
        )

        print(
            f"[INFO] epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"valid_loss={valid_result['loss']:.6f} "
            f"valid_auc={valid_result['auc_roc']} "
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

    print(f"[INFO] Entrenamiento ABMIL + Virchow2 finalizado. Best epoch: {best_epoch}")
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
