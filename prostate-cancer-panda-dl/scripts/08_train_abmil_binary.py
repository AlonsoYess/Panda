"""Train ABMIL on frozen UNI embeddings for binary cancer classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.abmil import ABMIL
from src.mil.dataset import MILDataset, mil_collate_fn
from src.mil.metrics import compute_binary_metrics, compute_youden_threshold
from src.mil.utils import ensure_dir, get_device, load_yaml, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ABMIL binary classifier on UNI embeddings.")
    parser.add_argument("--config", type=Path, required=True, help="Ruta a configs/abmil_uni_binary.yaml")
    return parser.parse_args()


def infer_input_dim(dataset: MILDataset) -> int:
    sample = dataset[0]
    return int(sample["features"].shape[1])


def compute_pos_weight(dataset: MILDataset) -> float:
    labels = np.asarray(dataset.labels, dtype=np.int64)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0:
        return 1.0
    return float(neg / pos)


def forward_batch(
    model: ABMIL,
    features_list: List[torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = []
    for feats in features_list:
        feats = feats.to(device, non_blocking=True)
        logit, _ = model(feats)
        logits.append(logit)

    logits_t = torch.stack(logits, dim=0).float()
    labels_t = labels.to(device, non_blocking=True).float()
    return logits_t, labels_t


def evaluate(
    model: ABMIL,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()
    losses = []
    all_true = []
    all_prob = []

    with torch.inference_mode():
        for batch in loader:
            logits, labels = forward_batch(
                model=model,
                features_list=batch["features"],
                labels=batch["labels"],
                device=device,
            )
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            all_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            all_prob.extend(probs.detach().cpu().numpy().astype(float).tolist())

    y_true = np.asarray(all_true, dtype=int)
    y_prob = np.asarray(all_prob, dtype=float)

    threshold_youden = compute_youden_threshold(y_true, y_prob)
    metrics_05 = compute_binary_metrics(y_true, y_prob, threshold=0.50)
    metrics_youden = compute_binary_metrics(y_true, y_prob, threshold=threshold_youden)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "y_true": y_true,
        "y_prob": y_prob,
        "metrics_05": metrics_05,
        "metrics_youden": metrics_youden,
        "threshold_youden": threshold_youden,
    }


def save_confusion_plot(cm_dict: Dict[str, int], path: Path, title: str) -> None:
    ensure_dir(path.parent)
    cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]], dtype=int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no_cancer", "cancer"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roc_plot(y_true: np.ndarray, y_prob: np.ndarray, path: Path, title: str) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "ROC no disponible: falta una clase", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return

    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    out_cfg = cfg["outputs"]

    set_seed(int(train_cfg.get("seed", 42)))
    device = get_device()

    embeddings_dir = Path(data_cfg["embeddings_dir"])
    base_out = Path(out_cfg["base_dir"])
    ckpt_dir = base_out / "checkpoints"
    metrics_dir = base_out / "metrics"
    plots_dir = base_out / "plots"
    for p in [ckpt_dir, metrics_dir, plots_dir]:
        ensure_dir(p)

    train_ds = MILDataset(embeddings_dir=embeddings_dir, split="train")
    valid_ds = MILDataset(embeddings_dir=embeddings_dir, split="valid")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size_slides", 8)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=mil_collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(train_cfg.get("batch_size_slides", 8)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=mil_collate_fn,
    )

    input_dim_cfg = model_cfg.get("input_dim", "auto")
    input_dim = infer_input_dim(train_ds) if str(input_dim_cfg).lower() == "auto" else int(input_dim_cfg)

    model = ABMIL(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.25)),
    ).to(device)

    pos_weight_val = compute_pos_weight(train_ds)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device))
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("patience", 7))

    history_rows = []
    best_score = -np.inf
    best_epoch = -1
    no_improve = 0
    best_valid_eval = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        epoch_true = []
        epoch_prob = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train", leave=False)
        for batch in loop:
            optimizer.zero_grad(set_to_none=True)

            logits, labels = forward_batch(
                model=model,
                features_list=batch["features"],
                labels=batch["labels"],
                device=device,
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            train_losses.append(float(loss.item()))
            epoch_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            epoch_prob.extend(probs.detach().cpu().numpy().astype(float).tolist())

        train_true_np = np.asarray(epoch_true, dtype=int)
        train_prob_np = np.asarray(epoch_prob, dtype=float)
        train_metrics = compute_binary_metrics(train_true_np, train_prob_np, threshold=0.5)

        valid_eval = evaluate(model=model, loader=valid_loader, criterion=criterion, device=device)
        valid_metrics_05 = valid_eval["metrics_05"]
        valid_metrics_youden = valid_eval["metrics_youden"]

        valid_auc = valid_metrics_05["auc"]
        if valid_auc is not None:
            score = float(valid_auc)
            score_name = "valid_auc"
        else:
            score = float(valid_metrics_youden["f1"])
            score_name = "valid_f1"

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "valid_loss": float(valid_eval["loss"]),
                "accuracy": float(valid_metrics_youden["accuracy"]),
                "precision": float(valid_metrics_youden["precision"]),
                "recall": float(valid_metrics_youden["recall"]),
                "specificity": float(valid_metrics_youden["specificity"]),
                "f1": float(valid_metrics_youden["f1"]),
                "auc": valid_metrics_youden["auc"],
                "gini": valid_metrics_youden["gini"],
                "selection_metric": score_name,
                "selection_score": score,
            }
        )

        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_valid_eval = valid_eval
            no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
                    "dropout": float(model_cfg.get("dropout", 0.25)),
                    "best_score": best_score,
                    "selection_metric": score_name,
                },
                ckpt_dir / "best_model.pt",
            )
        else:
            no_improve += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
                "dropout": float(model_cfg.get("dropout", 0.25)),
                "best_score": best_score,
                "selection_metric": score_name,
            },
            ckpt_dir / "last_model.pt",
        )

        print(
            f"[INFO] epoch={epoch} "
            f"train_loss={history_rows[-1]['train_loss']:.4f} "
            f"valid_loss={history_rows[-1]['valid_loss']:.4f} "
            f"valid_auc={valid_metrics_05['auc']} "
            f"valid_f1_youden={valid_metrics_youden['f1']:.4f}"
        )

        if no_improve >= patience:
            print(f"[INFO] Early stopping activado en epoch={epoch} (patience={patience})")
            break

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(metrics_dir / "train_history.csv", index=False)

    if best_valid_eval is None:
        print("[ERROR] No se pudo obtener evaluacion valida.")
        return 1

    valid_metrics_payload = {
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "metrics_threshold_0_50": best_valid_eval["metrics_05"],
        "metrics_threshold_youden": best_valid_eval["metrics_youden"],
    }
    save_json(valid_metrics_payload, metrics_dir / "valid_metrics.json")

    best_threshold_payload = {
        "best_epoch": best_epoch,
        "threshold_youden": float(best_valid_eval["threshold_youden"]),
    }
    save_json(best_threshold_payload, metrics_dir / "best_threshold.json")

    save_confusion_plot(
        cm_dict=best_valid_eval["metrics_youden"]["confusion_matrix"],
        path=plots_dir / "confusion_matrix_valid.png",
        title="Valid Confusion Matrix (Youden threshold)",
    )
    save_roc_plot(
        y_true=best_valid_eval["y_true"],
        y_prob=best_valid_eval["y_prob"],
        path=plots_dir / "roc_valid.png",
        title="Valid ROC",
    )

    print("\n[INFO] Entrenamiento ABMIL binario finalizado")
    print(f"- best_model: {ckpt_dir / 'best_model.pt'}")
    print(f"- last_model: {ckpt_dir / 'last_model.pt'}")
    print(f"- train_history: {metrics_dir / 'train_history.csv'}")
    print(f"- valid_metrics: {metrics_dir / 'valid_metrics.json'}")
    print(f"- best_threshold: {metrics_dir / 'best_threshold.json'}")

    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
