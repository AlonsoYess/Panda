"""Evaluate ABMIL binary model on test embeddings and export reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mil.abmil import ABMIL
from src.mil.dataset import MILDataset, mil_collate_fn
from src.mil.metrics import compute_binary_metrics
from src.mil.utils import ensure_dir, get_device, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ABMIL binary model on test split.")
    parser.add_argument("--config", type=Path, required=True, help="Ruta a configs/abmil_uni_binary.yaml")
    return parser.parse_args()


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
    out_cfg = cfg["outputs"]
    train_cfg = cfg["train"]

    base_out = Path(out_cfg["base_dir"])
    embeddings_dir = Path(data_cfg["embeddings_dir"])
    ckpt_path = base_out / "checkpoints" / "best_model.pt"
    threshold_path = base_out / "metrics" / "best_threshold.json"
    metrics_dir = base_out / "metrics"
    plots_dir = base_out / "plots"
    ensure_dir(metrics_dir)
    ensure_dir(plots_dir)

    if not ckpt_path.exists():
        print(f"[ERROR] No existe checkpoint: {ckpt_path}")
        return 1
    if not threshold_path.exists():
        print(f"[ERROR] No existe threshold file: {threshold_path}")
        return 1

    with threshold_path.open("r", encoding="utf-8") as f:
        threshold_payload = json.load(f)
    threshold = float(threshold_payload["threshold_youden"])

    test_ds = MILDataset(embeddings_dir=embeddings_dir, split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=int(train_cfg.get("batch_size_slides", 8)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=mil_collate_fn,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    input_dim = int(checkpoint.get("input_dim", model_cfg.get("input_dim", 1024)))
    hidden_dim = int(checkpoint.get("hidden_dim", model_cfg.get("hidden_dim", 512)))
    dropout = float(checkpoint.get("dropout", model_cfg.get("dropout", 0.25)))

    device = get_device()
    model = ABMIL(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows_pred: List[Dict] = []
    rows_attn: List[Dict] = []
    y_true: List[int] = []
    y_prob: List[float] = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluando test"):
            features_list = batch["features"]
            labels = batch["labels"].numpy().astype(int).tolist()
            slide_ids = batch["slide_ids"]
            metadata_list = batch["metadata"]

            for feats, label, slide_id, meta in zip(features_list, labels, slide_ids, metadata_list):
                feats = feats.to(device, non_blocking=True)
                logit, attn = model(feats)
                prob = float(torch.sigmoid(logit).detach().cpu().item())
                pred = int(prob >= threshold)

                y_true.append(int(label))
                y_prob.append(prob)

                rows_pred.append(
                    {
                        "slide_id": slide_id,
                        "cancer_label": int(label),
                        "pred_prob": prob,
                        "pred_label": pred,
                        "threshold": threshold,
                        "isup_grade": meta.get("isup_grade"),
                        "gleason_score": meta.get("gleason_score"),
                    }
                )

                tile_ids = meta.get("tile_ids", [])
                tile_paths = meta.get("tile_paths", [])
                attn_np = attn.detach().cpu().numpy().tolist()
                for i, score in enumerate(attn_np):
                    rows_attn.append(
                        {
                            "slide_id": slide_id,
                            "tile_id": tile_ids[i] if i < len(tile_ids) else f"{slide_id}_tile_{i}",
                            "tile_path": tile_paths[i] if i < len(tile_paths) else "",
                            "attention_score": float(score),
                            "cancer_label": int(label),
                            "pred_prob": prob,
                            "pred_label": pred,
                        }
                    )

    y_true_np = np.asarray(y_true, dtype=int)
    y_prob_np = np.asarray(y_prob, dtype=float)
    metrics = compute_binary_metrics(y_true_np, y_prob_np, threshold=threshold)

    test_metrics_payload = {
        "threshold_used": threshold,
        "metrics": metrics,
    }

    pred_df = pd.DataFrame(rows_pred)
    attn_df = pd.DataFrame(rows_attn)

    save_json(test_metrics_payload, metrics_dir / "test_metrics.json")
    pred_df.to_csv(metrics_dir / "test_predictions.csv", index=False)
    attn_df.to_csv(metrics_dir / "test_attention_scores.csv", index=False)

    save_confusion_plot(
        cm_dict=metrics["confusion_matrix"],
        path=plots_dir / "confusion_matrix_test.png",
        title="Test Confusion Matrix",
    )
    save_roc_plot(
        y_true=y_true_np,
        y_prob=y_prob_np,
        path=plots_dir / "roc_test.png",
        title="Test ROC",
    )

    print("\n[INFO] Evaluacion test completada")
    print(f"- test_metrics: {metrics_dir / 'test_metrics.json'}")
    print(f"- test_predictions: {metrics_dir / 'test_predictions.csv'}")
    print(f"- test_attention_scores: {metrics_dir / 'test_attention_scores.csv'}")
    print(f"- roc_test: {plots_dir / 'roc_test.png'}")
    print(f"- confusion_matrix_test: {plots_dir / 'confusion_matrix_test.png'}")
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
