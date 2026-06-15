"""Lightweight contracts for ABMIL training on UNI2-h embeddings."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.mil.abmil import ABMIL
from src.mil.engine import (
    compute_binary_metrics,
    evaluate_binary,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from src.mil.uni2h_dataset import (
    UNI2HDatasetError,
    UNI2HEmbeddingDataset,
    uni2h_bag_collate_fn,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_fake_embedding(
    path: Path,
    *,
    embedding_dim: int = 1536,
    encoder_family: str = "UNI2-h",
    label: int = 1,
    n_tiles: int = 3,
) -> None:
    payload = {
        "slide_id": path.stem,
        "features": torch.randn(n_tiles, embedding_dim),
        "cancer_label": label,
        "split": path.parent.name,
        "encoder_name": "MahmoodLab/UNI2-h",
        "encoder_family": encoder_family,
        "embedding_dim": embedding_dim,
        "isup_grade": 2 if label else 0,
        "gleason_score": "3+4" if label else "negative",
        "tile_ids": [f"tile-{index}" for index in range(n_tiles)],
        "tile_paths": [f"tile-{index}.png" for index in range(n_tiles)],
        "coordinates": torch.zeros(n_tiles, 2),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class UNI2HTrainingDatasetTests(unittest.TestCase):
    def test_accepts_valid_uni2h_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_fake_embedding(root / "train" / "slide-a.pt")

            dataset = UNI2HEmbeddingDataset(root, split="train")
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["features"].shape, (3, 1536))
            self.assertEqual(sample["label"], 1.0)

    def test_rejects_uni_classic_1024_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_fake_embedding(
                root / "train" / "slide-old.pt",
                embedding_dim=1024,
            )

            with self.assertRaisesRegex(UNI2HDatasetError, "UNI clasico"):
                UNI2HEmbeddingDataset(root, split="train")

    def test_rejects_wrong_encoder_family(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_fake_embedding(
                root / "train" / "slide-wrong.pt",
                encoder_family="UNI",
            )

            with self.assertRaisesRegex(UNI2HDatasetError, "encoder_family"):
                UNI2HEmbeddingDataset(root, split="train")


class EngineMetricTests(unittest.TestCase):
    def test_compute_binary_metrics(self) -> None:
        metrics = compute_binary_metrics(
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.8, 0.7, 0.9],
            threshold=0.5,
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["auc_roc"], 0.75)
        self.assertAlmostEqual(metrics["gini"], 0.5)
        self.assertEqual(
            metrics["confusion_matrix"],
            {"tn": 1, "fp": 1, "fn": 0, "tp": 2},
        )

    def test_cpu_training_and_evaluation_work_with_variable_bags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_fake_embedding(
                root / "train" / "negative.pt",
                label=0,
                n_tiles=2,
            )
            write_fake_embedding(
                root / "train" / "positive.pt",
                label=1,
                n_tiles=4,
            )
            dataset = UNI2HEmbeddingDataset(root, split="train")
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=uni2h_bag_collate_fn,
            )
            model = ABMIL(
                input_dim=1536,
                hidden_dim=8,
                attention_dim=4,
                dropout=0.0,
            )
            optimizer = AdamW(model.parameters(), lr=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            device = torch.device("cpu")

            loss = train_one_epoch(
                model,
                loader,
                optimizer,
                criterion,
                device,
                scaler=None,
                amp=False,
            )
            result = evaluate_binary(
                model,
                loader,
                criterion,
                device,
                threshold=0.5,
                amp=False,
            )

            self.assertGreaterEqual(loss, 0.0)
            self.assertEqual(len(result["slide_ids"]), 2)
            self.assertEqual(len(result["probabilities"]), 2)


class CheckpointTests(unittest.TestCase):
    def test_save_and_load_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last_checkpoint.pt"
            model = ABMIL(
                input_dim=1536,
                hidden_dim=8,
                attention_dim=4,
                dropout=0.0,
            )
            optimizer = AdamW(model.parameters(), lr=1e-4)
            history = [{"epoch": 1, "train_loss": 0.5, "valid_auc": 0.8}]

            save_checkpoint(
                path,
                epoch=1,
                model=model,
                optimizer=optimizer,
                best_metric=0.8,
                best_epoch=1,
                train_history=history,
                config={"input_dim": 1536},
                seed=42,
            )
            raw = torch.load(path, map_location="cpu", weights_only=False)

            self.assertIn("model_state_dict", raw)
            self.assertIn("optimizer_state_dict", raw)
            self.assertEqual(raw["epoch"], 1)
            self.assertEqual(raw["best_metric"], 0.8)

            restored_model = ABMIL(
                input_dim=1536,
                hidden_dim=8,
                attention_dim=4,
                dropout=0.0,
            )
            restored_optimizer = AdamW(restored_model.parameters(), lr=1e-4)
            restored = load_checkpoint(
                path,
                model=restored_model,
                optimizer=restored_optimizer,
                device="cpu",
            )

            self.assertEqual(restored["epoch"], 1)
            self.assertEqual(restored["train_history"], history)


class TrainingScriptDryRunTests(unittest.TestCase):
    def test_dry_run_builds_model_without_writing_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embeddings = root / "embeddings"
            write_fake_embedding(
                embeddings / "train" / "train-negative.pt",
                label=0,
            )
            write_fake_embedding(
                embeddings / "train" / "train-positive.pt",
                label=1,
            )
            write_fake_embedding(
                embeddings / "valid" / "valid-negative.pt",
                label=0,
            )
            write_fake_embedding(
                embeddings / "valid" / "valid-positive.pt",
                label=1,
            )

            config = {
                "experiment_name": "abmil_uni2h_binary",
                "task": "binary",
                "label_column": "cancer_label",
                "input_dim": 1536,
                "encoder_name": "MahmoodLab/UNI2-h",
                "encoder_family": "UNI2-h",
                "embeddings_root": str(embeddings),
                "output_root": str(root / "outputs"),
                "checkpoints_dir": str(root / "outputs" / "checkpoints"),
                "metrics_dir": str(root / "outputs" / "metrics"),
                "plots_dir": str(root / "outputs" / "plots"),
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size_bags": 1,
                "learning_rate": 0.0001,
                "weight_decay": 0.00001,
                "dropout": 0.25,
                "hidden_dim": 8,
                "attention_dim": 4,
                "num_workers": 0,
                "amp": False,
                "early_stopping_patience": 2,
                "monitor_metric": "valid_auc",
                "save_every_epoch": True,
                "resume": True,
                "threshold_default": 0.5,
                "max_train": None,
                "max_valid": None,
                "max_test": None,
            }
            config_path = root / "train_config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "11_train_abmil_uni2h_binary.py"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("Dry-run completado", result.stdout)
            checkpoint_dir = root / "outputs" / "checkpoints"
            self.assertFalse(list(checkpoint_dir.glob("*.pt")))


class EndToEndScriptTests(unittest.TestCase):
    def test_one_epoch_training_and_final_evaluation_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embeddings = root / "embeddings"
            for split in ("train", "valid", "test"):
                write_fake_embedding(
                    embeddings / split / f"{split}-negative.pt",
                    label=0,
                    n_tiles=2,
                )
                write_fake_embedding(
                    embeddings / split / f"{split}-positive.pt",
                    label=1,
                    n_tiles=3,
                )

            output_root = root / "outputs"
            config = {
                "experiment_name": "abmil_uni2h_binary",
                "task": "binary",
                "label_column": "cancer_label",
                "input_dim": 1536,
                "encoder_name": "MahmoodLab/UNI2-h",
                "encoder_family": "UNI2-h",
                "embeddings_root": str(embeddings),
                "output_root": str(output_root),
                "checkpoints_dir": str(output_root / "checkpoints"),
                "metrics_dir": str(output_root / "metrics"),
                "plots_dir": str(output_root / "plots"),
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size_bags": 1,
                "learning_rate": 0.0001,
                "weight_decay": 0.00001,
                "dropout": 0.0,
                "hidden_dim": 8,
                "attention_dim": 4,
                "num_workers": 0,
                "amp": False,
                "early_stopping_patience": 2,
                "monitor_metric": "valid_auc",
                "save_every_epoch": False,
                "resume": False,
                "threshold_default": 0.5,
                "max_train": None,
                "max_valid": None,
                "max_test": None,
            }
            config_path = root / "pipeline_config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            train_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "11_train_abmil_uni2h_binary.py"),
                    "--config",
                    str(config_path),
                    "--no-resume",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                train_result.returncode,
                0,
                msg=train_result.stdout + train_result.stderr,
            )
            self.assertTrue((output_root / "checkpoints" / "last_checkpoint.pt").is_file())
            self.assertTrue((output_root / "checkpoints" / "best_model.pt").is_file())
            self.assertTrue((output_root / "metrics" / "train_history.csv").is_file())

            config["epochs"] = 2
            config["resume"] = True
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            resume_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "11_train_abmil_uni2h_binary.py"),
                    "--config",
                    str(config_path),
                    "--resume",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                resume_result.returncode,
                0,
                msg=resume_result.stdout + resume_result.stderr,
            )
            self.assertIn("Resuming training from epoch 2", resume_result.stdout)
            last_checkpoint = torch.load(
                output_root / "checkpoints" / "last_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(last_checkpoint["epoch"], 2)
            self.assertEqual(len(last_checkpoint["train_history"]), 2)

            eval_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "12_evaluate_abmil_uni2h_binary.py"),
                    "--config",
                    str(config_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                eval_result.returncode,
                0,
                msg=eval_result.stdout + eval_result.stderr,
            )
            for filename in (
                "valid_predictions.csv",
                "test_predictions.csv",
                "valid_metrics.json",
                "test_metrics.json",
                "best_threshold.json",
            ):
                self.assertTrue((output_root / "metrics" / filename).is_file())
            for filename in (
                "confusion_matrix_valid.png",
                "confusion_matrix_test.png",
                "roc_valid.png",
                "roc_test.png",
            ):
                self.assertTrue((output_root / "plots" / filename).is_file())


if __name__ == "__main__":
    unittest.main()
