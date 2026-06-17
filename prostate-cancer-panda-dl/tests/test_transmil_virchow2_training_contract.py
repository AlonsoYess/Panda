"""Lightweight contracts for TransMIL training on Virchow2 embeddings."""

from __future__ import annotations

import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.mil.transmil import TransMILBinary
from src.mil.virchow2_dataset import Virchow2EmbeddingDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_fake_virchow2_embedding(
    path: Path,
    *,
    embedding_dim: int = 1280,
    encoder_family: str = "Virchow2",
    encoder_name: str = "paige-ai/Virchow2",
    label: int = 1,
    n_tiles: int = 4,
) -> None:
    payload = {
        "slide_id": path.stem,
        "features": torch.randn(n_tiles, embedding_dim, dtype=torch.float32),
        "cancer_label": label,
        "split": path.parent.name,
        "encoder_name": encoder_name,
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


def make_config(embeddings: Path, output_root: Path) -> dict:
    return {
        "experiment_name": "transmil_virchow2_binary",
        "task": "binary cancer detection",
        "label_key": "cancer_label",
        "label_column": "cancer_label",
        "input_dim": 1280,
        "encoder_name": "paige-ai/Virchow2",
        "encoder_family": "Virchow2",
        "embeddings_root": str(embeddings),
        "output_root": str(output_root),
        "checkpoints_dir": str(output_root / "checkpoints"),
        "metrics_dir": str(output_root / "metrics"),
        "plots_dir": str(output_root / "plots"),
        "logs_dir": str(output_root / "logs"),
        "entregables_dir": str(
            output_root.parent / "entregables" / "transmil_virchow2_binary_resultados"
        ),
        "hidden_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "max_tiles": 16,
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "early_stopping_patience": 2,
        "monitor": "valid_auc",
        "random_seed": 42,
        "threshold_default": 0.5,
        "threshold_selection": "Youden",
        "num_workers": 0,
        "pin_memory": False,
        "save_epoch_checkpoints": False,
        "resume": False,
        "mixed_precision": False,
        "device": "cpu",
        "max_train": None,
        "max_valid": None,
        "max_test": None,
        "expected_splits": ["train", "valid", "test"],
    }


class TransMILVirchow2ConfigAndScriptTests(unittest.TestCase):
    def test_expected_files_exist(self) -> None:
        expected_files = [
            PROJECT_ROOT / "configs" / "transmil_virchow2_train_binary.yaml",
            PROJECT_ROOT / "scripts" / "25_train_transmil_virchow2_binary.py",
            PROJECT_ROOT / "scripts" / "26_evaluate_transmil_virchow2_binary.py",
            PROJECT_ROOT / "docs" / "experiments" / "transmil_virchow2_training.md",
            PROJECT_ROOT / "src" / "mil" / "transmil.py",
            PROJECT_ROOT / "src" / "mil" / "virchow2_dataset.py",
        ]
        for path in expected_files:
            self.assertTrue(path.is_file(), msg=f"Falta archivo esperado: {path}")

    def test_scripts_compile(self) -> None:
        for script_name in (
            "25_train_transmil_virchow2_binary.py",
            "26_evaluate_transmil_virchow2_binary.py",
        ):
            py_compile.compile(
                str(PROJECT_ROOT / "scripts" / script_name),
                doraise=True,
            )

    def test_config_points_to_virchow2_embeddings_and_output(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "transmil_virchow2_train_binary.yaml"
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.assertEqual(config["experiment_name"], "transmil_virchow2_binary")
        self.assertEqual(config["input_dim"], 1280)
        self.assertEqual(config["encoder_family"], "Virchow2")
        self.assertEqual(config["label_key"], "cancer_label")
        self.assertEqual(config["label_column"], "cancer_label")
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["random_seed"], 42)
        self.assertEqual(
            config["embeddings_root"],
            "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings",
        )
        self.assertNotIn("uni2h", config["embeddings_root"].lower())
        self.assertNotIn("UNI2-h", config["encoder_family"])
        self.assertEqual(
            config["output_root"],
            "/content/drive/MyDrive/PANDA_PROSTATE/outputs/transmil_virchow2_binary",
        )
        self.assertEqual(
            config["entregables_dir"],
            "/content/drive/MyDrive/PANDA_PROSTATE/entregables/transmil_virchow2_binary_resultados",
        )

    def test_train_help_works(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "25_train_transmil_virchow2_binary.py"),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--output-root", result.stdout)
        self.assertIn("--max-train-slides", result.stdout)
        self.assertIn("--max-valid-slides", result.stdout)

    def test_evaluate_help_works(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "26_evaluate_transmil_virchow2_binary.py"),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("--max-test", result.stdout)
        self.assertIn("--output-root", result.stdout)

    def test_transmil_forward_accepts_virchow2_dim(self) -> None:
        model = TransMILBinary(
            input_dim=1280,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            dim_feedforward=16,
            dropout=0.0,
        )
        output = model(torch.randn(5, 1280))

        self.assertEqual(tuple(output["logit"].shape), ())
        self.assertEqual(tuple(output["attention"].shape), (5,))
        self.assertAlmostEqual(float(output["attention"].sum().item()), 1.0, places=5)

    def test_uses_existing_virchow2_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_fake_virchow2_embedding(root / "train" / "slide-a.pt")
            dataset = Virchow2EmbeddingDataset(root, split="train")
            self.assertEqual(dataset[0]["features"].shape, (4, 1280))

    def test_does_not_reprocess_images_zips_or_use_hf_token(self) -> None:
        for script_name in (
            "25_train_transmil_virchow2_binary.py",
            "26_evaluate_transmil_virchow2_binary.py",
        ):
            text = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
            self.assertNotIn("HF_TOKEN", text)
            self.assertNotIn("timm", text)
            self.assertNotIn("PIL", text)
            self.assertNotIn("Image.open", text)
            self.assertNotIn("zip_batches", text)
            self.assertNotIn("raw_batches", text)

    def test_existing_experiment_files_still_exist(self) -> None:
        for path in (
            PROJECT_ROOT / "scripts" / "15_train_transmil_uni2h_binary.py",
            PROJECT_ROOT / "scripts" / "16_evaluate_transmil_uni2h_binary.py",
            PROJECT_ROOT / "scripts" / "21_train_abmil_virchow2_binary.py",
            PROJECT_ROOT / "scripts" / "22_evaluate_abmil_virchow2_binary.py",
            PROJECT_ROOT / "scripts" / "23_train_clam_virchow2_binary.py",
            PROJECT_ROOT / "scripts" / "24_evaluate_clam_virchow2_binary.py",
            PROJECT_ROOT / "src" / "mil" / "transmil.py",
        ):
            self.assertTrue(path.is_file(), msg=f"Falta archivo existente: {path}")


class TransMILVirchow2ScriptIntegrationTests(unittest.TestCase):
    def test_dry_run_builds_model_without_writing_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embeddings = root / "embeddings"
            write_fake_virchow2_embedding(embeddings / "train" / "train-neg.pt", label=0)
            write_fake_virchow2_embedding(embeddings / "train" / "train-pos.pt", label=1)
            write_fake_virchow2_embedding(embeddings / "valid" / "valid-neg.pt", label=0)
            write_fake_virchow2_embedding(embeddings / "valid" / "valid-pos.pt", label=1)
            write_fake_virchow2_embedding(embeddings / "test" / "test-neg.pt", label=0)
            write_fake_virchow2_embedding(embeddings / "test" / "test-pos.pt", label=1)

            output_root = root / "outputs"
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(make_config(embeddings, output_root)),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "25_train_transmil_virchow2_binary.py"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=45,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("Dry-run TransMIL + Virchow2 completado", result.stdout)
            self.assertIn("Train WSI: 2", result.stdout)
            self.assertIn("Valid WSI: 2", result.stdout)
            self.assertIn("Test WSI: 2", result.stdout)
            self.assertIn("Sample slide_id:", result.stdout)
            self.assertIn("Features shape:", result.stdout)
            self.assertIn("Embedding dim: 1280", result.stdout)
            self.assertIn("Encoder family: Virchow2", result.stdout)
            self.assertIn("Attention shape:", result.stdout)
            self.assertIn("Output dir esperado:", result.stdout)
            self.assertFalse((output_root / "checkpoints").exists())

    def test_one_epoch_training_and_final_evaluation_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embeddings = root / "embeddings"
            for split in ("train", "valid", "test"):
                write_fake_virchow2_embedding(
                    embeddings / split / f"{split}-negative.pt",
                    label=0,
                    n_tiles=2,
                )
                write_fake_virchow2_embedding(
                    embeddings / split / f"{split}-positive.pt",
                    label=1,
                    n_tiles=3,
                )

            output_root = root / "outputs"
            config = make_config(embeddings, output_root)
            config_path = root / "pipeline_config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            train_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "25_train_transmil_virchow2_binary.py"),
                    "--config",
                    str(config_path),
                    "--no-resume",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(
                train_result.returncode,
                0,
                msg=train_result.stdout + train_result.stderr,
            )
            self.assertTrue((output_root / "checkpoints" / "best_model.pt").is_file())
            self.assertTrue((output_root / "metrics" / "train_history.csv").is_file())
            self.assertTrue((output_root / "metrics" / "run_metadata.json").is_file())

            eval_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "26_evaluate_transmil_virchow2_binary.py"),
                    "--config",
                    str(config_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(
                eval_result.returncode,
                0,
                msg=eval_result.stdout + eval_result.stderr,
            )
            valid_predictions = pd.read_csv(output_root / "metrics" / "valid_predictions.csv")
            self.assertTrue(
                (output_root / "metrics" / "evaluation_run_metadata.json").is_file()
            )
            self.assertEqual(
                list(valid_predictions.columns),
                [
                    "slide_id",
                    "cancer_label",
                    "pred_probability",
                    "pred_label_threshold_0_5",
                    "pred_label_threshold_youden",
                ],
            )
            self.assertTrue((output_root / "metrics" / "test_attention_scores.csv").is_file())
            entregable = output_root.parent / "entregables" / "transmil_virchow2_binary_resultados"
            self.assertTrue((entregable / "resumen_resultados_transmil_virchow2.json").is_file())
            self.assertTrue((entregable / "modelo" / "best_model.pt").is_file())
            self.assertTrue((entregable / "regiones_relevantes" / "test_attention_scores.csv").is_file())


if __name__ == "__main__":
    unittest.main()
