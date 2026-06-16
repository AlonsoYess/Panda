"""Lightweight contracts for TransMIL + UNI2-h training."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch
import yaml

from src.mil.transmil import TransMILBinary
from src.mil.uni2h_dataset import UNI2HEmbeddingDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_numbered_script(script_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo importar el script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransMILModelContractTests(unittest.TestCase):
    def test_forward_returns_scalar_logit_and_attention(self) -> None:
        model = TransMILBinary(
            input_dim=1536,
            hidden_dim=64,
            num_layers=1,
            num_heads=4,
            dim_feedforward=128,
            dropout=0.0,
        )
        output = model(torch.randn(32, 1536))

        self.assertEqual(tuple(output["logit"].shape), ())
        self.assertEqual(tuple(output["attention"].shape), (32,))

    def test_attention_sums_to_one(self) -> None:
        model = TransMILBinary(
            input_dim=1536,
            hidden_dim=64,
            num_layers=1,
            num_heads=4,
            dim_feedforward=128,
            dropout=0.0,
        )
        output = model(torch.randn(32, 1536))

        self.assertAlmostEqual(float(output["attention"].sum().item()), 1.0, places=5)

    def test_supports_variable_number_of_tiles(self) -> None:
        model = TransMILBinary(
            input_dim=1536,
            hidden_dim=64,
            num_layers=1,
            num_heads=4,
            dim_feedforward=128,
            dropout=0.0,
        )
        for n_tiles in (8, 32, 64):
            output = model(torch.randn(n_tiles, 1536))
            self.assertEqual(tuple(output["attention"].shape), (n_tiles,))

    def test_model_works_on_cpu(self) -> None:
        model = TransMILBinary(
            input_dim=1536,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            dim_feedforward=64,
            dropout=0.0,
        )
        output = model(torch.randn(4, 1536))

        self.assertEqual(output["logit"].device.type, "cpu")
        self.assertEqual(output["attention"].device.type, "cpu")

    def test_import_and_instantiation_do_not_require_cuda(self) -> None:
        model = TransMILBinary(input_dim=1536, hidden_dim=32, num_layers=1, num_heads=4)
        output = model(torch.randn(2, 1536))

        self.assertEqual(tuple(output["attention"].shape), (2,))

    def test_truncates_when_more_tiles_than_max_tiles(self) -> None:
        model = TransMILBinary(
            input_dim=1536,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            dim_feedforward=64,
            dropout=0.0,
            max_tiles=16,
        )
        output = model(torch.randn(20, 1536))

        self.assertTrue(output["truncated"])
        self.assertEqual(output["n_tiles_original"], 20)
        self.assertEqual(output["n_tiles_used"], 16)
        self.assertEqual(tuple(output["attention"].shape), (16,))


class TransMILConfigAndScriptTests(unittest.TestCase):
    def test_train_help_works(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "15_train_transmil_uni2h_binary.py"),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("--dry-run", result.stdout)

    def test_evaluate_help_works(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "16_evaluate_transmil_uni2h_binary.py"),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("--max-test", result.stdout)

    def test_config_uses_transmil_output_and_abmil_embeddings_read_only(self) -> None:
        transmil_config_path = PROJECT_ROOT / "configs" / "transmil_uni2h_train_binary.yaml"
        abmil_config_path = PROJECT_ROOT / "configs" / "abmil_uni2h_train_binary.yaml"
        clam_config_path = PROJECT_ROOT / "configs" / "clam_uni2h_train_binary.yaml"

        with transmil_config_path.open("r", encoding="utf-8") as file:
            transmil_config = yaml.safe_load(file)
        with abmil_config_path.open("r", encoding="utf-8") as file:
            abmil_config = yaml.safe_load(file)
        with clam_config_path.open("r", encoding="utf-8") as file:
            clam_config = yaml.safe_load(file)

        self.assertEqual(
            transmil_config["output_root"],
            "/content/drive/MyDrive/PANDA_PROSTATE/outputs/transmil_uni2h_binary",
        )
        self.assertEqual(
            transmil_config["embeddings_root"],
            "/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary/embeddings",
        )
        self.assertEqual(
            abmil_config["output_root"],
            "/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary",
        )
        self.assertEqual(
            clam_config["output_root"],
            "/content/drive/MyDrive/PANDA_PROSTATE/outputs/clam_uni2h_binary",
        )
        self.assertNotEqual(transmil_config["output_root"], abmil_config["output_root"])
        self.assertNotEqual(transmil_config["output_root"], clam_config["output_root"])

    def test_prediction_csv_contract(self) -> None:
        eval_script = load_numbered_script("16_evaluate_transmil_uni2h_binary.py")

        frame = eval_script.prediction_frame(
            {
                "slide_ids": ["a", "b"],
                "labels": [0, 1],
                "probabilities": [0.25, 0.75],
            },
            threshold_default=0.5,
            threshold_youden=0.6,
        )
        self.assertEqual(
            list(frame.columns),
            [
                "slide_id",
                "cancer_label",
                "pred_probability",
                "pred_label_threshold_0_5",
                "pred_label_threshold_youden",
            ],
        )

    def test_scripts_use_lazy_dataset_initialization(self) -> None:
        train_text = (
            PROJECT_ROOT / "scripts" / "15_train_transmil_uni2h_binary.py"
        ).read_text(encoding="utf-8")
        eval_text = (
            PROJECT_ROOT / "scripts" / "16_evaluate_transmil_uni2h_binary.py"
        ).read_text(encoding="utf-8")

        self.assertIn("validate_on_init=False", train_text)
        self.assertIn("validate_on_init=False", eval_text)


class TransMILLazyDatasetTests(unittest.TestCase):
    def test_lazy_dataset_does_not_load_all_embeddings_on_init(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = Path(temp_dir) / "embeddings" / "train"
            split_dir.mkdir(parents=True)
            for index in range(3):
                (split_dir / f"slide-{index}.pt").write_bytes(b"not-loaded")

            payload = {
                "slide_id": "slide-0",
                "features": torch.randn(4, 1536),
                "cancer_label": 1,
                "split": "train",
                "encoder_name": "MahmoodLab/UNI2-h",
                "encoder_family": "UNI2-h",
                "embedding_dim": 1536,
            }

            with mock.patch(
                "src.mil.uni2h_dataset.torch.load",
                return_value=payload,
            ) as mocked_load:
                dataset = UNI2HEmbeddingDataset(
                    Path(temp_dir) / "embeddings",
                    split="train",
                    validate_on_init=False,
                )

                self.assertEqual(len(dataset), 3)
                mocked_load.assert_not_called()

                sample = dataset[0]
                self.assertEqual(sample["slide_id"], "slide-0")
                self.assertEqual(tuple(sample["features"].shape), (4, 1536))
                self.assertEqual(mocked_load.call_count, 1)


class TransMILSmokeIntegrationTests(unittest.TestCase):
    def _write_fake_embedding(self, path: Path, label: int) -> None:
        payload = {
            "slide_id": path.stem,
            "features": torch.randn(4, 1536),
            "cancer_label": label,
            "split": path.parent.name,
            "encoder_name": "MahmoodLab/UNI2-h",
            "encoder_family": "UNI2-h",
            "embedding_dim": 1536,
            "isup_grade": 2 if label else 0,
            "gleason_score": "3+4" if label else "negative",
            "tile_ids": [f"tile-{index}" for index in range(4)],
            "tile_paths": [f"tile-{index}.png" for index in range(4)],
            "coordinates": torch.zeros(4, 2),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def test_train_and_evaluate_one_epoch_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embeddings = root / "embeddings"
            for split in ("train", "valid", "test"):
                self._write_fake_embedding(embeddings / split / f"{split}-neg.pt", 0)
                self._write_fake_embedding(embeddings / split / f"{split}-pos.pt", 1)

            output_root = root / "transmil_outputs"
            config = {
                "experiment_name": "transmil_uni2h_binary",
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
            }
            config_path = root / "transmil_config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            train_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "15_train_transmil_uni2h_binary.py"),
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

            eval_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "16_evaluate_transmil_uni2h_binary.py"),
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


if __name__ == "__main__":
    unittest.main()
