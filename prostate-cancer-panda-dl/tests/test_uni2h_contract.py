"""Lightweight contract tests for the official UNI2-h extraction flow."""

from __future__ import annotations

import inspect
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data.zip_batches import (
    ZipBatchError,
    cleanup_temporary_batch,
    extract_batch_temporarily,
    inspect_zip_structure,
    list_batch_zips,
    read_manifest_from_zip,
)
from src.encoders import uni2h
from src.encoders.uni2h import (
    EXPECTED_UNI2H_DIM,
    UNI2HContractError,
    validate_embedding_payload,
    validate_embedding_tensor,
)
from src.utils.provenance import (
    get_cuda_info,
    get_git_info,
    get_software_versions,
    sha256_file,
    utc_now_iso,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_extraction_script():
    script_path = PROJECT_ROOT / "scripts" / "10_extract_uni2h_embeddings.py"
    spec = importlib.util.spec_from_file_location("extract_uni2h_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo importar el script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimum_payload(features: torch.Tensor) -> dict:
    return {
        "slide_id": "slide-001",
        "features": features,
        "tile_ids": ["tile-001"],
        "tile_paths": ["selected_tiles/train/slide-001/tile-001.png"],
        "coordinates": torch.tensor([[0.0, 0.0]]),
        "split": "train",
        "cancer_label": 1,
        "isup_grade": 2,
        "gleason_score": "3+4",
        "encoder_name": "MahmoodLab/UNI2-h",
        "encoder_family": "UNI2-h",
        "embedding_dim": EXPECTED_UNI2H_DIM,
        "image_size": 224,
        "transform_info": {"image_size": 224},
        "source_zip": "batch_0000_0099.zip",
        "source_manifest_path": "batch_0000_0099/metadata/tile_manifest.csv",
        "manifest_hash": "abc123",
        "created_at": utc_now_iso(),
        "software_versions": {},
        "git": {},
        "cuda": {"available": False},
    }


class UNI2HContractTests(unittest.TestCase):
    def test_config_expected_embedding_dimension(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "abmil_uni2h_binary.yaml"
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.assertEqual(config["expected_embedding_dim"], 1536)
        self.assertEqual(config["encoder_name"], "MahmoodLab/UNI2-h")

    def test_rejects_uni_classic_1024_dimension(self) -> None:
        with self.assertRaisesRegex(UNI2HContractError, "UNI clasico"):
            validate_embedding_tensor(torch.zeros(2, 1024))

    def test_accepts_uni2h_dimension_and_minimum_metadata(self) -> None:
        payload = _minimum_payload(torch.zeros(1, EXPECTED_UNI2H_DIM))
        validate_embedding_payload(payload)

    def test_rejects_missing_metadata(self) -> None:
        payload = _minimum_payload(torch.zeros(1, EXPECTED_UNI2H_DIM))
        del payload["manifest_hash"]
        with self.assertRaisesRegex(UNI2HContractError, "manifest_hash"):
            validate_embedding_payload(payload)

    def test_new_flow_reads_token_without_assigning_it(self) -> None:
        encoder_source = inspect.getsource(uni2h)
        script_source = (
            PROJECT_ROOT / "scripts" / "10_extract_uni2h_embeddings.py"
        ).read_text(encoding="utf-8")
        combined = encoder_source + script_source
        self.assertIn('os.environ.get("HF_TOKEN")', encoder_source)
        self.assertNotIn('os.environ["HF_TOKEN"]', combined)
        self.assertNotIn("os.environ['HF_TOKEN']", combined)


class ProvenanceTests(unittest.TestCase):
    def test_provenance_functions_are_lightweight_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "manifest.csv"
            file_path.write_text("slide_id\nabc\n", encoding="utf-8")
            digest = sha256_file(file_path)

            self.assertEqual(len(digest), 64)
            self.assertIn("python", get_software_versions())
            self.assertIn("available", get_cuda_info())
            self.assertIn("commit", get_git_info(Path(temp_dir)))
            self.assertIn("+00:00", utc_now_iso())


class ZipBatchStructureTests(unittest.TestCase):
    def _create_zip(self, zip_path: Path, batch_prefix: str) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "slide_id": "slide-001",
                    "tile_id": "tile-001",
                    "tile_path": "tile-001.png",
                    "split": "train",
                    "cancer_label": 1,
                }
            ]
        )
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                f"{batch_prefix}/metadata/tile_manifest.csv",
                manifest.to_csv(index=False),
            )
            archive.writestr(
                f"{batch_prefix}/selected_tiles/train/slide-001/tile-001.png",
                b"small-test-png-placeholder",
            )
            archive.writestr(f"{batch_prefix}/summary.json", "{}")

    def test_detects_direct_batch_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "batch_0000_0099.zip"
            self._create_zip(zip_path, "batch_0000_0099")

            layout = inspect_zip_structure(zip_path)
            manifest = read_manifest_from_zip(layout)

            self.assertEqual(layout.batch_root.as_posix(), "batch_0000_0099")
            self.assertEqual(len(manifest), 1)

    def test_lists_batches_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in (
                "batch_10000_10099.zip",
                "batch_9800_9899.zip",
                "batch_9900_9999.zip",
            ):
                (root / name).touch()
            names = [path.name for path in list_batch_zips(root)]
            self.assertEqual(
                names,
                [
                    "batch_9800_9899.zip",
                    "batch_9900_9999.zip",
                    "batch_10000_10099.zip",
                ],
            )

    def test_detects_additional_root_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "batch_0100_0199.zip"
            self._create_zip(
                zip_path,
                "panda_outputs_batches/batch_0100_0199",
            )

            layout = inspect_zip_structure(zip_path)
            self.assertEqual(
                layout.batch_root.as_posix(),
                "panda_outputs_batches/batch_0100_0199",
            )

    def test_extracts_and_cleans_temporary_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "batch_0200_0299.zip"
            self._create_zip(zip_path, "batch_0200_0299")

            layout = inspect_zip_structure(zip_path)
            extracted = extract_batch_temporarily(layout, root / "work")
            self.assertTrue(extracted.tile_manifest.is_file())
            self.assertTrue(extracted.selected_tiles.is_dir())

            cleanup_temporary_batch(extracted.extraction_root)
            self.assertFalse(extracted.extraction_root.exists())

    def test_rejects_zip_without_selected_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "broken.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "batch_0000_0099/metadata/tile_manifest.csv",
                    "slide_id,tile_id,tile_path,split,cancer_label\n",
                )
            with self.assertRaisesRegex(ZipBatchError, "selected_tiles"):
                inspect_zip_structure(zip_path)


class DryRunIntegrationTests(unittest.TestCase):
    def test_dry_run_needs_no_token_and_creates_no_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            zip_path = raw_dir / "batch_0000_0099.zip"
            ZipBatchStructureTests()._create_zip(zip_path, "batch_0000_0099")

            config = {
                "experiment_name": "abmil_uni2h_binary",
                "task": "binary",
                "label_column": "cancer_label",
                "encoder_name": "MahmoodLab/UNI2-h",
                "encoder_family": "UNI2-h",
                "expected_embedding_dim": 1536,
                "image_size": 224,
                "batch_size_tiles": 1,
                "amp": True,
                "num_workers": 0,
                "pin_memory": False,
                "seed": 42,
                "device": "cpu",
                "drive_raw_batches_dir": str(raw_dir),
                "work_dir": str(root / "work"),
                "output_root": str(root / "outputs"),
                "embeddings_dir": str(root / "outputs" / "embeddings"),
                "metrics_dir": str(root / "outputs" / "metrics"),
                "plots_dir": str(root / "outputs" / "plots"),
                "checkpoints_dir": str(root / "outputs" / "checkpoints"),
                "resume": True,
                "max_wsi": None,
                "force": False,
                "dry_run": False,
                "limit_slides": None,
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            environment = dict(os.environ)
            environment.pop("HF_TOKEN", None)
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "10_extract_uni2h_embeddings.py"),
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--max-wsi",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("UNI2-h cargado: no", result.stdout)
            self.assertFalse(list((root / "outputs" / "embeddings").rglob("*.pt")))


class ExtractionIntegrationTests(unittest.TestCase):
    def test_fake_encoder_writes_valid_1536_dimension_artifact(self) -> None:
        class FakeEncoder:
            transform_info = {"model_id": "MahmoodLab/UNI2-h", "image_size": 224}

            def encode_paths(self, tile_paths, batch_size):
                return torch.zeros(len(tile_paths), EXPECTED_UNI2H_DIM)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            zip_path = raw_dir / "batch_0000_0099.zip"
            ZipBatchStructureTests()._create_zip(zip_path, "batch_0000_0099")

            config = {
                "experiment_name": "abmil_uni2h_binary",
                "task": "binary",
                "label_column": "cancer_label",
                "encoder_name": "MahmoodLab/UNI2-h",
                "encoder_family": "UNI2-h",
                "expected_embedding_dim": 1536,
                "image_size": 224,
                "batch_size_tiles": 1,
                "amp": True,
                "num_workers": 0,
                "pin_memory": False,
                "seed": 42,
                "device": "cpu",
                "drive_raw_batches_dir": str(raw_dir),
                "work_dir": str(root / "work"),
                "output_root": str(root / "outputs"),
                "embeddings_dir": str(root / "outputs" / "embeddings"),
                "metrics_dir": str(root / "outputs" / "metrics"),
                "plots_dir": str(root / "outputs" / "plots"),
                "checkpoints_dir": str(root / "outputs" / "checkpoints"),
                "resume": True,
                "max_wsi": 1,
                "force": False,
                "dry_run": False,
                "limit_slides": None,
            }

            script = _load_extraction_script()
            script.create_output_directories(config)
            summary = script.run_extraction([zip_path], config, FakeEncoder())

            output_path = root / "outputs" / "embeddings" / "train" / "slide-001.pt"
            payload = torch.load(output_path, map_location="cpu", weights_only=False)
            validate_embedding_payload(payload)

            self.assertFalse(summary["has_errors"])
            self.assertEqual(payload["features"].shape, (1, 1536))
            self.assertFalse((root / "work" / "batch_0000_0099").exists())


if __name__ == "__main__":
    unittest.main()
