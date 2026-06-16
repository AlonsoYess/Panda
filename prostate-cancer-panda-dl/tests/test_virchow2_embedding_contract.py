"""Local contract tests for Virchow2 embedding extraction."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.encoders.virchow2 import (
    VIRCHOW2_FAMILY,
    VIRCHOW2_MODEL_ID,
    Virchow2ContractError,
    normalize_virchow2_output,
    validate_embedding_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_numbered_script(script_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo importar el script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_payload(features: torch.Tensor) -> dict:
    n_tiles = int(features.shape[0])
    return {
        "slide_id": "slide-a",
        "features": features,
        "tile_ids": [f"tile-{index}" for index in range(n_tiles)],
        "tile_paths": [f"selected_tiles/train/slide-a/tile-{index}.png" for index in range(n_tiles)],
        "coordinates": [[float(index), float(index + 1)] for index in range(n_tiles)],
        "split": "train",
        "cancer_label": 1,
        "isup_grade": 2,
        "gleason_score": "3+4",
        "encoder_name": VIRCHOW2_MODEL_ID,
        "encoder_family": VIRCHOW2_FAMILY,
        "embedding_dim": int(features.shape[1]),
        "image_size": 224,
        "transform_info": {"transform": "fake"},
        "source_zip": "/tmp/batch_0000_0099.zip",
        "source_manifest_path": "batch_0000_0099/metadata/tile_manifest.csv",
        "manifest_hash": "abc123",
        "created_at": "2026-06-16T00:00:00+00:00",
        "software_versions": {"torch": torch.__version__},
        "git": {"commit": None, "branch": None},
        "cuda": {"available": False},
    }


class Virchow2OutputNormalizationTests(unittest.TestCase):
    def test_import_without_cuda(self) -> None:
        module = __import__("src.encoders.virchow2", fromlist=["Virchow2Encoder"])
        self.assertTrue(hasattr(module, "Virchow2Encoder"))

    def test_dict_cls_and_patch_tokens_concatenates_cls_and_mean_patch(self) -> None:
        cls = torch.randn(2, 4)
        patches = torch.randn(2, 3, 4)
        output = normalize_virchow2_output(
            {
                "x_norm_clstoken": cls,
                "x_norm_patchtokens": patches,
            }
        )

        expected = torch.cat([cls, patches.mean(dim=1)], dim=1)
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertTrue(torch.allclose(output, expected.float()))

    def test_dict_only_cls_token(self) -> None:
        cls = torch.randn(2, 4)
        output = normalize_virchow2_output({"x_norm_clstoken": cls})

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.allclose(output, cls.float()))

    def test_tensor_bnc_uses_first_token(self) -> None:
        tensor = torch.randn(2, 5, 4)
        output = normalize_virchow2_output(tensor)

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.allclose(output, tensor[:, 0, :].float()))

    def test_tensor_bc_uses_direct_tensor(self) -> None:
        tensor = torch.randn(2, 4)
        output = normalize_virchow2_output(tensor)

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.allclose(output, tensor.float()))

    def test_unrecognized_output_raises_clear_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no reconocido|no reconocida"):
            normalize_virchow2_output({"unexpected": torch.randn(2, 4)})


class Virchow2ConfigAndScriptTests(unittest.TestCase):
    def test_config_exists_and_uses_virchow2_output_root(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "virchow2_extract_binary.yaml"
        self.assertTrue(config_path.is_file())

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.assertEqual(
            config["output_root"],
            "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary",
        )
        self.assertEqual(
            config["embeddings_root"],
            "/content/drive/MyDrive/PANDA_PROSTATE/outputs/virchow2_binary/embeddings",
        )

    def test_script_help_works_without_token_gpu_or_download(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "20_extract_virchow2_embeddings.py"),
                "--help",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--force", result.stdout)

    def test_output_path_guard_rejects_uni2h_outputs(self) -> None:
        script = load_numbered_script("20_extract_virchow2_embeddings.py")
        config = {
            "output_root": "/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary",
            "embeddings_root": "/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary/embeddings",
            "metrics_dir": "/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary/metrics",
            "logs_dir": "/content/drive/MyDrive/PANDA_PROSTATE/outputs/abmil_uni2h_binary/logs",
        }

        with self.assertRaises(ValueError):
            script.assert_virchow2_output_paths(config)

    def test_script_source_does_not_target_uni2h_output_for_writes(self) -> None:
        script_text = (
            PROJECT_ROOT / "scripts" / "20_extract_virchow2_embeddings.py"
        ).read_text(encoding="utf-8")

        self.assertIn("FORBIDDEN_OUTPUT_MARKERS", script_text)
        self.assertNotIn("outputs/abmil_uni2h_binary/embeddings", script_text)
        self.assertNotIn("outputs/clam_uni2h_binary/checkpoints", script_text)
        self.assertNotIn("outputs/transmil_uni2h_binary/checkpoints", script_text)


class Virchow2PayloadValidationTests(unittest.TestCase):
    def test_valid_payload_accepts_float32_features(self) -> None:
        payload = make_payload(torch.randn(3, 12, dtype=torch.float32))

        validate_embedding_payload(payload)

    def test_invalid_feature_dtype_is_rejected(self) -> None:
        payload = make_payload(torch.randn(3, 12, dtype=torch.float64))

        with self.assertRaises(Virchow2ContractError):
            validate_embedding_payload(payload)

    def test_nan_features_are_rejected(self) -> None:
        features = torch.randn(3, 12, dtype=torch.float32)
        features[0, 0] = float("nan")
        payload = make_payload(features)

        with self.assertRaises(Virchow2ContractError):
            validate_embedding_payload(payload)

    def test_existing_embedding_validation_loads_and_checks_pt(self) -> None:
        script = load_numbered_script("20_extract_virchow2_embeddings.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "slide-a.pt"
            torch.save(make_payload(torch.randn(3, 12, dtype=torch.float32)), path)

            loaded = script.validate_existing_embedding(path)

        self.assertEqual(loaded["slide_id"], "slide-a")


if __name__ == "__main__":
    unittest.main()
