"""Path helpers for PANDA project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml


PathLike = Union[str, Path]


def get_project_root() -> Path:
    """Return repository root from the location of this module."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: PathLike | None = None) -> Dict[str, Any]:
    """Load YAML config file."""
    path = Path(config_path) if config_path else get_project_root() / "config.yaml"
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(base_root: Path, value: PathLike) -> Path:
    """Resolve path value to absolute path."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_root / path).resolve()


def build_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    """Build absolute paths from config values."""
    data_root = Path(str(config["data_root"]))
    outputs_dir = resolve_path(get_project_root(), str(config["outputs_dir"]))
    metadata_dir = outputs_dir / "metadata"
    logs_dir = outputs_dir / "logs"
    selected_tiles_dir = outputs_dir / "selected_tiles"

    paths = {
        "data_root": data_root,
        "train_csv": resolve_path(data_root, str(config["train_csv"])),
        "train_images_dir": resolve_path(data_root, str(config["train_images_dir"])),
        "train_label_masks_dir": resolve_path(data_root, str(config["train_label_masks_dir"])),
        "test_images_dir": resolve_path(data_root, str(config.get("test_images_dir", "test_images"))),
        "sample_submission_csv": resolve_path(data_root, str(config["sample_submission_csv"])),
        "outputs_dir": outputs_dir,
        "metadata_dir": metadata_dir,
        "logs_dir": logs_dir,
        "selected_tiles_dir": selected_tiles_dir,
        "splits_csv": metadata_dir / "splits.csv",
        "candidate_tiles_manifest_csv": metadata_dir / "candidate_tiles_manifest.csv",
        "tile_manifest_csv": metadata_dir / "tile_manifest.csv",
    }
    return paths


def get_split_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return split sub-config with defaults."""
    split_cfg = dict(config.get("split", {}))
    split_cfg.setdefault("train_size", 0.70)
    split_cfg.setdefault("valid_size", 0.15)
    split_cfg.setdefault("test_size", 0.15)
    split_cfg.setdefault("stratify_by", "isup_grade")
    return split_cfg
