"""Path helpers for PANDA project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import yaml


PathLike = Union[str, Path]


def get_project_root() -> Path:
    """Return repository root from the location of this module."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: PathLike | None = None) -> Dict[str, str]:
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


def build_paths(config: Dict[str, str]) -> Dict[str, Path]:
    """Build absolute paths from config values."""
    data_root = Path(config["data_root"])
    paths = {
        "data_root": data_root,
        "train_csv": resolve_path(data_root, config["train_csv"]),
        "train_images_dir": resolve_path(data_root, config["train_images_dir"]),
        "train_label_masks_dir": resolve_path(data_root, config["train_label_masks_dir"]),
        "test_images_dir": resolve_path(data_root, config["test_images_dir"]),
        "sample_submission_csv": resolve_path(data_root, config["sample_submission_csv"]),
        "outputs_dir": resolve_path(get_project_root(), config["outputs_dir"]),
    }
    return paths
