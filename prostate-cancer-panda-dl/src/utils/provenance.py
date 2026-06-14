"""Reproducibility metadata for scientific experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 hash without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_software_versions() -> Dict[str, str | None]:
    """Collect versions needed to reproduce feature extraction."""
    return {
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "PIL": _package_version("Pillow"),
        "huggingface_hub": _package_version("huggingface-hub"),
    }


def get_cuda_info() -> Dict[str, Any]:
    """Collect CUDA and GPU information without failing on CPU systems."""
    try:
        import torch
    except ImportError:
        return {"available": False, "error": "torch no esta instalado"}

    info: Dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if not torch.cuda.is_available():
        return info

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    info.update(
        {
            "device_index": int(device_index),
            "gpu_name": torch.cuda.get_device_name(device_index),
            "total_memory_bytes": int(properties.total_memory),
            "total_memory_gb": round(properties.total_memory / (1024**3), 3),
        }
    )
    return info


def _git_command(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def get_git_info(project_root: Path) -> Dict[str, str | None]:
    """Return commit and branch when the project is inside a Git repository."""
    root = Path(project_root)
    return {
        "commit": _git_command(root, "rev-parse", "HEAD"),
        "branch": _git_command(root, "rev-parse", "--abbrev-ref", "HEAD"),
    }


def build_experiment_metadata(
    config: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    """Build common provenance attached to every UNI2-h embedding."""
    return {
        "experiment_name": config.get("experiment_name"),
        "task": config.get("task"),
        "label_column": config.get("label_column"),
        "encoder_name": config.get("encoder_name"),
        "encoder_family": config.get("encoder_family"),
        "expected_embedding_dim": config.get("expected_embedding_dim"),
        "image_size": config.get("image_size"),
        "created_at": utc_now_iso(),
        "software_versions": get_software_versions(),
        "cuda": get_cuda_info(),
        "git": get_git_info(project_root),
    }
