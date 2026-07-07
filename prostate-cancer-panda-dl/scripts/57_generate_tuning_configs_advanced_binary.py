"""Generate validation-only tuning configs for selected advanced binary MIL runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_CONFIG_DIR = CONFIG_DIR / "tuning_advanced_binary"
DEFAULT_TUNING_OUTPUTS_ROOT = "/content/drive/MyDrive/PANDA_PROSTATE/outputs/tuning_advanced_binary"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_CONFIG_DIR / "tuning_manifest.csv"
SELECTION_OBJECTIVE = "valid_auc"

CANDIDATES = {
    "dtfdmil": {
        "base_config": CONFIG_DIR / "dtfdmil_virchow2_advanced_train_binary.yaml",
        "grid": {
            "learning_rate": [1e-4, 5e-5],
            "weight_decay": [1e-5, 1e-4],
            "dropout": [0.25, 0.35],
            "hidden_dim": [256, 512],
            "loss_function": ["bce", "weighted_bce", "focal"],
        },
    },
    "abmil": {
        "base_config": CONFIG_DIR / "abmil_virchow2_advanced_train_binary.yaml",
        "grid": {
            "learning_rate": [1e-4, 5e-5],
            "weight_decay": [1e-5, 1e-4],
            "dropout": [0.25, 0.35],
            "attention_dim": [128, 256],
            "loss_function": ["bce", "weighted_bce", "focal"],
        },
    },
    "clam": {
        "base_config": CONFIG_DIR / "clam_virchow2_advanced_train_binary.yaml",
        "grid": {
            "learning_rate": [1e-4, 5e-5],
            "weight_decay": [1e-5, 1e-4],
            "dropout": [0.25, 0.35],
            "attention_dim": [128, 256],
            "k_sample": [8, 16],
            "instance_loss_weight": [0.3, 0.5],
            "loss_function": ["bce", "weighted_bce", "focal"],
        },
    },
    "acmil": {
        "base_config": CONFIG_DIR / "acmil_virchow2_advanced_train_binary.yaml",
        "grid": {
            "learning_rate": [1e-4, 5e-5],
            "weight_decay": [1e-5, 1e-4],
            "dropout": [0.25, 0.35],
            "attention_dim": [128, 256],
            "loss_function": ["bce", "weighted_bce", "focal"],
        },
    },
}

MANIFEST_COLUMNS = [
    "tuning_id",
    "encoder",
    "model",
    "base_config",
    "config_path",
    "output_dir",
    "lr",
    "weight_decay",
    "dropout",
    "hidden_dim",
    "attention_dim",
    "k_sample",
    "instance_loss_weight",
    "loss_function",
    "selection_objective",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small deterministic tuning grid for advanced binary MIL."
    )
    parser.add_argument("--output-config-dir", type=Path, default=DEFAULT_OUTPUT_CONFIG_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=DEFAULT_TUNING_OUTPUTS_ROOT,
        help="Absolute root where tuning outputs will be written, ideally on Google Drive.",
    )
    parser.add_argument(
        "--output-root-prefix",
        dest="outputs_root",
        type=str,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-variants-per-model", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe config base: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML invalida: {path}")
    return data


def dump_yaml(path: Path, data: Dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {path}. Usa --overwrite si quieres regenerar configs de tuning."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def grid_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def deterministic_subset(items: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    """Select an evenly spaced deterministic subset from a full grid."""
    if max_items < 1:
        raise ValueError("--max-variants-per-model debe ser >= 1.")
    if len(items) <= max_items:
        return items
    if max_items == 1:
        return [items[0]]
    indices = [
        round(index * (len(items) - 1) / (max_items - 1))
        for index in range(max_items)
    ]
    return [items[index] for index in indices]


def apply_loss_policy(config: Dict[str, Any], loss_function: str) -> None:
    """Record loss intent while keeping current trainer compatibility."""
    config["loss_function"] = str(loss_function)
    if loss_function == "bce":
        config["pos_weight"] = 1.0
    elif loss_function == "weighted_bce":
        config["pos_weight"] = None
    elif loss_function == "focal":
        config["pos_weight"] = None
        config.setdefault("focal_gamma", 2.0)
        config.setdefault("focal_alpha", None)
    else:
        raise ValueError(f"loss_function no soportada: {loss_function}")


def set_output_paths(config: Dict[str, Any], output_dir: Path | PurePosixPath) -> None:
    output_dir_text = output_dir.as_posix()
    config["output_root"] = output_dir_text
    config["checkpoints_dir"] = (output_dir / "checkpoints").as_posix()
    config["metrics_dir"] = (output_dir / "metrics").as_posix()
    config["plots_dir"] = (output_dir / "plots").as_posix()
    config["logs_dir"] = (output_dir / "logs").as_posix()


def absolute_outputs_root(path: str | Path) -> Path | PurePosixPath:
    """Return an absolute output root so generated configs never write into repo by accident."""
    text = str(path).replace("\\", "/")
    if text.startswith("/"):
        return PurePosixPath(text)
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return candidate.resolve()


def display_path(path: Path) -> str:
    """Return a stable repo-relative path when possible."""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def create_variant_config(
    *,
    base_config: Dict[str, Any],
    model_name: str,
    variant_name: str,
    params: Dict[str, Any],
    output_root_prefix: Path | PurePosixPath,
) -> tuple[Dict[str, Any], Path | PurePosixPath]:
    config = dict(base_config)
    base_experiment = str(base_config.get("experiment_name", f"{model_name}_virchow2_advanced_binary"))
    output_dir = output_root_prefix / base_experiment / variant_name

    config["experiment_name"] = f"{base_experiment}_{variant_name}"
    config["tuning_id"] = f"virchow2_{model_name}_{variant_name}"
    config["tuning_base_experiment"] = base_experiment
    config["selection_objective"] = SELECTION_OBJECTIVE
    for key, value in params.items():
        if key == "loss_function":
            apply_loss_policy(config, str(value))
        else:
            config[key] = value

    if "monitor_metric" in config:
        config["monitor_metric"] = SELECTION_OBJECTIVE
    if "monitor" in config:
        config["monitor"] = SELECTION_OBJECTIVE
    set_output_paths(config, output_dir)
    return config, output_dir


def manifest_row(
    *,
    tuning_id: str,
    encoder: str,
    model: str,
    base_config: Path,
    config_path: Path,
    output_dir: Path | PurePosixPath,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "tuning_id": tuning_id,
        "encoder": encoder,
        "model": model,
        "base_config": display_path(base_config),
        "config_path": display_path(config_path),
        "output_dir": output_dir.as_posix(),
        "lr": params.get("learning_rate"),
        "weight_decay": params.get("weight_decay"),
        "dropout": params.get("dropout"),
        "hidden_dim": params.get("hidden_dim"),
        "attention_dim": params.get("attention_dim"),
        "k_sample": params.get("k_sample"),
        "instance_loss_weight": params.get("instance_loss_weight"),
        "loss_function": params.get("loss_function"),
        "selection_objective": SELECTION_OBJECTIVE,
    }


def write_manifest(path: Path, rows: Iterable[Dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {path}. Usa --overwrite si quieres regenerar el manifest."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_config_dir = Path(args.output_config_dir)
    manifest_path = Path(args.manifest)
    output_root_prefix = absolute_outputs_root(str(args.outputs_root))
    rows: list[Dict[str, Any]] = []

    for model_name, spec in CANDIDATES.items():
        base_config_path = Path(spec["base_config"])
        base_config = load_yaml(base_config_path)
        encoder = str(base_config.get("encoder_name", "virchow2"))
        full_grid = grid_product(spec["grid"])
        selected_grid = deterministic_subset(full_grid, int(args.max_variants_per_model))

        for index, params in enumerate(selected_grid, start=1):
            variant_name = f"{model_name}_v{index:02d}"
            config, output_dir = create_variant_config(
                base_config=base_config,
                model_name=model_name,
                variant_name=variant_name,
                params=params,
                output_root_prefix=output_root_prefix,
            )
            config_path = output_config_dir / f"{config['tuning_id']}.yaml"
            dump_yaml(config_path, config, overwrite=bool(args.overwrite))
            rows.append(
                manifest_row(
                    tuning_id=str(config["tuning_id"]),
                    encoder=encoder,
                    model=model_name,
                    base_config=base_config_path,
                    config_path=config_path,
                    output_dir=output_dir,
                    params=params,
                )
            )

    write_manifest(manifest_path, rows, overwrite=bool(args.overwrite))
    print(f"[OK] Configs generados: {len(rows)}")
    print(f"[OK] Directorio configs: {output_config_dir}")
    print(f"[OK] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
