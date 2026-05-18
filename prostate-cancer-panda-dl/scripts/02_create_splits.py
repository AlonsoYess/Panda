"""Phase 2 - Create reproducible train/valid/test splits for PANDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.split_data import BASE_COLUMNS, build_splits_dataframe, summarize_splits
from src.utils.io import ensure_output_structure, save_dataframe_csv
from src.utils.logger import setup_logger
from src.utils.paths import build_paths, get_split_config, load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified splits for PANDA dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Ruta a config.yaml",
    )
    return parser.parse_args()


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"train.csv no contiene columnas requeridas: {missing}")


def run(config_path: Path) -> int:
    try:
        config = load_config(config_path)
        paths = build_paths(config)
        split_cfg = get_split_config(config)
        random_seed = int(config.get("random_seed", 42))
        set_seed(random_seed)
    except Exception as ex:
        print(f"[ERROR] No se pudo cargar configuracion: {ex}")
        return 1

    ensure_output_structure(paths)
    logger = setup_logger("02_create_splits", paths["logs_dir"])
    logger.info("Iniciando Fase 2 - creacion de splits.")
    logger.info("Usando train_csv: %s", paths["train_csv"])
    logger.info("Split config: %s", split_cfg)
    logger.info("Random seed: %s", random_seed)

    try:
        train_df = pd.read_csv(paths["train_csv"])
        validate_required_columns(train_df)
    except Exception as ex:
        logger.exception("Error al leer/validar train.csv")
        print(f"[ERROR] No se pudo leer train.csv: {ex}")
        return 1

    try:
        splits_df = build_splits_dataframe(
            train_df=train_df[BASE_COLUMNS].copy(),
            split_cfg=split_cfg,
            random_seed=random_seed,
        )
        splits_df = splits_df[BASE_COLUMNS + ["cancer_label", "split"]]
    except Exception as ex:
        logger.exception("Error al crear splits estratificados")
        print(f"[ERROR] No se pudo crear splits: {ex}")
        return 1

    try:
        save_dataframe_csv(splits_df, paths["splits_csv"])
        logger.info("splits.csv guardado en: %s", paths["splits_csv"])
    except Exception as ex:
        logger.exception("Error al guardar splits.csv")
        print(f"[ERROR] No se pudo guardar splits.csv: {ex}")
        return 1

    summaries = summarize_splits(splits_df)

    print("\n[INFO] Cantidad total por split:")
    print(summaries["split_counts"].to_string(index=False))
    print("\n[INFO] Distribucion por isup_grade y split:")
    print(summaries["isup_by_split"].to_string())
    print("\n[INFO] Distribucion por cancer_label y split:")
    print(summaries["cancer_by_split"].to_string())

    logger.info("Cantidad total por split:\n%s", summaries["split_counts"].to_string(index=False))
    logger.info("Distribucion isup_grade por split:\n%s", summaries["isup_by_split"].to_string())
    logger.info("Distribucion cancer_label por split:\n%s", summaries["cancer_by_split"].to_string())
    logger.info("Fase 2 (splits) finalizada correctamente.")
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args.config))


if __name__ == "__main__":
    main()
