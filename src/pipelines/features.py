"""Feature builder that combines mission catalogs with BLS outputs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureBuilderConfig:
    catalog_path: Path
    bls_dir: Path | None = None
    output_path: Path = Path("data/processed/features.parquet")
    output_format: str = "parquet"  # "parquet" or "csv"


def build_features(config: FeatureBuilderConfig) -> pd.DataFrame:
    catalog_df = _load_catalog(config.catalog_path)
    logger.info("Loaded catalog with %d rows", len(catalog_df))

    if config.bls_dir and config.bls_dir.exists():
        bls_df = _load_bls_directory(config.bls_dir)
        logger.info("Loaded %d BLS result(s)", len(bls_df))
        if not bls_df.empty:
            catalog_df = catalog_df.merge(bls_df, on="object_id", how="left")
    else:
        logger.info("No BLS directory supplied or directory missing; skipping merge")

    _write_output(catalog_df, config.output_path, config.output_format)
    logger.info("Wrote feature table to %s", config.output_path)
    return catalog_df


def _load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {
        "mission",
        "object_id",
        "period_days",
        "duration_hours",
        "depth_ppm",
        "label",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Catalog missing expected columns: {sorted(missing)}")
    return df


def _load_bls_directory(directory: Path) -> pd.DataFrame:
    rows = []
    for json_path in sorted(directory.glob("*.json")):
        with json_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)

        object_id = payload.get("object_id")
        if not object_id:
            object_id = json_path.stem

        row = {
            "object_id": object_id,
            "bls_best_period_days": payload.get("best_period_days"),
            "bls_best_duration_hours": payload.get("best_duration_hours"),
            "bls_depth_ppm": payload.get("depth_ppm"),
            "bls_epoch_bjd": payload.get("epoch_bjd"),
            "bls_snr": payload.get("snr"),
        }

        diagnostics = payload.get("diagnostics") or {}
        for key, value in diagnostics.items():
            row[f"bls_diag_{key}"] = value

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["object_id"])

    return pd.DataFrame(rows)


def _write_output(df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = output_format.lower()
    if fmt == "parquet":
        df.to_parquet(output_path, index=False)
    elif fmt == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build modeling features from catalog and BLS outputs")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/processed/catalog_merged.csv"),
        help="Path to the unified mission catalog",
    )
    parser.add_argument(
        "--bls-dir",
        type=Path,
        help="Directory containing BLS JSON results (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Output file for the feature table",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output format (parquet or csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    config = FeatureBuilderConfig(
        catalog_path=args.catalog,
        bls_dir=args.bls_dir,
        output_path=args.output,
        output_format=args.format,
    )
    build_features(config)
    return config.output_path


if __name__ == "__main__":  # pragma: no cover
    main()

