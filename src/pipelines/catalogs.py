"""Utilities for harmonising NASA mission catalogues into a common schema."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

logger = logging.getLogger(__name__)

MissionName = Literal["kepler", "k2", "tess"]


@dataclass(slots=True)
class CataloguePaths:
    koi: Path
    k2: Path
    toi: Path


LABEL_POSITIVE = {
    "CONFIRMED",
    "CANDIDATE",
    "CP",
    "KP",
    "PC",
    "APC",
}
LABEL_NEGATIVE = {
    "FALSE POSITIVE",
    "FP",
    "FA",
    "REFUTED",
    "EPHEMERIS MATCHED",
}


def _read_csv(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", usecols=list(columns))
    return df


def _load_koi(path: Path) -> pd.DataFrame:
    df = _read_csv(
        path,
        (
            "kepid",
            "koi_pdisposition",
            "koi_period",
            "koi_duration",
            "koi_depth",
        ),
    )
    df = df.rename(
        columns={
            "kepid": "object_id",
            "koi_pdisposition": "label_raw",
            "koi_period": "period_days",
            "koi_duration": "duration_hours",
            "koi_depth": "depth_ppm",
        }
    )
    df["mission"] = "kepler"
    return df


def _load_k2(path: Path) -> pd.DataFrame:
    df = _read_csv(
        path,
        (
            "epic_hostname",
            "disposition",
            "pl_orbper",
            "pl_trandur",
            "pl_trandep",
        ),
    )
    df = df.rename(
        columns={
            "epic_hostname": "object_id",
            "disposition": "label_raw",
            "pl_orbper": "period_days",
            "pl_trandur": "duration_hours",
            "pl_trandep": "depth_percent",
        }
    )
    df["depth_ppm"] = df["depth_percent"].astype(float) * 1e4
    df.loc[df["depth_percent"].isna(), "depth_ppm"] = pd.NA
    df = df.drop(columns="depth_percent")
    df["mission"] = "k2"
    return df


def _load_toi(path: Path) -> pd.DataFrame:
    df = _read_csv(
        path,
        (
            "toi",
            "tfopwg_disp",
            "pl_orbper",
            "pl_trandurh",
            "pl_trandep",
        ),
    )
    df = df.rename(
        columns={
            "toi": "object_id",
            "tfopwg_disp": "label_raw",
            "pl_orbper": "period_days",
            "pl_trandurh": "duration_hours",
            "pl_trandep": "depth_ppm",
        }
    )
    df["mission"] = "tess"
    return df


def harmonise_catalogues(paths: CataloguePaths) -> pd.DataFrame:
    koi_df = _load_koi(paths.koi)
    k2_df = _load_k2(paths.k2)
    toi_df = _load_toi(paths.toi)

    combined = pd.concat([koi_df, k2_df, toi_df], ignore_index=True, sort=False)

    combined["label"] = combined["label_raw"].map(_normalise_label)
    combined = combined.dropna(subset=["label"])

    combined["duration_hours"] = combined["duration_hours"].astype(float)
    combined["period_days"] = combined["period_days"].astype(float)
    combined["depth_ppm"] = combined["depth_ppm"].astype(float)

    return combined[["mission", "object_id", "period_days", "duration_hours", "depth_ppm", "label", "label_raw"]]


def _normalise_label(value: str | float) -> str | None:
    if not isinstance(value, str):
        return None
    upper = value.strip().upper()
    if upper in LABEL_POSITIVE:
        return "planet-candidate"
    if upper in LABEL_NEGATIVE:
        return "false-positive"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harmonise mission catalogues into a unified Table")
    parser.add_argument("--koi", type=Path, default=Path("data/raw/koi.csv"), help="Path to KOI CSV")
    parser.add_argument("--k2", type=Path, default=Path("data/raw/k2.csv"), help="Path to K2 CSV")
    parser.add_argument("--toi", type=Path, default=Path("data/raw/toi.csv"), help="Path to TOI CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/catalog_merged.csv"),
        help="Output CSV path",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    paths = CataloguePaths(koi=args.koi, k2=args.k2, toi=args.toi)
    df = harmonise_catalogues(paths)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote unified catalogue to %s (%d rows)", output_path, len(df))
    return output_path


if __name__ == "__main__":  # pragma: no cover
    main()

