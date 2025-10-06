
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipelines.features import FeatureBuilderConfig, build_features


def test_build_features_without_bls(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.csv"
    df = pd.DataFrame(
        {
            "mission": ["kepler"],
            "object_id": ["KOI-0001"],
            "period_days": [5.0],
            "duration_hours": [2.4],
            "depth_ppm": [600.0],
            "label": ["planet-candidate"],
        }
    )
    df.to_csv(catalog_path, index=False)

    config = FeatureBuilderConfig(catalog_path=catalog_path, bls_dir=None, output_path=tmp_path / "features.parquet")
    result = build_features(config)

    assert "mission" in result.columns
    assert result.loc[0, "object_id"] == "KOI-0001"


def test_build_features_with_bls(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.csv"
    catalog_df = pd.DataFrame(
        {
            "mission": ["kepler", "tess"],
            "object_id": ["KOI-0001", "TOI-1000"],
            "period_days": [5.0, 2.0],
            "duration_hours": [2.4, 1.2],
            "depth_ppm": [600.0, 800.0],
            "label": ["planet-candidate", "false-positive"],
        }
    )
    catalog_df.to_csv(catalog_path, index=False)

    bls_dir = tmp_path / "bls"
    bls_dir.mkdir()
    payload = {
        "object_id": "KOI-0001",
        "best_period_days": 5.01,
        "best_duration_hours": 2.5,
        "depth_ppm": 580.0,
        "epoch_bjd": 2455000.1,
        "snr": 12.3,
        "diagnostics": {"frequency_bins": 5000},
    }
    (bls_dir / "KOI-0001.json").write_text(json.dumps(payload), encoding="utf-8")

    config = FeatureBuilderConfig(
        catalog_path=catalog_path,
        bls_dir=bls_dir,
        output_path=tmp_path / "features.csv",
        output_format="csv",
    )
    result = build_features(config)

    assert "bls_best_period_days" in result.columns
    row = result[result["object_id"] == "KOI-0001"].iloc[0]
    assert pytest.approx(row["bls_best_period_days"], rel=1e-6) == 5.01
    assert pd.isna(result[result["object_id"] == "TOI-1000"]["bls_best_period_days"]).all()
