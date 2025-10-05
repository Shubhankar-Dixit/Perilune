"""FastAPI application exposing Perilune services."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from src.pipelines.bls import BLSConfig, BLSResult, run_bls
from src.pipelines.lightcurves import LightcurveFetchConfig, fetch_lightcurve

try:  # Baseline model is optional
    from src.models.baseline import BaselinePredictor
except RuntimeError:  # pragma: no cover - scikit-learn not installed
    BaselinePredictor = None  # type: ignore

logger = logging.getLogger(__name__)

BASELINE_ARTIFACT_DIR = Path(os.getenv("PERILUNE_BASELINE_DIR", "artifacts/baseline"))
FEATURE_TABLE_PATH = Path(os.getenv("PERILUNE_FEATURE_TABLE", "data/processed/features.parquet"))
BASELINE_THRESHOLD = float(os.getenv("PERILUNE_BASELINE_THRESHOLD", "0.5"))

app = FastAPI(title="Perilune API", version="0.0.1")


class PredictRequest(BaseModel):
    object_id: str | None = Field(default=None, alias="objectId")
    times: list[float] | None = None
    flux: list[float] | None = None
    mission: str | None = None
    dry_run: bool = Field(default=True, alias="dryRun")

    @model_validator(mode="after")
    def validate_inputs(self) -> "PredictRequest":
        provided_series = self.times is not None or self.flux is not None
        if provided_series and (self.times is None or self.flux is None):
            raise ValueError("Both times and flux must be provided together.")
        if not self.object_id and not provided_series:
            raise ValueError("Provide either objectId or both times and flux.")
        if provided_series and not self.object_id and not self.mission:
            raise ValueError("mission must be provided when supplying times/flux.")
        if self.times is not None and self.flux is not None:
            if len(self.times) != len(self.flux):
                raise ValueError("times and flux arrays must have the same length.")
        return self


class FeatureSummary(BaseModel):
    period_days: float = Field(..., ge=0.0)
    duration_hours: float = Field(..., ge=0.0)
    depth_ppm: float = Field(..., ge=0.0)
    snr: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    features: FeatureSummary
    evidence: list[str]


class SearchTransitsRequest(BaseModel):
    times: list[float] = Field(..., min_length=2)
    flux: list[float] = Field(..., min_length=2)
    dry_run: bool = Field(default=True, alias="dryRun")

    @field_validator("flux")
    @classmethod
    def check_length(cls, v: list[float], info: ValidationInfo) -> list[float]:
        times = info.data.get("times") if info.data else None
        if times is not None and len(times) != len(v):
            raise ValueError("times and flux arrays must have the same length.")
        return v


class SearchTransitsResponse(BaseModel):
    result: dict[str, Any]


class ObjectResponse(BaseModel):
    object_id: str = Field(alias="objectId")
    mission: str
    status: str
    notes: str


@lru_cache(maxsize=1)
def _get_baseline_predictor() -> BaselinePredictor | None:
    if BaselinePredictor is None:
        return None
    if not BASELINE_ARTIFACT_DIR.exists():
        return None
    try:
        return BaselinePredictor(BASELINE_ARTIFACT_DIR)
    except Exception as exc:  # pragma: no cover - optional dependency failures
        logger.warning("Failed to load baseline model: %s", exc)
        return None


@lru_cache(maxsize=1)
def _load_feature_table() -> pd.DataFrame | None:
    if not FEATURE_TABLE_PATH.exists():
        return None
    try:
        if FEATURE_TABLE_PATH.suffix.lower() == ".parquet":
            return pd.read_parquet(FEATURE_TABLE_PATH)
        if FEATURE_TABLE_PATH.suffix.lower() == ".csv":
            return pd.read_csv(FEATURE_TABLE_PATH)
    except Exception as exc:  # pragma: no cover - IO errors
        logger.warning("Failed to load feature table %s: %s", FEATURE_TABLE_PATH, exc)
        return None
    logger.warning("Unsupported feature table format: %s", FEATURE_TABLE_PATH)
    return None


def _lookup_feature_row(object_id: str) -> dict[str, Any] | None:
    table = _load_feature_table()
    if table is None:
        return None
    matches = table[table["object_id"].astype(str) == str(object_id)]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _build_feature_row_from_bls(mission: str, result: BLSResult) -> dict[str, Any]:
    return {
        "mission": mission,
        "period_days": result.best_period_days,
        "duration_hours": result.best_duration_hours,
        "depth_ppm": result.depth_ppm,
        "bls_best_period_days": result.best_period_days,
        "bls_best_duration_hours": result.best_duration_hours,
        "bls_depth_ppm": result.depth_ppm,
        "bls_snr": result.snr,
    }


def _merge_bls_into_features(feature_row: dict[str, Any], result: BLSResult) -> None:
    def _set_if_missing(key: str, value: Any) -> None:
        if value is None:
            return
        current = feature_row.get(key)
        if current is None or (isinstance(current, float) and np.isnan(current)):
            feature_row[key] = value

    _set_if_missing("period_days", result.best_period_days)
    _set_if_missing("duration_hours", result.best_duration_hours)
    _set_if_missing("depth_ppm", result.depth_ppm)
    feature_row["bls_best_period_days"] = result.best_period_days
    feature_row["bls_best_duration_hours"] = result.best_duration_hours
    feature_row["bls_depth_ppm"] = result.depth_ppm
    feature_row["bls_snr"] = result.snr


def _predict_with_baseline(feature_row: dict[str, Any], fallback: float) -> tuple[float, str]:
    predictor = _get_baseline_predictor()
    if predictor is None or not feature_row:
        return fallback, "heuristic"
    try:
        probability = predictor.predict_proba(feature_row)
        return probability, "baseline"
    except Exception as exc:  # pragma: no cover - inference errors
        logger.warning("Baseline prediction failed: %s", exc)
        return fallback, "heuristic"


def _run_placeholder_bls(times: list[float], flux: list[float], dry_run: bool) -> BLSResult:
    times_arr = np.asarray(times, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    try:
        return run_bls(times_arr, flux_arr, BLSConfig(), dry_run=dry_run)
    except NotImplementedError:  # pragma: no cover - real implementation pending
        logger.warning("Falling back to dry-run BLS placeholder")
        return run_bls(times_arr, flux_arr, BLSConfig(), dry_run=True)


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    dry_run = request.dry_run
    mission = request.mission or "kepler"

    feature_row: dict[str, Any] = {}
    bls_dry_run = dry_run

    if request.object_id:
        lc_config = LightcurveFetchConfig(
            object_id=request.object_id,
            mission=mission.lower(),
            dry_run=dry_run,
        )
        try:
            lightcurve = fetch_lightcurve(lc_config)
        except Exception as exc:
            if not dry_run:
                logger.warning(
                    "Falling back to synthetic light curve for %s (%s)", request.object_id, exc
                )
                lc_config = LightcurveFetchConfig(
                    object_id=request.object_id,
                    mission=mission.lower(),
                    dry_run=True,
                )
                lightcurve = fetch_lightcurve(lc_config)
                bls_dry_run = True
            else:
                raise
        times = lightcurve.times_days
        flux = lightcurve.flux_normalised
        catalog_row = _lookup_feature_row(request.object_id)
        if catalog_row:
            feature_row.update({k: v for k, v in catalog_row.items() if k != "label"})
    elif request.times is not None and request.flux is not None:
        times = request.times
        flux = request.flux
    else:  # pragma: no cover - validated already
        raise HTTPException(status_code=422, detail="Missing input data")

    result = _run_placeholder_bls(times, flux, dry_run=bls_dry_run)
    if not feature_row:
        feature_row.update(_build_feature_row_from_bls(mission, result))
    feature_row.setdefault("mission", mission)
    _merge_bls_into_features(feature_row, result)

    summary_period = feature_row.get("bls_best_period_days") or feature_row.get("period_days") or result.best_period_days
    summary_duration = feature_row.get("bls_best_duration_hours") or feature_row.get("duration_hours") or result.best_duration_hours
    summary_depth = feature_row.get("bls_depth_ppm") or feature_row.get("depth_ppm") or result.depth_ppm
    summary_snr = feature_row.get("bls_snr") or result.snr

    features = FeatureSummary(
        period_days=float(summary_period or 0.0),
        duration_hours=float(summary_duration or 0.0),
        depth_ppm=float(summary_depth or 0.0),
        snr=float(summary_snr or 0.0),
    )
    evidence = [
        f"BLS best period {features.period_days:.2f} d",
        f"Depth {features.depth_ppm:.0f} ppm",
        f"SNR {features.snr:.1f}",
    ]

    fallback_probability = min(features.snr / 20.0, 0.99)
    if dry_run:
        probability, source = fallback_probability, "heuristic"
    else:
        probability, source = _predict_with_baseline(feature_row, fallback_probability)
        if source == "baseline":
            evidence.append(
                f"Baseline model probability {probability:.2f} (threshold {BASELINE_THRESHOLD:.2f})"
            )

    label = "planet-candidate" if probability >= BASELINE_THRESHOLD else "false-positive"
    return PredictResponse(
        label=label,
        probability=probability,
        threshold=BASELINE_THRESHOLD,
        features=features,
        evidence=evidence,
    )

@app.get("/api/object/{object_id}", response_model=ObjectResponse)
async def get_object(object_id: str) -> ObjectResponse:
    return ObjectResponse(
        objectId=object_id,
        mission="kepler",
        status="unknown",
        notes="Placeholder object metadata. Real implementation will query mission tables.",
    )


@app.post("/api/search-transits", response_model=SearchTransitsResponse)
async def search_transits(request: SearchTransitsRequest) -> SearchTransitsResponse:
    result = _run_placeholder_bls(request.times, request.flux, dry_run=request.dry_run)
    return SearchTransitsResponse(result=result.to_dict())


__all__ = ["app"]
