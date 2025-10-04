"""FastAPI application exposing Perilune services."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationInfo

from src.pipelines.bls import BLSConfig, BLSResult, run_bls
from src.pipelines.lightcurves import LightcurveFetchConfig, fetch_lightcurve

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


def _run_placeholder_bls(times: list[float], flux: list[float], dry_run: bool) -> BLSResult:
    times_arr = np.asarray(times, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    return run_bls(times_arr, flux_arr, BLSConfig(), dry_run=dry_run)


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    dry_run = request.dry_run
    mission = request.mission or "kepler"

    if request.object_id:
        lc_config = LightcurveFetchConfig(
            object_id=request.object_id,
            mission=mission.lower(),
            dry_run=dry_run,
        )
        lightcurve = fetch_lightcurve(lc_config)
        times = lightcurve.times_days
        flux = lightcurve.flux_normalised
    elif request.times is not None and request.flux is not None:
        times = request.times
        flux = request.flux
    else:  # pragma: no cover - validated already
        raise HTTPException(status_code=422, detail="Missing input data")

    result = _run_placeholder_bls(times, flux, dry_run=dry_run)
    features = FeatureSummary(
        period_days=result.best_period_days,
        duration_hours=result.best_duration_hours,
        depth_ppm=result.depth_ppm,
        snr=result.snr,
    )
    evidence = [
        f"BLS best period {result.best_period_days:.2f} d",
        f"Depth {result.depth_ppm:.0f} ppm",
        f"SNR {result.snr:.1f}",
    ]
    probability = min(result.snr / 20.0, 0.99)
    return PredictResponse(
        label="planet-candidate" if probability >= 0.5 else "false-positive",
        probability=probability,
        threshold=0.5,
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
