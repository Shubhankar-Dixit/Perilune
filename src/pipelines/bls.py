"""Box Least Squares (BLS) scaffolding utilities."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BLSConfig:
    """Configuration for the (future) BLS search."""

    min_period_days: float = 0.5
    max_period_days: float = 40.0
    frequency_bins: int = 5000


@dataclass(slots=True)
class BLSResult:
    """Summary of the BLS search results."""

    best_period_days: float
    best_duration_hours: float
    depth_ppm: float
    epoch_bjd: float
    snr: float
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["diagnostics"] = dict(self.diagnostics)
        return payload


def run_bls(
    times_days: np.ndarray,
    flux_normalised: np.ndarray,
    config: BLSConfig | None = None,
    *,
    dry_run: bool = False,
) -> BLSResult:
    """Run a BLS search (placeholder implementation).

    The dry-run path fabricates a plausible-looking result so downstream systems can be
    developed without the heavy dependency stack. A real implementation should replace
    this branch with a call into ``astropy.timeseries.BoxLeastSquares`` or similar.
    """

    cfg = config or BLSConfig()

    if dry_run:
        if times_days.size == 0 or flux_normalised.size == 0:
            times_days = np.linspace(0.0, 20.0, 200)
            flux_normalised = np.ones_like(times_days)
            flux_normalised[50:55] -= 0.001

        span = float(times_days.max() - times_days.min()) if times_days.size else 10.0
        best_period = float(np.clip(span / 4.0, cfg.min_period_days, cfg.max_period_days))
        best_duration = float(np.clip(best_period * 0.05 * 24.0, 0.5, 12.0))
        depth_ppm = float((1.0 - np.min(flux_normalised)) * 1e6)
        epoch = float(times_days.min()) if times_days.size else 0.0
        snr = float(max(depth_ppm / 200.0, 5.0))
        diagnostics = {
            "frequency_bins": cfg.frequency_bins,
            "span_days": span,
        }
        return BLSResult(
            best_period_days=best_period,
            best_duration_hours=best_duration,
            depth_ppm=depth_ppm,
            epoch_bjd=epoch,
            snr=snr,
            diagnostics=diagnostics,
        )

    raise NotImplementedError("Real BLS search not yet implemented. Use dry_run=True for scaffolding.")


def _load_lightcurve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["times_days"], dtype=float), np.asarray(payload["flux_normalised"], dtype=float)


def _write_result(result: BLSResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BLS on a light curve")
    parser.add_argument("input", type=Path, help="Path to light curve JSON (times_days, flux_normalised)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for the BLS result JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the placeholder implementation instead of a real BLS search",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="bls",
        help="Filename prefix for the output JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    times, flux = _load_lightcurve(args.input)
    result = run_bls(times, flux, dry_run=args.dry_run)
    output_file = args.output / f"{args.prefix}.json"
    _write_result(result, output_file)
    logger.info("Saved BLS results to %s", output_file)
    return output_file


if __name__ == "__main__":  # pragma: no cover
    main()

