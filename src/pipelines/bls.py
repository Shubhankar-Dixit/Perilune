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

try:  # pragma: no cover - optional dependency
    from astropy.timeseries import BoxLeastSquares
except ImportError:  # pragma: no cover - handled at runtime
    BoxLeastSquares = None  # type: ignore


@dataclass(slots=True)
class BLSConfig:
    """Configuration for the (future) BLS search."""

    min_period_days: float = 0.5
    max_period_days: float = 40.0
    frequency_bins: int = 5000
    duration_grid_hours: tuple[float, float] = (1.0, 12.0)


@dataclass(slots=True)
class BLSResult:
    """Summary of the BLS search results."""

    best_period_days: float
    best_duration_hours: float
    depth_ppm: float
    epoch_bjd: float
    snr: float
    diagnostics: dict[str, Any]
    sde: float | None = None
    odd_even_ratio: float | None = None
    secondary_flag: bool | None = None

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

    if dry_run or BoxLeastSquares is None:
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
            sde=None,
            odd_even_ratio=None,
            secondary_flag=None,
        )

    mask = np.isfinite(times_days) & np.isfinite(flux_normalised)
    times = times_days[mask]
    flux = flux_normalised[mask]

    if times.size < 10:
        raise ValueError("Not enough valid samples to run BLS (need >=10)")

    order = np.argsort(times)
    times = times[order]
    flux = flux[order]

    flux = flux / np.nanmedian(flux)
    flux -= np.mean(flux)

    duration_min_hours = max(cfg.duration_grid_hours[0], 0.2)
    duration_max_hours = max(cfg.duration_grid_hours[1], duration_min_hours + 0.1)
    # Ensure durations remain shorter than the minimum period to satisfy astropy's constraints
    max_allowed_hours = (cfg.min_period_days * 0.9) * 24.0
    duration_max_hours = min(duration_max_hours, max_allowed_hours)
    if duration_max_hours <= duration_min_hours:
        duration_max_hours = duration_min_hours + 0.1
    durations = np.linspace(duration_min_hours, duration_max_hours, num=5) / 24.0

    bls = BoxLeastSquares(times, flux)
    freq = np.linspace(1.0 / cfg.max_period_days, 1.0 / cfg.min_period_days, cfg.frequency_bins)

    try:
        power = bls.autopower(durations, frequency=freq)
    except TypeError:
        frequency_factor = max(int(cfg.frequency_bins / max(int(durations.size), 1)), 5)
        power = bls.autopower(
            durations,
            minimum_period=cfg.min_period_days,
            maximum_period=cfg.max_period_days,
            frequency_factor=frequency_factor,
        )
    except Exception:
        power = bls.power(freq, durations)
    best_index = int(np.argmax(power.power))

    best_period = float(power.period[best_index])
    best_duration = float(power.duration[best_index])
    depth = float(power.depth[best_index])
    transit_time = float(power.transit_time[best_index])
    if hasattr(power, "snr"):
        snr = float(power.snr[best_index])
    else:
        scatter = float(np.std(flux)) if flux.size else 1.0
        snr = float(abs(depth) / (scatter + 1e-12))

    # Compute Signal Detection Efficiency (SDE) on the power spectrum
    try:
        pwr = np.asarray(power.power, dtype=float)
        sde = float((np.max(pwr) - np.mean(pwr)) / (np.std(pwr) + 1e-9))
    except Exception:  # pragma: no cover - numerical edge
        sde = None

    # Secondary eclipse heuristic: check relative power near half period
    secondary_flag = None
    odd_even_ratio = None
    try:
        half_period = best_period / 2.0
        # Find closest index to half-period in grid
        half_idx = int(np.argmin(np.abs(power.period - half_period)))
        rel_power = float(power.power[half_idx]) / float(power.power[best_index] + 1e-12)
        secondary_flag = bool(rel_power > 0.7)

        # Odd-even proxy: evaluate power near double period
        double_period = best_period * 2.0
        double_idx = int(np.argmin(np.abs(power.period - double_period)))
        # Ratio of inferred depths at 2P vs P as a crude proxy
        depth_2p = float(np.abs(power.depth[double_idx]))
        depth_p = float(np.abs(power.depth[best_index])) + 1e-12
        odd_even_ratio = float(min(max(depth_2p / depth_p, 0.0), 10.0))
    except Exception:  # pragma: no cover - guard for degenerate grids
        pass

    diagnostics = {
        "frequency_bins": cfg.frequency_bins,
        "max_power": float(power.power[best_index]),
        "sde": sde,
    }

    return BLSResult(
        best_period_days=best_period,
        best_duration_hours=best_duration * 24.0,
        depth_ppm=abs(depth) * 1e6,
        epoch_bjd=transit_time,
        snr=snr,
        diagnostics=diagnostics,
        sde=sde,
        odd_even_ratio=odd_even_ratio,
        secondary_flag=secondary_flag,
    )


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

