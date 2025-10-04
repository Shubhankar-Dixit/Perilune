"""Light curve acquisition helpers."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

MissionName = Literal["kepler", "k2", "tess"]


@dataclass(slots=True)
class LightcurveData:
    """Container for light curve samples.

    Attributes
    ----------
    times_days:
        Barycentric Julian Date (BJD) times measured in days.
    flux_normalised:
        Normalised, unitless flux measurements (baseline ~1.0).
    metadata:
        Free-form dictionary with mission metadata.
    """

    times_days: list[float]
    flux_normalised: list[float]
    metadata: dict[str, str]


@dataclass(slots=True)
class LightcurveFetchConfig:
    object_id: str
    mission: MissionName
    output_dir: Path = Path("data/raw/lightcurves")
    dry_run: bool = False


def fetch_lightcurve(config: LightcurveFetchConfig) -> LightcurveData:
    """Fetch a light curve for the requested object.

    When ``dry_run`` is enabled we emit a deterministic synthetic sinusoid to keep tests fast.
    """

    if config.dry_run:
        logger.info("Dry-run: generating synthetic light curve for %s", config.object_id)
        times = [2455197.0 + i * 0.02 for i in range(200)]  # 0.02 days cadence
        flux = [1.0 - 0.001 if 50 <= i % 100 <= 54 else 1.0 for i in range(200)]
        metadata = {
            "object_id": config.object_id,
            "mission": config.mission,
            "source": "synthetic",
        }
        return LightcurveData(times_days=times, flux_normalised=flux, metadata=metadata)

    try:
        from lightkurve import search_lightcurve
    except ImportError as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError(
            "Lightkurve is required for real light curve downloads. Install with 'uv sync --extra astro'."
        ) from exc

    mission_lookup = {
        "kepler": "Kepler",
        "k2": "K2",
        "tess": "TESS",
    }
    mission_name = mission_lookup.get(config.mission.lower())
    if mission_name is None:
        raise ValueError(f"Unsupported mission '{config.mission}'")

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Searching Lightkurve for %s (%s)", config.object_id, mission_name)
    search = search_lightcurve(config.object_id, mission=mission_name)
    if search is None or len(search) == 0:
        raise RuntimeError(f"No light curves found for {config.object_id} ({mission_name})")

    logger.info("Found %d light curve file(s); downloading...", len(search))
    collection = search.download_all(download_dir=str(output_dir))
    if collection is None:
        raise RuntimeError(f"Failed to download light curves for {config.object_id}")

    stitched = collection.stitch()
    cleaned = stitched.remove_nans().normalize()

    times = cleaned.time.value.tolist()
    flux = cleaned.flux.value.tolist()

    metadata = {
        "object_id": config.object_id,
        "mission": config.mission,
        "n_files": len(collection),
        "source": "lightkurve",
    }
    return LightcurveData(times_days=times, flux_normalised=flux, metadata=metadata)


def save_lightcurve(data: LightcurveData, output_dir: Path, filename: str | None = None) -> Path:
    """Persist light curve samples to a JSON file compatible with our pipelines."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = filename or f"{data.metadata.get('object_id', 'lightcurve')}.json"
    output_path = output_dir / file_name

    payload = {
        "times_days": data.times_days,
        "flux_normalised": data.flux_normalised,
        "metadata": data.metadata,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved light curve to %s", output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch a light curve for an object")
    parser.add_argument("object_id", help="KOI/TOI/K2 identifier")
    parser.add_argument(
        "--mission",
        choices=("kepler", "k2", "tess"),
        required=True,
        help="Mission name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/lightcurves"),
        help="Directory where the light curve JSON will be stored",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate a synthetic light curve instead of downloading",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    config = LightcurveFetchConfig(
        object_id=args.object_id,
        mission=args.mission,
        output_dir=args.output,
        dry_run=args.dry_run,
    )
    data = fetch_lightcurve(config)
    return save_lightcurve(data, output_dir=config.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
