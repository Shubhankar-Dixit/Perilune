# Perilune

End-to-end exoplanet transit discovery and explanation on open NASA data.

- Start with the project overview in `Info.md`.
- See the technical plan and API contracts in `docs/plan.md`.
- Contributor guidelines live in `AGENTS.md`.

## Getting Data

1. Install dependencies (including astronomy extras): `uv sync --extra astro`.
2. Fetch a mission catalog (example downloads the first 5k KOI rows to `data/raw/koi.csv`):
   `uv run python -m src.pipelines.fetch_data --table koi --limit 5000`
3. Download and cache a light curve with Lightkurve (writes JSON under `data/raw/lightcurves/`):
   `uv run python -m src.pipelines.lightcurves KOI-0001 --mission kepler`
4. (Optional) Once the BLS implementation is swapped from the placeholder, call `/api/predict` with `dryRun=false` to exercise the full pipeline.
5. Harmonise the catalogues into `data/processed/catalog_merged.csv`:
   `uv run python -m src.pipelines.catalogs --output data/processed/catalog_merged.csv`
