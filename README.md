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
6. Build the modeling feature table (optionally pointing to a directory of BLS JSON outputs):
   `uv run python -m src.pipelines.features --catalog data/processed/catalog_merged.csv --output data/processed/features.parquet`
7. Train the baseline model (requires `uv sync --extra ml` once):
   `uv run python -m src.models.baseline --features data/processed/features.parquet --output-dir artifacts/baseline`
   - The API automatically looks for artifacts under `artifacts/baseline` (override with `PERILUNE_BASELINE_DIR`).
8. Train the GBM model (recommended):
   `uv run python -m src.models.gbm --features data/processed/features.parquet --output-dir artifacts/gbm-v1`
   - The API prefers `artifacts/gbm-v1` if present (override with `PERILUNE_GBM_DIR`).

## Frontend

Inside `webapp/` run:

```bash
npm install
npm run dev
```

Set `NEXT_PUBLIC_PERILUNE_API` to the FastAPI base URL when calling a remote backend.

## API runtime notes

- By default the API treats requests as non-dry-run (`dryRun` defaults to `false`).
- If Lightkurve or network lookups fail during `/api/predict` with `dryRun=false`, the server now returns `HTTP 502` instead of silently switching to synthetic data.
- To allow a development-only fallback to synthetic light curves on failures, set `PERILUNE_ALLOW_SYNTHETIC_FALLBACK=true`. When a fallback or explicit dry-run is used, the response `evidence` includes: "Synthetic data path used (fallback or dry-run)".
