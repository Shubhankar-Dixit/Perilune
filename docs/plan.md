# Perilune — Technical Plan and Spec (v0.1)

Date: 2025-10-04
Owner: perilune team

## Goals
- Classify exoplanet candidates with calibrated probabilities and clear evidence.
- Ship an API + UI that runs BLS, visualizes results, and explains model output in plain language using an LLM grounded by our own context.
- Keep the system reproducible, testable, and secure (no raw data in git; secrets via env).

## Non-Goals
- Full, automated scientific validation without human review.
- Publishing to external archives or coordinating follow-up observations.

## Architecture Overview
- `src/pipelines/`: data loaders, cleaning, BLS, and feature builders.
- `src/models/`: baseline ensemble (LightGBM/XGBoost) and hybrid deep model; calibration; export.
- `src/common/`: IO, configs, logging, utilities.
- `webapp/`: Next.js app with discovery canvas and explain panel.
- `artifacts/`: versioned models, calibrators, and encoders (LFS/DVC-backed).
- `data/raw/` (gitignored): NASA tables, light curves; `data/processed/`: derived features and BLS results.

## Delivery Milestones
1) M0 Repo setup (today)
- Update docs (Info.md, this plan) and AGENTS.md. Add .gitignore and env examples.

2) M1 Data ingestion and BLS (2–3 days)
- Implement `src/pipelines/fetch_data` to download KOI, K2, TOI tables.
- Implement `src/pipelines/lightcurves` to fetch and cache light curves with Lightkurve.
- Implement `src/pipelines/bls` to run Box Least Squares, produce periodogram, best period/epoch/depth/duration, odd/even stats; persist plots and a compact feature table.
- CLI: `uv run python -m src.pipelines.fetch_data --table koi` etc.

3) M2 Baseline calibrated ensemble (2–4 days)
- Feature builder unifying table + BLS features; mission normalization (done).
- Train calibrated baseline classifier (LogReg for v0, upgradeable to LightGBM/XGBoost) with class weights; fit a probability calibrator.
- Evaluate PR-AUC, F1; confusion by mission; select operating thresholds.
- Export artifacts to `artifacts/mission_model-v1/`.

4) M3 API + minimal UI (2 days)
- FastAPI endpoints:
  - `POST /api/predict` (payload: KOI/TOI/K2 ID or times/flux array) -> calibrated probability + evidence summary; falls back to heuristic if the baseline model is unavailable.
  - `GET /api/object/{id}` -> merged mission metadata + latest classification.
  - `POST /api/search-transits` -> run BLS and return periodogram + phase-folded points.
- Next.js pages for search/upload and a discovery canvas (charts + summary chips).

5) M4 LLM “Explain” (1–2 days)
- Server-side route that builds a grounded context (key features, metrics, chart annotations, mission docs snippets) and calls an LLM to produce a short explanation with citations to our evidence.
- Safety: strict prompt with “answer only from context”; truncate/strip PII; redact secrets.

6) M5 Hybrid deep model (stretch, 5–7 days)
- Phase-folded 1D-CNN (or light Transformer) + tabular fusion; calibrate and compare.

## API Contracts (v0)
### POST `/api/predict`
Request (choose one of the inputs):
```json
{
  "objectId": "KOI-1234"
}
```
```json
{
  "times": [ ... ],
  "flux": [ ... ],
  "mission": "TESS"
}
```
Response:
```json
{
  "label": "planet-candidate",
  "probability": 0.91,
  "threshold": 0.80,
  "features": {
    "period_days": 12.34,
    "depth_ppm": 820,
    "odd_even_ratio": 1.02,
    "secondary_flag": false
  },
  "evidence": [
    "BLS peak at 12.34 d (SDE 9.1)",
    "Odd/even depths within 2%",
    "No strong secondary at 0.5 phase"
  ]
}
```

### GET `/api/object/{id}`
Returns canonical mission metadata merged with our latest classification and links to stored plots.

### POST `/api/search-transits`
Runs BLS and returns a periodogram, best period/epoch, and a phase-folded series suitable for plotting.

## Data Contracts
- Feature schema (initial): `period_days`, `duration_hours`, `depth_ppm`, `mes_like`, `odd_even_ratio`, `secondary_flag`, `crowding`, `centroid_flag`, `teff_k`, `logg_cgs`, `radius_solar`, `mission_onehot[*]`.
- All numeric features must include units in docstrings (NumPy-style) and be validated in tests.

## Evaluation & Targets
- Metrics: PR-AUC (primary), ROC-AUC, F1 at selected operating points, confusion by mission.
- Targets (initial): PR-AUC >= 0.92 on Kepler/K2 validation; >= 0.85 on a TESS-only test subset.
- Calibration: reliability curves; ECE < 0.05 preferred.

## LLM Integration (grounded explainability)
- Input context: object metadata, top features with values/units, operating threshold and probability, textual annotations extracted from our own plots (period, depth, odd/even, secondary), and short mission helptext.
- Output: 4–6 sentence explanation with 2–4 inline citations back to our evidence items.
- Guardrails: keep answers within context; if insufficient evidence, respond with a graceful “not enough data” message.

## Security & Privacy
- No raw data in git; keep in `data/raw/` (gitignored).
- Secrets in `.env.local`; never log secrets; server-side calls to LLM provider.
- Add basic rate limiting and size limits for uploads.

## Next Actions (short list)
- Implement `src/pipelines/fetch_data` + tests.
- Implement `src/pipelines/bls` + tests using small fixtures.
- Draft API stubs and a mock UI binding to those stubs.


\r
## Model v1 (updated Oct 5, 2025)
- Preferred baseline: LightGBM + isotonic calibration over mission + BLS features (period/duration/depth, BLS SNR/SDE, odd_even_ratio, secondary_flag).
- Train via: uv run python -m src.models.gbm --features data/processed/features.parquet --output-dir artifacts/gbm-v1.
- API preference: rtifacts/gbm-v1 > rtifacts/baseline.
- Metrics to report: PR-AUC (primary), ROC-AUC, reliability/ECE, confusion by mission; optimize threshold for desired recall.

