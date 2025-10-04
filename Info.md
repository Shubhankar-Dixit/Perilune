# Mission Summary

Perilune is an end-to-end system that helps users find and understand exoplanet transit signals using open NASA data. We will: (1) build reliable, calibrated ML models that classify exoplanet candidates; (2) expose them behind a simple API; and (3) add a web UI that explains predictions in plain language using an LLM, grounded in mission metadata and plots.

# Problem Background

Space-based surveys such as Kepler, K2, and TESS detect exoplanets via the transit method: a periodic dip in a star’s brightness when a planet crosses the stellar disk. These missions have released rich public tables (confirmed planets, candidates, and false positives) along with time-series light curves. Much of the vetting is still manual; our goal is to automate the tedious parts while keeping scientists in the loop with interpretable outputs.

# Project Objectives

- Train at least one high-quality supervised classifier on mission tables and BLS-derived features to separate planets/candidates from false positives.
- Provide a low-latency endpoint that can: (a) look up an object by KOI/TOI/K2 ID and return a fresh classification with evidence; or (b) accept a user-supplied light curve to run transit search (BLS) and classification.
- Build a web UI for search, upload, visualization (raw, periodogram, phase-folded), and explanations.
- Add an LLM-backed “Explain” panel that answers questions about the object, features, and model decision with citations to the plotted evidence and mission docs.

# In-Scope Datasets and Labels

- Kepler KOI cumulative: label via Disposition Using Kepler Data (CONFIRMED/CANDIDATE vs FALSE POSITIVE).
- K2 planets and candidates: label via Archive Disposition (CONFIRMED/CANDIDATE vs FALSE POSITIVE/REFUTED).
- TESS TOI: label via TFOPWG Disposition (CP/KP/PC/APC vs FP/FA). Kept primarily as a realistic holdout/test due to domain shift.

# Modeling Strategy

Two complementary models share common preprocessing and evaluation:

1) Ensemble Tabular Baseline (ships first)
- Inputs: mission table fields (period, duration, depth proxies, stellar Teff/logg/R, crowding/centroid flags) + fast BLS features (periodogram SDE, best period, odd/even depth ratio, secondary scan flags).
- Model: Gradient-boosted trees (LightGBM/XGBoost) with class weighting and probability calibration (isotonic/Platt).
- Why: Strong accuracy with limited compute, interpretable feature importances, robust to missingness.

2) Hybrid Deep Classifier (stretch)
- Inputs: phase-folded, binned light curve channels (all, odd, even, secondary window) + tabular features.
- Model: Small 1D-CNN (or lightweight Transformer) fused with an MLP head over tabular features; calibrated output.
- Why: Captures learned transit shapes while preserving expert cues from tables.

# Data Pipeline (units in days, hours, ppm as applicable)

1) Fetch: scripted pulls of KOI/K2/TOI tables into `data/raw/` (gitignored). For time series, fetch light curves via Lightkurve (PDCSAP where possible). Pixel cutouts (TESSCut) optional for visuals.
2) Clean: schema harmonization across missions; unit normalization; mission flags; quality masks for flux.
3) Transit Search: run Box Least Squares (BLS) to estimate candidate period/epoch/depth/duration and compute quick vetting statistics (odd/even, secondary check, SNR proxy). Persist to `data/processed/`.
4) Features: build a reproducible feature table used by both models; save encoders/scalers.
5) Train/Eval: split by star or by mission; primary metric PR-AUC; report precision/recall trade-offs and confusion by mission.
6) Package: export model weights (e.g., ONNX/TorchScript), feature pipeline, and calibrator to `artifacts/` (use LFS/DVC).

# Web UI and LLM Layer

- Frontend: Next.js app (Node 20) in `webapp/` with pages for search/upload, discovery canvas (raw curve, periodogram, phase-fold), and results.
- Backend: FastAPI in `src/` exposes `/predict`, `/object/{id}`, and `/search-transits` endpoints. Jobs run asynchronously for long BLS scans.
- LLM “Explain”: a server-side route that assembles a grounded context (object metadata, key features, plots/metrics) and asks an LLM to summarize how the evidence supports the classification. Responses include citations (e.g., “periodogram peak at 12.34 d; odd/even depths within 2%”).
- Safety: never send raw secrets to the client; redact identifiers; bound the context to on-disk evidence to minimize hallucinations.

# Deliverables

- Reproducible data loaders and feature builders under `src/pipelines/` with CLI entry points.
- Baseline calibrated ensemble model with saved artifacts, feature schema, and evaluation report.
- API server with the three endpoints above and JSON contracts.
- Web UI implementing the discovery canvas and explain panel.
- Documentation: model card, data documentation, and setup instructions.

# Success Criteria (initial targets)

- PR-AUC >= 0.92 on Kepler/K2 validation; >= 0.85 on TESS-only test set.
- Latency: <= 2 s for table-only prediction; <= 20 s for BLS+prediction on a typical light curve.
- Explanations: include at least three grounded references (feature values, chart annotations, or mission flags) for each answer.

# Out of Scope (for now)

- Automatic planet validation claims without human review.
- Centroid/pixel-level vetting beyond simple cutouts and flags.
- Autonomous retraining from arbitrary user uploads.

# Notes

- This document describes what we are building. See `docs/plan.md` for the day-by-day delivery plan and API contracts.
