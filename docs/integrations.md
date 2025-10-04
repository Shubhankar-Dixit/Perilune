# Integrations

This document lists the external data sources and services used by Perilune.

## NASA Exoplanet Archive (tables)
- Usage: fetch KOI, K2, and TOI tables via the public API/TAP endpoints.
- TAP table names: `koi`, `k2pandc`, `toi`.
- Example query: `SELECT TOP 5000 kepid,koi_disposition FROM koi` with `format=csv`.
- We rely on the following disposition columns: KOI Disposition Using Kepler Data (`koi_disposition`), K2 Archive Disposition (`k2_disposition`), TOI TFOPWG Disposition (`tfopwg_disp`).
- Store raw responses in `data/raw/` and record the query used for provenance.

## Light Curves
- Usage: retrieve Kepler/TESS light curves via the Lightkurve Python library where possible (PDCSAP preferred).
- Apply quality masks and basic detrending; record the version of the library and mission data.

## BLS Transit Search
- Usage: compute period, epoch, depth, duration via a Box Least Squares implementation; persist a compact result table and plots for the UI.

## LLM Provider
- Usage: server-side only; read provider key from environment (e.g., `OPENAI_API_KEY`).
- Ground answers in our own computed context (features, metrics, chart annotations, and mission helptext). If context is insufficient, return a safe fallback.
