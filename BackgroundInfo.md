# Background and Demo Plan (revised)

This document provides context and a concise plan for the live demo. For the full specification and delivery milestones, see `docs/plan.md`.

## Why this project now
- NASA missions (Kepler, K2, TESS) provide labeled catalogs and light curves, but analysis still leans on manual vetting.
- We can combine fast, interpretable ML with a modern UI and a grounded LLM layer to accelerate insight without sacrificing scientific caution.

## Demo narrative (what we will show)
1) Search or upload: enter a KOI/TOI/K2 ID or upload times/flux.
2) Discovery canvas: visualize raw flux, BLS periodogram, and phase-folded curve with odd/even overlay and secondary window.
3) Classify: run calibrated baseline to produce a probability and a thresholded label.
4) Explain: ask natural-language questions; receive answers grounded in the displayed evidence and mission metadata (citations included).

## Minimal scope for the demo
- Data: a small, cached subset of objects from KOI/K2 (for speed), with a TESS-only holdout.
- Compute: precomputed BLS and features for the demo set; on-demand BLS for a small upload.
- UI: one page with search/upload, charts, and results; a right-side “Explain” drawer.
- Safety: explanations only cite our own evidence items; if insufficient context, we say so explicitly.

## Risks and mitigations
- Domain shift (Kepler/K2 -> TESS): keep TESS as holdout; report metric by mission; consider light fine-tuning.
- False certainty: calibrate probabilities; show reliability curve during review; present thresholds transparently.
- Latency: precompute and cache; stream charts first, then classification/explanation.

## What’s next after the demo
- Expand the dataset; add the hybrid deep model; add centroid/pixel-based heuristics; add user annotations and export.

---
Previous brainstorming content is superseded by this version and by `docs/plan.md`.
