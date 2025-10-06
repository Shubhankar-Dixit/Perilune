References informing model choices (accessed Oct 5, 2025):

- Malik et al., 2022, MNRAS 513(4):5505 — shows strong performance of feature-based ML for exoplanet vetting and recommends recall-optimized thresholds with probability calibration.
- Electronics (MDPI), 2024, 13(19):3950 — ensemble methods (including stacking) outperform single learners on KOI; supports moving beyond simple logistic to gradient boosting and ensembles.

Implications for Perilune v1:
- Prefer LightGBM baseline with isotonic calibration and report PR-AUC as primary metric.
- Add BLS-derived diagnostics (SNR, SDE, odd/even proxy, secondary flag) to the tabular feature set.
- Optimize decision thresholds for high recall in triage scenarios; monitor reliability via ECE.

