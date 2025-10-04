import numpy as np

from src.pipelines.bls import run_bls


def test_run_bls_dry_run() -> None:
    times = np.linspace(0.0, 20.0, 200)
    flux = np.ones_like(times)
    flux[50:55] -= 0.001

    result = run_bls(times, flux, dry_run=True)

    assert result.best_period_days > 0
    assert result.depth_ppm > 0
    assert "frequency_bins" in result.diagnostics
