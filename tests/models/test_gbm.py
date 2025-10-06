from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import lightgbm  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

if not LIGHTGBM_AVAILABLE:
    pytest.skip("lightgbm not installed", allow_module_level=True)

from src.models.gbm import GBMPredictor, TrainConfig, train_gbm  # type: ignore


def test_train_gbm(tmp_path: Path) -> None:
    n = 60
    rng = np.random.default_rng(1)
    labels = np.array(["planet-candidate"] * (n // 2) + ["false-positive"] * (n // 2))
    rng.shuffle(labels)
    df = pd.DataFrame(
        {
            "mission": rng.choice(["kepler", "tess"], size=n),
            "object_id": [f"OBJ-{i:04d}" for i in range(n)],
            "period_days": rng.uniform(1, 30, size=n),
            "duration_hours": rng.uniform(1, 12, size=n),
            "depth_ppm": rng.uniform(100, 5000, size=n),
            "bls_best_period_days": rng.uniform(1, 30, size=n),
            "bls_best_duration_hours": rng.uniform(1, 12, size=n),
            "bls_depth_ppm": rng.uniform(100, 5000, size=n),
            "bls_snr": rng.uniform(1, 25, size=n),
            "bls_sde": rng.uniform(5, 15, size=n),
            "bls_odd_even_ratio": rng.uniform(0.8, 1.2, size=n),
            "bls_secondary_flag": rng.integers(0, 2, size=n),
            "label": labels,
        }
    )

    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)
    out = tmp_path / "artifacts"
    cfg = TrainConfig(features_path=features_path, output_dir=out, test_size=0.25, random_state=42)
    metrics = train_gbm(cfg)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert (out / "model.joblib").exists()
    assert (out / "metrics.json").exists()
    assert (out / "metadata.json").exists()
    predictor = GBMPredictor(out)
    proba = predictor.predict_proba(df.iloc[0].to_dict())
    assert 0.0 <= proba <= 1.0

