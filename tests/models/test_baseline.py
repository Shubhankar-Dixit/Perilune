from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import baseline as baseline_module

if not baseline_module.SKLEARN_AVAILABLE:
    pytest.skip("scikit-learn not installed", allow_module_level=True)

from src.models.baseline import BaselinePredictor, TrainConfig, train_baseline



def test_train_baseline(tmp_path: Path) -> None:
    n = 40
    rng = np.random.default_rng(0)
    labels = np.array(["planet-candidate"] * (n // 2) + ["false-positive"] * (n // 2))
    rng.shuffle(labels)

    df = pd.DataFrame(
        {
            "mission": rng.choice(["kepler", "tess"], size=n),
            "object_id": [f"OBJ-{i:04d}" for i in range(n)],
            "period_days": rng.uniform(1, 30, size=n),
            "duration_hours": rng.uniform(1, 10, size=n),
            "depth_ppm": rng.uniform(100, 5000, size=n),
            "bls_best_period_days": rng.uniform(1, 30, size=n),
            "bls_best_duration_hours": rng.uniform(1, 12, size=n),
            "bls_depth_ppm": rng.uniform(100, 5000, size=n),
            "bls_snr": rng.uniform(1, 20, size=n),
            "label": labels,
        }
    )

    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    output_dir = tmp_path / "artifacts"
    config = TrainConfig(features_path=features_path, output_dir=output_dir, test_size=0.25, random_state=123)
    metrics = train_baseline(config)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metadata.json").exists()
    predictor = BaselinePredictor(output_dir)
    proba = predictor.predict_proba(df.iloc[0].to_dict())
    assert 0.0 <= proba <= 1.0
