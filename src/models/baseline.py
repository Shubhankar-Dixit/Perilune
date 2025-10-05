"""Baseline ensemble (logistic) training and inference utilities."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - runtime guard for artifact IO
    from joblib import dump, load
except ImportError:  # pragma: no cover - handled dynamically
    dump = load = None  # type: ignore

try:  # pragma: no cover - scikit-learn optional dependency
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - training requires scikit-learn
    SKLEARN_AVAILABLE = False
    CalibratedClassifierCV = ColumnTransformer = SimpleImputer = LogisticRegression = None  # type: ignore
    average_precision_score = classification_report = roc_auc_score = None  # type: ignore
    train_test_split = Pipeline = OneHotEncoder = None  # type: ignore


@dataclass(slots=True)
class TrainConfig:
    features_path: Path
    output_dir: Path = Path("artifacts/baseline")
    test_size: float = 0.2
    random_state: int = 42
    positive_label: str = "planet-candidate"
    negative_label: str = "false-positive"


NUMERIC_FEATURES: Sequence[str] = (
    "period_days",
    "duration_hours",
    "depth_ppm",
    "bls_best_period_days",
    "bls_best_duration_hours",
    "bls_depth_ppm",
    "bls_snr",
)

CATEGORICAL_FEATURES: Sequence[str] = ("mission",)


def load_features(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature file format: {path}")


def train_baseline(config: TrainConfig) -> dict[str, float | dict[str, Any]]:
    if not SKLEARN_AVAILABLE or dump is None:
        raise RuntimeError("scikit-learn is required for training. Install with 'uv sync --extra ml'.")

    df = load_features(config.features_path)
    logger.info("Loaded %d rows from %s", len(df), config.features_path)

    label_map = {
        config.positive_label: 1,
        config.negative_label: 0,
    }
    df = df[df["label"].isin(label_map.keys())].copy()
    df["target"] = df["label"].map(label_map)

    feature_columns = [col for col in NUMERIC_FEATURES if col in df.columns] + list(CATEGORICAL_FEATURES)

    X = df[feature_columns]
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    numeric_cols = [col for col in NUMERIC_FEATURES if col in X.columns]
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    base_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    pipeline.fit(X_train, y_train)

    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_pred_prob)),
        "pr_auc": float(average_precision_score(y_test, y_pred_prob)),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    _save_artifacts(pipeline, metrics, feature_columns, config)
    return metrics


def _save_artifacts(pipeline: Any, metrics: dict[str, Any], feature_columns: Sequence[str], config: TrainConfig) -> None:
    if dump is None:
        raise RuntimeError("joblib is required to persist artifacts.")

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", metrics_path)

    metadata = {
        "features": list(feature_columns),
        "positive_label": config.positive_label,
        "negative_label": config.negative_label,
        "test_size": config.test_size,
        "random_state": config.random_state,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class BaselinePredictor:
    """Load a trained baseline pipeline for inference."""

    def __init__(self, artifact_dir: Path):
        if load is None:
            raise RuntimeError("joblib is required for inference.")

        self.artifact_dir = artifact_dir.expanduser().resolve()
        model_path = self.artifact_dir / "model.joblib"
        metadata_path = self.artifact_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.pipeline = load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_columns: list[str] = list(metadata.get("features", []))
        if not self.feature_columns:
            raise RuntimeError("Model metadata missing feature list.")

    def predict_proba(self, feature_row: dict[str, Any]) -> float:
        data = {column: feature_row.get(column, np.nan) for column in self.feature_columns}
        frame = pd.DataFrame([data])
        probabilities = self.pipeline.predict_proba(frame)
        return float(probabilities[0][1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the baseline logistic model")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Path to the feature table (parquet or csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/baseline"),
        help="Directory to store model artifacts",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out proportion for evaluation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    config = TrainConfig(
        features_path=args.features,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics = train_baseline(config)
    logger.info("Training complete: ROC-AUC %.3f, PR-AUC %.3f", metrics["roc_auc"], metrics["pr_auc"])
    return metrics


__all__ = [
    "TrainConfig",
    "train_baseline",
    "load_features",
    "BaselinePredictor",
]


if __name__ == "__main__":  # pragma: no cover
    main()
