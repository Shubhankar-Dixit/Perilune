"""LightGBM classifier with probability calibration and metrics.

Implements a stronger baseline inspired by recent literature that favors
gradient-boosted tree ensembles over simple logistic regression for tabular
and BLS-derived features.
"""

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

try:  # Optional dependencies
    from joblib import dump, load
except Exception:  # pragma: no cover
    dump = load = None  # type: ignore

try:
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    SKLEARN_OK = True
except Exception:  # pragma: no cover
    SKLEARN_OK = False
    LGBMClassifier = CalibratedClassifierCV = ColumnTransformer = SimpleImputer = None  # type: ignore
    average_precision_score = roc_auc_score = None  # type: ignore
    train_test_split = Pipeline = OneHotEncoder = None  # type: ignore


NUMERIC_FEATURES: Sequence[str] = (
    "period_days",
    "duration_hours",
    "depth_ppm",
    "bls_best_period_days",
    "bls_best_duration_hours",
    "bls_depth_ppm",
    "bls_snr",
    "bls_sde",
    "bls_odd_even_ratio",
    "bls_secondary_flag",
)

CATEGORICAL_FEATURES: Sequence[str] = ("mission",)


@dataclass(slots=True)
class TrainConfig:
    features_path: Path
    output_dir: Path = Path("artifacts/gbm-v1")
    test_size: float = 0.2
    random_state: int = 42
    positive_label: str = "planet-candidate"
    negative_label: str = "false-positive"


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    bin_stats: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            bin_stats.append({"bin": float(b), "frac": 0.0, "acc": 0.0, "conf": 0.0})
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        frac = float(np.mean(mask))
        ece += abs(acc - conf) * frac
        bin_stats.append({"bin": float(b), "frac": frac, "acc": acc, "conf": conf})
    return {"ece": float(ece), "bins": bin_stats}


def _load_features(path: Path) -> pd.DataFrame:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature file format: {path}")


def train_gbm(config: TrainConfig) -> dict[str, Any]:
    if not SKLEARN_OK or dump is None:
        raise RuntimeError("lightgbm and scikit-learn required. Install with 'uv sync --extra ml'.")

    df = _load_features(config.features_path)
    logger.info("Loaded %d rows from %s", len(df), config.features_path)

    # Label normalization
    labels = df["label"].astype(str).str.strip().str.lower()
    labels = labels.str.replace(r"[\s_]+", "-", regex=True)
    labels = labels.replace({
        "candidate": "planet-candidate",
        "falsepositive": "false-positive",
        "fp": "false-positive",
    })
    df = df.assign(label=labels)
    label_map = {config.positive_label: 1, config.negative_label: 0}
    df = df[df["label"].isin(label_map.keys())].copy()
    if df.empty:
        raise ValueError("No samples with recognized labels after normalization.")
    df["target"] = df["label"].map(label_map)

    feature_columns = [c for c in NUMERIC_FEATURES if c in df.columns] + [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if not feature_columns:
        raise ValueError("No expected feature columns found in feature table.")

    X = df[feature_columns]
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    gbm = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,
    )

    clf = CalibratedClassifierCV(gbm, method="isotonic", cv=5)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    roc = float(roc_auc_score(y_test, proba))
    pr = float(average_precision_score(y_test, proba))
    cal = _ece(y_test, proba, n_bins=10)

    # Operating threshold: maximize recall subject to precision >= 0.9
    prec, rec, thr = precision_recall_curve(y_test, proba)
    best_thr = 0.5
    best_rec = -1.0
    for p, r, t in zip(prec[:-1], rec[:-1], thr):
        if p >= 0.9 and r > best_rec:
            best_rec = float(r)
            best_thr = float(t)

    metrics = {"roc_auc": roc, "pr_auc": pr, "calibration": cal, "operating_threshold": best_thr}

    _save(config, pipe, metrics, feature_columns)
    return metrics


def _save(config: TrainConfig, model: Any, metrics: dict[str, Any], feature_columns: Sequence[str]) -> None:
    if dump is None:
        raise RuntimeError("joblib required to persist artifacts.")
    out = config.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    meta = {
        "features": list(feature_columns),
        "positive_label": config.positive_label,
        "negative_label": config.negative_label,
        "random_state": config.random_state,
        "model": "lightgbm-isotonic-cv5",
        "threshold": metrics.get("operating_threshold", 0.5),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class GBMPredictor:
    def __init__(self, artifact_dir: Path):
        if load is None:
            raise RuntimeError("joblib required for inference.")
        self.dir = artifact_dir.expanduser().resolve()
        model_path = self.dir / "model.joblib"
        meta_path = self.dir / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError("GBM artifacts not found.")
        self.model = load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_columns: list[str] = list(meta.get("features", []))
        self.model_name: str = str(meta.get("model", "gbm"))
        self.threshold: float = float(meta.get("threshold", 0.5))

    def predict_proba(self, feature_row: dict[str, Any]) -> float:
        data = {col: feature_row.get(col, np.nan) for col in self.feature_columns}
        frame = pd.DataFrame([data])
        prob = self.model.predict_proba(frame)[0][1]
        return float(prob)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LightGBM model with calibration")
    p.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/gbm-v1"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = TrainConfig(
        features_path=args.features,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics = train_gbm(cfg)
    logger.info("GBM: ROC-AUC %.3f PR-AUC %.3f ECE %.3f", metrics["roc_auc"], metrics["pr_auc"], metrics["calibration"]["ece"])
    return metrics


__all__ = ["TrainConfig", "train_gbm", "GBMPredictor"]


if __name__ == "__main__":  # pragma: no cover
    main()
