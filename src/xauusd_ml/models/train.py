"""Dataset prep, time-ordered split, and training loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from xauusd_ml.features.engineering import build_feature_frame
from xauusd_ml.labels.targets import make_labels
from xauusd_ml.models.baseline import build_model


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta: pd.DataFrame  # time/close aligned to full clean index


@dataclass
class TrainResult:
    model: BaseEstimator
    task: str
    feature_names: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    split: SplitData | None = None
    frame: pd.DataFrame | None = None


def prepare_dataset(ohlc: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    feats = build_feature_frame(ohlc, cfg.get("features"))
    labels = make_labels(ohlc, cfg.get("label"))
    frame = pd.concat(
        [
            ohlc[["time", "open", "high", "low", "close"]].reset_index(drop=True),
            feats.reset_index(drop=True),
            labels[["y", "future_log_return"]].reset_index(drop=True),
        ],
        axis=1,
    )
    frame["task"] = labels["task"].iloc[0]
    return frame.dropna().reset_index(drop=True)


def time_split(frame: pd.DataFrame, split_cfg: dict[str, Any] | None = None) -> SplitData:
    cfg = split_cfg or {}
    train_ratio = float(cfg.get("train_ratio", 0.70))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    n = len(frame)
    if n < 100:
        raise ValueError(f"Need at least 100 clean rows after dropna; got {n}")

    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    feature_cols = [c for c in frame.columns if c not in {"time", "open", "high", "low", "close", "y", "future_log_return", "task"}]

    def _xy(sl: slice) -> tuple[pd.DataFrame, pd.Series]:
        return frame.loc[sl, feature_cols], frame.loc[sl, "y"]

    X_train, y_train = _xy(slice(0, i_train))
    X_val, y_val = _xy(slice(i_train, i_val))
    X_test, y_test = _xy(slice(i_val, n))
    meta = frame[["time", "close", "future_log_return", "y"]].copy()
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test, meta)


def _classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate_split(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, task: str) -> dict[str, float]:
    if task == "classification":
        pred = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        return _classification_metrics(y, pred, proba)
    pred = model.predict(X)
    return _regression_metrics(y, pred)


def train_model(ohlc: pd.DataFrame, cfg: dict[str, Any]) -> TrainResult:
    frame = prepare_dataset(ohlc, cfg)
    task = str(frame["task"].iloc[0])
    split = time_split(frame, cfg.get("split"))
    model = build_model(cfg.get("model", {}), task=task)
    model.fit(split.X_train, split.y_train)

    metrics = {
        "train": evaluate_split(model, split.X_train, split.y_train, task),
        "val": evaluate_split(model, split.X_val, split.y_val, task),
        "test": evaluate_split(model, split.X_test, split.y_test, task),
        "n_train": len(split.X_train),
        "n_val": len(split.X_val),
        "n_test": len(split.X_test),
        "task": task,
        "features": list(split.X_train.columns),
    }
    return TrainResult(
        model=model,
        task=task,
        feature_names=list(split.X_train.columns),
        metrics=metrics,
        split=split,
        frame=frame,
    )


def save_artifact(result: TrainResult, artifacts_dir: str | Path, name: str = "model") -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{name}.joblib"
    payload = {
        "model": result.model,
        "task": result.task,
        "feature_names": result.feature_names,
        "metrics": result.metrics,
    }
    joblib.dump(payload, path)
    return path


def load_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
