"""Unit tests for features, labels, and offline training smoke path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xauusd_ml.config import load_config
from xauusd_ml.data.sample import generate_sample_ohlc
from xauusd_ml.evaluate.backtest import run_signal_backtest
from xauusd_ml.features.engineering import build_feature_frame
from xauusd_ml.labels.targets import make_labels
from xauusd_ml.models.train import prepare_dataset, time_split, train_model


@pytest.fixture
def ohlc() -> pd.DataFrame:
    return generate_sample_ohlc(bars=800, timeframe="M15", seed=7)


@pytest.fixture
def cfg() -> dict:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "default.yaml")


def test_generate_sample_ohlc_shape(ohlc: pd.DataFrame) -> None:
    assert len(ohlc) == 800
    assert set(["time", "open", "high", "low", "close"]).issubset(ohlc.columns)
    assert (ohlc["high"] >= ohlc[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (ohlc["low"] <= ohlc[["open", "close"]].min(axis=1) + 1e-9).all()


def test_features_and_labels(ohlc: pd.DataFrame, cfg: dict) -> None:
    feats = build_feature_frame(ohlc, cfg["features"])
    labels = make_labels(ohlc, cfg["label"])
    assert feats.shape[0] == len(ohlc)
    assert "rsi_14" in feats.columns
    assert labels["y"].notna().sum() > 100
    assert set(labels["y"].dropna().unique()).issubset({0.0, 1.0})


def test_train_and_backtest_smoke(ohlc: pd.DataFrame, cfg: dict) -> None:
    cfg = {**cfg, "model": {"type": "sklearn_rf", "params": {"n_estimators": 30, "max_depth": 3, "random_state": 0}}}
    result = train_model(ohlc, cfg)
    assert result.task == "classification"
    assert "accuracy" in result.metrics["test"]
    assert result.split is not None
    bt = run_signal_backtest(result.model, result.split, task=result.task, backtest_cfg=cfg["backtest"])
    assert bt["n_bars"] == result.metrics["n_test"]
    assert np.isfinite(bt["total_net_return"])


def test_time_split_is_ordered(ohlc: pd.DataFrame, cfg: dict) -> None:
    frame = prepare_dataset(ohlc, cfg)
    split = time_split(frame, cfg["split"])
    # last train time < first val time < first test time
    t = frame["time"]
    i_train = len(split.X_train)
    i_val = i_train + len(split.X_val)
    assert t.iloc[i_train - 1] < t.iloc[i_train]
    assert t.iloc[i_val - 1] < t.iloc[i_val]
