"""OHLC feature engineering suitable for XAUUSD / FX-style bars."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = []  # populated after build for introspection


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def build_feature_frame(df: pd.DataFrame, feature_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Build a feature matrix from OHLC (+ optional volume).

    Returns a DataFrame aligned to *df* index with NaNs in the warm-up region.
    Caller should dropna after joining labels.
    """
    cfg = feature_cfg or {}
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    log_close = np.log(close.replace(0.0, np.nan))

    # Candle shape
    out["log_return_1"] = log_close.diff()
    out["hl_range"] = (high - low) / close
    out["oc_body"] = (close - open_) / close
    out["upper_wick"] = (high - np.maximum(open_, close)) / close
    out["lower_wick"] = (np.minimum(open_, close) - low) / close

    for w in cfg.get("returns_windows", [1, 2, 3, 5, 10, 20]):
        if w == 1:
            continue  # already have log_return_1
        out[f"log_return_{w}"] = log_close.diff(w)

    for w in cfg.get("sma_windows", [5, 10, 20, 50]):
        sma = close.rolling(w).mean()
        out[f"close_sma_ratio_{w}"] = close / sma - 1.0

    for w in cfg.get("ema_windows", [8, 21]):
        ema = close.ewm(span=w, adjust=False).mean()
        out[f"close_ema_ratio_{w}"] = close / ema - 1.0

    # EMA cross distance
    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    out["ema_spread_8_21"] = (ema_fast - ema_slow) / close

    rsi_period = int(cfg.get("rsi_period", 14))
    out[f"rsi_{rsi_period}"] = _rsi(close, rsi_period) / 100.0

    atr_period = int(cfg.get("atr_period", 14))
    atr = _atr(df, atr_period)
    out[f"atr_ratio_{atr_period}"] = atr / close

    bb_period = int(cfg.get("bollinger_period", 20))
    bb_std = float(cfg.get("bollinger_std", 2.0))
    bb_mid = close.rolling(bb_period).mean()
    bb_sigma = close.rolling(bb_period).std()
    out["bb_pct_b"] = (close - (bb_mid - bb_std * bb_sigma)) / (2 * bb_std * bb_sigma.replace(0.0, np.nan))
    out["bb_width"] = (2 * bb_std * bb_sigma) / bb_mid

    for w in cfg.get("volatility_windows", [10, 20]):
        out[f"realized_vol_{w}"] = out["log_return_1"].rolling(w).std()

    if "tick_volume" in df.columns:
        vol = df["tick_volume"].astype(float)
        vw = int(cfg.get("volume_sma_window", 20))
        vol_sma = vol.rolling(vw).mean()
        out[f"volume_ratio_{vw}"] = vol / vol_sma.replace(0.0, np.nan)

    if "spread" in df.columns:
        out["spread_norm"] = df["spread"].astype(float) / close

    global FEATURE_COLUMNS
    FEATURE_COLUMNS = list(out.columns)
    return out
