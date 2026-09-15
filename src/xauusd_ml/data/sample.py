"""Synthetic XAUUSD-like OHLC for offline training when MT5 is unavailable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_ml.config import TIMEFRAME_MINUTES


def generate_sample_ohlc(
    bars: int = 5000,
    timeframe: str = "M15",
    *,
    seed: int = 42,
    start_price: float = 2300.0,
) -> pd.DataFrame:
    """
    Geometric random-walk OHLC with mild mean reversion and volume noise.

    Not real market data — enough structure for pipeline smoke tests and demos.
    """
    rng = np.random.default_rng(seed)
    minutes = TIMEFRAME_MINUTES.get(timeframe.upper(), 15)
    # Align to a fixed UTC start so files are reproducible
    start = pd.Timestamp("2023-01-02 00:00:00", tz="UTC")
    times = pd.date_range(start, periods=bars, freq=f"{minutes}min")

    # Gold-ish daily vol scaled to bar size (~1% daily ≈ 0.01 / sqrt(96) for M15)
    bars_per_day = max(1, 1440 // minutes)
    bar_vol = 0.01 / np.sqrt(bars_per_day)
    shocks = rng.normal(0.0, bar_vol, size=bars)
    # mild AR(1) mean reversion toward 0 drift
    rets = np.zeros(bars)
    for i in range(1, bars):
        rets[i] = 0.15 * rets[i - 1] + shocks[i]

    log_price = np.log(start_price) + np.cumsum(rets)
    close = np.exp(log_price)
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0.0, bar_vol * start_price * 0.35, size=bars))
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    tick_volume = rng.integers(800, 5000, size=bars)
    spread = rng.integers(20, 60, size=bars)  # points-ish; informational only

    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tick_volume,
            "spread": spread,
            "real_volume": 0,
        }
    )


def ensure_sample_csv(
    sample_dir: str | Path,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    bars: int = 5000,
) -> Path:
    """Write sample CSV if missing; return path."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / f"{symbol.lower()}_{timeframe.lower()}_sample.csv"
    if not path.exists():
        df = generate_sample_ohlc(bars=bars, timeframe=timeframe)
        df.to_csv(path, index=False)
    return path
