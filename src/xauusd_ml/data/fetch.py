"""Fetch OHLC data with MT5 → CSV → sample fallback chain."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from xauusd_ml.config import resolve_paths
from xauusd_ml.data.mt5_client import MT5UnavailableError, copy_rates, is_mt5_available
from xauusd_ml.data.sample import ensure_sample_csv

logger = logging.getLogger(__name__)

OHLC_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


def load_ohlc_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"CSV must include a 'time' column: {path}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"CSV missing OHLC column {col!r}: {path}")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0
    if "spread" not in df.columns:
        df["spread"] = 0
    if "real_volume" not in df.columns:
        df["real_volume"] = 0
    return df.sort_values("time").reset_index(drop=True)


def save_ohlc_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.to_csv(path, index=False)
    logger.info("Wrote %d bars to %s", len(out), path)
    return path


def _fetch_from_mt5(cfg: dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    return copy_rates(
        symbol=cfg["symbol"],
        timeframe=cfg["timeframe"],
        bars=int(cfg.get("bars", 5000)),
        path=data_cfg.get("mt5_path"),
        login=data_cfg.get("mt5_login"),
        password=data_cfg.get("mt5_password"),
        server=data_cfg.get("mt5_server"),
    )


def fetch_ohlc(cfg: dict[str, Any], *, persist: bool = True) -> pd.DataFrame:
    """
    Load OHLC according to cfg['data']['source']:

    - mt5: require MetaTrader5
    - csv: load cfg['data']['csv_path']
    - sample: bundled / generated synthetic XAUUSD-like bars
    - auto: try MT5 → csv_path → sample
    """
    paths = resolve_paths(cfg)
    data_cfg = cfg.get("data", {})
    source = str(data_cfg.get("source", "auto")).lower()
    csv_path = data_cfg.get("csv_path")
    symbol = cfg.get("symbol", "XAUUSD")
    timeframe = cfg.get("timeframe", "M15")
    bars = int(cfg.get("bars", 5000))

    df: pd.DataFrame | None = None
    used_source = source

    if source == "mt5":
        df = _fetch_from_mt5(cfg)
        used_source = "mt5"
    elif source == "csv":
        if not csv_path:
            raise ValueError("data.source=csv requires data.csv_path")
        df = load_ohlc_csv(csv_path)
        used_source = "csv"
    elif source == "sample":
        sample_path = ensure_sample_csv(
            paths.sample_data_dir,
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
        )
        df = load_ohlc_csv(sample_path)
        used_source = "sample"
    elif source == "auto":
        if is_mt5_available():
            try:
                df = _fetch_from_mt5(cfg)
                used_source = "mt5"
            except MT5UnavailableError as exc:
                logger.warning("MT5 available but fetch failed (%s); falling back", exc)
        if df is None and csv_path:
            df = load_ohlc_csv(csv_path)
            used_source = "csv"
        if df is None:
            sample_path = ensure_sample_csv(
                paths.sample_data_dir,
                symbol=symbol,
                timeframe=timeframe,
                bars=bars,
            )
            df = load_ohlc_csv(sample_path)
            used_source = "sample"
            logger.warning(
                "Using sample/synthetic data (source=%s). "
                "For live history, run on Windows with MT5 or provide data.csv_path.",
                used_source,
            )
    else:
        raise ValueError(f"Unknown data.source={source!r}")

    assert df is not None
    df = df.sort_values("time").reset_index(drop=True)
    df.attrs["source"] = used_source
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = timeframe

    if persist and used_source == "mt5":
        paths.raw_data_dir.mkdir(parents=True, exist_ok=True)
        out = paths.raw_data_dir / f"{symbol.lower()}_{timeframe.lower()}.csv"
        save_ohlc_csv(df, out)

    logger.info("Loaded %d bars via source=%s (%s %s)", len(df), used_source, symbol, timeframe)
    return df
