"""Market data ingestion: MetaTrader 5 with CSV / sample fallback."""

from __future__ import annotations

from .fetch import fetch_ohlc, load_ohlc_csv, save_ohlc_csv
from .sample import ensure_sample_csv, generate_sample_ohlc

__all__ = [
    "fetch_ohlc",
    "load_ohlc_csv",
    "save_ohlc_csv",
    "ensure_sample_csv",
    "generate_sample_ohlc",
]
