"""Thin wrapper around the official MetaTrader5 package (Windows only)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Official MT5 timeframe constants mirrored for docs / fallback when package missing
TIMEFRAME_MAP: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}


class MT5UnavailableError(RuntimeError):
    """Raised when MetaTrader5 cannot be imported or initialized."""


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise MT5UnavailableError(
            "MetaTrader5 package is not installed. "
            "It is Windows-only and requires a running MT5 terminal. "
            "Install with: pip install MetaTrader5  (on Windows). "
            "Use data.source=csv|sample on other platforms."
        ) from exc
    return mt5


def initialize_mt5(
    path: str | None = None,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
) -> Any:
    """
    Initialize connection to a local MetaTrader 5 terminal.

    Prefer launching MT5 and logging in via the terminal UI, then calling
    initialize() with no credentials. Only pass login/password/server if you
    intentionally want the API to log in (never commit secrets).
    """
    mt5 = _import_mt5()
    kwargs: dict[str, Any] = {}
    if path:
        kwargs["path"] = path
    if login is not None:
        kwargs["login"] = int(login)
    if password is not None:
        kwargs["password"] = password
    if server is not None:
        kwargs["server"] = server

    ok = mt5.initialize(**kwargs) if kwargs else mt5.initialize()
    if not ok:
        err = mt5.last_error()
        raise MT5UnavailableError(f"mt5.initialize() failed: {err}")
    logger.info("MT5 initialized: %s", mt5.terminal_info())
    return mt5


def shutdown_mt5(mt5: Any | None = None) -> None:
    if mt5 is None:
        try:
            mt5 = _import_mt5()
        except MT5UnavailableError:
            return
    mt5.shutdown()


def copy_rates(
    symbol: str,
    timeframe: str,
    bars: int,
    *,
    path: str | None = None,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLC bars from MT5 via copy_rates_from_pos.

    Returns a DataFrame with columns:
    time, open, high, low, close, tick_volume, spread, real_volume
    """
    tf_key = timeframe.upper()
    if tf_key not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Choose from {list(TIMEFRAME_MAP)}")

    mt5 = initialize_mt5(path=path, login=login, password=password, server=server)
    try:
        tf_const = getattr(mt5, f"TIMEFRAME_{tf_key}", TIMEFRAME_MAP[tf_key])
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, int(bars))
        if rates is None or len(rates) == 0:
            raise MT5UnavailableError(
                f"No rates for {symbol} {tf_key}: {mt5.last_error()}. "
                "Ensure the symbol is visible in Market Watch."
            )
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df
    finally:
        shutdown_mt5(mt5)


def is_mt5_available() -> bool:
    try:
        _import_mt5()
        return True
    except MT5UnavailableError:
        return False
