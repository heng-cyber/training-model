"""Evaluation metrics and simple signal backtest."""

from __future__ import annotations

from xauusd_ml.evaluate.backtest import run_signal_backtest
from xauusd_ml.evaluate.metrics import format_metrics

__all__ = ["run_signal_backtest", "format_metrics"]
