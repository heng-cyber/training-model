"""Label generation for supervised trading targets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_labels(df: pd.DataFrame, label_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Create labels from OHLC.

    Targets:
    - next_direction: 1 if future close > current close (after horizon), else 0
    - next_return: log return over horizon bars

    Rows near the end (insufficient future) are NaN.
    """
    cfg = label_cfg or {}
    target = str(cfg.get("target", "next_direction"))
    horizon = int(cfg.get("horizon", 1))
    min_abs = float(cfg.get("min_abs_return", 0.0))

    close = df["close"].astype(float)
    future = close.shift(-horizon)
    log_ret = np.log(future / close)

    out = pd.DataFrame(index=df.index)
    out["future_log_return"] = log_ret

    if target == "next_return":
        out["y"] = log_ret
        out["task"] = "regression"
    elif target == "next_direction":
        direction = (log_ret > 0).astype(float)
        if min_abs > 0:
            direction = direction.where(log_ret.abs() >= min_abs, np.nan)
        out["y"] = direction
        out["task"] = "classification"
    else:
        raise ValueError(f"Unknown label.target={target!r}")

    return out
