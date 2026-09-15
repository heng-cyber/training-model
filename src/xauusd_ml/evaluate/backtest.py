"""Simple next-bar signal backtest for model evaluation (not production execution)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from xauusd_ml.models.train import SplitData


def run_signal_backtest(
    model: BaseEstimator,
    split: SplitData,
    *,
    task: str,
    backtest_cfg: dict[str, Any] | None = None,
    on: str = "test",
) -> dict[str, Any]:
    """
    Evaluate directional signals on a held-out slice.

    Classification: long (+1) if P(up) >= threshold, short (-1) if P(up) <= 1-threshold, else flat.
    Regression: sign of predicted return, flat if |pred| is tiny relative to median |y_train|.

    PnL approximates sum(position_t * future_log_return_t) minus turnover * cost_rate.
    This is a research sanity check, not a full broker simulation.
    """
    cfg = backtest_cfg or {}
    threshold = float(cfg.get("probability_threshold", 0.55))
    cost_rate = float(cfg.get("cost_rate", 0.0002))
    position_size = float(cfg.get("position_size", 1.0))

    if on == "test":
        X, y = split.X_test, split.y_test
        start = len(split.X_train) + len(split.X_val)
    elif on == "val":
        X, y = split.X_val, split.y_val
        start = len(split.X_train)
    else:
        raise ValueError("on must be 'test' or 'val'")

    end = start + len(X)
    meta = split.meta.iloc[start:end].reset_index(drop=True)
    future_ret = meta["future_log_return"].to_numpy(dtype=float)

    if task == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        position = np.zeros(len(X), dtype=float)
        position[proba >= threshold] = 1.0
        position[proba <= (1.0 - threshold)] = -1.0
        signal_score = proba
    else:
        pred = np.asarray(model.predict(X), dtype=float)
        if task == "classification":
            position = np.where(pred >= 0.5, 1.0, -1.0)
            signal_score = pred
        else:
            dead = np.median(np.abs(split.y_train.to_numpy())) * 0.25
            position = np.sign(pred)
            position[np.abs(pred) < dead] = 0.0
            signal_score = pred

    position = position * position_size
    turnover = np.abs(np.diff(position, prepend=0.0))
    gross = position * future_ret
    costs = turnover * cost_rate
    net = gross - costs

    equity = np.cumsum(net)
    ann_factor = np.sqrt(252 * 24 * 4)  # rough for M15; informational only
    sharpe = float(net.mean() / (net.std() + 1e-12) * ann_factor) if len(net) > 2 else float("nan")
    hit = float((np.sign(position) == np.sign(future_ret))[position != 0].mean()) if np.any(position != 0) else float("nan")

    curve = pd.DataFrame(
        {
            "time": meta["time"].values,
            "position": position,
            "future_log_return": future_ret,
            "net_return": net,
            "equity": equity,
            "signal": signal_score,
        }
    )

    return {
        "n_bars": int(len(X)),
        "coverage": float(np.mean(position != 0)),
        "total_net_return": float(net.sum()),
        "total_gross_return": float(gross.sum()),
        "total_cost": float(costs.sum()),
        "hit_rate": hit,
        "sharpe_approx": sharpe,
        "max_drawdown": float(_max_drawdown(equity)),
        "threshold": threshold,
        "cost_rate": cost_rate,
        "curve": curve,
    }


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())
