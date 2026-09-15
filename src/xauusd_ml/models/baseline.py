"""Baseline sklearn models for classification / regression."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(model_cfg: dict[str, Any], *, task: str) -> Pipeline:
    """
    Build a sklearn Pipeline: StandardScaler + estimator.

    model_cfg['type']: sklearn_gb | sklearn_rf | sklearn_mlp
    task: classification | regression
    """
    mtype = str(model_cfg.get("type", "sklearn_gb"))
    params = dict(model_cfg.get("params") or {})
    # Defaults that are safe for both clf/reg where names overlap
    params.setdefault("random_state", 42)

    if mtype == "sklearn_gb":
        if task == "classification":
            est = GradientBoostingClassifier(**_filter_params(GradientBoostingClassifier, params))
        else:
            est = GradientBoostingRegressor(**_filter_params(GradientBoostingRegressor, params))
    elif mtype == "sklearn_rf":
        params.setdefault("n_estimators", 200)
        params.setdefault("n_jobs", -1)
        if task == "classification":
            est = RandomForestClassifier(**_filter_params(RandomForestClassifier, params))
        else:
            est = RandomForestRegressor(**_filter_params(RandomForestRegressor, params))
    elif mtype == "sklearn_mlp":
        mlp_params = {
            "hidden_layer_sizes": params.get("hidden_layer_sizes", (64, 32)),
            "max_iter": params.get("max_iter", 200),
            "random_state": params.get("random_state", 42),
            "early_stopping": True,
            "validation_fraction": 0.1,
        }
        if task == "classification":
            est = MLPClassifier(**mlp_params)
        else:
            est = MLPRegressor(**mlp_params)
    else:
        raise ValueError(f"Unknown model.type={mtype!r}")

    return Pipeline([("scaler", StandardScaler()), ("model", est)])


def _filter_params(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that the estimator __init__ does not accept."""
    import inspect

    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters) - {"self"}
    # sklearn estimators often accept **kwargs via _parameter_constraints; keep intersection
    return {k: v for k, v in params.items() if k in allowed}
