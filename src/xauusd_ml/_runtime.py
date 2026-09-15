"""Install numpy-backed stand-ins when pandas / sklearn / joblib are missing.

Cloud and some CI environments block PyPI. The training pipeline still needs
DataFrame-style ops and a sklearn-like estimator API. When the real packages
are importable they are used unchanged; otherwise lightweight numpy shims are
registered in sys.modules so fetch / train / eval can run on sample data.
"""

from __future__ import annotations

import sys
import types
from importlib import import_module


def ensure_runtime() -> None:
    """Idempotent: real packages win; shims fill gaps only."""
    _ensure_pandas()
    _ensure_sklearn()
    _ensure_joblib()
    if using_numpy_shims():
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger("xauusd_ml").info(
            "pandas/scikit-learn not installed — using numpy runtime shims "
            "(sample/CSV training still works; install pandas+scikit-learn for the full sklearn models)"
        )


def using_numpy_shims() -> bool:
    pandas = sys.modules.get("pandas")
    return bool(pandas is not None and getattr(pandas, "__xauusd_shim__", False))


def _ensure_pandas() -> None:
    if "pandas" in sys.modules:
        return
    try:
        import_module("pandas")
    except ImportError:
        from xauusd_ml.vendor import minipandas as shim

        sys.modules["pandas"] = shim


def _ensure_joblib() -> None:
    if "joblib" in sys.modules:
        return
    try:
        import_module("joblib")
    except ImportError:
        from xauusd_ml.vendor import minijoblib as shim

        sys.modules["joblib"] = shim


def _ensure_sklearn() -> None:
    if "sklearn" in sys.modules and "sklearn.ensemble" in sys.modules:
        return
    try:
        import_module("sklearn")
        import_module("sklearn.ensemble")
        import_module("sklearn.neural_network")
        import_module("sklearn.pipeline")
        import_module("sklearn.preprocessing")
        import_module("sklearn.base")
        import_module("sklearn.metrics")
        return
    except ImportError:
        pass

    from xauusd_ml.vendor.minisklearn import (
        BaseEstimator,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        MLPClassifier,
        MLPRegressor,
        Pipeline,
        RandomForestClassifier,
        RandomForestRegressor,
        StandardScaler,
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        roc_auc_score,
    )

    def _mod(name: str) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        return mod

    sklearn = _mod("sklearn")
    sklearn.__xauusd_shim__ = True
    ensemble = _mod("sklearn.ensemble")
    neural = _mod("sklearn.neural_network")
    pipeline = _mod("sklearn.pipeline")
    preprocessing = _mod("sklearn.preprocessing")
    base = _mod("sklearn.base")
    metrics = _mod("sklearn.metrics")

    sklearn.ensemble = ensemble
    sklearn.neural_network = neural
    sklearn.pipeline = pipeline
    sklearn.preprocessing = preprocessing
    sklearn.base = base
    sklearn.metrics = metrics

    ensemble.GradientBoostingClassifier = GradientBoostingClassifier
    ensemble.GradientBoostingRegressor = GradientBoostingRegressor
    ensemble.RandomForestClassifier = RandomForestClassifier
    ensemble.RandomForestRegressor = RandomForestRegressor
    neural.MLPClassifier = MLPClassifier
    neural.MLPRegressor = MLPRegressor
    pipeline.Pipeline = Pipeline
    preprocessing.StandardScaler = StandardScaler
    base.BaseEstimator = BaseEstimator
    metrics.accuracy_score = accuracy_score
    metrics.f1_score = f1_score
    metrics.mean_absolute_error = mean_absolute_error
    metrics.mean_squared_error = mean_squared_error
    metrics.roc_auc_score = roc_auc_score
