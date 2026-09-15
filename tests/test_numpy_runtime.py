"""Stdlib tests for the numpy runtime (no pytest / pandas required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xauusd_ml._runtime import ensure_runtime, using_numpy_shims  # noqa: E402

ensure_runtime()

import pandas as pd  # noqa: E402
from xauusd_ml.config import load_config  # noqa: E402
from xauusd_ml.data.sample import generate_sample_ohlc  # noqa: E402
from xauusd_ml.models.train import train_model  # noqa: E402


class NumpyRuntimeTests(unittest.TestCase):
    def test_scalar_column_broadcast(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": 0})
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["b"].values), [0, 0, 0])

    def test_concat_uniquifies_series_names(self) -> None:
        a = pd.Series([1.0, 2.0], name="high")
        b = pd.Series([3.0, 4.0], name="high")
        out = pd.concat([a, b], axis=1)
        self.assertEqual(len(out.columns), 2)
        self.assertEqual(out.shape, (2, 2))

    def test_train_smoke_on_sample(self) -> None:
        cfg = load_config(ROOT / "configs" / "default.yaml")
        cfg["model"] = {"type": "sklearn_gb", "params": {"n_estimators": 8, "max_depth": 2, "random_state": 0}}
        ohlc = generate_sample_ohlc(bars=400, timeframe="M15", seed=3)
        result = train_model(ohlc, cfg)
        self.assertEqual(result.task, "classification")
        self.assertGreater(result.metrics["n_train"], 100)
        self.assertIn("accuracy", result.metrics["test"])


if __name__ == "__main__":
    print("using_numpy_shims", using_numpy_shims())
    unittest.main()
