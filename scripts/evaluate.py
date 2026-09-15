#!/usr/bin/env python3
"""Evaluate saved model (thin wrapper around xauusd-eval)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xauusd_ml.cli import eval_main

if __name__ == "__main__":
    raise SystemExit(eval_main())
