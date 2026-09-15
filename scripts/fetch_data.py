#!/usr/bin/env python3
"""Fetch market data (thin wrapper around xauusd-fetch)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xauusd_ml.cli import fetch_main

if __name__ == "__main__":
    raise SystemExit(fetch_main())
