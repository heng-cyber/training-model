#!/usr/bin/env python3
"""Fetch market data (thin wrapper around xauusd-fetch)."""

from xauusd_ml.cli import fetch_main

if __name__ == "__main__":
    raise SystemExit(fetch_main())
