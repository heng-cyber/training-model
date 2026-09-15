#!/usr/bin/env python3
"""Train baseline model (thin wrapper around xauusd-train)."""

from xauusd_ml.cli import train_main

if __name__ == "__main__":
    raise SystemExit(train_main())
