# XAUUSD ML (MetaTrader 5)

Train a **baseline ML model** on XAUUSD OHLC bars from MetaTrader 5.  
When MT5 is unavailable (typical on Linux/macOS CI), the same pipeline runs on **CSV or bundled synthetic sample data**.

This is a research/training starter — not a live trading bot and not financial advice.

## Layout

```
configs/default.yaml      # symbol, timeframe, features, model, backtest
src/xauusd_ml/
  data/                   # MT5 client + CSV/sample fallback
  features/               # FX/gold-style OHLC features
  labels/                 # next-bar direction / return targets
  models/                 # sklearn baselines (GB / RF / MLP)
  evaluate/               # metrics + simple signal backtest
  cli.py                  # fetch / train / eval entrypoints
scripts/                  # thin wrappers for the same CLIs
data/raw/                 # MT5 pulls land here
data/sample/              # auto-generated synthetic CSV
artifacts/                # trained models + metrics
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

If PyPI is unavailable (restricted CI/cloud) but `numpy` and `PyYAML` are present, `scripts/fetch_data.py`, `scripts/train.py`, and `scripts/evaluate.py` still run using a numpy fallback for pandas/scikit-learn. Prefer the real packages when you can install them.

### MetaTrader 5 (Windows only)

The official [`MetaTrader5`](https://pypi.org/project/MetaTrader5/) package talks to a **local MT5 terminal**. It does **not** run on Linux/macOS.

1. Install [MetaTrader 5](https://www.metatrader5.com/) and open a demo or live account in the terminal UI.
2. Add **XAUUSD** to Market Watch (right-click → Show All / symbols).
3. On the same Windows machine:

   ```bash
   pip install MetaTrader5
   # or: pip install -e ".[mt5]"
   ```

4. Leave `data.mt5_login` / `password` / `server` as `null` in `configs/default.yaml` so the API uses the **already logged-in terminal**. Do not commit brokerage credentials.
5. Optional: set `data.mt5_path` to the `terminal64.exe` path if multiple installs exist.

## Config

Edit `configs/default.yaml`:

| Key | Default | Notes |
|-----|---------|--------|
| `symbol` | `XAUUSD` | Broker symbol name must match Market Watch |
| `timeframe` | `M15` | `M1` `M5` `M15` `M30` `H1` `H4` `D1` |
| `bars` | `5000` | History depth for `copy_rates_from_pos` |
| `data.source` | `auto` | `auto` → MT5 → CSV → sample |
| `label.target` | `next_direction` | or `next_return` |
| `model.type` | `sklearn_gb` | also `sklearn_rf`, `sklearn_mlp` |

## Fetch data

```bash
# Prefer MT5 when installed; otherwise write/use sample CSV
python scripts/fetch_data.py
# or: xauusd-fetch

# Force sample (works everywhere)
python scripts/fetch_data.py --source sample

# Force MT5 (Windows + terminal running)
python scripts/fetch_data.py --source mt5

# Use your own CSV (columns: time, open, high, low, close [, tick_volume, spread])
# Set data.csv_path in the YAML, or copy into data/raw/
```

Fetched MT5 bars are saved under `data/raw/xauusd_m15.csv`.

## Train

```bash
python scripts/train.py --source sample
# or: xauusd-train --source auto
```

Outputs:

- `artifacts/xauusd_model.joblib` — scaler + model
- `artifacts/xauusd_model_metrics.json` — train/val/test scores
- `artifacts/xauusd_model_backtest.csv` — simple signal equity curve

Split is **time-ordered** (no shuffle): 70% train / 15% val / 15% test.

**Target:** next-bar direction (`close[t+1] > close[t]`).  
**Features:** log returns, SMA/EMA ratios, RSI, ATR, Bollinger %B/width, realized vol, volume ratio, candle shape.

## Evaluate

```bash
python scripts/evaluate.py --model artifacts/xauusd_model.joblib --source sample
```

Backtest is a research check: long/short/flat from predicted probabilities, with a flat cost rate. It is **not** a full broker simulator (no slippage model, swaps, or partial fills).

## Tests

```bash
pytest -q
```

## Platform notes / blockers

| Constraint | Impact |
|------------|--------|
| MetaTrader5 Python API is **Windows-only** | Linux/macOS/cloud agents use `--source sample` or CSV |
| Terminal must be installed & running | `mt5.initialize()` fails otherwise |
| Symbol naming varies by broker | e.g. `XAUUSD`, `XAUUSDm`, `GOLD` — set `symbol` to match |
| Sample data is synthetic | Useful for plumbing; retrain on real MT5/CSV history before any decision use |

## License

MIT
