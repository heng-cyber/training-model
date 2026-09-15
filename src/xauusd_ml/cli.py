"""Command-line entry points: fetch, train, evaluate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from xauusd_ml.config import load_config, resolve_paths
from xauusd_ml.data.fetch import fetch_ohlc, save_ohlc_csv
from xauusd_ml.evaluate.backtest import run_signal_backtest
from xauusd_ml.evaluate.metrics import format_metrics
from xauusd_ml.models.train import load_artifact, save_artifact, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("xauusd_ml")


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )


def fetch_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch XAUUSD OHLC (MT5 / CSV / sample)")
    _add_config_arg(parser)
    parser.add_argument(
        "--source",
        choices=["auto", "mt5", "csv", "sample"],
        default=None,
        help="Override data.source from config",
    )
    parser.add_argument("-o", "--output", default=None, help="Optional CSV output path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.source:
        cfg.setdefault("data", {})["source"] = args.source

    df = fetch_ohlc(cfg, persist=True)
    paths = resolve_paths(cfg)
    out = Path(args.output) if args.output else paths.raw_data_dir / (
        f"{cfg['symbol'].lower()}_{cfg['timeframe'].lower()}.csv"
    )
    save_ohlc_csv(df, out)
    print(f"Fetched {len(df)} bars via source={df.attrs.get('source')} → {out}")
    return 0


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline model on XAUUSD features")
    _add_config_arg(parser)
    parser.add_argument(
        "--source",
        choices=["auto", "mt5", "csv", "sample"],
        default=None,
        help="Override data.source",
    )
    parser.add_argument("--name", default="xauusd_model", help="Artifact basename")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.source:
        cfg.setdefault("data", {})["source"] = args.source

    df = fetch_ohlc(cfg, persist=True)
    result = train_model(df, cfg)
    paths = resolve_paths(cfg)
    art = save_artifact(result, paths.artifacts_dir, name=args.name)

    metrics_path = paths.artifacts_dir / f"{args.name}_metrics.json"
    serializable = {k: v for k, v in result.metrics.items() if k != "features"}
    serializable["features"] = result.feature_names
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print(format_metrics(result.metrics))
    print(f"Saved model → {art}")
    print(f"Saved metrics → {metrics_path}")

    if result.split is not None:
        bt = run_signal_backtest(
            result.model,
            result.split,
            task=result.task,
            backtest_cfg=cfg.get("backtest"),
            on="test",
        )
        curve_path = paths.artifacts_dir / f"{args.name}_backtest.csv"
        bt["curve"].to_csv(curve_path, index=False)
        summary = {k: v for k, v in bt.items() if k != "curve"}
        bt_json = paths.artifacts_dir / f"{args.name}_backtest.json"
        bt_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            "Backtest (test): "
            f"net={summary['total_net_return']:.4f} "
            f"hit={summary['hit_rate']:.3f} "
            f"coverage={summary['coverage']:.3f} "
            f"sharpe≈{summary['sharpe_approx']:.2f}"
        )
        print(f"Saved backtest → {curve_path}")
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved model artifact")
    _add_config_arg(parser)
    parser.add_argument(
        "--model",
        default=None,
        help="Path to .joblib artifact (default: artifacts/xauusd_model.joblib)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "mt5", "csv", "sample"],
        default=None,
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.source:
        cfg.setdefault("data", {})["source"] = args.source
    paths = resolve_paths(cfg)
    model_path = Path(args.model) if args.model else paths.artifacts_dir / "xauusd_model.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}. Run train first.", file=sys.stderr)
        return 1

    payload = load_artifact(model_path)
    df = fetch_ohlc(cfg, persist=False)
    # Re-run training pipeline prep for aligned split metrics without refitting
    from xauusd_ml.models.train import evaluate_split, prepare_dataset, time_split

    frame = prepare_dataset(df, cfg)
    split = time_split(frame, cfg.get("split"))
    model = payload["model"]
    task = payload.get("task") or frame["task"].iloc[0]

    metrics = {
        "train": evaluate_split(model, split.X_train, split.y_train, task),
        "val": evaluate_split(model, split.X_val, split.y_val, task),
        "test": evaluate_split(model, split.X_test, split.y_test, task),
        "n_train": len(split.X_train),
        "n_val": len(split.X_val),
        "n_test": len(split.X_test),
        "task": task,
    }
    print(format_metrics(metrics))

    bt = run_signal_backtest(model, split, task=task, backtest_cfg=cfg.get("backtest"), on="test")
    summary = {k: v for k, v in bt.items() if k != "curve"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    train_main()
