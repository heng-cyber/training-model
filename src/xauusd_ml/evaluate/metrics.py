"""Pretty-print helpers for training metrics."""

from __future__ import annotations

from typing import Any


def format_metrics(metrics: dict[str, Any]) -> str:
    lines = [
        f"task={metrics.get('task')}  "
        f"n_train={metrics.get('n_train')}  "
        f"n_val={metrics.get('n_val')}  "
        f"n_test={metrics.get('n_test')}"
    ]
    for split_name in ("train", "val", "test"):
        block = metrics.get(split_name) or {}
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in block.items()]
        lines.append(f"  {split_name}: " + ", ".join(parts))
    return "\n".join(lines)
