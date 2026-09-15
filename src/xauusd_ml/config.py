"""Configuration loading and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# MT5 timeframe name -> minutes (used for sample data and docs)
TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    raw_data_dir: Path
    sample_data_dir: Path
    artifacts_dir: Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until pyproject.toml is found."""
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cur


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config; default is configs/default.yaml under the project root."""
    root = find_project_root()
    cfg_path = Path(path) if path else root / "configs" / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    cfg["_config_path"] = str(cfg_path)
    cfg["_project_root"] = str(root)
    return cfg


def resolve_paths(cfg: dict[str, Any]) -> ProjectPaths:
    root = Path(cfg.get("_project_root", find_project_root()))
    paths = cfg.get("paths", {})
    return ProjectPaths(
        root=root,
        config=Path(cfg.get("_config_path", root / "configs" / "default.yaml")),
        raw_data_dir=root / paths.get("raw_data_dir", "data/raw"),
        sample_data_dir=root / paths.get("sample_data_dir", "data/sample"),
        artifacts_dir=root / paths.get("artifacts_dir", "artifacts"),
    )
