"""Minimal joblib.dump / joblib.load using pickle."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

__xauusd_shim__ = True


def dump(obj: Any, filename: str | Path, **_kwargs: Any) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename: str | Path, **_kwargs: Any) -> Any:
    with Path(filename).open("rb") as fh:
        return pickle.load(fh)
