"""Tiny pandas subset backed by numpy — enough for the XAUUSD pipeline."""

from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

__xauusd_shim__ = True
__version__ = "0.0.1-shim"


def _as_1d(values: Any, n: int | None = None) -> np.ndarray:
    if isinstance(values, Series):
        arr = np.array(values.values, copy=True)
    elif isinstance(values, (list, tuple)):
        arr = np.array(values)
    else:
        arr = np.asarray(values)
    if arr.ndim == 0:
        if n is None:
            return arr.reshape(1)
        arr = np.full(n, arr.item() if hasattr(arr, "item") else values)
        return arr
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _is_na(val: Any) -> bool:
    if val is None:
        return True
    try:
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _na_mask(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind in "f":
        return np.isnan(arr.astype(float, copy=False))
    if arr.dtype.kind in "Mm":
        return np.isnat(arr)
    if arr.dtype == object:
        out = np.empty(len(arr), dtype=bool)
        for i, v in enumerate(arr):
            out[i] = _is_na(v)
        return out
    return np.zeros(len(arr), dtype=bool)


class Index:
    def __init__(self, data: Iterable[Any] | None = None):
        if data is None:
            self._values = np.array([], dtype=object)
        elif isinstance(data, Index):
            self._values = np.array(data._values, copy=True)
        else:
            self._values = np.asarray(list(data) if not isinstance(data, np.ndarray) else data)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __getitem__(self, key: Any) -> Any:
        result = self._values[key]
        if np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0):
            return result.item() if hasattr(result, "item") else result
        return Index(result)

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)

    @property
    def values(self) -> np.ndarray:
        return self._values

    def tolist(self) -> list[Any]:
        return list(self._values)


class Series:
    def __init__(self, data=None, index=None, name: str | None = None, dtype=None):
        if isinstance(data, Series):
            values = np.array(data.values, copy=True)
            index = data.index if index is None else index
            name = data.name if name is None else name
        else:
            values = _as_1d(data) if data is not None else np.array([])
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        if index is None:
            self._index = Index(np.arange(len(values)))
        else:
            self._index = index if isinstance(index, Index) else Index(index)
        if len(self._index) != len(values) and len(values) != 0:
            raise ValueError("Series data and index must be same length")
        self._values = values
        self.name = name

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def index(self) -> Index:
        return self._index

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        arrays = []
        for inp in inputs:
            if isinstance(inp, Series):
                arrays.append(inp._values)
            elif isinstance(inp, DataFrame):
                return NotImplemented
            else:
                arrays.append(inp)
        result = ufunc(*arrays, **kwargs)
        if isinstance(result, np.ndarray) and result.shape == self._values.shape:
            return Series(result, index=self._index, name=self.name)
        return result

    def _wrap(self, values: np.ndarray, name: str | None = None) -> Series:
        return Series(values, index=self._index, name=self.name if name is None else name)

    def _binary(self, other: Any, op) -> Series:
        if isinstance(other, Series):
            other = other._values
        return self._wrap(op(self._values, other))

    def __add__(self, other: Any) -> Series:
        return self._binary(other, np.add)

    def __radd__(self, other: Any) -> Series:
        return self._wrap(np.add(other, self._values))

    def __sub__(self, other: Any) -> Series:
        return self._binary(other, np.subtract)

    def __rsub__(self, other: Any) -> Series:
        return self._wrap(np.subtract(other, self._values))

    def __mul__(self, other: Any) -> Series:
        return self._binary(other, np.multiply)

    def __rmul__(self, other: Any) -> Series:
        return self._wrap(np.multiply(other, self._values))

    def __truediv__(self, other: Any) -> Series:
        return self._binary(other, np.divide)

    def __rtruediv__(self, other: Any) -> Series:
        return self._wrap(np.divide(other, self._values))

    def __neg__(self) -> Series:
        return self._wrap(-self._values)

    def __eq__(self, other: Any) -> Series:
        return self._binary(other, np.equal)

    def __ne__(self, other: Any) -> Series:
        return self._binary(other, np.not_equal)

    def __gt__(self, other: Any) -> Series:
        return self._binary(other, np.greater)

    def __ge__(self, other: Any) -> Series:
        return self._binary(other, np.greater_equal)

    def __lt__(self, other: Any) -> Series:
        return self._binary(other, np.less)

    def __le__(self, other: Any) -> Series:
        return self._binary(other, np.less_equal)

    def __and__(self, other: Any) -> Series:
        return self._binary(other, np.logical_and)

    def __or__(self, other: Any) -> Series:
        return self._binary(other, np.logical_or)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            idx = self._index[key]
            return Series(self._values[key], index=idx, name=self.name)
        if isinstance(key, (Series, np.ndarray, list)):
            mask = np.asarray(key, dtype=bool) if not isinstance(key, Series) else np.asarray(key._values, dtype=bool)
            return Series(self._values[mask], index=self._index[mask], name=self.name)
        if isinstance(key, (int, np.integer)):
            return self._values[int(key)]
        raise TypeError(f"Invalid Series indexer {type(key)}")

    def astype(self, dtype) -> Series:
        if dtype is float or dtype == float or dtype == "float":
            return self._wrap(self._values.astype(float, copy=False))
        return self._wrap(self._values.astype(dtype))

    def diff(self, periods: int = 1) -> Series:
        values = self._values.astype(float, copy=False)
        out = np.empty_like(values, dtype=float)
        out[:] = np.nan
        if periods > 0 and periods < len(values):
            out[periods:] = values[periods:] - values[:-periods]
        elif periods < 0:
            p = -periods
            if p < len(values):
                out[:-p] = values[:-p] - values[p:]
        return self._wrap(out)

    def shift(self, periods: int = 1) -> Series:
        values = np.array(self._values, copy=True)
        out = np.empty_like(values)
        if out.dtype.kind in "f":
            out[:] = np.nan
        elif out.dtype.kind in "Mm":
            out[:] = np.datetime64("NaT")
        else:
            out = np.empty(len(values), dtype=object)
            out[:] = None
            if values.dtype.kind in "f":
                values = values.astype(object)
        if periods > 0:
            out[periods:] = values[:-periods]
        elif periods < 0:
            p = -periods
            out[:-p] = values[p:]
        else:
            out = values
        return self._wrap(out)

    def clip(self, lower=None, upper=None) -> Series:
        values = self._values.astype(float, copy=False)
        if lower is not None:
            values = np.maximum(values, lower)
        if upper is not None:
            values = np.minimum(values, upper)
        return self._wrap(values)

    def replace(self, to_replace, value) -> Series:
        values = np.array(self._values, copy=True)
        if isinstance(to_replace, (int, float, np.floating)) and values.dtype.kind in "fc":
            mask = np.isclose(values.astype(float), float(to_replace), equal_nan=False)
            # exact 0.0 should match
            mask = values.astype(float) == float(to_replace)
            values = values.astype(float, copy=True)
            values[mask] = value
        else:
            mask = values == to_replace
            values = values.astype(object, copy=True) if value is None or _is_na(value) else values
            values[mask] = value
        return self._wrap(values)

    def abs(self) -> Series:
        return self._wrap(np.abs(self._values.astype(float, copy=False)))

    def ewm(self, alpha: float | None = None, span: float | None = None, min_periods: int = 0, adjust: bool = True):
        if alpha is None:
            if span is None:
                raise ValueError("ewm requires alpha or span")
            alpha = 2.0 / (float(span) + 1.0)
        return _EWM(self, float(alpha), int(min_periods), bool(adjust))

    def rolling(self, window: int, min_periods: int | None = None):
        return _Rolling(self, int(window), min_periods)

    def where(self, cond, other) -> Series:
        mask = np.asarray(cond._values if isinstance(cond, Series) else cond, dtype=bool)
        values = np.array(self._values, dtype=float, copy=True)
        other_v = other._values if isinstance(other, Series) else other
        values = np.where(mask, values, other_v)
        return self._wrap(values.astype(float))

    def dropna(self) -> Series:
        mask = ~_na_mask(self._values)
        return Series(self._values[mask], index=self._index[mask], name=self.name)

    def notna(self) -> Series:
        return self._wrap(~_na_mask(self._values))

    def unique(self) -> np.ndarray:
        vals = self._values
        mask = ~_na_mask(vals)
        return np.unique(vals[mask])

    def all(self) -> bool:
        return bool(np.all(np.asarray(self._values, dtype=bool)))

    def any(self) -> bool:
        return bool(np.any(np.asarray(self._values, dtype=bool)))

    def sum(self) -> float:
        vals = self._values.astype(float, copy=False)
        return float(np.nansum(vals))

    def mean(self) -> float:
        return float(np.nanmean(self._values.astype(float, copy=False)))

    def std(self, ddof: int = 1) -> float:
        return float(np.nanstd(self._values.astype(float, copy=False), ddof=ddof))

    def max(self) -> Any:
        return np.nanmax(self._values)

    def min(self) -> Any:
        return np.nanmin(self._values)

    def to_numpy(self, dtype=None) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)

    def reset_index(self, drop: bool = False) -> Series | DataFrame:
        if drop:
            return Series(self._values, index=Index(np.arange(len(self))), name=self.name)
        raise NotImplementedError("Series.reset_index(drop=False) is not implemented in the shim")

    def copy(self) -> Series:
        return Series(np.array(self._values, copy=True), index=Index(self._index._values), name=self.name)

    @property
    def iloc(self) -> _SeriesILoc:
        return _SeriesILoc(self)

    def fillna(self, value) -> Series:
        values = np.array(self._values, copy=True)
        values[_na_mask(values)] = value
        return self._wrap(values)


class _SeriesILoc:
    def __init__(self, series: Series):
        self._s = series

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return self._s[key]
        if isinstance(key, (int, np.integer)):
            return self._s._values[int(key)]
        raise TypeError(key)


class _EWM:
    def __init__(self, series: Series, alpha: float, min_periods: int, adjust: bool):
        self.series = series
        self.alpha = alpha
        self.min_periods = min_periods
        self.adjust = adjust

    def mean(self) -> Series:
        x = self.series._values.astype(float, copy=False)
        n = len(x)
        out = np.empty(n, dtype=float)
        out[:] = np.nan
        alpha = self.alpha
        started = False
        prev = 0.0
        count = 0
        for i in range(n):
            xi = x[i]
            if np.isnan(xi):
                continue
            count += 1
            if not started:
                prev = xi
                started = True
            else:
                prev = alpha * xi + (1.0 - alpha) * prev
            if count >= max(self.min_periods, 1):
                out[i] = prev
        return self.series._wrap(out)


class _Rolling:
    def __init__(self, series: Series, window: int, min_periods: int | None):
        self.series = series
        self.window = window
        self.min_periods = window if min_periods is None else min_periods

    def mean(self) -> Series:
        x = self.series._values.astype(float, copy=False)
        n = len(x)
        out = np.empty(n, dtype=float)
        out[:] = np.nan
        w = self.window
        mp = self.min_periods
        for i in range(n):
            start = max(0, i - w + 1)
            window = x[start : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= mp:
                out[i] = valid.mean()
        return self.series._wrap(out)

    def std(self, ddof: int = 1) -> Series:
        x = self.series._values.astype(float, copy=False)
        n = len(x)
        out = np.empty(n, dtype=float)
        out[:] = np.nan
        w = self.window
        mp = self.min_periods
        for i in range(n):
            start = max(0, i - w + 1)
            window = x[start : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= max(mp, ddof + 1):
                out[i] = valid.std(ddof=ddof)
        return self.series._wrap(out)

    def max(self) -> Series:
        x = self.series._values.astype(float, copy=False)
        n = len(x)
        out = np.empty(n, dtype=float)
        out[:] = np.nan
        w = self.window
        mp = self.min_periods
        for i in range(n):
            start = max(0, i - w + 1)
            window = x[start : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= mp:
                out[i] = valid.max()
        return self.series._wrap(out)


class _ILoc:
    def __init__(self, frame: DataFrame):
        self._f = frame

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            rows, cols = key
            sub = self._f._slice_rows(rows)
            return sub._select_cols(cols)
        return self._f._slice_rows(key)


class _Loc:
    def __init__(self, frame: DataFrame):
        self._f = frame

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            rows, cols = key
            # integer position slices are used by this project (RangeIndex)
            sub = self._f._slice_rows(rows)
            return sub._select_cols(cols)
        return self._f._slice_rows(key)


class DataFrame:
    def __init__(self, data=None, index=None, columns=None):
        self.attrs: dict[str, Any] = {}
        self._columns: list[str] = []
        self._data: dict[str, np.ndarray] = {}

        if isinstance(data, DataFrame):
            self._index = Index(data._index._values)
            self._columns = list(data._columns)
            self._data = {c: np.array(data._data[c], copy=True) for c in self._columns}
            self.attrs = dict(data.attrs)
            return

        if data is None:
            data = {}

        if isinstance(data, dict):
            cols = list(columns) if columns is not None else list(data.keys())
            arrays = {}
            lengths: list[int] = []
            for c in cols:
                if c in data:
                    arr = _as_1d(data[c])
                else:
                    arr = None
                arrays[c] = arr
                if arr is not None:
                    lengths.append(len(arr))
            if index is not None:
                n = len(index)
            elif lengths:
                n = max(lengths)
            else:
                n = 0
            for c in cols:
                arr = arrays[c]
                if arr is None:
                    arrays[c] = np.full(n, np.nan)
                elif len(arr) == 1 and n != 1:
                    arrays[c] = np.full(n, arr[0])
                elif len(arr) != n:
                    raise ValueError(f"Column {c!r} length {len(arr)} != {n}")
            self._columns = [str(c) for c in cols]
            self._data = {str(c): arrays[c] for c in cols}
            self._index = Index(index) if index is not None else Index(np.arange(n))
            return

        if isinstance(data, np.ndarray):
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            n, k = arr.shape
            cols = list(columns) if columns is not None else [str(i) for i in range(k)]
            self._columns = [str(c) for c in cols]
            self._data = {self._columns[j]: arr[:, j] for j in range(k)}
            self._index = Index(index) if index is not None else Index(np.arange(n))
            return

        raise TypeError(f"Unsupported DataFrame data type: {type(data)}")

    @property
    def index(self) -> Index:
        return self._index

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._index), len(self._columns))

    @property
    def values(self) -> np.ndarray:
        return self.__array__()

    def __len__(self) -> int:
        return len(self._index)

    def __array__(self, dtype=None):
        if not self._columns:
            return np.empty((len(self._index), 0), dtype=dtype or float)
        cols = [np.asarray(self._data[c]) for c in self._columns]
        # numeric stack when possible
        try:
            stacked = np.column_stack([c.astype(float) for c in cols])
            return stacked if dtype is None else stacked.astype(dtype)
        except (TypeError, ValueError):
            stacked = np.empty((len(self._index), len(cols)), dtype=object)
            for j, c in enumerate(cols):
                stacked[:, j] = c
            return stacked

    def __contains__(self, item: Any) -> bool:
        return str(item) in self._data

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, list):
            return self._select_cols(key)
        if isinstance(key, str):
            return Series(self._data[key], index=self._index, name=key)
        if isinstance(key, (Series, np.ndarray)):
            mask = np.asarray(key._values if isinstance(key, Series) else key, dtype=bool)
            return self._slice_rows(mask)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        n = len(self._index)
        if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
            arr = np.full(n, value)
        else:
            arr = _as_1d(value, n=n)
            if len(arr) == 1 and n != 1:
                arr = np.full(n, arr[0])
            if len(arr) != n:
                raise ValueError(f"Length mismatch setting column {key}: {len(arr)} vs {n}")
        key = str(key)
        self._data[key] = arr
        if key not in self._columns:
            self._columns.append(key)

    def _select_cols(self, cols: Any) -> Any:
        if isinstance(cols, str):
            return Series(self._data[cols], index=self._index, name=cols)
        if isinstance(cols, slice):
            selected = self._columns[cols]
        elif isinstance(cols, (int, np.integer)):
            selected = [self._columns[int(cols)]]
            return Series(self._data[selected[0]], index=self._index, name=selected[0])
        else:
            selected = [str(c) for c in cols]
        out = DataFrame(
            {c: self._data[c] for c in selected},
            index=self._index,
        )
        out.attrs = dict(self.attrs)
        return out

    def _slice_rows(self, rows: Any) -> DataFrame:
        if isinstance(rows, slice):
            new_index = self._index[rows]
            data = {c: self._data[c][rows] for c in self._columns}
            # Index.__getitem__ may return scalar for weird cases; normalize
            if isinstance(new_index, Index):
                idx = new_index
            else:
                idx = Index([new_index])
            out = DataFrame(data, index=idx)
            out.attrs = dict(self.attrs)
            return out
        if isinstance(rows, (int, np.integer)):
            i = int(rows)
            data = {c: np.array([self._data[c][i]]) for c in self._columns}
            out = DataFrame(data, index=Index([self._index[i]]))
            out.attrs = dict(self.attrs)
            return out
        mask = np.asarray(rows)
        if mask.dtype == bool:
            data = {c: self._data[c][mask] for c in self._columns}
            out = DataFrame(data, index=Index(self._index._values[mask]))
            out.attrs = dict(self.attrs)
            return out
        idx = mask.astype(int)
        data = {c: self._data[c][idx] for c in self._columns}
        out = DataFrame(data, index=Index(self._index._values[idx]))
        out.attrs = dict(self.attrs)
        return out

    @property
    def loc(self) -> _Loc:
        return _Loc(self)

    @property
    def iloc(self) -> _ILoc:
        return _ILoc(self)

    def astype(self, dtype) -> DataFrame:
        out = self.copy()
        for c in out._columns:
            out._data[c] = out._data[c].astype(dtype)
        return out

    def copy(self) -> DataFrame:
        out = DataFrame({c: np.array(self._data[c], copy=True) for c in self._columns}, index=Index(self._index._values))
        out.attrs = dict(self.attrs)
        return out

    def sort_values(self, by: str, **_kwargs: Any) -> DataFrame:
        order = np.argsort(self._data[by], kind="stable")
        return self._slice_rows(order)

    def reset_index(self, drop: bool = False) -> DataFrame:
        if not drop:
            raise NotImplementedError("reset_index(drop=False) is not implemented in the shim")
        out = DataFrame({c: np.array(self._data[c], copy=True) for c in self._columns}, index=Index(np.arange(len(self))))
        out.attrs = dict(self.attrs)
        return out

    def dropna(self) -> DataFrame:
        n = len(self)
        mask = np.ones(n, dtype=bool)
        for c in self._columns:
            mask &= ~_na_mask(self._data[c])
        return self._slice_rows(mask)

    def max(self, axis: int | None = None) -> Any:
        if axis == 1:
            arr = np.column_stack([self._data[c].astype(float) for c in self._columns])
            with np.errstate(all="ignore"):
                vals = np.nanmax(arr, axis=1)
            return Series(vals, index=self._index)
        if axis == 0 or axis is None:
            return Series([np.nanmax(self._data[c].astype(float)) for c in self._columns], index=Index(self._columns))
        raise ValueError(axis)

    def min(self, axis: int | None = None) -> Any:
        if axis == 1:
            arr = np.column_stack([self._data[c].astype(float) for c in self._columns])
            return Series(np.nanmin(arr, axis=1), index=self._index)
        raise ValueError(axis)

    def to_csv(self, path: str | Path, index: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            header = (["index"] if index else []) + self._columns
            writer.writerow(header)
            for i in range(len(self)):
                row: list[Any] = []
                if index:
                    row.append(_csv_cell(self._index._values[i]))
                for c in self._columns:
                    row.append(_csv_cell(self._data[c][i]))
                writer.writerow(row)


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (np.datetime64, datetime)):
        return _fmt_dt(value)
    if isinstance(value, np.generic):
        if _is_na(value.item() if hasattr(value, "item") else value):
            return ""
        return value.item()
    if _is_na(value):
        return ""
    return value


def _fmt_dt(value: Any) -> str:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return ""
        # ns -> python datetime UTC
        ts = value.astype("datetime64[us]").astype(datetime)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat().replace("T", " ")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat().replace("T", " ")
    return str(value)


def read_csv(path: str | Path) -> DataFrame:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return DataFrame()
    header = rows[0]
    cols: dict[str, list[Any]] = {h: [] for h in header}
    for row in rows[1:]:
        if not row:
            continue
        for i, h in enumerate(header):
            cell = row[i] if i < len(row) else ""
            cols[h].append(_parse_cell(cell))
    return DataFrame(cols)


def _parse_cell(cell: str) -> Any:
    if cell is None or cell == "":
        return np.nan
    try:
        if cell.isdigit() or (cell.startswith("-") and cell[1:].isdigit()):
            return int(cell)
        return float(cell)
    except ValueError:
        return cell


def to_datetime(arg, utc: bool = False, unit: str | None = None) -> Series | np.ndarray:
    series = isinstance(arg, Series)
    values = arg._values if series else np.asarray(arg, dtype=object)

    out = np.empty(len(values), dtype="datetime64[ns]")
    for i, v in enumerate(values):
        out[i] = _parse_datetime(v, unit=unit)
    if series:
        return Series(out, index=arg.index, name=arg.name)
    return out


def _parse_datetime(v: Any, unit: str | None = None) -> np.datetime64:
    if isinstance(v, np.datetime64):
        return v.astype("datetime64[ns]")
    if isinstance(v, datetime):
        if v.tzinfo is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(v, "ns")
    if isinstance(v, (int, float, np.integer, np.floating)) and unit:
        if _is_na(v):
            return np.datetime64("NaT")
        return np.datetime64(int(v), unit).astype("datetime64[ns]")
    if isinstance(v, str):
        text = v.replace("T", " ").replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = datetime.strptime(text.split("+")[0].split(".")[0].strip(), "%Y-%m-%d %H:%M:%S")
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(dt, "ns")
    return np.datetime64("NaT")


class Timestamp:
    def __init__(self, ts: Any, tz: str | None = None):
        if isinstance(ts, datetime):
            dt = ts
        else:
            text = str(ts).replace("T", " ")
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        if tz in ("UTC", "utc") and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        self._dt = dt

    def to_datetime64(self) -> np.datetime64:
        dt = self._dt
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(dt, "ns")

    def __array__(self, dtype=None):
        return np.asarray(self.to_datetime64(), dtype=dtype)


def date_range(start, periods: int, freq: str = "15min") -> np.ndarray:
    if isinstance(start, Timestamp):
        start_dt = start._dt
    elif isinstance(start, datetime):
        start_dt = start
    else:
        start_dt = Timestamp(start, tz="UTC")._dt
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    minutes = _freq_minutes(freq)
    times = [start_dt + timedelta(minutes=minutes * i) for i in range(int(periods))]
    naive = [t.astimezone(timezone.utc).replace(tzinfo=None) for t in times]
    return np.array(naive, dtype="datetime64[ns]")


def _freq_minutes(freq: str) -> int:
    freq = str(freq).strip().lower().replace("t", "min")
    if freq.endswith("min"):
        return int(freq[:-3])
    if freq.endswith("h"):
        return int(freq[:-1]) * 60
    if freq.endswith("d"):
        return int(freq[:-1]) * 1440
    raise ValueError(f"Unsupported freq {freq!r}")


def concat(objs: Sequence[Series | DataFrame], axis: int = 0) -> DataFrame | Series:
    objs = list(objs)
    if axis == 1:
        cols: dict[str, np.ndarray] = {}
        index = None
        col_i = 0
        for obj in objs:
            if isinstance(obj, Series):
                name = obj.name if obj.name is not None else f"col_{col_i}"
                name = str(name)
                while name in cols:
                    col_i += 1
                    name = f"{name}_{col_i}"
                cols[name] = obj._values
                index = obj.index
                col_i += 1
            else:
                for c in obj.columns:
                    name = c
                    while name in cols:
                        name = f"{c}_{col_i}"
                        col_i += 1
                    cols[name] = obj._data[c]
                    index = obj.index
        return DataFrame(cols, index=index)
    raise NotImplementedError("concat(axis=0) is not implemented in the shim")
