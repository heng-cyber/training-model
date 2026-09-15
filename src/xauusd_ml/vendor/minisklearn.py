"""Minimal sklearn-like API: scaler, pipeline, trees, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__xauusd_shim__ = True


def _as_xy(X, y=None) -> tuple[np.ndarray, np.ndarray | None]:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = None if y is None else np.asarray(y, dtype=float).reshape(-1)
    return X_arr, y_arr


class BaseEstimator:
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(getattr(self, "__dict__", {}))

    def set_params(self, **params: Any) -> BaseEstimator:
        for k, v in params.items():
            setattr(self, k, v)
        return self


class StandardScaler(BaseEstimator):
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X, y=None) -> StandardScaler:
        X_arr, _ = _as_xy(X)
        self.mean_ = np.nanmean(X_arr, axis=0)
        scale = np.nanstd(X_arr, axis=0, ddof=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X) -> np.ndarray:
        X_arr, _ = _as_xy(X)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class Pipeline(BaseEstimator):
    def __init__(self, steps: list[tuple[str, Any]]):
        self.steps = list(steps)

    def _transform(self, X):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return Xt

    def fit(self, X, y=None) -> Pipeline:
        Xt = X
        y_arr = y
        for _, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y_arr)
            else:
                step.fit(Xt, y_arr)
                Xt = step.transform(Xt)
        self.steps[-1][1].fit(Xt, y_arr)
        return self

    def predict(self, X):
        return self.steps[-1][1].predict(self._transform(X))

    def predict_proba(self, X):
        final = self.steps[-1][1]
        if not hasattr(final, "predict_proba"):
            raise AttributeError("final estimator has no predict_proba")
        return final.predict_proba(self._transform(X))


@dataclass
class _Node:
    left: _Node | None = None
    right: _Node | None = None
    feature: int | None = None
    threshold: float | None = None
    value: float = 0.0
    is_leaf: bool = True


def _best_split(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_thresholds: int = 24) -> tuple[int, float, float] | None:
    n, n_features = X.shape
    if n < 2:
        return None
    best: tuple[int, float, float] | None = None  # feature, threshold, sse
    best_sse = np.inf
    y_sum = float(y.sum())
    y_sq = float((y * y).sum())

    for feat in range(n_features):
        col = X[:, feat]
        if not np.isfinite(col).all():
            finite = np.isfinite(col)
            if finite.sum() < 2:
                continue
        # unique-ish candidate thresholds via quantiles
        qs = np.linspace(0.05, 0.95, num=min(max_thresholds, n))
        candidates = np.unique(np.quantile(col, qs))
        if len(candidates) < 2:
            continue
        # use midpoints between sorted unique values when few uniques
        for thr in candidates:
            left = col <= thr
            n_left = int(left.sum())
            n_right = n - n_left
            if n_left < 1 or n_right < 1:
                continue
            sl = float(y[left].sum())
            sr = y_sum - sl
            sse = (y_sq - (sl * sl) / n_left - (sr * sr) / n_right)
            if sse < best_sse:
                best_sse = sse
                best = (feat, float(thr), sse)
    return best


def _grow_tree(X: np.ndarray, y: np.ndarray, depth: int, max_depth: int, rng: np.random.Generator) -> _Node:
    node = _Node(value=float(np.mean(y)) if len(y) else 0.0, is_leaf=True)
    if depth >= max_depth or len(y) < 4 or np.allclose(y, y[0]):
        return node
    split = _best_split(X, y, rng)
    if split is None:
        return node
    feat, thr, _ = split
    left_mask = X[:, feat] <= thr
    right_mask = ~left_mask
    if left_mask.sum() < 1 or right_mask.sum() < 1:
        return node
    node.is_leaf = False
    node.feature = feat
    node.threshold = thr
    node.left = _grow_tree(X[left_mask], y[left_mask], depth + 1, max_depth, rng)
    node.right = _grow_tree(X[right_mask], y[right_mask], depth + 1, max_depth, rng)
    return node


def _predict_tree(node: _Node, X: np.ndarray) -> np.ndarray:
    out = np.full(len(X), node.value, dtype=float)
    if node.is_leaf or len(X) == 0:
        return out
    left = X[:, int(node.feature)] <= float(node.threshold)  # type: ignore[arg-type]
    out[left] = _predict_tree(node.left, X[left])  # type: ignore[arg-type]
    out[~left] = _predict_tree(node.right, X[~left])  # type: ignore[arg-type]
    return out


class _GradientBooster(BaseEstimator):
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int | None = 42,
        **_kwargs: Any,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.random_state = random_state
        self.trees_: list[_Node] = []
        self.init_: float = 0.0
        self._is_classifier = True

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    def fit(self, X, y) -> _GradientBooster:
        X_arr, y_arr = _as_xy(X, y)
        assert y_arr is not None
        rng = self._rng()
        if self._is_classifier:
            p = float(np.clip(y_arr.mean(), 1e-6, 1.0 - 1e-6))
            self.init_ = float(np.log(p / (1.0 - p)))
        else:
            self.init_ = float(y_arr.mean())
        F = np.full(len(y_arr), self.init_, dtype=float)
        self.trees_ = []
        lr = self.learning_rate
        for _ in range(self.n_estimators):
            if self._is_classifier:
                p_hat = 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))
                residual = y_arr - p_hat
            else:
                residual = y_arr - F
            tree = _grow_tree(X_arr, residual, 0, self.max_depth, rng)
            update = _predict_tree(tree, X_arr)
            F = F + lr * update
            self.trees_.append(tree)
        return self

    def decision_function(self, X) -> np.ndarray:
        X_arr, _ = _as_xy(X)
        F = np.full(len(X_arr), self.init_, dtype=float)
        lr = self.learning_rate
        for tree in self.trees_:
            F = F + lr * _predict_tree(tree, X_arr)
        return F


class GradientBoostingClassifier(_GradientBooster):
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int | None = 42,
        **kwargs: Any,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self._is_classifier = True

    def predict_proba(self, X) -> np.ndarray:
        F = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


class GradientBoostingRegressor(_GradientBooster):
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int | None = 42,
        **kwargs: Any,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self._is_classifier = False

    def predict(self, X) -> np.ndarray:
        return self.decision_function(X)


class RandomForestClassifier(GradientBoostingClassifier):
    """Approximate RF with bagged shallow trees averaged as log-odds."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = 42,
        n_jobs: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=1.0,
            max_depth=3 if max_depth is None else int(max_depth),
            random_state=random_state,
            **kwargs,
        )
        self.n_jobs = n_jobs

    def fit(self, X, y) -> RandomForestClassifier:
        X_arr, y_arr = _as_xy(X, y)
        assert y_arr is not None
        rng = self._rng()
        p = float(np.clip(y_arr.mean(), 1e-6, 1.0 - 1e-6))
        self.init_ = 0.0
        self.trees_ = []
        n = len(y_arr)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = _grow_tree(X_arr[idx], y_arr[idx], 0, self.max_depth, rng)
            self.trees_.append(tree)
        self.learning_rate = 1.0 / max(len(self.trees_), 1)
        # map leaf 0/1 probabilities into log-odds at predict time via averaging probs
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_arr, _ = _as_xy(X)
        acc = np.zeros(len(X_arr), dtype=float)
        for tree in self.trees_:
            acc += np.clip(_predict_tree(tree, X_arr), 0.0, 1.0)
        p = acc / max(len(self.trees_), 1)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])


class RandomForestRegressor(GradientBoostingRegressor):
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = 42,
        n_jobs: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=1.0,
            max_depth=3 if max_depth is None else int(max_depth),
            random_state=random_state,
            **kwargs,
        )
        self.n_jobs = n_jobs


class _LogisticMLP(BaseEstimator):
    """Small numpy MLP used as sklearn_mlp fallback."""

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 200,
        random_state: int | None = 42,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        **_kwargs: Any,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self._is_classifier = True
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []

    def fit(self, X, y) -> _LogisticMLP:
        X_arr, y_arr = _as_xy(X, y)
        assert y_arr is not None
        rng = np.random.default_rng(self.random_state)
        if self._is_classifier:
            y_arr = y_arr.reshape(-1, 1)
        else:
            y_arr = y_arr.reshape(-1, 1)
        layers = [X_arr.shape[1], *self.hidden_layer_sizes, 1]
        self.weights_ = []
        self.biases_ = []
        for a, b in zip(layers[:-1], layers[1:]):
            self.weights_.append(rng.normal(0, 0.2, size=(a, b)))
            self.biases_.append(np.zeros(b))
        lr = 0.05
        n = len(X_arr)
        for _ in range(min(self.max_iter, 80)):
            idx = rng.integers(0, n, size=min(256, n))
            xb, yb = X_arr[idx], y_arr[idx]
            acts = [xb]
            h = xb
            for i, (w, b) in enumerate(zip(self.weights_, self.biases_)):
                h = h @ w + b
                if i < len(self.weights_) - 1:
                    h = np.tanh(h)
                else:
                    h = 1.0 / (1.0 + np.exp(-np.clip(h, -30, 30))) if self._is_classifier else h
                acts.append(h)
            pred = acts[-1]
            delta = pred - yb
            for i in range(len(self.weights_) - 1, -1, -1):
                a_prev = acts[i]
                grad_w = a_prev.T @ delta / len(xb)
                grad_b = delta.mean(axis=0)
                self.weights_[i] -= lr * grad_w
                self.biases_[i] -= lr * grad_b
                if i > 0:
                    delta = (delta @ self.weights_[i].T) * (1.0 - acts[i] ** 2)
        return self

    def _forward(self, X) -> np.ndarray:
        h = np.asarray(X, dtype=float)
        for i, (w, b) in enumerate(zip(self.weights_, self.biases_)):
            h = h @ w + b
            if i < len(self.weights_) - 1:
                h = np.tanh(h)
            elif self._is_classifier:
                h = 1.0 / (1.0 + np.exp(-np.clip(h, -30, 30)))
        return h.reshape(-1)

    def predict_proba(self, X) -> np.ndarray:
        p = np.clip(self._forward(X), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        if self._is_classifier:
            return (self._forward(X) >= 0.5).astype(float)
        return self._forward(X)


class MLPClassifier(_LogisticMLP):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._is_classifier = True


class MLPRegressor(_LogisticMLP):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._is_classifier = False


def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_true == y_pred))


def f1_score(y_true, y_pred, zero_division: float = 0) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return float(zero_division)
    return float(2 * tp / denom)


def roc_auc_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype(float)
    y_score = np.asarray(y_score).reshape(-1).astype(float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("ROC AUC requires both classes")
    # Mann–Whitney / Wilcoxon
    # P(score_pos > score_neg) + 0.5 P(eq)
    n = 0.0
    for p in pos:
        n += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(n / (len(pos) * len(neg)))


def mean_absolute_error(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mean_squared_error(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
