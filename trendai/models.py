"""Forecasting models.

All models share the tiny :class:`Forecaster` interface::

    m = make_model("holt_damped").fit(y, season_length=12)
    yhat = m.predict(h=6)          # np.ndarray of length 6

``y`` is a 1-D float ``np.ndarray`` with no NaNs (regular spacing assumed).
Models must never raise on short or constant input once ``len(y) >= min_obs``:
they degrade gracefully (e.g. drop seasonality) instead.  Predictions must be
finite.

The registry maps a name to a zero-argument factory so the backtester can
create fresh, unfitted instances for every rolling window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Forecaster(ABC):
    """Base class: ``fit(y, season_length) -> self`` then ``predict(h) -> ndarray``."""

    name: str = "base"
    min_obs: int = 2          # minimum len(y) needed to fit
    needs_season: bool = False  # True if the model only makes sense with seasonality
    requires_positive: bool = False  # True if the model needs all y > 0 (log / multiplicative)

    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self.season_length: int = 1

    def fit(self, y: np.ndarray, season_length: int = 1) -> "Forecaster":
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1-D")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")
        if len(y) < self.min_obs:
            raise ValueError(f"{self.name} needs at least {self.min_obs} observations, got {len(y)}")
        self._y = y
        self.season_length = max(1, int(season_length))
        self._fit(y, self.season_length)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("call fit() first")
        if h < 1:
            raise ValueError("h must be >= 1")
        out = np.asarray(self._predict(int(h)), dtype=float)
        if out.shape != (h,):
            raise RuntimeError(f"{self.name} returned shape {out.shape}, expected ({h},)")
        if not np.all(np.isfinite(out)):
            # last-resort guard: never leak NaN/inf to callers
            out = np.where(np.isfinite(out), out, self._y[-1])
        return out

    @abstractmethod
    def _fit(self, y: np.ndarray, m: int) -> None: ...

    @abstractmethod
    def _predict(self, h: int) -> np.ndarray: ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


# --------------------------------------------------------------------------- simple baselines


class Naive(Forecaster):
    """Repeat the last observation."""

    name = "naive"
    min_obs = 1

    def _fit(self, y, m):
        self._last = y[-1]

    def _predict(self, h):
        return np.full(h, self._last)


class SeasonalNaive(Forecaster):
    """Repeat the last full season (falls back to naive without enough data)."""

    name = "seasonal_naive"
    min_obs = 1
    needs_season = True

    def _fit(self, y, m):
        self._m = m if (m > 1 and len(y) >= m) else 1
        self._tail = y[-self._m:]

    def _predict(self, h):
        reps = int(np.ceil(h / self._m))
        return np.tile(self._tail, reps)[:h]


class Drift(Forecaster):
    """Random walk with drift: extrapolate the line from first to last value."""

    name = "drift"
    min_obs = 2

    def _fit(self, y, m):
        self._last = y[-1]
        self._slope = (y[-1] - y[0]) / (len(y) - 1)

    def _predict(self, h):
        return self._last + self._slope * np.arange(1, h + 1)


# --------------------------------------------------------------------------- registry

MODEL_REGISTRY: dict[str, Callable[[], Forecaster]] = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "drift": Drift,
}


def register(name: str, factory: Callable[[], Forecaster]) -> None:
    MODEL_REGISTRY[name] = factory


def make_model(name: str) -> Forecaster:
    """Create a fresh, unfitted model by registry name."""
    try:
        return MODEL_REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}") from None


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def default_candidates(n_obs: int, season_length: int, positive: bool) -> list[str]:
    """Registry names worth trying for a series of this shape.

    Only returns models whose ``min_obs`` is satisfied; seasonal models only
    when at least two full seasons are available.  ``positive`` means all
    observations are > 0 (enables log-space variants if registered).
    """
    names = []
    for nm, factory in MODEL_REGISTRY.items():
        m = factory()
        if n_obs < m.min_obs:
            continue
        if m.needs_season and (season_length <= 1 or n_obs < 2 * season_length):
            continue
        if (m.requires_positive or nm.endswith("@log")) and not positive:
            continue
        names.append(nm)
    return names
