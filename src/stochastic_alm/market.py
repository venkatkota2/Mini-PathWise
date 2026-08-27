"""Correlated economic scenario generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _correlation_root(correlation: FloatArray) -> FloatArray:
    """Return a symmetric square root for a validated PSD correlation matrix."""
    symmetric = 0.5 * (correlation + correlation.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if eigenvalues.min() < -1e-10:
        raise ValueError("correlation must be positive semidefinite")
    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T


def _default_correlation() -> FloatArray:
    return np.array(
        [
            [1.00, -0.20, 0.25, 0.30],
            [-0.20, 1.00, -0.10, -0.35],
            [0.25, -0.10, 1.00, 0.10],
            [0.30, -0.35, 0.10, 1.00],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class MarketAssumptions:
    initial_short_rate: float = 0.035
    long_run_short_rate: float = 0.035
    rate_mean_reversion: float = 0.35
    rate_volatility: float = 0.012
    initial_equity_index: float = 100.0
    equity_risk_premium: float = 0.045
    equity_dividend_yield: float = 0.018
    equity_volatility: float = 0.18
    initial_inflation_rate: float = 0.022
    long_run_inflation_rate: float = 0.020
    inflation_mean_reversion: float = 0.45
    inflation_volatility: float = 0.008
    initial_credit_spread: float = 0.012
    long_run_credit_spread: float = 0.013
    spread_mean_reversion: float = 0.60
    spread_volatility: float = 0.007
    correlation: FloatArray = field(default_factory=_default_correlation)

    def __post_init__(self) -> None:
        numeric = (
            self.initial_short_rate,
            self.long_run_short_rate,
            self.rate_mean_reversion,
            self.rate_volatility,
            self.initial_equity_index,
            self.equity_risk_premium,
            self.equity_dividend_yield,
            self.equity_volatility,
            self.initial_inflation_rate,
            self.long_run_inflation_rate,
            self.inflation_mean_reversion,
            self.inflation_volatility,
            self.initial_credit_spread,
            self.long_run_credit_spread,
            self.spread_mean_reversion,
            self.spread_volatility,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("market assumptions must be finite")
        if self.initial_equity_index <= 0:
            raise ValueError("initial equity index must be positive")
        if any(
            value < 0
            for value in (
                self.rate_mean_reversion,
                self.rate_volatility,
                self.equity_volatility,
                self.inflation_mean_reversion,
                self.inflation_volatility,
                self.initial_credit_spread,
                self.long_run_credit_spread,
                self.spread_mean_reversion,
                self.spread_volatility,
            )
        ):
            raise ValueError("mean reversion, volatility, and credit spreads must be non-negative")
        corr = np.asarray(self.correlation, dtype=float)
        if corr.shape != (4, 4):
            raise ValueError("correlation must be a 4x4 matrix")
        if not np.all(np.isfinite(corr)):
            raise ValueError("correlation must contain finite values")
        if not np.allclose(corr, corr.T, atol=1e-12):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-12):
            raise ValueError("correlation must have ones on the diagonal")
        if np.linalg.eigvalsh(corr).min() < -1e-10:
            raise ValueError("correlation must be positive semidefinite")
        object.__setattr__(self, "correlation", corr)


@dataclass(frozen=True)
class MarketPaths:
    short_rate: FloatArray
    equity_index: FloatArray
    inflation_rate: FloatArray
    inflation_index: FloatArray
    credit_spread: FloatArray
    dt: float

    @property
    def scenarios(self) -> int:
        return int(self.short_rate.shape[0])

    @property
    def steps(self) -> int:
        return int(self.short_rate.shape[1] - 1)


def simulate_market(
    assumptions: MarketAssumptions | None = None,
    *,
    scenarios: int = 10_000,
    years: float = 1.0,
    steps_per_year: int = 12,
    seed: int = 7,
) -> MarketPaths:
    """Simulate correlated economic factors with Euler/log-Euler discretization."""
    if (
        not isinstance(scenarios, int)
        or isinstance(scenarios, bool)
        or scenarios <= 0
        or not isinstance(steps_per_year, int)
        or isinstance(steps_per_year, bool)
        or steps_per_year <= 0
        or not isfinite(years)
        or years <= 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise ValueError("scenarios, years, and steps_per_year must be positive")

    a = assumptions or MarketAssumptions()
    steps = max(1, round(years * steps_per_year))
    dt = years / steps
    root = _correlation_root(np.asarray(a.correlation))
    rng = np.random.default_rng(seed)

    short_rate = np.empty((scenarios, steps + 1))
    equity = np.empty_like(short_rate)
    inflation_rate = np.empty_like(short_rate)
    inflation_index = np.empty_like(short_rate)
    spread = np.empty_like(short_rate)

    short_rate[:, 0] = a.initial_short_rate
    equity[:, 0] = a.initial_equity_index
    inflation_rate[:, 0] = a.initial_inflation_rate
    inflation_index[:, 0] = 1.0
    spread[:, 0] = a.initial_credit_spread

    sqrt_dt = np.sqrt(dt)
    for step in range(1, steps + 1):
        z = rng.standard_normal((scenarios, 4)) @ root.T
        previous_rate = short_rate[:, step - 1]
        previous_inflation = inflation_rate[:, step - 1]
        previous_spread = spread[:, step - 1]

        short_rate[:, step] = (
            previous_rate
            + a.rate_mean_reversion * (a.long_run_short_rate - previous_rate) * dt
            + a.rate_volatility * sqrt_dt * z[:, 0]
        )
        equity_drift = (
            previous_rate
            + a.equity_risk_premium
            - a.equity_dividend_yield
            - 0.5 * a.equity_volatility**2
        )
        equity[:, step] = equity[:, step - 1] * np.exp(
            equity_drift * dt + a.equity_volatility * sqrt_dt * z[:, 1]
        )
        inflation_rate[:, step] = (
            previous_inflation
            + a.inflation_mean_reversion * (a.long_run_inflation_rate - previous_inflation) * dt
            + a.inflation_volatility * sqrt_dt * z[:, 2]
        )
        inflation_index[:, step] = inflation_index[:, step - 1] * np.exp(
            inflation_rate[:, step] * dt
        )
        spread[:, step] = np.maximum(
            0.0,
            previous_spread
            + a.spread_mean_reversion * (a.long_run_credit_spread - previous_spread) * dt
            + a.spread_volatility * sqrt_dt * z[:, 3],
        )

    return MarketPaths(
        short_rate=short_rate,
        equity_index=equity,
        inflation_rate=inflation_rate,
        inflation_index=inflation_index,
        credit_spread=spread,
        dt=dt,
    )
