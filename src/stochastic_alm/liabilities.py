"""Liability cash-flow construction and valuation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LiabilityCashflows:
    times: FloatArray
    amounts: FloatArray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        amounts = np.asarray(self.amounts, dtype=float)
        if times.ndim != 1 or amounts.ndim != 1 or len(times) != len(amounts):
            raise ValueError("times and amounts must be equally sized one-dimensional arrays")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(amounts)):
            raise ValueError("cash-flow inputs must be finite")
        if len(times) == 0 or np.any(times <= 0) or np.any(amounts < 0):
            raise ValueError("cash flows require positive times and non-negative amounts")
        if np.any(np.diff(times) <= 0):
            raise ValueError("cash-flow times must be strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "amounts", amounts)

    @classmethod
    def level_annuity(
        cls,
        *,
        annual_payment: float,
        years: int,
        growth: float = 0.02,
    ) -> LiabilityCashflows:
        if annual_payment <= 0 or years <= 0 or growth <= -1:
            raise ValueError("invalid annuity assumptions")
        times = np.arange(1, years + 1, dtype=float)
        amounts = annual_payment * np.power(1.0 + growth, times - 1.0)
        return cls(times=times, amounts=amounts)

    def present_value(self, discount_rate: float) -> float:
        """Present value under a continuously compounded flat rate."""
        if not isfinite(discount_rate):
            raise ValueError("discount_rate must be finite")
        return float(np.sum(self.amounts * np.exp(-discount_rate * self.times)))

    def payments_by_step(self, horizon: float, steps: int) -> FloatArray:
        """Aggregate payments into `(start, end]` projection intervals.

        Market evolution is applied over each interval before payments at or
        before the interval end are deducted. A payment exactly at the risk
        horizon is therefore paid before ending assets are reported.
        """
        if not isfinite(horizon) or horizon <= 0 or not isinstance(steps, int) or steps <= 0:
            raise ValueError("horizon and steps must be positive")
        boundaries = np.linspace(0.0, horizon, steps + 1)
        payments = np.zeros(steps + 1, dtype=float)
        for step in range(1, steps + 1):
            due = (self.times > boundaries[step - 1] + 1e-12) & (
                self.times <= boundaries[step] + 1e-12
            )
            payments[step] = float(np.sum(self.amounts[due]))
        return payments

    def value_at_horizon(
        self,
        horizon: float,
        scenario_discount_rates: FloatArray,
    ) -> FloatArray:
        """Value remaining cash flows at a horizon using scenario-specific flat rates."""
        rates = np.asarray(scenario_discount_rates, dtype=float)
        if not isfinite(horizon) or horizon < 0:
            raise ValueError("horizon must be finite and non-negative")
        if rates.ndim != 1 or not np.all(np.isfinite(rates)):
            raise ValueError("scenario discount rates must be a finite vector")
        remaining = self.times > horizon + 1e-12
        if not np.any(remaining):
            return np.zeros_like(rates)
        durations = self.times[remaining] - horizon
        return np.sum(
            self.amounts[remaining][None, :] * np.exp(-rates[:, None] * durations[None, :]),
            axis=1,
        )
