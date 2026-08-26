"""Liability cash-flow construction and valuation."""

from __future__ import annotations

from dataclasses import dataclass

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
        if len(times) == 0 or np.any(times <= 0) or np.any(amounts < 0):
            raise ValueError("cash flows require positive times and non-negative amounts")
        if np.any(np.diff(times) <= 0):
            raise ValueError("cash-flow times must be strictly increasing")

    @classmethod
    def level_annuity(
        cls,
        *,
        annual_payment: float,
        years: int,
        growth: float = 0.02,
    ) -> "LiabilityCashflows":
        if annual_payment <= 0 or years <= 0 or growth <= -1:
            raise ValueError("invalid annuity assumptions")
        times = np.arange(1, years + 1, dtype=float)
        amounts = annual_payment * np.power(1.0 + growth, times - 1.0)
        return cls(times=times, amounts=amounts)

    def present_value(self, discount_rate: float) -> float:
        if discount_rate <= -1:
            raise ValueError("discount_rate must be greater than -100%")
        return float(np.sum(self.amounts / np.power(1.0 + discount_rate, self.times)))

    def value_at_horizon(
        self,
        horizon: float,
        scenario_discount_rates: FloatArray,
    ) -> FloatArray:
        """Value remaining cash flows at a horizon using scenario-specific flat rates."""
        rates = np.asarray(scenario_discount_rates, dtype=float)
        remaining = self.times > horizon + 1e-12
        if not np.any(remaining):
            return np.zeros_like(rates)
        durations = self.times[remaining] - horizon
        return np.sum(
            self.amounts[remaining][None, :] * np.exp(-rates[:, None] * durations[None, :]),
            axis=1,
        )

