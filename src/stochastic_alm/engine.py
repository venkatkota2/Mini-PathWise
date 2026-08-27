"""End-to-end ALM study orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from .assets import project_constant_mix_with_shortfall
from .liabilities import LiabilityCashflows
from .market import MarketAssumptions, simulate_market

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ALMConfig:
    initial_assets: float = 120_000_000.0
    equity_weight: float = 0.35
    bond_duration: float = 6.0
    spread_duration: float = 5.0
    risk_horizon_years: float = 1.0
    steps_per_year: int = 12
    scenarios: int = 10_000
    confidence: float = 0.995
    liability_discount_spread: float = 0.006
    seed: int = 7


@dataclass(frozen=True)
class ALMResult:
    funding_ratio: FloatArray
    deficit: FloatArray
    assets_at_horizon: FloatArray
    liabilities_at_horizon: FloatArray
    deficit_var: float
    deficit_cvar: float
    probability_of_deficit: float
    stress_funding_ratios: dict[str, float]
    liability_payments_through_horizon: float
    unpaid_liability_payments: FloatArray

    @property
    def mean_funding_ratio(self) -> float:
        return float(np.mean(self.funding_ratio))


def _tail_metrics(losses: FloatArray, confidence: float) -> tuple[float, float]:
    """Return linear-interpolated VaR and fractional-tail empirical ES.

    Expected Shortfall averages exactly the worst `(1-confidence)` empirical
    mass. If that mass ends within an observation, the boundary observation is
    included fractionally. This handles atoms at VaR, including zero deficits.
    """
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("losses must be a non-empty finite vector")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    value_at_risk = float(np.quantile(values, confidence))
    descending = np.sort(values)[::-1]
    tail_observations = (1.0 - confidence) * len(descending)
    whole = int(np.floor(tail_observations))
    fraction = tail_observations - whole
    tail_total = float(np.sum(descending[:whole]))
    if fraction > 0.0:
        tail_total += fraction * float(descending[whole])
    expected_shortfall = tail_total / tail_observations
    return value_at_risk, expected_shortfall


def _stress_ratios(
    assets: FloatArray,
    liabilities: LiabilityCashflows,
    discount_rates: FloatArray,
    *,
    horizon: float,
    equity_weight: float,
    bond_duration: float,
    spread_duration: float,
    unpaid_payments: FloatArray,
) -> dict[str, float]:
    base_assets = float(np.mean(assets))
    overdue = float(np.mean(unpaid_payments))
    base_liabilities = overdue + float(
        np.mean(liabilities.value_at_horizon(horizon, discount_rates))
    )
    rates_up_liabilities = (
        float(np.mean(liabilities.value_at_horizon(horizon, discount_rates + 0.01))) + overdue
    )
    rates_down_liabilities = (
        float(np.mean(liabilities.value_at_horizon(horizon, discount_rates - 0.01))) + overdue
    )

    def ratio(asset_value: float, liability_value: float) -> float:
        return asset_value / liability_value if liability_value > 0 else float("inf")

    return {
        "rates_up_100bp": ratio(
            base_assets * (1.0 - (1.0 - equity_weight) * bond_duration * 0.01),
            rates_up_liabilities,
        ),
        "rates_down_100bp": ratio(
            base_assets * (1.0 + (1.0 - equity_weight) * bond_duration * 0.01),
            rates_down_liabilities,
        ),
        "equity_down_30pct": ratio(base_assets * (1.0 - equity_weight * 0.30), base_liabilities),
        "spreads_up_150bp": ratio(
            base_assets * (1.0 - (1.0 - equity_weight) * spread_duration * 0.015),
            base_liabilities,
        ),
    }


def run_alm(
    liabilities: LiabilityCashflows,
    config: ALMConfig | None = None,
    assumptions: MarketAssumptions | None = None,
) -> ALMResult:
    """Run a stochastic ALM study and return scenario-level and summary risk results."""
    c = config or ALMConfig()
    numeric_config = (
        c.initial_assets,
        c.equity_weight,
        c.bond_duration,
        c.spread_duration,
        c.risk_horizon_years,
        c.confidence,
        c.liability_discount_spread,
    )
    if not all(isfinite(value) for value in numeric_config):
        raise ValueError("ALM configuration must be finite")
    if (
        c.initial_assets <= 0
        or c.risk_horizon_years <= 0
        or not 0.0 <= c.equity_weight <= 1.0
        or c.bond_duration < 0
        or c.spread_duration < 0
        or not 0.5 < c.confidence < 1.0
    ):
        raise ValueError("invalid ALM configuration")
    if (
        not isinstance(c.scenarios, int)
        or isinstance(c.scenarios, bool)
        or c.scenarios <= 0
        or not isinstance(c.steps_per_year, int)
        or isinstance(c.steps_per_year, bool)
        or c.steps_per_year <= 0
        or not isinstance(c.seed, int)
        or isinstance(c.seed, bool)
        or c.seed < 0
    ):
        raise ValueError("scenario counts, steps, and seed must be valid integers")

    paths = simulate_market(
        assumptions,
        scenarios=c.scenarios,
        years=c.risk_horizon_years,
        steps_per_year=c.steps_per_year,
        seed=c.seed,
    )
    payments = liabilities.payments_by_step(c.risk_horizon_years, paths.steps)
    asset_paths, unpaid_payments = project_constant_mix_with_shortfall(
        paths,
        initial_assets=c.initial_assets,
        equity_weight=c.equity_weight,
        bond_duration=c.bond_duration,
        spread_duration=c.spread_duration,
        payments=payments,
    )
    assets = asset_paths[:, -1]
    discount_rates = paths.short_rate[:, -1] + c.liability_discount_spread
    remaining_liability_values = liabilities.value_at_horizon(c.risk_horizon_years, discount_rates)
    liability_values = remaining_liability_values + unpaid_payments
    funding_ratio = np.divide(
        assets,
        liability_values,
        out=np.full_like(assets, np.inf),
        where=liability_values > 0,
    )
    deficit = np.maximum(liability_values - assets, 0.0)
    deficit_var, deficit_cvar = _tail_metrics(deficit, c.confidence)

    return ALMResult(
        funding_ratio=funding_ratio,
        deficit=deficit,
        assets_at_horizon=assets,
        liabilities_at_horizon=liability_values,
        deficit_var=deficit_var,
        deficit_cvar=deficit_cvar,
        probability_of_deficit=float(np.mean(deficit > 0)),
        stress_funding_ratios=_stress_ratios(
            assets,
            liabilities,
            discount_rates,
            horizon=c.risk_horizon_years,
            equity_weight=c.equity_weight,
            bond_duration=c.bond_duration,
            spread_duration=c.spread_duration,
            unpaid_payments=unpaid_payments,
        ),
        liability_payments_through_horizon=float(np.sum(payments)),
        unpaid_liability_payments=unpaid_payments,
    )
