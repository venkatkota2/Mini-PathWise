"""End-to-end ALM study orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .assets import project_constant_mix
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

    @property
    def mean_funding_ratio(self) -> float:
        return float(np.mean(self.funding_ratio))


def _tail_metrics(losses: FloatArray, confidence: float) -> tuple[float, float]:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    value_at_risk = float(np.quantile(losses, confidence))
    tail = losses[losses >= value_at_risk]
    return value_at_risk, float(np.mean(tail))


def _stress_ratios(
    assets: FloatArray,
    liabilities: FloatArray,
    *,
    equity_weight: float,
    bond_duration: float,
    spread_duration: float,
) -> dict[str, float]:
    base_assets = float(np.mean(assets))
    base_liabilities = float(np.mean(liabilities))
    return {
        "rates_up_100bp": base_assets * (1.0 - (1.0 - equity_weight) * bond_duration * 0.01)
        / (base_liabilities * 0.94),
        "rates_down_100bp": base_assets * (1.0 + (1.0 - equity_weight) * bond_duration * 0.01)
        / (base_liabilities * 1.07),
        "equity_down_30pct": base_assets * (1.0 - equity_weight * 0.30) / base_liabilities,
        "spreads_up_150bp": base_assets
        * (1.0 - (1.0 - equity_weight) * spread_duration * 0.015)
        / base_liabilities,
    }


def run_alm(
    liabilities: LiabilityCashflows,
    config: ALMConfig | None = None,
    assumptions: MarketAssumptions | None = None,
) -> ALMResult:
    """Run a stochastic ALM study and return scenario-level and summary risk results."""
    c = config or ALMConfig()
    if c.initial_assets <= 0 or c.scenarios <= 0 or c.risk_horizon_years <= 0:
        raise ValueError("invalid ALM configuration")

    paths = simulate_market(
        assumptions,
        scenarios=c.scenarios,
        years=c.risk_horizon_years,
        steps_per_year=c.steps_per_year,
        seed=c.seed,
    )
    asset_paths = project_constant_mix(
        paths,
        initial_assets=c.initial_assets,
        equity_weight=c.equity_weight,
        bond_duration=c.bond_duration,
        spread_duration=c.spread_duration,
    )
    assets = asset_paths[:, -1]
    discount_rates = paths.short_rate[:, -1] + c.liability_discount_spread
    liability_values = liabilities.value_at_horizon(c.risk_horizon_years, discount_rates)
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
            liability_values,
            equity_weight=c.equity_weight,
            bond_duration=c.bond_duration,
            spread_duration=c.spread_duration,
        ),
    )
