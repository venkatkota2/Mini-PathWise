"""Asset portfolio projection."""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import NDArray

from .market import MarketPaths

FloatArray = NDArray[np.float64]


def project_constant_mix(
    paths: MarketPaths,
    *,
    initial_assets: float,
    equity_weight: float = 0.35,
    bond_duration: float = 6.0,
    spread_duration: float = 5.0,
    payments: FloatArray | None = None,
) -> FloatArray:
    """Project a rebalanced portfolio and deduct end-of-interval payments."""
    values, _ = project_constant_mix_with_shortfall(
        paths,
        initial_assets=initial_assets,
        equity_weight=equity_weight,
        bond_duration=bond_duration,
        spread_duration=spread_duration,
        payments=payments,
    )
    return values


def project_constant_mix_with_shortfall(
    paths: MarketPaths,
    *,
    initial_assets: float,
    equity_weight: float = 0.35,
    bond_duration: float = 6.0,
    spread_duration: float = 5.0,
    payments: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Project assets and retain cumulative nominal unpaid liability payments."""
    if not np.isfinite(initial_assets) or initial_assets <= 0:
        raise ValueError("initial_assets must be positive")
    if not isfinite(equity_weight) or not 0.0 <= equity_weight <= 1.0:
        raise ValueError("equity_weight must be between zero and one")
    if not isfinite(bond_duration) or not isfinite(spread_duration):
        raise ValueError("durations must be finite")
    if bond_duration < 0 or spread_duration < 0:
        raise ValueError("durations must be non-negative")

    values = np.empty_like(paths.short_rate)
    values[:, 0] = initial_assets
    bond_weight = 1.0 - equity_weight
    scheduled_payments = (
        np.zeros(paths.steps + 1, dtype=float)
        if payments is None
        else np.asarray(payments, dtype=float)
    )
    if scheduled_payments.shape != (paths.steps + 1,):
        raise ValueError("payments must contain one amount for each projection boundary")
    if not np.all(np.isfinite(scheduled_payments)) or np.any(scheduled_payments < 0):
        raise ValueError("payments must be finite and non-negative")
    if scheduled_payments[0] != 0:
        raise ValueError("payments at the initial valuation time are not supported")

    unpaid = np.zeros(paths.scenarios, dtype=float)

    for step in range(1, paths.steps + 1):
        equity_return = paths.equity_index[:, step] / paths.equity_index[:, step - 1] - 1.0
        rate_change = paths.short_rate[:, step] - paths.short_rate[:, step - 1]
        spread_change = paths.credit_spread[:, step] - paths.credit_spread[:, step - 1]
        carry = (paths.short_rate[:, step - 1] + paths.credit_spread[:, step - 1]) * paths.dt
        bond_return = carry - bond_duration * rate_change - spread_duration * spread_change
        portfolio_return = equity_weight * equity_return + bond_weight * bond_return
        assets_before_payment = np.maximum(0.0, values[:, step - 1] * (1.0 + portfolio_return))
        payment = scheduled_payments[step]
        paid = np.minimum(assets_before_payment, payment)
        unpaid += payment - paid
        values[:, step] = assets_before_payment - paid

    return values, unpaid
