"""Asset portfolio projection."""

from __future__ import annotations

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
    if not np.isfinite(initial_assets) or initial_assets <= 0:
        raise ValueError("initial_assets must be positive")
    if not 0.0 <= equity_weight <= 1.0:
        raise ValueError("equity_weight must be between zero and one")
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

    for step in range(1, paths.steps + 1):
        equity_return = paths.equity_index[:, step] / paths.equity_index[:, step - 1] - 1.0
        rate_change = paths.short_rate[:, step] - paths.short_rate[:, step - 1]
        spread_change = paths.credit_spread[:, step] - paths.credit_spread[:, step - 1]
        carry = (paths.short_rate[:, step - 1] + paths.credit_spread[:, step - 1]) * paths.dt
        bond_return = carry - bond_duration * rate_change - spread_duration * spread_change
        portfolio_return = equity_weight * equity_return + bond_weight * bond_return
        assets_before_payment = values[:, step - 1] * (1.0 + portfolio_return)
        values[:, step] = np.maximum(0.0, assets_before_payment - scheduled_payments[step])

    return values
