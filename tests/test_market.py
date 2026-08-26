import numpy as np
import pytest

from stochastic_alm.market import MarketAssumptions, simulate_market


def test_paths_are_reproducible_and_well_shaped():
    first = simulate_market(scenarios=64, years=2, steps_per_year=4, seed=11)
    second = simulate_market(scenarios=64, years=2, steps_per_year=4, seed=11)

    assert first.short_rate.shape == (64, 9)
    assert np.array_equal(first.equity_index, second.equity_index)
    assert np.all(first.equity_index > 0)
    assert np.all(first.credit_spread >= 0)


def test_invalid_correlation_is_rejected():
    with pytest.raises(ValueError):
        MarketAssumptions(correlation=np.ones((3, 3)))

