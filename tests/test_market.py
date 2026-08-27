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

    changed_seed = simulate_market(scenarios=64, years=2, steps_per_year=4, seed=12)
    assert not np.array_equal(first.equity_index, changed_seed.equity_index)


def test_invalid_correlation_is_rejected():
    with pytest.raises(ValueError):
        MarketAssumptions(correlation=np.ones((3, 3)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"equity_volatility": float("nan")},
        {"rate_volatility": -0.01},
        {"initial_equity_index": 0.0},
        {"initial_credit_spread": -0.01},
    ],
)
def test_invalid_scalar_market_assumptions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MarketAssumptions(**kwargs)


def test_positive_semidefinite_correlation_is_supported():
    perfectly_correlated = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    assumptions = MarketAssumptions(correlation=perfectly_correlated)

    paths = simulate_market(assumptions, scenarios=32, seed=7)

    assert paths.short_rate.shape == (32, 13)


@pytest.mark.parametrize(
    "correlation",
    [
        np.array(
            [
                [1.0, 0.2, 0.0, 0.0],
                [0.1, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        np.diag([1.0, 1.0, 1.0, 0.9]),
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.2],
                [0.0, 0.0, 1.2, 1.0],
            ]
        ),
    ],
)
def test_structurally_invalid_correlations_are_rejected(correlation):
    with pytest.raises(ValueError):
        MarketAssumptions(correlation=correlation)
