import numpy as np
import pytest

from stochastic_alm import ALMConfig, LiabilityCashflows, run_alm
from stochastic_alm.engine import _tail_metrics
from stochastic_alm.market import MarketAssumptions


def test_alm_result_has_consistent_tail_metrics():
    liabilities = LiabilityCashflows.level_annuity(
        annual_payment=7_500_000,
        years=20,
        growth=0.02,
    )
    result = run_alm(liabilities, ALMConfig(scenarios=1_000, seed=4))

    assert result.funding_ratio.shape == (1_000,)
    assert 0 <= result.probability_of_deficit <= 1
    assert result.deficit_cvar >= result.deficit_var >= 0
    assert set(result.stress_funding_ratios) == {
        "rates_up_100bp",
        "rates_down_100bp",
        "equity_down_30pct",
        "spreads_up_150bp",
    }
    assert np.all(result.deficit >= 0)


def test_more_assets_improve_the_funding_ratio():
    liabilities = LiabilityCashflows.level_annuity(annual_payment=5_000_000, years=15)
    lower = run_alm(liabilities, ALMConfig(initial_assets=60_000_000, scenarios=500, seed=8))
    higher = run_alm(liabilities, ALMConfig(initial_assets=90_000_000, scenarios=500, seed=8))
    assert higher.mean_funding_ratio > lower.mean_funding_ratio


def test_liability_payment_at_horizon_is_deducted_from_assets():
    liabilities = LiabilityCashflows(
        times=np.array([1.0, 2.0]),
        amounts=np.array([10.0, 10.0]),
    )
    deterministic = MarketAssumptions(
        initial_short_rate=0.0,
        long_run_short_rate=0.0,
        rate_mean_reversion=0.0,
        rate_volatility=0.0,
        equity_risk_premium=0.0,
        equity_dividend_yield=0.0,
        equity_volatility=0.0,
        initial_inflation_rate=0.0,
        long_run_inflation_rate=0.0,
        inflation_mean_reversion=0.0,
        inflation_volatility=0.0,
        initial_credit_spread=0.0,
        long_run_credit_spread=0.0,
        spread_mean_reversion=0.0,
        spread_volatility=0.0,
    )
    result = run_alm(
        liabilities,
        ALMConfig(
            initial_assets=100.0,
            equity_weight=0.0,
            bond_duration=0.0,
            spread_duration=0.0,
            risk_horizon_years=1.0,
            steps_per_year=1,
            scenarios=4,
            liability_discount_spread=0.0,
        ),
        deterministic,
    )

    assert np.all(result.assets_at_horizon == 90.0)
    assert np.all(result.liabilities_at_horizon == 10.0)
    assert result.liability_payments_through_horizon == 10.0
    assert np.all(result.funding_ratio == 9.0)


def test_liability_valuation_uses_one_continuous_compounding_convention():
    liabilities = LiabilityCashflows(
        times=np.array([1.0, 3.0]),
        amounts=np.array([20.0, 80.0]),
    )

    present = liabilities.present_value(0.037)
    at_zero = liabilities.value_at_horizon(0.0, np.array([0.037]))[0]

    assert at_zero == pytest.approx(present)


@pytest.mark.parametrize(
    ("losses", "confidence", "expected_var", "expected_es"),
    [
        ([0.0] * 10, 0.8, 0.0, 0.0),
        ([0.0] * 9 + [10.0], 0.8, 0.0, 5.0),
        ([0.0] * 8 + [10.0, 20.0], 0.8, 2.0, 15.0),
        ([0.0] * 7 + [10.0, 20.0, 30.0], 0.8, 12.0, 25.0),
        ([0.0, 1.0, 2.0, 3.0, 4.0], 0.6, 2.4, 3.5),
    ],
)
def test_empirical_expected_shortfall_handles_quantile_boundary_mass(
    losses, confidence, expected_var, expected_es
):
    value_at_risk, expected_shortfall = _tail_metrics(np.array(losses), confidence)

    assert value_at_risk == pytest.approx(expected_var)
    assert expected_shortfall == pytest.approx(expected_es)
    assert expected_shortfall >= value_at_risk


def test_rate_stress_revalues_remaining_cashflows():
    liabilities = LiabilityCashflows(
        times=np.array([2.0, 5.0]),
        amounts=np.array([50.0, 100.0]),
    )
    base = liabilities.value_at_horizon(1.0, np.array([0.03]))[0]
    rates_up = liabilities.value_at_horizon(1.0, np.array([0.04]))[0]
    rates_down = liabilities.value_at_horizon(1.0, np.array([0.02]))[0]

    assert rates_up < base < rates_down
