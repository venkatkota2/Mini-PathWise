import numpy as np

from stochastic_alm import ALMConfig, LiabilityCashflows, run_alm


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

