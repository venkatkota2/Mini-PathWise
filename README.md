# stochastic-alm

A compact stochastic asset-liability modelling platform for insurance and pension risk work. It turns explicit economic assumptions into correlated market paths, projects an asset portfolio against liability cash flows, and reports funding-ratio and tail-risk outcomes.

This repository is the portfolio-ready continuation of **Mini PathWise**. It is an independent educational implementation and is not affiliated with Aon or PathWise.

## What it does

- Simulates correlated short rates, equities, inflation, and credit spreads.
- Projects a rebalanced equity/bond portfolio, then explicitly pays liability cash flows due in each interval.
- Values remaining liability cash flows in every scenario.
- Reports funding-ratio distributions, probability of deficit, 99.5% deficit VaR, and fractional-tail empirical Expected Shortfall.
- Revalues remaining cash flows under rate stresses and applies documented duration approximations to assets.
- Produces reproducible results from a fixed random seed.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
stochastic-alm --scenarios 10000 --seed 7
pytest
```

Example output:

```text
scenarios               10,000
mean funding ratio       1.019
probability of deficit  42.76%
99.5% deficit VaR       $24.5m
99.5% deficit CVaR      $28.6m
```

## Model design

The economic scenario generator uses mean-reverting rate, inflation, and spread processes with geometric Brownian equity returns. Innovations are linked through a validated positive-semidefinite correlation matrix. The asset projection applies constant-mix rebalancing and a duration approximation to the bond sleeve. Market evolution occurs first in each interval; cash flows due at the interval end (including exactly at the risk horizon) are then paid from assets. Remaining nominal cash flows are valued with continuously compounded scenario discount rates.

Expected Shortfall averages exactly the worst `1 - confidence` empirical mass,
including a fractional boundary observation where necessary. This prevents a
point mass of zero deficits from turning ES into an unconditional mean.

```text
Assumptions → Economic scenarios → Asset returns → Liability payments
            → Horizon revaluation → Funding distribution → VaR / ES / stresses
```

The implementation keeps assumptions, simulation, valuation, and risk reporting separate so each layer can be tested or replaced independently.

## Repository layout

```text
src/stochastic_alm/market.py       economic scenario generator
src/stochastic_alm/assets.py       portfolio projection
src/stochastic_alm/liabilities.py  liability cash-flow model
src/stochastic_alm/engine.py       ALM orchestration and risk metrics
examples/run_study.py              end-to-end study
tests/                             deterministic unit tests
```

## Scope and limitations

This is an educational/research implementation, not production actuarial software. It uses simplified market dynamics and duration approximations for the bond sleeve. It is not affiliated with Aon or PathWise and is not a full nested-stochastic insurance valuation system. It does not yet include policyholder behaviour, capital aggregation, or IFRS 17 measurement; those are future model layers rather than implied features.
