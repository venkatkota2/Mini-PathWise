# stochastic-alm

A compact stochastic asset-liability modelling platform for insurance and pension risk work. It turns explicit economic assumptions into correlated market paths, projects an asset portfolio against liability cash flows, and reports funding-ratio and tail-risk outcomes.

This repository is the portfolio-ready continuation of **Mini PathWise**. It is an independent educational implementation and is not affiliated with Aon or PathWise.

## What it does

- Simulates correlated short rates, equities, inflation, and credit spreads.
- Projects a rebalanced equity/bond portfolio with rate and spread duration effects.
- Values remaining liability cash flows in every scenario.
- Reports funding-ratio distributions, probability of deficit, 99.5% deficit VaR, and CVaR.
- Runs transparent rate, equity, and spread stresses.
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
mean funding ratio       1.08
probability of deficit  16.69%
99.5% deficit VaR       $17.3m
99.5% deficit CVaR      $21.2m
```

## Model design

The economic scenario generator uses mean-reverting rate, inflation, and spread processes with geometric Brownian equity returns. Innovations are linked through a validated correlation matrix. The asset projection applies constant-mix rebalancing and a duration approximation to the bond sleeve. Liability cash flows are revalued at the selected risk horizon in each scenario.

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

This is a research and portfolio project, not production actuarial software. It does not yet include policyholder behaviour, nested stochastic valuation, capital aggregation, or IFRS 17 measurement. Those are deliberately left as future model layers rather than implied by the current outputs.
