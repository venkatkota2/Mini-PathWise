"""Command-line interface for an example ALM study."""

from __future__ import annotations

import argparse

from .engine import ALMConfig, run_alm
from .liabilities import LiabilityCashflows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a stochastic ALM study")
    parser.add_argument("--scenarios", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    liabilities = LiabilityCashflows.level_annuity(
        annual_payment=7_500_000,
        years=20,
        growth=0.02,
    )
    result = run_alm(liabilities, ALMConfig(scenarios=args.scenarios, seed=args.seed))
    print(f"scenarios              {args.scenarios:>10,d}")
    print(f"mean funding ratio     {result.mean_funding_ratio:>10.3f}")
    print(f"probability of deficit {result.probability_of_deficit:>9.2%}")
    print(f"99.5% deficit VaR      ${result.deficit_var / 1e6:>8.1f}m")
    print(f"99.5% deficit CVaR     ${result.deficit_cvar / 1e6:>8.1f}m")


if __name__ == "__main__":
    main()

