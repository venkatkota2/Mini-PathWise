from stochastic_alm import ALMConfig, LiabilityCashflows, run_alm

cashflows = LiabilityCashflows.level_annuity(
    annual_payment=7_500_000,
    years=20,
    growth=0.02,
)
result = run_alm(cashflows, ALMConfig(scenarios=25_000, seed=42))

print(f"Mean funding ratio: {result.mean_funding_ratio:.3f}")
print(f"Probability of deficit: {result.probability_of_deficit:.2%}")
print(f"99.5% deficit CVaR: ${result.deficit_cvar:,.0f}")
for name, ratio in result.stress_funding_ratios.items():
    print(f"{name}: {ratio:.3f}")
