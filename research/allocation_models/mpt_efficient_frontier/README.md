# MPT Efficient Frontier (Live Data)

## Objective

Implement Modern Portfolio Theory workflow from `unit_4622.ipynb` with live
`yfinance` data, then extend it with explicit examples of where MPT can fail.

## What This Pipeline Covers

1. Annualized returns and annualized standard deviation.
2. Covariance-based portfolio risk formula.
3. Random long-only portfolio simulation.
4. Efficient frontier approximation from simulated portfolios.
5. Maximum Sharpe and minimum-volatility portfolio detection.
6. Limitation diagnostics for MPT assumptions:
   - estimation error (unstable optimal weights),
   - regime shift (out-of-sample underperformance),
   - tail risk (Gaussian VaR underestimation),
   - concentration risk (weight collapse into few assets).

## Run

```bash
pm/bin/python research/allocation_models/mpt_efficient_frontier/src/mpt_efficient_frontier.py
```

## Folder Layout

```
mpt_efficient_frontier/
├── src/
│   └── mpt_efficient_frontier.py
├── input/
│   └── raw/
│       └── prices.csv
└── output/
    ├── data/
    │   ├── prices_used.csv
    │   ├── annual_returns.csv
    │   ├── annual_std_dev.csv
    │   ├── annual_covariance_matrix.csv
    │   ├── pair_comparison_notebook_reference.csv
    │   ├── random_portfolios.csv
    │   ├── efficient_frontier.csv
    │   ├── optimal_weights.csv
    │   └── mpt_limitations.json
    ├── figures/
    │   └── efficient_frontier.png   (if matplotlib is available)
    └── reports/
        └── summary.txt
```

## Notes

- The optimization is long-only Monte Carlo search, matching the notebook style.
- Data quality and date alignment are enforced automatically.
- If `matplotlib` is missing, the figure is skipped but all tabular/report
  outputs are still generated.
