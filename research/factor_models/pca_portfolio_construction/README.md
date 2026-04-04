# PCA-Based Factor Model for Portfolio Construction

## Objective

Use principal component analysis (PCA) on a universe of equity returns to:

1. Identify the dominant latent risk factors.
2. Select the assets with the highest exposure to those factors.
3. Construct a **global minimum variance (GMV)** portfolio over the selected subset.
4. Decompose portfolio risk into per-factor contributions.

---

## Methodology

| Step | Description |
|------|-------------|
| 1 | Download adjusted close prices (`yfinance`) |
| 2 | Compute simple daily returns; drop NaN rows |
| 3 | Center returns by subtracting the column mean |
| 4 | Estimate the sample covariance matrix Σ |
| 5 | Eigendecompose Σ; sort eigenvalues descending |
| 6 | Select minimum *k* PCs that explain ≥ 95 % of variance |
| 7 | Score each asset: *score_i = Σ_j λ_j · q²_{ij}* |
| 8 | Retain the top-5 assets by score |
| 9 | Compute GMV weights: *w\* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)* |
| 10 | Decompose variance: *Var(r_p) = Σ_j λ_j (q_jᵀ w)²* |

---

## Folder Layout

```
pca_portfolio_construction/
├── src/
│   └── pca_portfolio.py      ← single-entry pipeline
├── input/
│   └── raw/
│       └── prices.csv        ← auto-generated on first run (gitignored)
└── output/
    ├── data/
    │   ├── portfolio_weights.csv
    │   └── factor_risk_decomposition.csv
    ├── figures/               ← reserved for plots
    └── reports/
        └── summary.txt
```

---

## Quickstart

```bash
# from repo root
pip install -r requirements.txt
python research/factor_models/pca_portfolio_construction/src/pca_portfolio.py
```

All configuration (tickers, dates, variance threshold, number of assets) lives at
the top of `src/pca_portfolio.py` under the **Configuration** section — no CLI
flags needed for exploratory use.

---

## Key Parameters

| Variable | Default | Meaning |
|----------|---------|---------|
| `TICKERS` | 10 large-cap US equities | Investment universe |
| `START_DATE` / `END_DATE` | 2022-01-01 → 2024-01-01 | Back-test window |
| `VARIANCE_THRESHOLD` | 0.95 | Minimum cumulative explained variance for PC selection |
| `N_ASSETS_SELECTED` | 5 | Assets retained for portfolio construction |

---

## Outputs

| File | Description |
|------|-------------|
| `output/data/portfolio_weights.csv` | GMV weights per selected asset |
| `output/data/factor_risk_decomposition.csv` | Per-PC exposure and risk contribution |
| `output/reports/summary.txt` | Human-readable run summary |
| `input/raw/prices.csv` | Cached price data (first-run download) |

---

## Limitations & Next Steps

- **Sample covariance** is noisy for small T/N. Consider Ledoit-Wolf shrinkage.
- **GMV** ignores expected returns entirely. A mean-variance or Black-Litterman
  extension is a natural next step.
- **No transaction costs** or turnover constraints applied.
- Results are in-sample. Add a walk-forward or rolling-window back-test.
