# Quantitative Portfolio Management

A systematic research repository for building and analysing portfolios using
rigorous quantitative methods — covariance modelling, PCA, optimisation, and
risk decomposition.

---

## Overview

This repo explores portfolio construction through a quantitative lens. Each
module is self-contained with separate layers for **source code**, **input
data**, and **output results**. The work is designed to be reproducible,
readable, and extensible for institutional-grade research.

**Core focus areas:**
- Covariance estimation and asset relationship modelling
- Principal Component Analysis for dimensionality reduction and factor discovery
- Mean-variance and minimum-variance portfolio optimisation
- Risk decomposition into interpretable factor contributions

---

## Key Concepts

| Concept | Role |
|---------|------|
| **Covariance matrix** | Captures pairwise asset return relationships |
| **Eigen decomposition** | Reveals the latent risk structure of the portfolio |
| **PCA** | Reduces dimensionality; identifies dominant risk factors |
| **Optimisation** | Allocates weights to minimise risk (or maximise Sharpe) |
| **Risk decomposition** | Attributes portfolio variance to individual factors |

---

## Repository Structure

```
Portfoliomanagement/
│
├── docs/                    ← methodology references and concept deep-dives
│   ├── basics.md            ← covariance, variance, eigen decomposition
│   ├── pca.md               ← PCA theory and finance applications
│   ├── optimization.md      ← GMV and mean-variance derivations
│   └── risk.md              ← risk decomposition framework
│
├── research/                ← quantitative strategies (each self-contained)
│   └── factor_models/
│       └── pca_portfolio_construction/
│           ├── src/         ← runnable Python pipeline
│           ├── input/       ← raw market data (gitignored)
│           └── output/      ← weights, risk reports, figures (gitignored)
│
├── notebooks/               ← exploratory Jupyter notebooks
├── examples/                ← minimal, standalone usage examples
├── utils/                   ← shared helpers (data loaders, plotters)
└── requirements.txt
```

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the PCA portfolio pipeline
python research/factor_models/pca_portfolio_construction/src/pca_portfolio.py
```

Results are written to `research/factor_models/pca_portfolio_construction/output/`.

---

## Documentation

- [Basics — Covariance, Variance & Eigen Decomposition](docs/basics.md)
- [PCA for Portfolio Construction](docs/pca.md)
- [Portfolio Optimisation](docs/optimization.md)
- [Risk Decomposition](docs/risk.md)

---

## Strategies

| # | Strategy | Path | Status |
|---|----------|------|--------|
| 1 | PCA Factor Model + GMV Portfolio | [`research/factor_models/pca_portfolio_construction`](research/factor_models/pca_portfolio_construction/README.md) | Active |

---

## Roadmap

- [ ] Factor models (Fama-French, BARRA-style)
- [ ] Risk parity and equal risk contribution
- [ ] Statistical arbitrage (pairs / cointegration)
- [ ] Walk-forward back-testing framework
- [ ] Ledoit-Wolf shrinkage for covariance estimation
