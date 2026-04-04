"""
PCA-Based Factor Model for Portfolio Construction
==================================================
Methodology:
    1. Download adjusted close prices for a universe of equities.
    2. Compute daily log/simple returns and center them.
    3. Perform eigendecomposition of the sample covariance matrix.
    4. Select the minimum number of principal components that explain
       a target fraction of total variance.
    5. Score each asset by its cumulative factor loading; retain the
       top-k assets.
    6. Construct the minimum-variance portfolio over the selected assets
       using the closed-form global minimum variance (GMV) weights.
    7. Decompose portfolio risk into per-factor contributions.

References:
    - Ledoit & Wolf (2004): "A well-conditioned estimator for
      large-dimensional covariance matrices."
    - Jolliffe (2002): "Principal Component Analysis", 2nd ed.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration  —  edit here, nowhere else
# ---------------------------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "TSLA", "NVDA", "JPM",  "XOM",  "UNH",
]
START_DATE          = "2022-01-01"
END_DATE            = "2024-01-01"
VARIANCE_THRESHOLD  = 0.95   # cumulative explained-variance target for factor selection
N_ASSETS_SELECTED   = 5      # number of assets to retain for portfolio construction

# Paths  (relative to this file's directory → keeps src/ self-contained)
_HERE        = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR    = os.path.join(_HERE, "..", "input",  "raw")
OUTPUT_DATA  = os.path.join(_HERE, "..", "output", "data")
OUTPUT_FIGS  = os.path.join(_HERE, "..", "output", "figures")
OUTPUT_REP   = os.path.join(_HERE, "..", "output", "reports")


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices and persist a local CSV cache."""
    cache_path = os.path.join(INPUT_DIR, "prices.csv")
    if os.path.exists(cache_path):
        print(f"[data] Loading cached prices from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"[data] Downloading prices for {tickers} ({start} → {end})")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = raw["Close"]
    os.makedirs(INPUT_DIR, exist_ok=True)
    prices.to_csv(cache_path)
    print(f"[data] Saved to {cache_path}")
    return prices


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, NaNs dropped."""
    return prices.pct_change().dropna()


def center_returns(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, mu) where X = returns - mu (column-wise)."""
    mu = returns.mean()
    X  = returns - mu
    return X, mu


# ---------------------------------------------------------------------------
# Factor model
# ---------------------------------------------------------------------------

def eigen_decompose(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the sample covariance and return (eigvals, eigvecs) sorted
    by descending eigenvalue.
    """
    Sigma             = np.cov(X.T)
    eigvals, eigvecs  = np.linalg.eigh(Sigma)
    idx               = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx], Sigma


def select_factors(eigvals: np.ndarray,
                   threshold: float = VARIANCE_THRESHOLD) -> int:
    """Minimum k such that the top-k components explain >= threshold variance."""
    explained     = eigvals / eigvals.sum()
    cum_explained = np.cumsum(explained)
    k             = int(np.argmax(cum_explained >= threshold)) + 1
    print(f"[pca] {k} factor(s) explain {cum_explained[k-1]:.2%} of variance "
          f"(threshold={threshold:.0%})")
    return k


def score_assets(eigvecs: np.ndarray,
                 eigvals: np.ndarray,
                 k: int) -> np.ndarray:
    """
    Asset importance score = sum of squared loadings weighted by eigenvalue.
    Higher score → asset is more exposed to the dominant risk factors.
    """
    Qk     = eigvecs[:, :k]
    Lk     = eigvals[:k]
    scores = (Qk ** 2) @ Lk
    return scores


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def global_min_variance(Sigma_sel: np.ndarray) -> np.ndarray:
    """
    Closed-form global minimum variance (GMV) weights.
    w* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)
    """
    ones    = np.ones(Sigma_sel.shape[0])
    inv_S   = np.linalg.inv(Sigma_sel)
    w       = inv_S @ ones
    w      /= ones @ inv_S @ ones
    return w


def factor_risk_decomposition(w: np.ndarray,
                               Q_sel: np.ndarray,
                               eigvals: np.ndarray,
                               k: int) -> np.ndarray:
    """
    Decompose portfolio variance into per-factor contributions.
    Var(r_p) = Σ_j λ_j (q_jᵀ w)²
    """
    z            = Q_sel.T @ w          # factor exposures
    risk_contrib = eigvals[:k] * (z ** 2)
    return z, risk_contrib


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(selected_assets: pd.Index,
                 weights: np.ndarray,
                 portfolio_var: float,
                 factor_exposure: np.ndarray,
                 risk_contrib: np.ndarray) -> pd.DataFrame:
    """Assemble a tidy summary DataFrame."""
    weight_df = pd.DataFrame({
        "asset":   selected_assets,
        "weight":  weights,
    }).set_index("asset")

    factor_df = pd.DataFrame({
        "factor":           [f"PC{i+1}" for i in range(len(factor_exposure))],
        "exposure_z":       factor_exposure,
        "risk_contribution": risk_contrib,
        "pct_risk":         risk_contrib / risk_contrib.sum(),
    }).set_index("factor")

    return weight_df, factor_df, portfolio_var


def save_outputs(weight_df: pd.DataFrame,
                 factor_df: pd.DataFrame,
                 portfolio_var: float) -> None:
    """Persist results to output/data/ and output/reports/."""
    os.makedirs(OUTPUT_DATA, exist_ok=True)
    os.makedirs(OUTPUT_REP,  exist_ok=True)

    weight_path = os.path.join(OUTPUT_DATA, "portfolio_weights.csv")
    factor_path = os.path.join(OUTPUT_DATA, "factor_risk_decomposition.csv")
    weight_df.to_csv(weight_path)
    factor_df.to_csv(factor_path)

    summary_path = os.path.join(OUTPUT_REP, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== PCA Portfolio Construction — Summary ===\n\n")
        f.write(f"Portfolio variance : {portfolio_var:.6f}\n")
        f.write(f"Portfolio vol (ann) : {(portfolio_var * 252) ** 0.5:.2%}\n\n")
        f.write("--- Weights ---\n")
        f.write(weight_df.to_string())
        f.write("\n\n--- Factor Risk Decomposition ---\n")
        f.write(factor_df.to_string())
        f.write("\n")

    print(f"[output] Weights        → {weight_path}")
    print(f"[output] Factor decomp  → {factor_path}")
    print(f"[output] Summary report → {summary_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    # 1. Data
    prices  = download_prices(TICKERS, START_DATE, END_DATE)

    # 2. Returns & centering
    returns = compute_returns(prices)
    X, mu   = center_returns(returns)

    # 3. Eigendecomposition
    eigvals, eigvecs, Sigma = eigen_decompose(X)

    # 4. Factor selection
    k = select_factors(eigvals, threshold=VARIANCE_THRESHOLD)

    # 5. Asset scoring & selection
    scores          = score_assets(eigvecs, eigvals, k)
    selection_idx   = np.argsort(scores)[-N_ASSETS_SELECTED:]
    selected_assets = returns.columns[selection_idx]
    print(f"[selection] Top {N_ASSETS_SELECTED} assets: {list(selected_assets)}")

    # 6. Subset covariance & GMV weights
    Sigma_sel = Sigma[np.ix_(selection_idx, selection_idx)]
    weights   = global_min_variance(Sigma_sel)

    # 7. Portfolio variance
    portfolio_var = float(weights.T @ Sigma_sel @ weights)
    print(f"[portfolio] Variance={portfolio_var:.6f}  "
          f"Vol(ann)={(portfolio_var*252)**0.5:.2%}")

    # 8. Factor risk decomposition
    Q_sel               = eigvecs[selection_idx, :k]
    z, risk_contrib     = factor_risk_decomposition(weights, Q_sel, eigvals, k)

    # 9. Reporting
    weight_df, factor_df, _ = build_report(
        selected_assets, weights, portfolio_var, z, risk_contrib
    )

    print("\n=== Portfolio Weights ===")
    print(weight_df.to_string())
    print("\n=== Factor Risk Decomposition ===")
    print(factor_df.to_string())

    # 10. Persist
    save_outputs(weight_df, factor_df, portfolio_var)


if __name__ == "__main__":
    run_pipeline()
