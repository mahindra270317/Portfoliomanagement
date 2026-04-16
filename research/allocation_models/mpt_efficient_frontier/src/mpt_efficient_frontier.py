"""
Modern Portfolio Theory (MPT) Efficient Frontier Pipeline
=========================================================

This module extends the concepts shown in `unit_4622.ipynb` using live
`yfinance` data:

1. Annualized returns and annualized volatility.
2. Covariance-based portfolio risk.
3. Random long-only portfolios and efficient frontier estimation.
4. Max Sharpe and minimum-volatility portfolio selection.
5. Explicit MPT limitation checks with code-driven examples.

Run:
    pm/bin/python research/allocation_models/mpt_efficient_frontier/src/mpt_efficient_frontier.py
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKERS = [
    "CVX", "MSFT", "GOOGL", "AAPL", "NVDA",
    "JPM", "XOM", "UNH", "PG", "JNJ",
]
START_DATE = "2016-01-01"
END_DATE = date.today().isoformat()
RISK_FREE_RATE = 0.02
N_RANDOM_PORTFOLIOS = 15000
RANDOM_SEED = 42

PAIR_HIGHER_COV = ("MSFT", "GOOGL")
PAIR_LOWER_COV = ("CVX", "MSFT")
EQUAL_WEIGHT = 0.5

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(_HERE, "..")
INPUT_DIR = os.path.join(PROJECT_DIR, "input", "raw")
OUTPUT_DATA = os.path.join(PROJECT_DIR, "output", "data")
OUTPUT_FIGS = os.path.join(PROJECT_DIR, "output", "figures")
OUTPUT_REPORTS = os.path.join(PROJECT_DIR, "output", "reports")


@dataclass
class PortfolioPoint:
    expected_return: float
    volatility: float
    sharpe: float
    weights: np.ndarray


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices and cache them."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    cache_path = os.path.join(INPUT_DIR, "prices.csv")

    print(f"[data] Downloading prices ({start} -> {end}) for {len(tickers)} tickers")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        if os.path.exists(cache_path):
            print("[data] Download failed; loading cached data")
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return cached
        raise RuntimeError("No price data downloaded and no cache available.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"]
        else:
            prices = raw.xs(raw.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        prices = raw.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.reindex(columns=tickers)
    available_cols = [c for c in prices.columns if not prices[c].dropna().empty]
    dropped_cols = [c for c in prices.columns if c not in available_cols]

    if dropped_cols:
        print(f"[data] Dropping tickers with no data: {dropped_cols}")

    prices = prices[available_cols].sort_index().ffill().dropna(how="any")

    if prices.shape[1] < 2:
        raise RuntimeError("Need at least two assets with valid price history.")

    prices.to_csv(cache_path)
    print(f"[data] Saved cleaned prices to {cache_path}")
    return prices


def annual_metrics_from_prices(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Compute daily returns, annualized returns, annualized vol, annualized covariance."""
    daily_returns = prices.pct_change().dropna()
    annual_returns = (prices.iloc[-1] / prices.iloc[0]) ** (252.0 / len(prices)) - 1.0
    annual_std = daily_returns.std() * math.sqrt(252.0)
    annual_cov = daily_returns.cov() * 252.0
    return daily_returns, annual_returns, annual_std, annual_cov


def annual_metrics_from_daily_returns(
    daily_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute annualized returns and covariance directly from daily returns."""
    annual_returns = (1.0 + daily_returns).prod() ** (252.0 / len(daily_returns)) - 1.0
    annual_cov = daily_returns.cov() * 252.0
    return annual_returns, annual_cov


def pair_portfolio_metrics(
    ticker_a: str,
    ticker_b: str,
    annual_returns: pd.Series,
    annual_std: pd.Series,
    annual_cov: pd.DataFrame,
    weight_a: float = EQUAL_WEIGHT,
) -> dict[str, float]:
    """Return portfolio metrics for a 2-asset portfolio using notebook formulas."""
    weight_b = 1.0 - weight_a

    portfolio_return = float(
        weight_a * annual_returns[ticker_a] + weight_b * annual_returns[ticker_b]
    )
    covariance_ab = float(annual_cov.loc[ticker_a, ticker_b])

    portfolio_std = math.sqrt(
        (weight_a ** 2) * (annual_std[ticker_a] ** 2)
        + (weight_b ** 2) * (annual_std[ticker_b] ** 2)
        + 2.0 * weight_a * weight_b * covariance_ab
    )

    return {
        "portfolio_return": portfolio_return,
        "portfolio_std_dev": float(portfolio_std),
        "return_to_risk": float(portfolio_return / portfolio_std),
        "covariance": covariance_ab,
        "weight_a": weight_a,
        "weight_b": weight_b,
    }


def simulate_random_portfolios(
    annual_returns: pd.Series,
    annual_cov: pd.DataFrame,
    n_portfolios: int,
    risk_free_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Generate long-only random portfolios with sum of weights = 1."""
    tickers = list(annual_returns.index)
    n_assets = len(tickers)

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    mu = annual_returns.to_numpy(dtype=float)
    sigma = annual_cov.to_numpy(dtype=float)

    portfolio_returns = weights @ mu
    portfolio_variances = np.einsum("ij,jk,ik->i", weights, sigma, weights)
    portfolio_volatility = np.sqrt(np.clip(portfolio_variances, a_min=0.0, a_max=None))

    return_to_risk = np.divide(
        portfolio_returns,
        portfolio_volatility,
        out=np.full_like(portfolio_returns, np.nan),
        where=portfolio_volatility > 0,
    )
    sharpe = np.divide(
        portfolio_returns - risk_free_rate,
        portfolio_volatility,
        out=np.full_like(portfolio_returns, np.nan),
        where=portfolio_volatility > 0,
    )

    simulations = pd.DataFrame(weights, columns=[f"w_{t}" for t in tickers])
    simulations["returns"] = portfolio_returns
    simulations["std_dev"] = portfolio_volatility
    simulations["returns/std_dev"] = return_to_risk
    simulations["sharpe"] = sharpe

    return simulations


def efficient_frontier_from_simulation(
    simulations: pd.DataFrame,
    n_bins: int = 120,
) -> pd.DataFrame:
    """Approximate efficient frontier as max return in each volatility bucket."""
    if simulations.empty:
        return pd.DataFrame(columns=["std_dev", "returns", "sharpe"])

    min_std = float(simulations["std_dev"].min())
    max_std = float(simulations["std_dev"].max())

    bins = np.linspace(min_std, max_std, n_bins + 1)
    rows: list[dict[str, float]] = []

    for left, right in zip(bins[:-1], bins[1:]):
        bucket = simulations[
            (simulations["std_dev"] >= left)
            & (simulations["std_dev"] < right)
        ]
        if bucket.empty:
            continue

        best = bucket.loc[bucket["returns"].idxmax()]
        rows.append(
            {
                "std_dev": float(best["std_dev"]),
                "returns": float(best["returns"]),
                "sharpe": float(best["sharpe"]),
            }
        )

    frontier = pd.DataFrame(rows).drop_duplicates(subset=["std_dev"])
    if frontier.empty:
        return pd.DataFrame(columns=["std_dev", "returns", "sharpe"])

    frontier = frontier.sort_values("std_dev").reset_index(drop=True)
    return frontier


def extract_portfolio_point(row: pd.Series, tickers: list[str]) -> PortfolioPoint:
    """Build a typed portfolio point from a simulation row."""
    weights = row[[f"w_{t}" for t in tickers]].to_numpy(dtype=float)
    return PortfolioPoint(
        expected_return=float(row["returns"]),
        volatility=float(row["std_dev"]),
        sharpe=float(row["sharpe"]),
        weights=weights,
    )


def annualized_return_and_vol(daily_series: pd.Series) -> tuple[float, float]:
    """Compute annualized geometric return and annualized volatility."""
    if daily_series.empty:
        return float("nan"), float("nan")

    annual_return = float((1.0 + daily_series).prod() ** (252.0 / len(daily_series)) - 1.0)
    annual_vol = float(daily_series.std() * math.sqrt(252.0))
    return annual_return, annual_vol


def optimize_max_sharpe_from_daily_returns(
    daily_returns: pd.DataFrame,
    risk_free_rate: float,
    n_portfolios: int,
    seed: int,
) -> PortfolioPoint:
    """Estimate max-Sharpe weights from a return window using random search."""
    annual_returns, annual_cov = annual_metrics_from_daily_returns(daily_returns)
    simulations = simulate_random_portfolios(
        annual_returns=annual_returns,
        annual_cov=annual_cov,
        n_portfolios=n_portfolios,
        risk_free_rate=risk_free_rate,
        seed=seed,
    )
    best = simulations.loc[simulations["sharpe"].idxmax()]
    return extract_portfolio_point(best, list(annual_returns.index))


def run_mpt_limitations(
    daily_returns: pd.DataFrame,
    tickers: list[str],
    max_sharpe_weights: np.ndarray,
    risk_free_rate: float,
) -> dict[str, dict[str, float | bool]]:
    """Run explicit examples where core MPT assumptions break."""
    results: dict[str, dict[str, float | bool]] = {}
    subset = daily_returns[tickers]

    # 1) Estimation error: nearby samples can produce very different weights.
    split = len(subset) // 2
    window_1 = subset.iloc[:split]
    window_2 = subset.iloc[split:]

    best_w1 = optimize_max_sharpe_from_daily_returns(
        window_1, risk_free_rate, n_portfolios=7000, seed=101
    )
    best_w2 = optimize_max_sharpe_from_daily_returns(
        window_2, risk_free_rate, n_portfolios=7000, seed=202
    )

    turnover = 0.5 * float(np.abs(best_w1.weights - best_w2.weights).sum())
    results["estimation_error"] = {
        "weight_turnover_between_windows": turnover,
        "window_1_sharpe": float(best_w1.sharpe),
        "window_2_sharpe": float(best_w2.sharpe),
        "failure_flag": bool(turnover > 0.35),
    }

    # 2) Non-stationarity: optimize in one regime, then evaluate out-of-sample.
    train_end = int(len(subset) * 0.65)
    train = subset.iloc[:train_end]
    test = subset.iloc[train_end:]

    trained = optimize_max_sharpe_from_daily_returns(
        train, risk_free_rate, n_portfolios=8000, seed=303
    )
    test_mpt = pd.Series(test.to_numpy() @ trained.weights, index=test.index)
    test_equal_weight = test.mean(axis=1)

    mpt_ret, mpt_vol = annualized_return_and_vol(test_mpt)
    eq_ret, eq_vol = annualized_return_and_vol(test_equal_weight)

    mpt_sharpe = float((mpt_ret - risk_free_rate) / mpt_vol) if mpt_vol > 0 else float("nan")
    eq_sharpe = float((eq_ret - risk_free_rate) / eq_vol) if eq_vol > 0 else float("nan")

    results["regime_shift"] = {
        "out_of_sample_mpt_sharpe": mpt_sharpe,
        "out_of_sample_equal_weight_sharpe": eq_sharpe,
        "out_of_sample_mpt_return": float(mpt_ret),
        "out_of_sample_equal_weight_return": float(eq_ret),
        "failure_flag": bool(mpt_sharpe < eq_sharpe),
    }

    # 3) Normality assumption: Gaussian VaR can understate tail losses.
    portfolio_daily = pd.Series(
        subset.to_numpy() @ max_sharpe_weights,
        index=subset.index,
        name="max_sharpe_portfolio",
    )
    mu_d = float(portfolio_daily.mean())
    sigma_d = float(portfolio_daily.std())
    z_5pct = -1.6448536269514729

    gaussian_var_95 = float(-(mu_d + z_5pct * sigma_d))
    q_5pct = float(portfolio_daily.quantile(0.05))
    historical_var_95 = float(-q_5pct)

    tail_slice = portfolio_daily[portfolio_daily <= q_5pct]
    historical_cvar_95 = float(-tail_slice.mean()) if not tail_slice.empty else float("nan")

    wealth = (1.0 + portfolio_daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    excess_kurtosis = float(portfolio_daily.kurt())

    results["tail_risk"] = {
        "gaussian_var_95_daily": gaussian_var_95,
        "historical_var_95_daily": historical_var_95,
        "historical_cvar_95_daily": historical_cvar_95,
        "excess_kurtosis": excess_kurtosis,
        "max_drawdown": max_drawdown,
        "failure_flag": bool(historical_var_95 > gaussian_var_95 * 1.2),
    }

    # 4) Concentration risk: max-Sharpe can collapse into a few names.
    hhi = float(np.sum(max_sharpe_weights ** 2))
    effective_assets = float(1.0 / hhi) if hhi > 0 else float("nan")
    max_weight = float(np.max(max_sharpe_weights))

    results["concentration"] = {
        "max_weight": max_weight,
        "effective_number_of_assets": effective_assets,
        "hhi": hhi,
        "failure_flag": bool(max_weight > 0.35 or effective_assets < len(tickers) / 2.0),
    }

    return results


def save_frontier_plot(
    simulations: pd.DataFrame,
    frontier: pd.DataFrame,
    max_sharpe: PortfolioPoint,
    min_vol: PortfolioPoint,
) -> str | None:
    """Save an efficient frontier chart if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed, skipping figure generation")
        return None

    os.makedirs(OUTPUT_FIGS, exist_ok=True)
    path = os.path.join(OUTPUT_FIGS, "efficient_frontier.png")

    plt.figure(figsize=(11, 7))
    plt.scatter(
        simulations["std_dev"],
        simulations["returns"],
        c=simulations["sharpe"],
        cmap="viridis",
        alpha=0.25,
        s=8,
        label="Random portfolios",
    )

    if not frontier.empty:
        plt.plot(
            frontier["std_dev"],
            frontier["returns"],
            color="black",
            linewidth=2.0,
            label="Approx. efficient frontier",
        )

    plt.scatter(
        [max_sharpe.volatility],
        [max_sharpe.expected_return],
        marker="*",
        s=280,
        color="red",
        label="Max Sharpe",
    )
    plt.scatter(
        [min_vol.volatility],
        [min_vol.expected_return],
        marker="*",
        s=280,
        color="darkorange",
        label="Min Volatility",
    )

    plt.title("MPT Efficient Frontier (Long-only Monte Carlo)")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

    print(f"[plot] Saved efficient frontier plot to {path}")
    return path


def save_outputs(
    prices: pd.DataFrame,
    annual_returns: pd.Series,
    annual_std: pd.Series,
    annual_cov: pd.DataFrame,
    pair_comparison: pd.DataFrame,
    simulations: pd.DataFrame,
    frontier: pd.DataFrame,
    max_sharpe: PortfolioPoint,
    min_vol: PortfolioPoint,
    limitations: dict[str, dict[str, float | bool]],
    tickers: list[str],
) -> None:
    """Persist all major artifacts for reproducibility."""
    os.makedirs(OUTPUT_DATA, exist_ok=True)
    os.makedirs(OUTPUT_REPORTS, exist_ok=True)

    prices.to_csv(os.path.join(OUTPUT_DATA, "prices_used.csv"))
    annual_returns.to_frame("annual_return").to_csv(
        os.path.join(OUTPUT_DATA, "annual_returns.csv")
    )
    annual_std.to_frame("annual_std_dev").to_csv(
        os.path.join(OUTPUT_DATA, "annual_std_dev.csv")
    )
    annual_cov.to_csv(os.path.join(OUTPUT_DATA, "annual_covariance_matrix.csv"))
    pair_comparison.to_csv(os.path.join(OUTPUT_DATA, "pair_comparison_notebook_reference.csv"), index=False)
    simulations.to_csv(os.path.join(OUTPUT_DATA, "random_portfolios.csv"), index=False)
    frontier.to_csv(os.path.join(OUTPUT_DATA, "efficient_frontier.csv"), index=False)

    weights_table = pd.DataFrame(
        {
            "ticker": tickers,
            "max_sharpe_weight": max_sharpe.weights,
            "min_vol_weight": min_vol.weights,
        }
    )
    weights_table.to_csv(os.path.join(OUTPUT_DATA, "optimal_weights.csv"), index=False)

    limitations_path = os.path.join(OUTPUT_DATA, "mpt_limitations.json")
    with open(limitations_path, "w", encoding="utf-8") as f:
        json.dump(limitations, f, indent=2)

    summary_path = os.path.join(OUTPUT_REPORTS, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== MPT Efficient Frontier Summary ===\n\n")
        f.write(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}\n")
        f.write(f"Assets used ({len(tickers)}): {tickers}\n\n")

        f.write("--- Notebook-Reference Pair Analysis ---\n")
        for _, row in pair_comparison.iterrows():
            f.write(
                f"{row['pair']}: cov={row['covariance']:.6f}, "
                f"ret={row['portfolio_return']:.2%}, vol={row['portfolio_std_dev']:.2%}, "
                f"ret/vol={row['return_to_risk']:.3f}\n"
            )
        f.write("\n")

        f.write("--- Optimized Portfolios (Long-only) ---\n")
        f.write(
            f"Max Sharpe: return={max_sharpe.expected_return:.2%}, "
            f"vol={max_sharpe.volatility:.2%}, sharpe={max_sharpe.sharpe:.3f}\n"
        )
        f.write(
            f"Min Vol  : return={min_vol.expected_return:.2%}, "
            f"vol={min_vol.volatility:.2%}, sharpe={min_vol.sharpe:.3f}\n\n"
        )

        f.write("Max Sharpe weights:\n")
        for ticker, weight in zip(tickers, max_sharpe.weights):
            f.write(f"  {ticker:<6}: {weight:.2%}\n")

        f.write("\n--- MPT Limitation Checks ---\n")
        for name, values in limitations.items():
            f.write(f"{name}:\n")
            for key, value in values.items():
                if isinstance(value, bool):
                    f.write(f"  {key}: {value}\n")
                elif "return" in key or "sharpe" in key or "drawdown" in key:
                    f.write(f"  {key}: {value:.4f}\n")
                elif "var" in key or "cvar" in key:
                    f.write(f"  {key}: {value:.4%}\n")
                else:
                    f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")

    print(f"[output] Saved data files to {OUTPUT_DATA}")
    print(f"[output] Saved summary report to {summary_path}")


def main() -> None:
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    tickers = list(prices.columns)

    daily_returns, annual_returns, annual_std, annual_cov = annual_metrics_from_prices(prices)

    print(f"[data] Final usable assets: {tickers}")
    print(f"[data] Observations: {len(prices)} daily prices, {len(daily_returns)} daily returns")

    pair_rows: list[dict[str, float | str]] = []
    for pair in (PAIR_HIGHER_COV, PAIR_LOWER_COV):
        a, b = pair
        if a in tickers and b in tickers:
            metrics = pair_portfolio_metrics(
                ticker_a=a,
                ticker_b=b,
                annual_returns=annual_returns,
                annual_std=annual_std,
                annual_cov=annual_cov,
                weight_a=EQUAL_WEIGHT,
            )
            pair_rows.append({"pair": f"{a}/{b}", **metrics})

    pair_comparison = pd.DataFrame(pair_rows)

    simulations = simulate_random_portfolios(
        annual_returns=annual_returns,
        annual_cov=annual_cov,
        n_portfolios=N_RANDOM_PORTFOLIOS,
        risk_free_rate=RISK_FREE_RATE,
        seed=RANDOM_SEED,
    )

    best_sharpe_row = simulations.loc[simulations["sharpe"].idxmax()]
    best_min_vol_row = simulations.loc[simulations["std_dev"].idxmin()]
    max_sharpe = extract_portfolio_point(best_sharpe_row, tickers)
    min_vol = extract_portfolio_point(best_min_vol_row, tickers)

    frontier = efficient_frontier_from_simulation(simulations)

    limitations = run_mpt_limitations(
        daily_returns=daily_returns,
        tickers=tickers,
        max_sharpe_weights=max_sharpe.weights,
        risk_free_rate=RISK_FREE_RATE,
    )

    save_outputs(
        prices=prices,
        annual_returns=annual_returns,
        annual_std=annual_std,
        annual_cov=annual_cov,
        pair_comparison=pair_comparison,
        simulations=simulations,
        frontier=frontier,
        max_sharpe=max_sharpe,
        min_vol=min_vol,
        limitations=limitations,
        tickers=tickers,
    )

    save_frontier_plot(
        simulations=simulations,
        frontier=frontier,
        max_sharpe=max_sharpe,
        min_vol=min_vol,
    )

    print("\n=== Key Results ===")
    if not pair_comparison.empty:
        for _, row in pair_comparison.iterrows():
            print(
                f"{row['pair']:<12} cov={row['covariance']:.5f}  "
                f"ret/vol={row['return_to_risk']:.3f}"
            )

    print(
        f"Max Sharpe: return={max_sharpe.expected_return:.2%}, "
        f"vol={max_sharpe.volatility:.2%}, sharpe={max_sharpe.sharpe:.3f}"
    )
    print(
        f"Min Vol   : return={min_vol.expected_return:.2%}, "
        f"vol={min_vol.volatility:.2%}, sharpe={min_vol.sharpe:.3f}"
    )

    print("\nMPT limitation flags:")
    for name, values in limitations.items():
        print(f"  {name:<18} -> failure_flag={values['failure_flag']}")


if __name__ == "__main__":
    main()
