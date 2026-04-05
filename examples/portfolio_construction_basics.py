"""
portfolio_construction_basics.py
=================================
Python equivalent of notebooks/portfolio_construction_basics.ipynb

Covers the fundamental building blocks of portfolio construction:
    1. Annualised returns
    2. Annualised standard deviation (volatility)
    3. Portfolio return (weighted average)
    4. Covariance
    5. Portfolio standard deviation (full formula with covariance term)

Data source: yfinance (replaces the original CSV dependency)
Tickers:     MSFT, GOOGL  — 2016-01-01 to 2017-12-31
             (matches the original course period)

Run:
    python examples/portfolio_construction_basics.py
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS    = ["MSFT", "GOOGL"]
START_DATE = "2016-01-01"
END_DATE   = "2017-12-31"
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. Download price data
# ---------------------------------------------------------------------------
print("=" * 55)
print("  1. Price Data")
print("=" * 55)

prices = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)["Close"][TICKERS]

print(f"\n  Period : {START_DATE} → {END_DATE}")
print(f"  Rows   : {len(prices)} trading days")
print(f"\n  Last 5 rows:")
print(prices.tail().to_string())


# ---------------------------------------------------------------------------
# 2. Annualised returns
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  2. Annualised Returns")
print("=" * 55)
print("""
  Formula:
    Annualised Return = ((1 + Total Return) ^ (252 / T)) - 1

  where Total Return = (P_last - P_first) / P_first
        T = number of trading days in the period
""")

annual_returns = (
    ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] + 1)
    ** (TRADING_DAYS / len(prices))
    - 1
)

for ticker in TICKERS:
    print(f"  Annualised return — {ticker}: {annual_returns[ticker]*100:.2f}%")


# ---------------------------------------------------------------------------
# 3. Daily returns and annualised standard deviation
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  3. Annualised Standard Deviation (Volatility)")
print("=" * 55)
print("""
  Formula:
    Daily returns      : r_t = (P_t - P_{t-1}) / P_{t-1}
    Annualised std dev : σ_annual = σ_daily × √252
""")

daily_returns = prices.pct_change().dropna()
annual_std    = daily_returns.std() * math.sqrt(TRADING_DAYS)

for ticker in TICKERS:
    print(f"  Annualised vol — {ticker}: {annual_std[ticker]*100:.2f}%")


# ---------------------------------------------------------------------------
# 4. Portfolio return (equal weights)
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  4. Portfolio Return")
print("=" * 55)
print("""
  Formula:
    R_p = w_A × R_A + w_B × R_B

  Using equal weights: w_A = w_B = 0.5
""")

w_msft  = 0.5
w_googl = 0.5

portfolio_return = (w_msft  * annual_returns["MSFT"] +
                    w_googl * annual_returns["GOOGL"])

print(f"  Weights         : MSFT={w_msft:.0%}, GOOGL={w_googl:.0%}")
print(f"  Portfolio return: {portfolio_return*100:.2f}%")


# ---------------------------------------------------------------------------
# 5. Covariance — toy example first, then real stocks
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  5. Covariance")
print("=" * 55)
print("""
  Formula (population):
    Cov(X, Y) = (1/n) Σ (x_i - x̄)(y_i - ȳ)

  numpy uses bias=True for population covariance (denominator = n)
  and bias=False (default) for sample covariance (denominator = n-1)
""")

# Toy example
X = [5, 2, 4, 5]
Y = [4, 0, 1, 7]
cov_toy = np.cov(X, Y, bias=True)
print(f"  Toy example — X={X}, Y={Y}")
print(f"  Cov matrix:\n{cov_toy}")
print(f"  Cov(X, Y) = {cov_toy[0, 1]:.2f}  "
      f"| Var(X)={cov_toy[0,0]:.2f}  Var(Y)={cov_toy[1,1]:.2f}")

# Real stocks — annualised covariance
cov_matrix = np.cov(daily_returns["MSFT"],
                    daily_returns["GOOGL"],
                    bias=True) * TRADING_DAYS

print(f"\n  Annualised covariance matrix (MSFT, GOOGL):")
print(f"  {cov_matrix}")
print(f"  Cov(MSFT, GOOGL) = {cov_matrix[0, 1]:.4f}")
print(f"  Var(MSFT)        = {cov_matrix[0, 0]:.4f}  "
      f"→ σ(MSFT) = {cov_matrix[0,0]**0.5*100:.2f}%")
print(f"  Var(GOOGL)       = {cov_matrix[1, 1]:.4f}  "
      f"→ σ(GOOGL) = {cov_matrix[1,1]**0.5*100:.2f}%")

# Correlation implied by the covariance
corr = cov_matrix[0, 1] / (cov_matrix[0, 0]**0.5 * cov_matrix[1, 1]**0.5)
print(f"  Correlation ρ    = {corr:.4f}")


# ---------------------------------------------------------------------------
# 6. Portfolio standard deviation
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("  6. Portfolio Standard Deviation")
print("=" * 55)
print("""
  Formula:
    σ_p = √( w_A²σ_A² + w_B²σ_B² + 2·w_A·w_B·Cov(A,B) )

  The covariance term is the key — it measures how much the two
  assets move together. Lower correlation → lower portfolio vol.
""")

sigma_p = math.sqrt(
    (w_msft  ** 2) * cov_matrix[0, 0] +
    (w_googl ** 2) * cov_matrix[1, 1] +
    2 * w_msft * w_googl * cov_matrix[0, 1]
)

print(f"  Weights          : MSFT={w_msft:.0%}, GOOGL={w_googl:.0%}")
print(f"  σ(MSFT)          : {annual_std['MSFT']*100:.2f}%")
print(f"  σ(GOOGL)         : {annual_std['GOOGL']*100:.2f}%")
print(f"  Cov(MSFT, GOOGL) : {cov_matrix[0, 1]:.4f}")
print(f"  Portfolio vol    : {sigma_p*100:.2f}%")

# Diversification benefit
naive_vol = math.sqrt(
    (w_msft ** 2)  * cov_matrix[0, 0] +
    (w_googl ** 2) * cov_matrix[1, 1]
)
print(f"\n  Naive vol (ignoring covariance): {naive_vol*100:.2f}%")
print(f"  Actual portfolio vol           : {sigma_p*100:.2f}%")
print(f"  Diversification benefit        : {(naive_vol - sigma_p)*100:.2f}%")
print(f"""
  The portfolio vol ({sigma_p*100:.2f}%) is below both individual vols
  ({annual_std['MSFT']*100:.2f}% and {annual_std['GOOGL']*100:.2f}%) because ρ={corr:.2f} < 1.
  This is the core benefit of diversification.
""")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 55)
print("  Summary")
print("=" * 55)
summary = pd.DataFrame({
    "Annualised Return": annual_returns * 100,
    "Annualised Vol (%)": annual_std * 100,
}).round(2)
summary.loc["Portfolio (50/50)"] = [portfolio_return * 100, sigma_p * 100]
print(summary.to_string())
print()
