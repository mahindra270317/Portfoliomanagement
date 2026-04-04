# Principal Component Analysis (PCA) for Portfolio Construction

---

## 1. What Is PCA in Finance?

PCA is a statistical technique that transforms correlated asset returns into a
smaller set of **uncorrelated latent factors** ordered by the amount of variance
they explain. In portfolio management it serves two purposes:

1. **Dimensionality reduction** — replace $N$ correlated assets with $k \ll N$
   orthogonal factors.
2. **Risk factor discovery** — identify the dominant systematic drivers of
   return co-movement (market beta, sector tilts, style spreads, etc.).

---

## 2. From Covariance to PCA

Given the sample covariance matrix $\Sigma \in \mathbb{R}^{N \times N}$,
the eigen decomposition is:

$$\boxed{\Sigma = Q \Lambda Q^\top}$$

| Symbol | Meaning |
|--------|---------|
| $Q = [\mathbf{q}_1, \ldots, \mathbf{q}_N]$ | Orthonormal eigenvectors (principal directions) |
| $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)$ | Eigenvalues, $\lambda_1 \geq \cdots \geq \lambda_N \geq 0$ |

The **factor return** of the $j$-th principal component at time $t$:

$$f_{j,t} = \mathbf{q}_j^\top \mathbf{r}_t$$

Because $Q$ is orthonormal, the factor returns are **uncorrelated** and their
variances are exactly the eigenvalues: $\text{Var}(f_j) = \lambda_j$.

---

## 3. Explained Variance Ratio

The fraction of total variance explained by the $j$-th component:

$$\text{EVR}_j = \frac{\lambda_j}{\sum_{i=1}^{N} \lambda_i}$$

Cumulative explained variance for the top $k$ components:

$$\text{CEV}_k = \sum_{j=1}^{k} \text{EVR}_j$$

---

## 4. Choosing the Number of Factors $k$

Three common approaches:

### 4.1 Variance Threshold (most common in practice)

Select the smallest $k$ such that:

$$\text{CEV}_k = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{N} \lambda_j} \geq \tau$$

Typical choice: $\tau = 0.95$ (95% of variance explained).

### 4.2 Scree Plot

Plot $\lambda_j$ vs $j$. Select $k$ at the **elbow** — the point where
incremental explained variance drops sharply.

```
λ  |  ●
   |    ●
   |      ● ← elbow at k=3
   |         ● ● ● ● ●
   +─────────────────── j
```

### 4.3 Eigenvalue Threshold

Discard components with $\lambda_j < 1$ (Kaiser criterion) or below the noise
floor estimated by random matrix theory (Marchenko-Pastur distribution).

---

## 5. Factor Portfolios (Eigen Portfolios)

The $j$-th **eigen portfolio** is the portfolio with weights proportional to
the $j$-th eigenvector:

$$\mathbf{w}^{(j)} = \mathbf{q}_j$$

Its return is $f_{j,t} = \mathbf{q}_j^\top \mathbf{r}_t$ and its variance is
$\lambda_j$.

These portfolios are:
- **Long-short** for $j \geq 2$ (eigenvectors are zero-sum after the first)
- **Uncorrelated** with each other by construction

---

## 6. Interpretation of Eigen Portfolios

| Component | Typical Loading Pattern | Economic Interpretation |
|-----------|------------------------|------------------------|
| $\mathbf{q}_1$ | All positive, roughly equal | **Market beta** — broad equity risk |
| $\mathbf{q}_2$ | Mixed signs across sectors | **Sector spread** (e.g., tech vs. energy) |
| $\mathbf{q}_3$ | Mixed signs within sectors | **Style / sub-sector** rotation |
| $\mathbf{q}_{k+1}$ onward | Near-zero, noisy | **Idiosyncratic noise** |

> Important: PCA does not label factors; the economic interpretation must be
> assigned by the analyst based on the loading structure.

---

## 7. Dimensionality Reduction — Low-Rank Approximation

Using only the top $k$ components, approximate the covariance matrix:

$$\Sigma \approx Q_k \Lambda_k Q_k^\top = \sum_{j=1}^{k} \lambda_j \mathbf{q}_j \mathbf{q}_j^\top$$

where $Q_k \in \mathbb{R}^{N \times k}$ and $\Lambda_k \in \mathbb{R}^{k \times k}$.

**Benefits:**
- Reduces parameter count from $O(N^2)$ to $O(Nk)$
- Produces a well-conditioned (invertible) approximate covariance
- Filters out estimation noise in the small eigenvalue components

---

## 8. Asset Selection Using PCA

### Projection-Error Minimisation

An asset $i$ is well-represented by the top-$k$ subspace if its return vector
can be accurately reconstructed from the first $k$ principal components.
The reconstruction error for asset $i$ is:

$$\epsilon_i = \|\mathbf{e}_i - Q_k Q_k^\top \mathbf{e}_i\|^2 = 1 - \sum_{j=1}^{k} q_{ij}^2$$

Equivalently, assets with **small projection error** (high $R^2$ onto the
factor subspace) are those with high cumulative squared loadings.

### Variance-Weighted Score

Weight the squared loadings by the eigenvalues to also prefer assets exposed
to the *most important* factors:

$$\text{score}_i = \sum_{j=1}^{k} \lambda_j \, q_{ij}^2$$

Select the top-scoring assets for portfolio construction.

---

## 9. Practical Use Cases

### 9.1 Statistical Arbitrage (Stat Arb)

Decompose returns into systematic ($Q_k \mathbf{f}_t$) and idiosyncratic
($\boldsymbol{\epsilon}_t$) components:

$$\mathbf{r}_t = Q_k \mathbf{f}_t + \boldsymbol{\epsilon}_t$$

Trade **mean-reverting residuals** $\epsilon_{i,t}$ — they represent
stock-specific deviations from factor-predicted returns.

### 9.2 Risk Modelling

Replace the noisy $N \times N$ sample covariance with the low-rank
approximation $Q_k \Lambda_k Q_k^\top + D$ where $D$ is diagonal (specific
variances). This is the structure underlying BARRA-style factor risk models.

### 9.3 Portfolio Construction

Use the factor subspace to select representative assets and to decorrelate the
portfolio from the largest systematic risks.

---

## 10. Python Example

```python
import numpy as np
import pandas as pd
import yfinance as yf

tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
prices  = yf.download(tickers, start="2022-01-01", end="2024-01-01",
                      auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()

# Center & covariance
X     = returns - returns.mean()
Sigma = np.cov(X.T)

# Eigen decomposition
eigvals, eigvecs = np.linalg.eigh(Sigma)
idx     = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Explained variance
evr = eigvals / eigvals.sum()
cev = np.cumsum(evr)
k   = int(np.argmax(cev >= 0.95)) + 1
print(f"Factors needed for 95% variance: {k}")
print(f"Explained: {evr[:k]}")

# Factor portfolios (eigen portfolios)
for j in range(k):
    print(f"\nPC{j+1} loadings: {dict(zip(tickers, eigvecs[:, j].round(3)))}")

# Low-rank covariance approximation
Qk        = eigvecs[:, :k]
Lk        = np.diag(eigvals[:k])
Sigma_approx = Qk @ Lk @ Qk.T

# Asset selection
scores        = (Qk**2) @ eigvals[:k]
top_assets    = np.argsort(scores)[-3:]
print(f"\nTop 3 assets by factor exposure: {[tickers[i] for i in top_assets]}")
```
