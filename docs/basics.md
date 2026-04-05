# Basics of Portfolio Risk: Covariance, Variance, and Eigen Decomposition

---

## 1. Returns and Mean

Let $P_{i,t}$ be the price of asset $i$ at time $t$. The simple daily return is:

$$r_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$$

For a universe of $N$ assets observed over $T$ periods, we collect returns into
a matrix $R \in \mathbb{R}^{T \times N}$.

The **sample mean return** for asset $i$:

$$\mu_i = \frac{1}{T} \sum_{t=1}^{T} r_{i,t}$$

In matrix form: $\boldsymbol{\mu} = \frac{1}{T} R^\top \mathbf{1}_T$

---

## 2. Centering

Define the **centered return matrix**:

$$X_{i,t} = r_{i,t} - \mu_i \quad \Longrightarrow \quad X = R - \mathbf{1}_T \boldsymbol{\mu}^\top$$

**Why center?**  
The covariance between two random variables measures co-movement around their
means, not around zero. Failing to center conflates the level of returns with
their dispersion — the covariance matrix would not be positive semi-definite in
general.

---

## 3. Covariance Matrix

### Definition

The covariance between assets $i$ and $j$ is:

$$\sigma_{ij} = \text{Cov}(r_i, r_j) = \mathbb{E}[(r_i - \mu_i)(r_j - \mu_j)]$$

### Sample Estimator

$$\hat{\sigma}_{ij} = \frac{1}{T-1} \sum_{t=1}^{T} (r_{i,t} - \mu_i)(r_{j,t} - \mu_j)$$

### Matrix Form

$$\Sigma = \frac{1}{T-1} X^\top X \quad \in \mathbb{R}^{N \times N}$$

**Diagonal entries** are variances: $\Sigma_{ii} = \sigma_i^2$.  
**Off-diagonal entries** are covariances: $\Sigma_{ij} = \sigma_{ij}$.

### Why divide by $(T-1)$?

Using $T$ gives the MLE estimator, which is biased for finite samples.
Dividing by $T-1$ (Bessel's correction) yields an **unbiased** estimator of the
population covariance because one degree of freedom is consumed by estimating
the mean.

**Key properties of $\Sigma$:**
- Symmetric: $\Sigma = \Sigma^\top$
- Positive semi-definite: $\mathbf{w}^\top \Sigma \mathbf{w} \geq 0$ for all $\mathbf{w}$
- Dimension: $N \times N$ — grows quadratically with the number of assets

---

## 4. Portfolio Return

Given a weight vector $\mathbf{w} \in \mathbb{R}^N$ with $\mathbf{1}^\top \mathbf{w} = 1$,
the portfolio return at time $t$ is:

$$r_{p,t} = \mathbf{w}^\top \mathbf{r}_t = \sum_{i=1}^{N} w_i r_{i,t}$$

Expected portfolio return:

$$\mu_p = \mathbf{w}^\top \boldsymbol{\mu}$$

---

## 5. Portfolio Variance — Full Derivation

**Starting point:**

$$\text{Var}(r_p) = \text{Var}\!\left(\sum_i w_i r_i\right)$$

**Expand using the definition of variance:**

$$= \mathbb{E}\!\left[\left(\sum_i w_i r_i - \mu_p\right)^2\right]
  = \mathbb{E}\!\left[\left(\sum_i w_i (r_i - \mu_i)\right)^2\right]$$

**Expand the square:**

$$= \mathbb{E}\!\left[\sum_i \sum_j w_i w_j (r_i-\mu_i)(r_j-\mu_j)\right]$$

**Bring expectation inside (linearity):**

$$= \sum_i \sum_j w_i w_j \,\underbrace{\mathbb{E}[(r_i-\mu_i)(r_j-\mu_j)]}_{\sigma_{ij}}$$

**Compact scalar form:**

$$\boxed{\text{Var}(r_p) = \sum_i \sum_j w_i w_j \sigma_{ij}}$$

**Matrix form:**

$$\boxed{\text{Var}(r_p) = \mathbf{w}^\top \Sigma \mathbf{w}}$$

---

## 6. Why the Full Covariance Matrix Is Needed

Consider two assets $A$ and $B$ that individually have low variance but move
together almost perfectly ($\sigma_{AB} \approx \sigma_A \sigma_B$). A portfolio
of only $A$ and $B$ offers almost no diversification. Ignoring the off-diagonal
term $\sigma_{AB}$ would drastically **underestimate** portfolio risk.

$$\text{Var}(r_p) = w_A^2 \sigma_A^2 + w_B^2 \sigma_B^2 + 2 w_A w_B \sigma_{AB}$$

The cross term $2 w_A w_B \sigma_{AB}$ can be large and positive — omitting it
is dangerous.

---

## 7. Limitations of the Covariance Matrix

| Issue | Detail |
|-------|--------|
| **Curse of dimensionality** | $N$ assets → $N(N+1)/2$ unique parameters to estimate |
| **Estimation error** | For small $T/N$, sample $\Sigma$ is poorly conditioned or singular |
| **Hard to interpret** | $N \times N$ dense matrix; no obvious structure |
| **Non-stationarity** | Correlations shift across market regimes |

These limitations motivate **shrinkage estimators** (Ledoit-Wolf) and
**factor models** (PCA).

---

## 8. Eigen Decomposition

Any real symmetric positive semi-definite matrix $\Sigma$ can be decomposed as:

$$\boxed{\Sigma = Q \Lambda Q^\top}$$

where:

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| $Q$ | Matrix of **eigenvectors** (columns = principal directions) | $N \times N$, orthonormal |
| $\Lambda$ | Diagonal matrix of **eigenvalues** $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_N \geq 0$ | $N \times N$ |

**Eigenvectors** $\mathbf{q}_i$ are the directions of maximum variance in
return space — they are orthogonal to each other.

**Eigenvalues** $\lambda_i$ measure the variance explained by each direction.
Large $\lambda_i$ → large systematic risk along $\mathbf{q}_i$.

---

## 9. Risk Decomposition via Eigen Decomposition

Substitute $\Sigma = Q \Lambda Q^\top$ into the variance formula:

$$\text{Var}(r_p)
  = \mathbf{w}^\top Q \Lambda Q^\top \mathbf{w}
  = \mathbf{z}^\top \Lambda \mathbf{z}
  = \sum_{j=1}^{N} \lambda_j z_j^2$$

where $\mathbf{z} = Q^\top \mathbf{w}$ are the **factor exposures** of the portfolio
and $z_j = \mathbf{q}_j^\top \mathbf{w}$ is the projection of the weight vector
onto the $j$-th principal component.

$$\boxed{\text{Var}(r_p) = \sum_{j=1}^{N} \lambda_j z_j^2}$$

Each term $\lambda_j z_j^2$ is the **risk contribution of factor $j$**.

---

## 10. Interpretation of Principal Components

| Component | Typical interpretation |
|-----------|----------------------|
| $\mathbf{q}_1$ | **Market factor** — all assets load with same sign; represents broad market beta |
| $\mathbf{q}_2$ | **Sector / style spread** — long some sectors, short others |
| $\mathbf{q}_3$ | **Relative-value** — finer cross-sectional differentiation |
| $\mathbf{q}_{k+1}, \ldots$ | **Idiosyncratic noise** — small eigenvalues, stock-specific |

---

## 11. Asset Selection Rule

Score each asset $i$ by its aggregate exposure to the top $k$ factors,
weighted by the variance each factor represents:

$$\text{score}_i = \sum_{j=1}^{k} \lambda_j \, q_{ij}^2$$

Assets with high scores are strongly driven by the dominant systematic risks.
Selecting the top-scoring assets ensures the portfolio captures the main
variance structure of the universe.

---

## 12. Complete Numerical Example

### Setup

Consider $N = 3$ assets, $T = 5$ daily observations.

**Returns matrix $R$ (%):**

| Day | Asset A | Asset B | Asset C |
|-----|---------|---------|---------|
| 1   | 1.0     | 0.8     | 1.2     |
| 2   | -0.5    | -0.4    | -0.6    |
| 3   | 0.8     | 1.0     | 0.9     |
| 4   | -1.0    | -0.9    | -1.1    |
| 5   | 0.3     | 0.2     | 0.4     |

**Step 1 — Means:** $\boldsymbol{\mu} = [0.12,\ 0.14,\ 0.16]$

**Step 2 — Centered matrix $X = R - \boldsymbol{\mu}$**

**Step 3 — Sample covariance $\Sigma = \frac{1}{4} X^\top X$:**

$$\Sigma \approx \begin{pmatrix} 0.577 & 0.538 & 0.622 \\ 0.538 & 0.527 & 0.587 \\ 0.622 & 0.587 & 0.677 \end{pmatrix}$$

**Step 4 — Eigendecomposition:**

$$\lambda_1 \approx 1.77,\quad \lambda_2 \approx 0.01,\quad \lambda_3 \approx 0.00$$

$\lambda_1$ accounts for ~99% of total variance — one dominant market factor.

**Step 5 — Leading eigenvector:**

$$\mathbf{q}_1 \approx [0.574,\ 0.551,\ 0.606]^\top$$

All positive — confirms market-factor interpretation.

**Step 6 — Asset scores** (using $k=1$):

$$\text{score}_i = \lambda_1 q_{1i}^2 \approx [0.584,\ 0.538,\ 0.650]$$

Top asset by score: **Asset C**.

**Step 7 — GMV weights over all 3 assets:**

$$\mathbf{w}^{\ast} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}
  \approx [0.29,\ 0.38,\ 0.33]$$

**Step 8 — Portfolio variance:**

$$(\mathbf{w}^{\ast})^\top \Sigma\, \mathbf{w}^{\ast} \approx 0.566$$

---

**Step 9 — Why are the weights near-equal $[0.29,\ 0.38,\ 0.33]$?**

Look at the return data again. All three assets move almost perfectly together
(correlation $\approx 0.99$). When assets are nearly identical in terms of risk
and co-movement, the optimiser has very little reason to prefer one over another
— the **inverse covariance matrix becomes nearly flat**, and the resulting
weights converge toward $1/N$.

To see this precisely, consider the extreme: if $\Sigma = \sigma^2 \mathbf{I}$
(all assets independent, equal variance), then:

$$\Sigma^{-1} = \frac{1}{\sigma^2} \mathbf{I}
  \quad \Longrightarrow \quad
  \mathbf{w}^{\ast} = \frac{\mathbf{1}}{N}$$

Exactly equal weights. Our example is close to this degenerate case because
all three assets are driven almost entirely by the **same single factor (PC1)**.

**Step 10 — When do weights diverge from equal?**

Weights become unequal when assets differ in either:

| Condition | Effect on weights |
|-----------|------------------|
| Different variances ($\sigma_i \neq \sigma_j$) | Lower-variance asset gets more weight |
| Imperfect correlation ($\rho_{ij} < 1$) | Asset that diversifies best gets more weight |
| One asset provides a hedge | It can receive a **negative** (short) weight |

In practice with real equity data (e.g. NVDA vs XOM), correlations are much
lower ($\rho \approx 0.3$–$0.6$) and variances differ significantly — so the
GMV weights will diverge meaningfully from $1/N$.

---

## 13. Python Code

```python
import numpy as np

# Returns (T x N)
R = np.array([
    [1.0,  0.8,  1.2],
    [-0.5, -0.4, -0.6],
    [0.8,  1.0,  0.9],
    [-1.0, -0.9, -1.1],
    [0.3,  0.2,  0.4],
])

# Mean & centering
mu = R.mean(axis=0)
X  = R - mu

# Sample covariance
T, N = X.shape
Sigma = X.T @ X / (T - 1)

# Eigen decomposition
eigvals, eigvecs = np.linalg.eigh(Sigma)
idx     = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

print("Eigenvalues:", eigvals)
print("Explained variance:", eigvals / eigvals.sum())

# GMV weights
ones  = np.ones(N)
inv_S = np.linalg.inv(Sigma)
w     = inv_S @ ones / (ones @ inv_S @ ones)
print("GMV weights:", w)
print("Portfolio variance:", w.T @ Sigma @ w)

# Risk decomposition
z            = eigvecs.T @ w
risk_contrib = eigvals * z**2
print("Factor exposures:", z)
print("Risk contributions:", risk_contrib)
```
