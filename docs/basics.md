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

## 13. Unequal-Weight Example

To make weights diverge, we need two things to be true simultaneously:
**different variances** and **imperfect correlation**.

### Setup

| Asset | Daily Vol ($\sigma$) | Variance ($\sigma^2$) | Interpretation |
|-------|---------------------|----------------------|---------------|
| A | 1% | 0.0001 | Low-vol bond-like asset |
| B | 3% | 0.0009 | Mid-vol equity |
| C | 6% | 0.0036 | High-vol growth stock |

Correlation matrix (realistic, not near-1):

$$\rho = \begin{pmatrix} 1.00 & 0.20 & 0.05 \\ 0.20 & 1.00 & 0.40 \\ 0.05 & 0.40 & 1.00 \end{pmatrix}$$

### Covariance Matrix

Convert using $\Sigma_{ij} = \rho_{ij} \sigma_i \sigma_j$:

$$\Sigma = \begin{pmatrix}
0.0001 & 0.00006 & 0.00003 \\
0.00006 & 0.0009 & 0.00072 \\
0.00003 & 0.00072 & 0.0036
\end{pmatrix}$$

### Step-by-Step Weight Calculation

The GMV formula is:

$$\mathbf{w}^{\ast} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}$$

So the weight of each asset is proportional to the **row sum of the precision
matrix** $\Sigma^{-1}$.

---

**Step 1 — Precision matrix $\Sigma^{-1}$**

$$\Sigma^{-1} = \begin{pmatrix}
 10428 & -745 &   62 \\
  -745 &  1376 & -269 \\
    62 & -269 &  331
\end{pmatrix}$$

*Note: positive diagonal entries, negative off-diagonal entries — standard for
a precision matrix of positively correlated assets.*

---

**Step 2 — Row sums = unnormalised weights**

$$\tilde{w}_i = \sum_j (\Sigma^{-1})_{ij}$$

| Asset | Diagonal | Off-diag sum | Row sum $\tilde{w}_i$ |
|-------|----------|--------------|-----------------------|
| A | +10428 | $-745 + 62 = -683$ | **9745** |
| B | +1376 | $-745 - 269 = -1014$ | **362** |
| C | +331 | $62 - 269 = -207$ | **124** |
| **Total** | | | **10232** |

---

**Step 3 — Normalise**

$$w_i = \frac{\tilde{w}_i}{\sum_j \tilde{w}_j}$$

| Asset | $\tilde{w}_i$ | Weight |
|-------|---------------|--------|
| A | 9745 | $9745 / 10232 = $ **95.2%** |
| B | 362 | $362 / 10232 = $ **3.5%** |
| C | 124 | $124 / 10232 = $ **1.2%** |

---

**Why does Asset A dominate so heavily?**

Two compounding effects:

**Effect 1 — Diagonal (variance effect)**

The diagonal of $\Sigma^{-1}$ is approximately $1/\sigma_i^2$ (adjusted for
correlations). Since variances differ by 36×:

| Asset | $\sigma^2$ | $1/\sigma^2$ (naive) | Actual $(\Sigma^{-1})_{ii}$ |
|-------|------------|---------------------|--------------------------|
| A | 0.0001 | 10,000 | 10,428 |
| B | 0.0009 | 1,111 | 1,376 |
| C | 0.0036 | 278 | 331 |

Asset A's diagonal alone is **31× larger** than Asset C's.

**Effect 2 — Off-diagonal (correlation penalty)**

When asset $i$ is correlated with others, the negative off-diagonal entries
*reduce* its row sum. Asset B pays the heaviest penalty:

- B is correlated with both A ($\rho=0.20$) and C ($\rho=0.40$)
- Its off-diagonal drain: $-745 - 269 = -1014$
- This wipes out 74% of its diagonal (1376 → 362)

Asset A is barely correlated with C ($\rho=0.05$), so its drain is only $-683$
— just 6.5% of its diagonal (10428 → 9745).

**Net result — row sum ratios:**

$$\tilde{w}_A : \tilde{w}_B : \tilde{w}_C = 9745 : 362 : 124 \approx 78 : 3 : 1$$

After normalisation → **95.2% : 3.5% : 1.2%**

---

**Why not just use $1/\sigma^2$ (inverse-variance) weights?**

Inverse-variance weighting ignores correlations entirely:

$$w_i^{\text{IVW}} = \frac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}$$

$$\mathbf{w}^{\text{IVW}} \approx [\ 87.8\%,\ 9.8\%,\ 2.4\%\ ]$$

This assigns Asset C 2.4% instead of 1.2% — twice as much — because it doesn't
account for the fact that B and C are correlated ($\rho=0.40$), making C
redundant to B. The full GMV correctly penalises C further.

| Method | Asset A | Asset B | Asset C | Portfolio Vol |
|--------|---------|---------|---------|--------------|
| Equal weight ($1/N$) | 33% | 33% | 33% | 2.63% |
| Inverse-variance ($1/\sigma^2$) | 87.8% | 9.8% | 2.4% | 1.01% |
| **GMV ($\Sigma^{-1}$)** | **95.2%** | **3.5%** | **1.2%** | **0.99%** |

GMV squeezes out the last bit of variance by recognising that B and C are
correlated — so there is no point holding both of them materially.

---

## 14. Python Code

```python
import numpy as np

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def gmv(Sigma):
    """Global minimum variance weights."""
    ones  = np.ones(Sigma.shape[0])
    inv_S = np.linalg.inv(Sigma)
    w     = inv_S @ ones
    return w / (ones @ inv_S @ ones)


def summarise(label, R_or_Sigma, from_returns=True):
    """Print weights, vol, and factor risk for a scenario."""
    print(f"\n{'='*55}")
    print(f" {label}")
    print(f"{'='*55}")

    if from_returns:
        mu    = R_or_Sigma.mean(axis=0)
        X     = R_or_Sigma - mu
        T, N  = X.shape
        Sigma = X.T @ X / (T - 1)
    else:
        Sigma = R_or_Sigma
        N     = Sigma.shape[0]

    w         = gmv(Sigma)
    port_var  = float(w.T @ Sigma @ w)

    print(f"Weights      : {w.round(3)}")
    print(f"Equal weights: {np.ones(N)/N}")
    print(f"Portfolio vol: {port_var**0.5:.4%}")
    print(f"Equal-wt vol : {(np.ones(N)/N @ Sigma @ np.ones(N)/N)**0.5:.4%}")

    # Factor decomposition
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx     = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    z       = eigvecs.T @ w

    print("Factor risk  :")
    for j, (lam, zj) in enumerate(zip(eigvals, z)):
        pct = lam * zj**2 / port_var
        print(f"  PC{j+1}: {pct:.1%}")


# ---------------------------------------------------------------------------
# Example A — high correlation, similar variance  →  near-equal weights
# ---------------------------------------------------------------------------
R_equal = np.array([
    [1.0,  0.8,  1.2],
    [-0.5, -0.4, -0.6],
    [0.8,  1.0,  0.9],
    [-1.0, -0.9, -1.1],
    [0.3,  0.2,  0.4],
])
summarise("Example A: high correlation, similar variance", R_equal)


# ---------------------------------------------------------------------------
# Example B — low correlation, very different variance  →  unequal weights
# ---------------------------------------------------------------------------
# Asset vols: A=1%, B=3%, C=6%
vols = np.array([0.01, 0.03, 0.06])

# Correlation matrix
rho = np.array([
    [1.00, 0.20, 0.05],
    [0.20, 1.00, 0.40],
    [0.05, 0.40, 1.00],
])

# Convert to covariance: Sigma_ij = rho_ij * vol_i * vol_j
Sigma_unequal = rho * np.outer(vols, vols)
summarise("Example B: low correlation, different variance", Sigma_unequal,
          from_returns=False)


# ---------------------------------------------------------------------------
# Side-by-side summary
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print(" Side-by-side: Equal-weight vol vs GMV vol")
print("="*55)

for label, Sigma in [
    ("High corr / similar var (Example A)",
     np.cov(R_equal.T)),
    ("Low corr  / diff var    (Example B)",
     Sigma_unequal),
]:
    w_gmv = gmv(Sigma)
    w_eq  = np.ones(len(w_gmv)) / len(w_gmv)
    print(f"\n{label}")
    print(f"  GMV weights    : {w_gmv.round(3)}")
    print(f"  GMV vol        : {(w_gmv @ Sigma @ w_gmv)**0.5:.4%}")
    print(f"  Equal-wt vol   : {(w_eq  @ Sigma @ w_eq )**0.5:.4%}")
    print(f"  Vol improvement: "
          f"{((w_eq@Sigma@w_eq)**0.5 - (w_gmv@Sigma@w_gmv)**0.5):.4%}")
```

---

## 15. Case 3 — Selecting 4 Assets from a Universe of 10 Using PCA

### Problem

We have **10 assets** but want a focused portfolio of only **4**. How do we
decide which 4 to keep?

Naively picking the 4 lowest-variance assets ignores correlations and
systematic risk structure. Instead we use **PCA scores** to select the 4 assets
most representative of the dominant risk factors in the universe.

---

### Setup — Universe of 10 Assets

Three groups with different volatility profiles:

| Asset | Group | Daily Vol | $\sigma^2$ |
|-------|-------|-----------|-----------|
| Growth-1 | Growth | 3.0% | 0.000900 |
| Growth-2 | Growth | 3.5% | 0.001225 |
| Growth-3 | Growth | 2.8% | 0.000784 |
| Growth-4 | Growth | 4.0% | 0.001600 |
| Value-1 | Value | 1.8% | 0.000324 |
| Value-2 | Value | 2.2% | 0.000484 |
| Value-3 | Value | 1.5% | 0.000225 |
| Def-1 | Defensive | 0.8% | 0.000064 |
| Def-2 | Defensive | 1.0% | 0.000100 |
| Def-3 | Defensive | 1.2% | 0.000144 |

**Correlation structure (assumed, not derived from data):**
- Within-group: $\rho = 0.70$ — chosen to reflect typical same-sector pairwise correlations (0.60–0.80 in real equity markets)
- Cross-group: $\rho = 0.15$ — chosen to reflect weak cross-sector linkage (0.10–0.25 in real markets)

> **Note:** These values are synthetic assumptions made to keep the example clean and illustrative.
> In a real project you would estimate $\Sigma$ directly from historical return data.

---

### Step 1 — Eigendecompose the $10 \times 10$ Covariance Matrix

$$\Sigma = Q \Lambda Q^\top, \quad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{10} \geq 0$$

**Explained variance per principal component:**

| PC | Eigenvalue $\lambda_j$ | Explained | Cumulative |
|----|----------------------|-----------|-----------|
| PC1 | 0.000954 | 61.1% | 61.1% |
| PC2 | 0.000215 | 13.8% | 74.9% |
| PC3 | 0.000114 | 7.3% | 82.2% |
| PC4 | 0.000082 | 5.3% | 87.5% |
| PC5 | 0.000065 | 4.2% | 91.7% |
| PC6 | 0.000063 | 4.0% | **95.7%** ← threshold |
| … | … | … | … |

**k = 6 principal components** are needed to explain ≥ 95% of variance.

*Why 6 and not 3?* Because within each of the 3 groups there are sub-patterns
(e.g. Growth-4 is more volatile than Growth-3) that require extra PCs to capture.

---

### Step 2 — Score Every Asset

For each asset $i$, compute:

$$\text{score}_i = \sum_{j=1}^{k=6} \lambda_j \, q_{ij}^2$$

This measures how much of asset $i$'s total variance is explained by the top-6
systematic factors. A high score means the asset is a major driver of market
risk — it is highly "systematic."

**Scores (bar = proportion of max score):**

| Asset | Score | Relative bar |
|-------|-------|-------------|
| Growth-4 | 0.001600 | ████████████████████ ← highest |
| Growth-2 | 0.001225 | ███████████████ |
| Growth-1 | 0.000900 | ████████████ |
| Growth-3 | 0.000784 | ██████████ |
| Value-2 | 0.000428 | █████ |
| Value-1 | 0.000250 | ███ |
| Value-3 | 0.000160 | ██ |
| Def-3 | 0.000127 | █ |
| Def-2 | 0.000078 | █ |
| Def-1 | 0.000045 | |

**Why does Growth dominate?**  
PCA score $\approx \sigma_i^2$ when within-group correlations are uniform.
Growth assets have the highest variances → highest scores. The score is not
about "safety" — it ranks assets by **systematic risk contribution**.

---

### Step 3 — Select Top 4 by Score

$$\text{Selected} = \{\text{Growth-4},\ \text{Growth-2},\ \text{Growth-1},\ \text{Growth-3}\}$$

All four come from the Growth group. Their $4 \times 4$ sub-covariance:

$$\Sigma_{\text{sel}} = \begin{pmatrix}
0.00160 & 0.00098 & 0.00084 & 0.00112 \\
0.00098 & 0.00122 & 0.00073 & 0.00098 \\
0.00084 & 0.00073 & 0.00090 & 0.00084 \\
0.00112 & 0.00098 & 0.00084 & 0.00160
\end{pmatrix}$$

*(rows/columns: Growth-4, Growth-2, Growth-1, Growth-3)*

Within-group correlation is 0.70 — these 4 assets are highly correlated with
each other, so GMV will again push toward unequal weights.

---

### Step 4 — GMV Weights on the 4 Selected Assets

$$\mathbf{w}^{\ast} \approx [\ -15.5\%,\ +5.8\%,\ +43.5\%,\ +66.2\%\ ]$$

*(Growth-4, Growth-2, Growth-1, Growth-3)*

**Why is Growth-4 negative?**  
Growth-4 has the **highest variance** (4.0% daily vol). Within a group of
highly correlated assets, GMV shorts the most volatile member and overweights
the least volatile (Growth-3 at 2.8%, Growth-1 at 3.0%) because shorting
the high-vol name reduces overall portfolio variance.

**Factor risk of the selected portfolio:**

| Factor | % of Portfolio Variance |
|--------|------------------------|
| PC1 (market) | 76.4% |
| PC3 | 12.9% |
| PC2 | 7.0% |
| PC4 | 3.7% |

76% of risk comes from PC1 — the portfolio is still market-dominated.

---

### Step 5 — Compare Across Methods

| Method | Portfolio Vol | Notes |
|--------|--------------|-------|
| Equal weight (all 10 assets) | 1.44% | Defensive assets dilute vol |
| GMV on PCA-selected 4 (Growth) | 2.63% | Higher vol — all systematic assets |
| GMV on lowest-vol 4 (Defensives) | ~0.85% | Lowest vol, but minimal factor exposure |

**Key insight:** PCA selection picks the *most systematic* assets — those that
best represent the dominant market risk factors. This is useful for factor
modelling and stat arb. If the goal is purely minimising portfolio volatility,
you would instead select the 4 lowest-variance assets (the Defensive group).

| Goal | Best selection method |
|------|----------------------|
| Capture market factor exposure | PCA score ranking |
| Minimise portfolio variance | Select lowest-vol assets |
| Balance both | Hybrid: PCA within each vol tier |

---

### Summary — All Three Cases

| Case | Assets | Method | GMV Vol | Equal-wt Vol |
|------|--------|--------|---------|-------------|
| A: Equal vol, high corr | 3 | Direct GMV | 1.93% | 1.93% |
| B: Diff vol, low corr | 3 | Direct GMV | **0.99%** | 2.63% |
| C: PCA top-4 of 10 | 4 of 10 | PCA select → GMV | 2.63% | 1.44% (all 10) |

---

## 16. Case 4 — Diversified Selection: Constrained PCA Across Groups

### Problem

Case 3 selected all 4 assets from the Growth group — high systematic exposure
but high volatility (2.63% daily). In practice, an investor wants **cross-group
diversification**: a mix of Growth, Value, and Defensive assets.

**Constraint:** select the top-scoring asset from each group tier:
- **2 from Growth** (highest factor exposure)
- **1 from Value** (mid-vol anchor)
- **1 from Defensive** (vol dampener)

---

### Step 1 — Score All 10 Assets (same as Case 3, k=6)

| Asset | Group | Score | Rank within group |
|-------|-------|-------|------------------|
| Growth-4 | Growth | 0.001600 | 1st ← selected |
| Growth-2 | Growth | 0.001225 | 2nd ← selected |
| Growth-1 | Growth | 0.000900 | 3rd |
| Growth-3 | Growth | 0.000784 | 4th |
| Value-2 | Value | 0.000428 | 1st ← selected |
| Value-1 | Value | 0.000250 | 2nd |
| Value-3 | Value | 0.000160 | 3rd |
| Def-3 | Defensive | 0.000127 | 1st ← selected |
| Def-2 | Defensive | 0.000078 | 2nd |
| Def-1 | Defensive | 0.000045 | 3rd |

**Selected: Growth-4, Growth-2, Value-2, Def-3**

---

### Step 2 — Sub-Covariance of the 4 Selected Assets

$$\Sigma_{\text{sel}} = \begin{pmatrix}
0.001600 & 0.000980 & 0.000126 & 0.000084 \\
0.000980 & 0.001225 & 0.000154 & 0.000102 \\
0.000126 & 0.000154 & 0.000484 & 0.000040 \\
0.000084 & 0.000102 & 0.000040 & 0.000144
\end{pmatrix}$$

*(rows/columns: Growth-4, Growth-2, Value-2, Def-3)*

**Notice:** Growth-Growth entries (~0.00098) are large — high within-group
correlation. Growth-Def entries (~0.000084) are tiny — low cross-group
correlation. This structure is what makes Def-3 so valuable as a diversifier.

---

### Step 3 — Precision Matrix & Row Sums

$$\Sigma_{\text{sel}}^{-1} \approx \begin{pmatrix}
\cdot & \cdot & \cdot & \cdot \\
\cdot & \cdot & \cdot & \cdot \\
\cdot & \cdot & \cdot & \cdot \\
\cdot & \cdot & \cdot & \cdot
\end{pmatrix}$$

**Row sums (unnormalised weights $\tilde{w}_i$):**

| Asset | Row sum $\tilde{w}_i$ | Normalised weight |
|-------|----------------------|------------------|
| Growth-4 | 5.2 | **0.1%** |
| Growth-2 | 346 | **4.2%** |
| Value-2 | 1,459 | **17.8%** |
| Def-3 | 6,389 | **77.9%** |
| **Total** | **8,199** | 100% |

---

### Step 4 — Why Does Def-3 Dominate at 77.9%?

The same two-force logic from Case B applies, now amplified by cross-group
structure:

**Force 1 — Variance:** Def-3 has the lowest variance in the selected set.

| Asset | $\sigma$ | $\sigma^2$ | Ratio vs Def-3 |
|-------|----------|-----------|---------------|
| Growth-4 | 4.0% | 0.001600 | **25×** |
| Growth-2 | 3.5% | 0.001225 | **19×** |
| Value-2 | 2.2% | 0.000484 | **7.5×** |
| Def-3 | 1.2% | 0.000144 | 1× |

**Force 2 — Correlation:** Def-3 has near-zero correlation with Growth assets
($\rho = 0.15$) and very low correlation with Value-2 ($\rho = 0.15$).
Its row sum suffers almost no off-diagonal drain.

**Force 3 (new in this example) — Growth-4 collapses to ~0%:**

Growth-4 and Growth-2 have within-group correlation $\rho = 0.70$. To the
precision matrix, they are near-redundant. GMV assigns essentially all the
Growth allocation to Growth-2 (slightly lower vol, 3.5% vs 4.0%) and nearly
zeros out Growth-4 (0.1%). This is the same short-the-most-volatile logic as
Case 3, but without going negative because Value-2 and Def-3 are available
as diversifiers.

---

### Step 5 — Portfolio Characteristics

**Weights:**

| Asset | Group | Weight |
|-------|-------|--------|
| Def-3 | Defensive | **77.9%** |
| Value-2 | Value | **17.8%** |
| Growth-2 | Growth | **4.2%** |
| Growth-4 | Growth | **0.1%** |

**Factor risk:**

| Factor | % of Variance |
|--------|--------------|
| PC3 (Defensive factor) | 64.3% |
| PC2 (Value factor) | 23.8% |
| PC1 (Market / Growth) | 11.8% |

The diversified portfolio has **inverted** risk attribution vs Case 3: instead
of 76% in PC1 (market/growth), it now puts 64% in PC3 (the defensive factor)
and 24% in PC2 (the value factor).

---

### Step 6 — Compare All Methods on the Same 10-Asset Universe

| Method | Selected | GMV Weights | Portfolio Vol | EQ-wt baseline |
|--------|----------|-------------|--------------|----------------|
| C: PCA top-4 (all Growth) | G1,G2,G3,G4 | -15.5 / 5.8 / 43.5 / 66.2% | 2.63% | 1.44% (all 10) |
| **D: Constrained PCA (2G+1V+1D)** | **G4,G2,V2,D3** | **0.1 / 4.2 / 17.8 / 77.9%** | **1.10%** | **1.44% (all 10)** |
| Min-vol 4 (Defensives only) | D1,D2,D3 + V3 | ~equal | ~0.85% | 1.44% (all 10) |

**The constrained diversified portfolio (Case D) beats the full equal-weight
universe by 0.34% in daily vol while maintaining meaningful cross-sector factor
exposure.** It is the practical middle ground between the pure factor-exposure
approach (Case C) and the pure vol-minimisation approach.

---

### Key Takeaways Across All Cases

| Case | Selection logic | What GMV does after | Vol outcome |
|------|----------------|---------------------|-------------|
| A | N/A (3 assets) | Equal weights (symmetric) | Same as EQ |
| B | N/A (3 assets) | Tilts 95% to lowest-vol | 63% below EQ |
| C | PCA top-4 (all Growth) | Shorts highest-vol Growth | Higher than EQ (all 10) |
| **D** | **Constrained PCA (2G+1V+1D)** | **Tilts to Def, nearly zeros Growth-4** | **24% below EQ (all 10)** |

**Selection method determines the opportunity set. GMV then squeezes the minimum
variance out of whatever assets you give it.**

---

## 17. Run the Code

```bash
python examples/basics.py
```

All four cases are implemented in [`examples/basics.py`](../examples/basics.py)
with step-by-step printed output matching this document.
