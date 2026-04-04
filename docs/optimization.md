# Portfolio Optimisation using the Covariance Matrix

---

## 1. Problem Statement

Given $N$ assets with return covariance $\Sigma$, find the weight vector
$\mathbf{w} \in \mathbb{R}^N$ that **minimises portfolio variance**:

$$\min_{\mathbf{w}} \quad \mathbf{w}^\top \Sigma \mathbf{w}$$

This is the **Global Minimum Variance (GMV)** problem — it requires no estimate
of expected returns, making it robust to the noisy mean estimates that plague
full mean-variance optimisation.

---

## 2. Constraints

### 2.1 Fully Invested (required)

$$\mathbf{1}^\top \mathbf{w} = 1$$

### 2.2 Long-Only (optional)

$$w_i \geq 0 \quad \forall\, i$$

When long-only constraints are active, the problem becomes a Quadratic Programme
(QP) with no closed-form solution — numerical solvers (e.g., `cvxpy`) are
required.

Without the long-only constraint, the unconstrained GMV has a clean
closed-form solution derived below.

---

## 3. Lagrangian Derivation (Unconstrained)

**Lagrangian:**

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^\top \Sigma \mathbf{w}
  - \lambda \left(\mathbf{1}^\top \mathbf{w} - 1\right)$$

**First-order condition — differentiate w.r.t. $\mathbf{w}$ and set to zero:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\Sigma \mathbf{w} - \lambda \mathbf{1} = \mathbf{0}$$

$$\Sigma \mathbf{w} = \frac{\lambda}{2} \mathbf{1}
  \quad \Longrightarrow \quad
  \mathbf{w} = \frac{\lambda}{2} \Sigma^{-1} \mathbf{1}$$

**Apply the budget constraint $\mathbf{1}^\top \mathbf{w} = 1$:**

$$\mathbf{1}^\top \frac{\lambda}{2} \Sigma^{-1} \mathbf{1} = 1
  \quad \Longrightarrow \quad
  \frac{\lambda}{2} = \frac{1}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}$$

**Substitute back:**

$$\boxed{\mathbf{w}^* = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}}$$

---

## 4. Interpretation — Inverse Covariance Weighting

$\Sigma^{-1}$ is the **precision matrix**. Asset $i$'s unnormalised weight is
the $i$-th row-sum of the precision matrix:

$$\tilde{w}_i = \sum_j (\Sigma^{-1})_{ij}$$

Assets that are *less correlated* with others and have *lower variance* receive
higher weights. Normalising by $\mathbf{1}^\top \Sigma^{-1} \mathbf{1}$ ensures
the budget constraint is satisfied.

---

## 5. Numerical Example

### Inputs ($N=2$)

$$\Sigma = \begin{pmatrix} 0.04 & 0.01 \\ 0.01 & 0.02 \end{pmatrix}$$

### Inverse

$$\Sigma^{-1} = \frac{1}{0.04 \times 0.02 - 0.01^2}
  \begin{pmatrix} 0.02 & -0.01 \\ -0.01 & 0.04 \end{pmatrix}
  = \frac{1}{0.0007}
  \begin{pmatrix} 0.02 & -0.01 \\ -0.01 & 0.04 \end{pmatrix}$$

$$\Sigma^{-1} \approx \begin{pmatrix} 28.57 & -14.29 \\ -14.29 & 57.14 \end{pmatrix}$$

### Weights

$$\Sigma^{-1} \mathbf{1} = \begin{pmatrix} 28.57 - 14.29 \\ -14.29 + 57.14 \end{pmatrix}
  = \begin{pmatrix} 14.29 \\ 42.86 \end{pmatrix}$$

$$\mathbf{1}^\top \Sigma^{-1} \mathbf{1} = 14.29 + 42.86 = 57.14$$

$$\mathbf{w}^* = \frac{1}{57.14} \begin{pmatrix} 14.29 \\ 42.86 \end{pmatrix}
  \approx \begin{pmatrix} 0.25 \\ 0.75 \end{pmatrix}$$

### Portfolio variance

$$\mathbf{w}^{*\top} \Sigma \mathbf{w}^* = 0.0175$$

> Asset B (lower variance, 0.02) receives the larger weight. The positive
> covariance (0.01) shifts weight further toward the lower-variance asset.

---

## 6. Via Eigen Decomposition

Substitute $\Sigma = Q \Lambda Q^\top$:

$$\mathbf{w}^{*\top} \Sigma \mathbf{w}^* = \mathbf{w}^{*\top} Q \Lambda Q^\top \mathbf{w}^*
  = \sum_{j=1}^{N} \lambda_j z_j^2$$

where $\mathbf{z} = Q^\top \mathbf{w}^*$ are the **factor exposures** of the GMV
portfolio.

The GMV portfolio automatically minimises this sum — it de-weights the
high-$\lambda$ (high-risk) factors by making $z_j$ small for large $\lambda_j$.

---

## 7. Risk-Adjusted Weights

An alternative heuristic: weight each asset inversely proportional to its
**total variance** (diagonal of $\Sigma$):

$$w_i^{\text{IVW}} = \frac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}$$

This ignores correlations entirely but is:
- Robust when $\Sigma$ is poorly estimated
- A sensible initialisation for more complex methods

For a correlation-aware but analytically tractable approach, Ledoit-Wolf
shrinkage is preferred over the raw inverse of the sample covariance.

---

## 8. Python Implementation

```python
import numpy as np

def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum variance weights (unconstrained)."""
    N     = Sigma.shape[0]
    ones  = np.ones(N)
    inv_S = np.linalg.inv(Sigma)
    w     = inv_S @ ones
    return w / (ones @ inv_S @ ones)


def ivw_weights(Sigma: np.ndarray) -> np.ndarray:
    """Inverse variance weights (ignores correlations)."""
    variances = np.diag(Sigma)
    inv_var   = 1.0 / variances
    return inv_var / inv_var.sum()


# --- Example ---
Sigma = np.array([[0.04, 0.01],
                  [0.01, 0.02]])

w_gmv = gmv_weights(Sigma)
w_ivw = ivw_weights(Sigma)

print("GMV weights :", w_gmv.round(4))
print("IVW weights :", w_ivw.round(4))
print("GMV variance:", (w_gmv.T @ Sigma @ w_gmv).round(6))
print("IVW variance:", (w_ivw.T @ Sigma @ w_ivw).round(6))

# Long-only GMV with cvxpy
try:
    import cvxpy as cp

    w   = cp.Variable(2)
    obj = cp.Minimize(cp.quad_form(w, Sigma))
    con = [cp.sum(w) == 1, w >= 0]
    cp.Problem(obj, con).solve()
    print("Long-only GMV:", w.value.round(4))
except ImportError:
    print("cvxpy not installed — long-only QP skipped")
```
