# Risk Decomposition in Portfolio Management

---

## 1. Portfolio Variance

Given weights $\mathbf{w} \in \mathbb{R}^N$ and covariance matrix
$\Sigma \in \mathbb{R}^{N \times N}$:

$$\text{Var}(r_p) = \mathbf{w}^\top \Sigma \mathbf{w} = \sum_i \sum_j w_i w_j \sigma_{ij}$$

This single number measures total portfolio risk but tells us nothing about
*where* that risk comes from.

---

## 2. Marginal Risk Contribution

The **marginal risk contribution** of asset $i$ is the sensitivity of portfolio
variance to a small change in $w_i$:

$$\text{MRC}_i = \frac{\partial\,\text{Var}(r_p)}{\partial w_i}
  = 2 (\Sigma \mathbf{w})_i$$

or in terms of portfolio volatility $\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}$:

$$\text{MRC}_i^{\sigma} = \frac{(\Sigma \mathbf{w})_i}{\sigma_p}$$

In vector form: $\nabla_{\mathbf{w}} \text{Var}(r_p) = 2 \Sigma \mathbf{w}$

---

## 3. Component Risk Contribution

The **component (absolute) risk contribution** of asset $i$:

$$\text{CRC}_i = w_i \cdot \text{MRC}_i = w_i \cdot 2(\Sigma \mathbf{w})_i$$

**Key identity — Euler's theorem** (variance is a degree-2 homogeneous function
of $\mathbf{w}$):

$$\sum_i w_i \cdot (\Sigma \mathbf{w})_i = \mathbf{w}^\top \Sigma \mathbf{w} = \text{Var}(r_p)$$

Therefore:

$$\sum_i \text{CRC}_i = 2\,\text{Var}(r_p)$$

The **percentage risk contribution** of asset $i$:

$$\text{PRC}_i = \frac{w_i (\Sigma \mathbf{w})_i}{\mathbf{w}^\top \Sigma \mathbf{w}}$$

with $\sum_i \text{PRC}_i = 1$.

---

## 4. Eigen Risk Decomposition

Substitute $\Sigma = Q \Lambda Q^\top$ into the variance formula:

$$\text{Var}(r_p) = \mathbf{w}^\top Q \Lambda Q^\top \mathbf{w}
  = \mathbf{z}^\top \Lambda \mathbf{z}
  = \sum_{j=1}^{N} \lambda_j z_j^2$$

where:
- $\mathbf{z} = Q^\top \mathbf{w}$ — **factor exposure vector**
- $z_j = \mathbf{q}_j^\top \mathbf{w}$ — exposure to the $j$-th principal component
- $\lambda_j z_j^2$ — **risk contribution of factor $j$**

$$\boxed{\text{Var}(r_p) = \sum_{j=1}^{N} \lambda_j z_j^2}$$

---

## 5. Interpretation

### Risk Concentration

| Metric | Interpretation |
|--------|---------------|
| $\lambda_j z_j^2$ large | Portfolio is heavily exposed to factor $j$ |
| $z_1 \gg z_2, z_3, \ldots$ | Risk is concentrated in the market factor |
| All $\lambda_j z_j^2$ equal | **Equal risk contribution** across factors |

### Reducing Factor Risk

To reduce exposure to the dominant factor ($j=1$, typically the market):

$$z_1 = \mathbf{q}_1^\top \mathbf{w} \approx 0$$

This requires a **long-short** portfolio (weights sum to zero) or explicit
factor neutralisation.

---

## 6. Worked Example

**Setup:** $N=3$ assets, covariance below, GMV weights.

$$\Sigma = \begin{pmatrix} 0.10 & 0.04 & 0.03 \\ 0.04 & 0.08 & 0.02 \\ 0.03 & 0.02 & 0.05 \end{pmatrix},
\qquad
\mathbf{w}^{\ast} \approx \begin{pmatrix} 0.24 \\ 0.31 \\ 0.45 \end{pmatrix}$$

**Step 1 — Portfolio variance:**

$$\text{Var}(r_p) = \mathbf{w}^{\ast\top} \Sigma \mathbf{w}^{\ast} \approx 0.046$$

**Step 2 — $\Sigma \mathbf{w}^{\ast}$:**

$$\Sigma \mathbf{w}^{\ast} \approx \begin{pmatrix} 0.040 \\ 0.036 \\ 0.033 \end{pmatrix}$$

**Step 3 — Component risk contributions:**

| Asset | $w_i$ | $(\Sigma\mathbf{w})_i$ | $\text{CRC}_i$ | $\text{PRC}_i$ |
|-------|--------|------------------------|----------------|----------------|
| A | 0.24 | 0.040 | 0.0096 | 20.9% |
| B | 0.31 | 0.036 | 0.0112 | 24.3% |
| C | 0.45 | 0.033 | 0.0149 | 32.4% |
| **Sum** | | | **0.046** | **≈ 1** |

**Step 4 — Eigen decomposition:**

Assuming $\lambda_1 = 0.18,\ \lambda_2 = 0.05,\ \lambda_3 = 0.01$ and
$\mathbf{z} = Q^\top \mathbf{w}^{\ast} \approx [0.48, 0.12, 0.05]$:

| Factor | $\lambda_j$ | $z_j$ | $\lambda_j z_j^2$ | % of Var |
|--------|-------------|--------|-------------------|---------|
| PC1 | 0.18 | 0.48 | 0.041 | 89.4% |
| PC2 | 0.05 | 0.12 | 0.001 | 1.6%  |
| PC3 | 0.01 | 0.05 | 0.000 | 0.0%  |

> ~89% of portfolio risk comes from the first factor (market). The GMV weights
> minimised total variance but did not eliminate market exposure.

---

## 7. Visualisation Ideas

| Chart | What to show |
|-------|-------------|
| **Bar chart** | $\text{PRC}_i$ per asset — identify concentrated positions |
| **Stacked bar** | $\lambda_j z_j^2$ per factor — factor risk budget |
| **Pie chart** | Factor risk shares for a single portfolio |
| **Heatmap** | $w_i \sigma_{ij} w_j$ matrix — asset-pair risk attribution |
| **Line chart** | Rolling $\text{PRC}_i$ over time — track how risk attribution evolves |

---

## 8. Python Code

```python
import numpy as np

def risk_decomposition(w: np.ndarray, Sigma: np.ndarray) -> dict:
    """
    Full risk decomposition: total variance, marginal/component
    contributions per asset, and eigen factor decomposition.
    """
    port_var = float(w.T @ Sigma @ w)
    sigma_w  = Sigma @ w

    # Asset-level
    mrc = 2 * sigma_w
    crc = w * sigma_w
    prc = crc / port_var

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx              = np.argsort(eigvals)[::-1]
    eigvals          = eigvals[idx]
    eigvecs          = eigvecs[:, idx]

    z             = eigvecs.T @ w         # factor exposures
    factor_risk   = eigvals * z**2        # per-factor variance contribution
    factor_pct    = factor_risk / port_var

    return {
        "portfolio_variance": port_var,
        "portfolio_vol":      port_var**0.5,
        "marginal_risk":      mrc,
        "component_risk":     crc,
        "pct_risk":           prc,
        "factor_exposures":   z,
        "factor_risk":        factor_risk,
        "factor_pct":         factor_pct,
    }


# --- Example ---
Sigma = np.array([
    [0.10, 0.04, 0.03],
    [0.04, 0.08, 0.02],
    [0.03, 0.02, 0.05],
])

ones  = np.ones(3)
inv_S = np.linalg.inv(Sigma)
w     = inv_S @ ones / (ones @ inv_S @ ones)   # GMV weights

rd = risk_decomposition(w, Sigma)

print(f"Portfolio variance : {rd['portfolio_variance']:.6f}")
print(f"Portfolio vol (ann): {(rd['portfolio_variance']*252)**0.5:.2%}")
print("\nAsset-level % risk contribution:")
for i, prc in enumerate(rd['pct_risk']):
    print(f"  Asset {i+1}: {prc:.2%}")
print("\nFactor-level % risk contribution:")
for j, fp in enumerate(rd['factor_pct']):
    print(f"  PC{j+1}: {fp:.2%}  (z={rd['factor_exposures'][j]:.4f})")
```
