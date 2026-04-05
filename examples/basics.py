"""
basics.py — Four worked examples from docs/basics.md
=====================================================

Case A: High correlation, equal variance        → near-equal GMV weights
Case B: Low correlation, different variance     → heavily skewed GMV weights
Case C: 10-asset universe, PCA top-4 → GMV     → all-Growth, high vol
Case D: Constrained PCA (2 Growth+1 Value+1 Def) → diversified, lower vol

Key concept:
    GMV weights depend on BOTH variance and correlation.
    When assets are identical in both dimensions → equal weights (1/N).
    When they differ → weights tilt heavily toward the lower-vol,
    better-diversifying assets.

Run:
    python examples/basics.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

DIVIDER = "=" * 62


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Closed-form global minimum variance weights (unconstrained)."""
    ones  = np.ones(Sigma.shape[0])
    inv_S = np.linalg.inv(Sigma)
    w     = inv_S @ ones
    return w / (ones @ inv_S @ ones)


def eigen_decompose(Sigma: np.ndarray):
    """Return (eigvals, eigvecs) sorted by descending eigenvalue."""
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]


def pca_asset_scores(eigvecs: np.ndarray,
                     eigvals: np.ndarray,
                     k: int) -> np.ndarray:
    """
    score_i = sum_{j=1}^{k} lambda_j * q_ij^2
    Measures how strongly asset i loads on the top-k risk factors.
    """
    return (eigvecs[:, :k] ** 2) @ eigvals[:k]


def print_weights(w: np.ndarray, names: list, Sigma: np.ndarray,
                  Sigma_eq_ref: np.ndarray | None = None) -> None:
    """Print weights, GMV vol, and equal-weight vol."""
    port_var = float(w @ Sigma @ w)
    Sigma_ref = Sigma_eq_ref if Sigma_eq_ref is not None else Sigma
    n_ref = Sigma_ref.shape[0]
    eq_w  = np.ones(n_ref) / n_ref
    eq_var = float(eq_w @ Sigma_ref @ eq_w)

    print("  Weights:")
    for name, wi in zip(names, w):
        print(f"    {name:<22}: {wi:+.4f}")
    print(f"  GMV portfolio vol  : {port_var**0.5:.4%}")
    print(f"  Equal-weight vol   : {eq_var**0.5:.4%}  "
          f"({'all assets' if Sigma_eq_ref is not None else 'same assets'})")
    saving = eq_var**0.5 - port_var**0.5
    direction = "saving" if saving >= 0 else "HIGHER (selection trades vol for factor exposure)"
    print(f"  Difference         : {saving:+.4%}  ← {direction}")


def print_factor_risk(w: np.ndarray, Sigma: np.ndarray) -> None:
    """Print per-factor variance contributions (only those > 0.5%)."""
    eigvals, eigvecs = eigen_decompose(Sigma)
    port_var = float(w @ Sigma @ w)
    z        = eigvecs.T @ w
    print("  Factor risk breakdown:")
    for j, (lam, zj) in enumerate(zip(eigvals, z)):
        pct = lam * zj**2 / port_var * 100
        if pct > 0.5:
            print(f"    PC{j+1}: {pct:5.1f}%")


# ===========================================================================
# CASE A — Equal variance, high correlation  →  near-equal weights
# ===========================================================================
print(f"\n{DIVIDER}")
print("  CASE A: High correlation, equal variance")
print("  Expected result: weights ≈ 1/3 each (near equal)")
print(DIVIDER)

# Build covariance directly: all vols = 2%, all pairwise correlations = 0.90
vol_A = 0.02
rho_A = 0.90
Sigma_A = np.array([
    [vol_A**2,            rho_A*vol_A**2,   rho_A*vol_A**2],
    [rho_A*vol_A**2,      vol_A**2,         rho_A*vol_A**2],
    [rho_A*vol_A**2,      rho_A*vol_A**2,   vol_A**2      ],
])
names_A = ["Asset A", "Asset B", "Asset C"]

corr_A = Sigma_A / np.outer(np.sqrt(np.diag(Sigma_A)),
                             np.sqrt(np.diag(Sigma_A)))
print(f"\n  All vols = {vol_A:.0%},  all pairwise ρ = {rho_A}")
print(f"  Correlation matrix:\n{corr_A.round(3)}")

eigvals_A, eigvecs_A = eigen_decompose(Sigma_A)
expl_A = eigvals_A / eigvals_A.sum()
print(f"\n  Explained variance per PC: {expl_A.round(3)}")
print(f"  → PC1 explains {expl_A[0]:.1%} — one dominant factor")

w_A = gmv_weights(Sigma_A)
print()
print_weights(w_A, names_A, Sigma_A)
print_factor_risk(w_A, Sigma_A)

print(f"""
  Why equal weights?
  When all assets have the same vol AND the same pairwise correlation,
  the precision matrix Σ⁻¹ is perfectly symmetric across all assets.
  Every asset contributes identically to reducing risk → w* = 1/N.

  Mathematical proof:
    If Σ = σ²[(1-ρ)I + ρ11ᵀ], then Σ⁻¹1 ∝ 1
    → w* = 1/N exactly.
""")


# ===========================================================================
# CASE B — Different variance, low correlation  →  skewed weights
# ===========================================================================
print(DIVIDER)
print("  CASE B: Low correlation, different variance")
print("  Expected result: weight heavily skewed to lowest-vol asset")
print(DIVIDER)

vols_B = np.array([0.01, 0.03, 0.06])    # 1%, 3%, 6% daily vol
rho_B  = np.array([
    [1.00, 0.20, 0.05],
    [0.20, 1.00, 0.40],
    [0.05, 0.40, 1.00],
])
Sigma_B = rho_B * np.outer(vols_B, vols_B)
names_B = ["Asset A (1% vol)", "Asset B (3% vol)", "Asset C (6% vol)"]

print(f"\n  Vols: {vols_B*100} %")
print(f"  Correlation matrix:\n{rho_B}")

eigvals_B, _ = eigen_decompose(Sigma_B)
expl_B = eigvals_B / eigvals_B.sum()
print(f"\n  Explained variance per PC: {expl_B.round(3)}")

w_B = gmv_weights(Sigma_B)
print()
print_weights(w_B, names_B, Sigma_B)
print_factor_risk(w_B, Sigma_B)

# ------------------------------------------------------------------
# Full step-by-step breakdown of WHY these weights come out this way
# ------------------------------------------------------------------
print(f"\n  --- Step-by-step weight derivation ---")

inv_S_B = np.linalg.inv(Sigma_B)
print(f"\n  Precision matrix Σ⁻¹:")
labels = ["A(1%)", "B(3%)", "C(6%)"]
header = f"  {'':10}" + "".join(f"  {l:>10}" for l in labels)
print(header)
for i, row in enumerate(inv_S_B):
    print(f"  {labels[i]:10}" + "".join(f"  {v:>10.1f}" for v in row))

print(f"\n  Row sums of Σ⁻¹  (= unnormalised weights w̃ᵢ):")
row_sums = inv_S_B.sum(axis=1)
total    = row_sums.sum()
for name, diag, row_s, w_i in zip(
        labels,
        np.diag(inv_S_B),
        row_sums,
        w_B):
    off_diag = row_s - diag
    print(f"  {name:10}  diag={diag:>8.1f}  off-diag={off_diag:>9.1f}"
          f"  row_sum={row_s:>8.1f}  → weight={w_i:.4f}  ({w_i:.1%})")
print(f"  {'Total':10}  {'':>8}  {'':>9}  total   ={total:>8.1f}")

print(f"\n  Effect 1 — Variance: diagonal of Σ⁻¹ vs naive 1/σ²")
print(f"  {'Asset':10}  {'σ²':>10}  {'1/σ² (naive)':>14}  {'Prec diag':>10}")
for i, (name, vol) in enumerate(zip(labels, vols_B)):
    var = vol**2
    print(f"  {name:10}  {var:>10.6f}  {1/var:>14.1f}  "
          f"{inv_S_B[i,i]:>10.1f}")

print(f"\n  Effect 2 — Correlation penalty (negative off-diagonal drain)")
print(f"  {'Asset':10}  {'Diag':>8}  {'Drain':>9}  {'RowSum':>8}  "
      f"{'Drain %':>8}")
for name, diag, row_s in zip(labels, np.diag(inv_S_B), row_sums):
    drain = row_s - diag
    print(f"  {name:10}  {diag:>8.1f}  {drain:>9.1f}  {row_s:>8.1f}  "
          f"{drain/diag*100:>7.1f}%")

print(f"\n  GMV vs Inverse-Variance (1/σ²) comparison:")
inv_var_B = 1 / np.diag(Sigma_B)
w_ivw_B   = inv_var_B / inv_var_B.sum()
var_ivw   = float(w_ivw_B @ Sigma_B @ w_ivw_B)
var_gmv   = float(w_B @ Sigma_B @ w_B)
print(f"  {'Asset':10}  {'IVW weight':>12}  {'GMV weight':>12}")
for name, wi, wg in zip(labels, w_ivw_B, w_B):
    print(f"  {name:10}  {wi:>11.4f}   {wg:>11.4f}")
print(f"  {'IVW vol':10}  {var_ivw**0.5:.4%}")
print(f"  {'GMV vol':10}  {var_gmv**0.5:.4%}")
print(f"  GMV saves extra {var_ivw**0.5 - var_gmv**0.5:.4%} over IVW by")
print(f"  penalising B+C correlation (ρ_BC={rho_B[1,2]:.2f}).")


# ===========================================================================
# CASE C — 10 assets, PCA selects top 4, GMV on subset
# ===========================================================================
print(DIVIDER)
print("  CASE C: 10-asset universe, PCA selection of top 4 → GMV")
print("  Expected result: top 4 = highest-vol (Growth) group,")
print("  because they score highest on systematic risk factors.")
print(DIVIDER)

# Universe: 3 groups
#   Growth    (A1-A4): high vol, within-group ρ=0.70, cross-group ρ=0.15
#   Value     (B1-B3): mid vol
#   Defensive (C1-C3): low vol
asset_names = [
    "Growth-1", "Growth-2", "Growth-3", "Growth-4",
    "Value-1",  "Value-2",  "Value-3",
    "Def-1",    "Def-2",    "Def-3",
]
N = len(asset_names)

vols_C = np.array([
    0.030, 0.035, 0.028, 0.040,   # Growth: 3–4%
    0.018, 0.022, 0.015,           # Value:  1.5–2.2%
    0.008, 0.010, 0.012,           # Defensive: 0.8–1.2%
])

rho_C = np.full((N, N), 0.15)
np.fill_diagonal(rho_C, 1.0)
for g in [range(0, 4), range(4, 7), range(7, 10)]:
    for i in g:
        for j in g:
            if i != j:
                rho_C[i, j] = 0.70

Sigma_C = rho_C * np.outer(vols_C, vols_C)

print(f"\n  Universe: {N} assets  |  3 groups  |  within-ρ=0.70, cross-ρ=0.15")
print(f"  Vols (% daily): {(vols_C*100).round(1)}")

# Eigendecompose
eigvals_C, eigvecs_C = eigen_decompose(Sigma_C)
expl_C     = eigvals_C / eigvals_C.sum()
cum_expl_C = np.cumsum(expl_C)
k = int(np.argmax(cum_expl_C >= 0.95)) + 1

print(f"\n  Explained variance (top 6 PCs): {expl_C[:6].round(3)}")
print(f"  Cumulative        (top 6 PCs): {cum_expl_C[:6].round(3)}")
print(f"  → k={k} PCs selected to explain ≥95% of variance")

# Score every asset
scores_C = pca_asset_scores(eigvecs_C, eigvals_C, k)
print(f"\n  PCA scores (systematic risk exposure per asset):")
max_score = scores_C.max()
for name, score in zip(asset_names, scores_C):
    bar = "█" * int(score / max_score * 24)
    print(f"    {name:<12}: {score:.5f}  {bar}")

# Select top 4
n_select = 4
sel_idx   = np.argsort(scores_C)[-n_select:][::-1]
sel_names = [asset_names[i] for i in sel_idx]
print(f"\n  → Top {n_select} selected: {sel_names}")
print(f"    (All from Growth group — highest vol = highest factor exposure)")

# GMV on selected subset
Sigma_sel = Sigma_C[np.ix_(sel_idx, sel_idx)]
w_C       = gmv_weights(Sigma_sel)

print()
print_weights(w_C, sel_names, Sigma_sel, Sigma_eq_ref=Sigma_C)
print_factor_risk(w_C, Sigma_sel)

print(f"""
  Key insight — PCA selection ≠ low-vol selection:
  PCA scores rank assets by systematic risk exposure, not by volatility.
  High-vol Growth assets load heavily on PC1 (market) → high scores.
  Low-vol Defensive assets are idiosyncratic → low scores, not selected.

  Consequence: the selected portfolio of 4 Growth stocks has HIGHER vol
  than equal-weighting all 10 (which includes defensive names).

  This is intentional — PCA selection captures the dominant risk factors
  of the universe. If the goal were pure vol minimisation, you would
  instead select the 4 lowest-vol assets (the Defensive group).

  PCA selection is best used for:
    ✓ Factor-representative sub-portfolio construction
    ✓ Stat arb (trade the residuals of the selected assets)
    ✓ Risk model building (represent the universe with fewer names)
  Not for:
    ✗ Pure minimum-variance investing → use all assets or select by vol
""")


# ===========================================================================
# CASE D — Constrained PCA: 2 Growth + 1 Value + 1 Defensive
# ===========================================================================
print(f"\n{DIVIDER}")
print("  CASE D: Constrained PCA — diversified cross-group selection")
print("  Rule: top-2 Growth + top-1 Value + top-1 Defensive by PCA score")
print("  Expected result: Def-3 dominates (~78%), Growth nearly zeroed out")
print(DIVIDER)

# Reuse Sigma_C and scores_C from Case C above
groups = {
    "Growth":    list(range(0, 4)),
    "Value":     list(range(4, 7)),
    "Defensive": list(range(7, 10)),
}
quota = {"Growth": 2, "Value": 1, "Defensive": 1}

div_idx = []
for group_name, members in groups.items():
    top_n = sorted(members, key=lambda i: scores_C[i], reverse=True)[:quota[group_name]]
    div_idx.extend(top_n)

div_names = [asset_names[i] for i in div_idx]
print(f"\n  Selected assets (by group quota):")
for name, idx in zip(div_names, div_idx):
    group = next(g for g, m in groups.items() if idx in m)
    print(f"    {name:<12}  group={group:<12}  score={scores_C[idx]:.6f}"
          f"  vol={vols_C[idx]*100:.1f}%")

Sigma_div = Sigma_C[np.ix_(div_idx, div_idx)]
w_div     = gmv_weights(Sigma_div)

print(f"\n  Sub-covariance (4×4):")
header = f"  {'':14}" + "".join(f"  {n:>12}" for n in div_names)
print(header)
for i, row in enumerate(Sigma_div):
    print(f"  {div_names[i]:<14}" + "".join(f"  {v:>12.6f}" for v in row))

# Precision matrix breakdown
inv_S_div = np.linalg.inv(Sigma_div)
row_sums_div = inv_S_div.sum(axis=1)
total_div    = row_sums_div.sum()

print(f"\n  Precision matrix row sums → weights:")
print(f"  {'Asset':<14}  {'σ (daily)':>10}  {'Row sum':>10}  {'Weight':>10}  {'Drain %':>8}")
for name, idx, rs, wi in zip(div_names, div_idx, row_sums_div, w_div):
    diag  = inv_S_div[div_names.index(name), div_names.index(name)]
    drain = (rs - diag) / diag * 100
    print(f"  {name:<14}  {vols_C[idx]*100:>9.1f}%  {rs:>10.1f}  "
          f"{wi:>9.1%}  {drain:>+8.1f}%")

port_var_div = float(w_div @ Sigma_div @ w_div)
eq_w_all10   = np.ones(N) / N
eq_var_all10 = float(eq_w_all10 @ Sigma_C @ eq_w_all10)
eq_w_4       = np.ones(4) / 4
eq_var_4     = float(eq_w_4 @ Sigma_div @ eq_w_4)

print(f"\n  Portfolio vols:")
print(f"    GMV diversified 4     : {port_var_div**0.5:.4%}")
print(f"    Equal-wt these 4      : {eq_var_4**0.5:.4%}")
print(f"    Equal-wt all 10       : {eq_var_all10**0.5:.4%}")
print(f"    Case C (PCA top-4 GMV): {(w_C @ Sigma_sel @ w_C)**0.5:.4%}")

# Factor risk
eigvals_d, eigvecs_d = eigen_decompose(Sigma_div)
z_d        = eigvecs_d.T @ w_div
risk_d     = eigvals_d * z_d**2
print(f"\n  Factor risk breakdown:")
for j, (lam, zj, rc) in enumerate(zip(eigvals_d, z_d, risk_d)):
    pct = rc / port_var_div * 100
    if pct > 0.5:
        print(f"    PC{j+1}: {pct:5.1f}%")

print(f"""
  Why Def-3 dominates at ~78%:
    1. Variance: σ²(Def-3)=0.000144 vs σ²(Growth-4)=0.001600 → 11× difference
    2. Correlation: Def-3 has ρ=0.15 with Growth/Value → tiny off-diagonal drain
    3. Growth-4 vs Growth-2 are ρ=0.70 correlated → GMV nearly zeros Growth-4
       (redundant given Growth-2 is present and has lower vol)

  Result: diversified selection gives {eq_var_all10**0.5 - port_var_div**0.5:.3%} vol saving
  vs equal-weighting all 10, while maintaining cross-sector exposure.
  Compare to Case C (all-Growth GMV) which was {(w_C@Sigma_sel@w_C)**0.5 - eq_var_all10**0.5:.3%} ABOVE equal-weight.
""")


# ===========================================================================
# Final summary table — all four cases
# ===========================================================================
print(DIVIDER)
print("  SUMMARY — All Four Cases")
print(DIVIDER)
print(f"  {'Case':<42} {'GMV vol':>8}  {'EQ vol (ref)':>13}  {'Diff':>8}")
print(f"  {'-'*42} {'-'*8}  {'-'*13}  {'-'*8}")

cases_summary = [
    ("A: Equal vol, high corr  (3 assets)",    Sigma_A,   Sigma_A,   "same 3"),
    ("B: Diff vol,  low corr   (3 assets)",    Sigma_B,   Sigma_B,   "same 3"),
    ("C: PCA top-4 Growth only (GMV on 4)",    Sigma_sel, Sigma_C,   "all 10"),
    ("D: Constrained PCA 2G+1V+1D (GMV on 4)",Sigma_div, Sigma_C,   "all 10"),
]
for label, Sigma_opt, Sigma_ref, ref_label in cases_summary:
    w     = gmv_weights(Sigma_opt)
    gvol  = (w @ Sigma_opt @ w) ** 0.5
    n_ref = Sigma_ref.shape[0]
    eq_w  = np.ones(n_ref) / n_ref
    eqvol = (eq_w @ Sigma_ref @ eq_w) ** 0.5
    diff  = eqvol - gvol
    note  = "↓ better" if diff > 0 else "↑ higher"
    print(f"  {label:<42} {gvol:>7.3%}   {eqvol:>7.3%} ({ref_label})  "
          f"{diff:>+7.3%}  {note}")

print()
