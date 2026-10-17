"""Phase 2 hardening: OLS with standard errors/p-values (not just point
estimates -- the Checkpoint 2 version only had coefficients, no way to tell
whether entropy "survives" controlling for the others), and bootstrap 95%
CIs over users for each entropy decile."""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def ols_with_pvalues(y: np.ndarray, x_cols: dict) -> dict:
    names = list(x_cols.keys())
    X = np.column_stack([np.ones(len(y))] + [x_cols[n] for n in names])
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(np.sum(resid ** 2) / dof) if dof > 0 else float("nan")
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0, None))
    t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stats), max(dof, 1)))
    ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")

    out = dict(n=n, r2=r2, adj_r2=adj_r2,
                intercept=dict(coef=float(beta[0]), se=float(se[0]), p=float(p_values[0])))
    for i, name in enumerate(names):
        out[name] = dict(coef=float(beta[i + 1]), se=float(se[i + 1]), p=float(p_values[i + 1]))
    return out


def bootstrap_decile_ci(values: np.ndarray, deciles: np.ndarray, n_boot: int = 2000,
                         seed: int = 0, ci: float = 0.95) -> dict:
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2
    out = {}
    for d in sorted(np.unique(deciles)):
        vals = values[deciles == d]
        n = len(vals)
        if n == 0:
            continue
        boots = np.array([vals[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
        out[int(d)] = dict(mean=float(vals.mean()), ci_lo=float(np.quantile(boots, alpha)),
                            ci_hi=float(np.quantile(boots, 1 - alpha)), n_users=n)
    return out
