# demographic_funcs.py

import numpy as np
from numba import njit

@njit
def reproduction( a, n_lag, Nlag_sum, rho, b, alpha, tau_idx):
    """
    Nonlocal birth term:
      ∫ rho(a)*n_lag(a)*max(1-alpha1*n_lag(a),0) da
    Returns a full vector with that same integral.
    """
    birth_comp = np.maximum(1.0 - alpha * Nlag_sum[tau_idx:], 0.0)
    # Only integrate from age-index tau_idx_i to end
    integral = b*np.trapz(rho[tau_idx:] * n_lag[tau_idx:] * birth_comp, a[tau_idx:])
    return np.full(n_lag.shape, integral)

@njit
def death(n, mu):
    """
    Death rate mu applied uniformly across ages.
    """
    return mu * np.ones_like(n)

@njit
def flux(u):
    """
    Minmod flux‐limiter for age advection, vectorized interior.
    """
    N = u.shape[0]
    F = np.empty_like(u)

    # boundary stencils
    F[0] = u[0]
    F[1] = u[1]
    F[-2] = u[-2]
    F[-1] = u[-1]

    # compute diffs for j=2..N-3
    diff_p = u[3:N-1] - u[2:N-2]
    diff_m = u[2:N-2] - u[1:N-3]
    mm = 0.5 * (np.sign(diff_p) + np.sign(diff_m)) * np.minimum(np.abs(diff_p), np.abs(diff_m))

    # interior update
    F[2:N-2] = u[2:N-2] + 0.5 * mm

    return F
