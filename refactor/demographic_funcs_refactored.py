# demographic_funcs.py

import numpy as np
from numba import njit

@njit
def reproduction(n_lag, Nlag_sum, w_rho_da, b, alpha, tau_idx):
    # """
    # Nonlocal birth term:
    #   ∫ rho(a)*n_lag(a)*max(1-alpha1*n_lag(a),0) da
    # Returns a full vector with that same integral.
    # """
    # birth_comp = np.maximum(1.0 - alpha * Nlag_sum[tau_idx:], 0.0)
    # # Only integrate from age-index tau_idx_i to end
    # integral = b*np.trapz(rho[tau_idx:] * n_lag[tau_idx:] * birth_comp, a[tau_idx:])
    # return integral

    s = 0.0
    # integrate from the delay index to the end
    for j in range(tau_idx, n_lag.shape[0]):
        # scalar v = 1 - α·Nlag_sum[j]
        v     = 1.0 - alpha * Nlag_sum[j]
        # branchless positive part: max(v,0) = 0.5*(v + |v|)
        v_pos = 0.5 * (v + abs(v))
        # accumulate weighted
        s    += w_rho_da[j] * n_lag[j] * v_pos
    return b * s

#@njit
#def death(n, mu):
    """
    Death rate mu applied uniformly across ages.
    """
#    return mu * np.ones_like(n)

@njit
def flux(u):
    N = u.shape[0]
    F = np.empty_like(u)
    # copy boundaries
    F[0], F[1], F[-2], F[-1] = u[0], u[1], u[-2], u[-1]
    # interior via one pass
    for j in range(2, N-2):
        dp = u[j+1] - u[j]
        dm = u[j]   - u[j-1]
        mm = 0.5*(np.sign(dp) + np.sign(dm)) * min(abs(dp), abs(dm))
        F[j] = u[j] + 0.5*mm
    return F
