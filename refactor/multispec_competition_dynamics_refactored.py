#multispec_competition_dynamics.py

# %% imports
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from ddeint import ddeint
# ensure: pip install ddeint

from multispec_config_refactored import a, da, rho, tau_idx, tmax, tau, birth_rate, death_rate, alpha, init_history, k, Na
from demographic_funcs_refactored import reproduction, death, flux

b = birth_rate
mu = death_rate

# %% Multi species RHS func, history func, and solver function




@njit(parallel=True)
def compute_rhs_numba(y_now, lagged_n_i, S_i_arr, a, da, rho, birth_rate, alpha, tau_idx, death_rate):
    k, Na1 = y_now.shape
    dydt = np.zeros_like(y_now)
    for i in prange(k):
        n_i_lag = lagged_n_i[i, :]   # (Na+1,)
        S_i     = S_i_arr[i, :]      # (Na+1,)

        Death_i = death(y_now[i], death_rate[i])
        Flux_i  = flux(y_now[i])

        Repo_i = reproduction(a, n_i_lag, S_i, rho, birth_rate[i], alpha[i], tau_idx[i])

        ddt = np.zeros(Na1)
        ddt[1:tau_idx[i]] = -(Flux_i[2:tau_idx[i]+1] - Flux_i[1:tau_idx[i]]) / da
        adv = -(Flux_i[tau_idx[i]+1:] - Flux_i[tau_idx[i]:-1]) / da
        ddt[tau_idx[i]+1:] = Repo_i[tau_idx[i]+1:] \
                             - Death_i[tau_idx[i]+1:] * y_now[i][tau_idx[i]+1:] \
                             + adv
        ddt[0] = 0
        dydt[i] = ddt
    return dydt

def rhs_multi(Y, t):
    y_now = np.reshape(Y(t), (k, Na+1))
    # Precompute, for each species i, all n_j(a, t-tau_i)
    lagged_states = []
    lagged_n_i    = []
    S_i_list      = []
    for i in range(k):
        lagged = np.reshape(Y(t - tau[i]), (k, Na+1))
        lagged_states.append(lagged)
        lagged_n_i.append(lagged[i, :])       # shape (Na+1,)
        S_i_list.append(np.sum(lagged, axis=0))  # shape (Na+1,)
    # Convert to arrays for Numba
    lagged_n_i = np.array(lagged_n_i)    # shape (k, Na+1)
    S_i_arr    = np.array(S_i_list)      # shape (k, Na+1)

    dydt = compute_rhs_numba(y_now, lagged_n_i, S_i_arr, ...)
    return dydt.reshape(k*(Na+1))


def history_multi(t):
    # build history function (values that functions take outside natural time domain. I.e. t<0)
    # Here we assume the history function is the initial value repeated
    # build initial n0 for each species (e.g., using init_history)
    # here we assume same init for all k
    base = init_history(a)        # shape (Na+1,)
    #stacked = np.vstack([base]*k) # shape (k, Na+1)
    #print("stacked.shape =", stacked.shape)   # debug print
    #print("stacked.size =", stacked.size)     # debug print
    #stacked.reshape(k*(Na+1))
    return base

# %% Call solver
t = np.arange(0, tmax, .001)
multi_sol = ddeint(rhs_multi, history_multi, t)
# sol has shape (len(t), k*(Na+1))
# you can reshape each row:
multi_sol_matrix = multi_sol.reshape(len(t), k, Na+1)

# multi_sol_matrix[t_idx, i, :] is exactly the age‐profile of species i at time index t_idx.

# %% compute quantities for plotting/other 

# Reshape for plotting
Na1 = len(a)                  # Na+1
sol = multi_sol.reshape(len(t), k, Na1)
# Compute N_t with shape (Nt, k)
N_t = np.trapezoid(rho[np.newaxis, np.newaxis, :] * sol, x=a, axis=2)

# %%Plotting
# — Plot total abundances N_i(t) for each species —

plt.figure(figsize=(7, 4))
for i in range(k):
    plt.plot(t, N_t[:, i], label=f"Species {i+1}")
plt.xlabel("time $t$")
plt.ylabel("Total abundance $N_i(t)$")
plt.legend(loc="best")
plt.title("Total abundance vs. time")
plt.tight_layout()
plt.show()


# — Plot final age‐distribution n_i(a, t_max) for each species —

plt.figure(figsize=(7, 4))
for i in range(k):
    n_final_i = sol[-1, i, :]     # sol has shape (Nt, k, Na1)
    plt.plot(a, n_final_i, label=f"Species {i+1}")
plt.xlabel("age $a$")
plt.ylabel(r"$n_i(a, t_{\max})$")
plt.legend(loc="best")
plt.title("Age distribution at final time")
plt.tight_layout()
plt.show()
