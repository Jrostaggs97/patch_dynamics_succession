#multispec_competition_dynamics.py

# %% imports
import numpy as np
import matplotlib.pyplot as plt

# ensure: pip install ddeint
from ddeint import ddeint
from multispec_config import k, a, da, Na, rho, tau, tau_idx, tmax, birth_rate, death_rate, alpha, init_history
from demographic_funcs import reproduction, death, flux

b = birth_rate
mu = death_rate

# %% Multi species RHS func, history func, and solver function
def rhs_multi(Y, t):
    # 1) pull back the flat state and reshape
    y     = np.reshape(Y(t),(k,Na+1))        # shape (k*(n+1),) -> (k,Na+1)

    # prepare storage
    dydt = np.zeros_like(y)

    # 2) common term: at each age, sum all species at lagged time
    #Nlag_sum = np.sum(y_lag, axis=0)   # shape (Na+1,)

    for i in range(k):
       
        y_lag = np.reshape( Y(t-tau[i]), (k,Na+1) )
        
        lag_sum = np.sum(y_lag)
        # 2a) death and flux as before
        Death_i = death(y[i], mu[i])            # vector length Na+1
        Flux_i  = flux(y[i])                    # vector length Na+1

        # 2b) nonlocal reproduction for species i
        #    ∫ rho(a) * nlag_i(a) * [1 - alpha_i * Σ_j nlag_j(a)] da
        birth_integral = np.trapezoid(
            rho * y_lag[i] * (1 - alpha[i] * lag_sum),
            a
        )
        Repo_i = b[i]*np.full_like(y[i], birth_integral)

        # 2c) build the advection + demo derivative
        #     same splitting at tau_idx
        ddt = np.zeros_like(y[i])
        # advection only for ages < tau
        ddt[1:tau_idx[i]] = -(Flux_i[2:tau_idx[i]+1] - Flux_i[1:tau_idx[i]]) / da
        # full dynamics for ages >= tau
        adv = -(Flux_i[tau_idx[i]+1:] - Flux_i[tau_idx[i]:-1]) / da
        ddt[tau_idx[i]+1:] = Repo_i[tau_idx[i]+1:] \
                          - Death_i[tau_idx[i]+1:]*y[i][tau_idx[i]+1:] \
                          + adv
        ddt[0] = 0.0

        dydt[i] = ddt

    # 3) flatten back to 1-D for the solver
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
