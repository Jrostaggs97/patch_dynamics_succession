#multispec_competition_dynamics.py

# %% imports
import time
import numpy as np
import matplotlib.pyplot as plt
import numba
import os
import json
from numba import njit, prange
from datetime import datetime





from my_ddeint.ddeint import ddeint #if you want to customize (e.g. use a stiff solver)

#from ddeint import ddeint
# ensure: pip install ddeint


from multispec_config_refactored import a, da, rho, tau_idx, tmax, tau, birth_rate, death_rate, alpha, gamma, init_history, generate_initial_profiles, k, Na
from demographic_funcs_refactored import reproduction, flux #,death
from analytic_funcs import one_spec_analytic_eq, one_spec_analytic_total_density_eq, two_spec_analytic_eq, two_spec_analytica_totaldensity_eq


#for notational convenience
b = birth_rate
mu = death_rate

##-------------------------- testing--------------------------##
k = 1
b = [3]
mu = [1]
tau = [.75]
tau_idx = [int(tau[0]/da)]
##------------------------------------------------------------##

# create initial condition
profiles = generate_initial_profiles(a, k)    # list of k arrays
initial_flat = np.concatenate(profiles)        # shape (k*(Na+1),)

# Global buffers for reuse
lagged_n_i_buffer = np.empty((k, Na+1))
S_i_arr_buffer    = np.empty((k, Na+1))


# %% Multi species RHS func, history func, and solver function


print("Numba is set to use", numba.config.NUMBA_NUM_THREADS, "threads by default.")


@njit(parallel=False)
def compute_rhs_numba(y_now, lagged_n_i, S_i_arr, a, da, rho, b, alpha, tau_idx, mu):
    k, Na1 = y_now.shape
    dydt = np.zeros_like(y_now)

    for i in prange(k):
        n_i_lag = lagged_n_i[i, :]   # (Na+1,)
        S_i     = S_i_arr[i, :]      # (Na+1,)

        #Death_i = death(y_now[i], mu[i])
        Flux_i  = flux(y_now[i])

        Repo_i = reproduction(n_i_lag, S_i, w_rho_da, b[i], alpha[i], tau_idx[i])

        ddt = np.empty(Na1)
        ddt[:] = 0.0
        ddt[1:tau_idx[i]] = -(Flux_i[2:tau_idx[i]+1] - Flux_i[1:tau_idx[i]]) / da
        adv = -(Flux_i[tau_idx[i]+1:] - Flux_i[tau_idx[i]:-1]) / da
        ddt[tau_idx[i]+1:] = Repo_i \
                             - mu[i]* y_now[i][tau_idx[i]+1:] \
                             + adv
        ddt[0] = 0
        dydt[i] = ddt
    return dydt


def rhs_multi(Y, t):
    y_now = Y(t).reshape(k, Na+1) 

    for i in range(k):
        lagged =  Y(t - tau[i]).reshape(k, Na+1) 
        lagged_n_i_buffer[i, :] = lagged[i, :]

        S_i_arr_buffer[i, :].fill(0.0)
        for j in range(k):
            S_i_arr_buffer[i, :] += lagged[j, :]


    dydt = compute_rhs_numba(y_now, lagged_n_i_buffer, S_i_arr_buffer, a, da, rho, b, alpha, tau_idx, mu)
    return dydt.reshape(k*(Na+1))

    #return dydt.reshape(k*(Na+1))

@njit
def history_multi(t):
    # Return the same precomputed initial condition for all t ≤ 0
    return initial_flat


# Analytic equilibrium
n_eq = one_spec_analytic_eq(tau[0], b[0], mu[0], alpha[0], gamma, a)
N_eq = one_spec_analytic_total_density_eq(n_eq, a, gamma)


#unique id for saving files
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")



# %% Call solver
t = np.arange(0, tmax, .001)

# For birth function 
# a: your age grid (length Na+1), rho: kernel array, da = a[1]-a[0]
trapz_w     = np.ones_like(a)
trapz_w[0]  = 0.5
trapz_w[-1] = 0.5
w_rho_da    = trapz_w * rho * da    # shape (Na+1,)


start_solve  = time.perf_counter()

multi_sol = ddeint(rhs_multi, history_multi, t)

end_solve = time.perf_counter()
print(f"[Timing] DDE solve time: {end_solve - start_solve:.3f} s")

# sol has shape (len(t), k*(Na+1))
# you can reshape each row:
multi_sol_matrix = multi_sol.reshape(len(t), k, Na+1)

# multi_sol_matrix[t_idx, i, :] is exactly the age‐profile of species i at time index t_idx.

# %% compute quantities for plotting/other 

# Reshape for plotting
Na1 = len(a)                  # Na+1
sol = multi_sol.reshape(len(t), k, Na1)
# Compute N_t with shape (Nt, k)
N_t = np.trapezoid(rho * sol, x=a, axis=2)



# %%Plotting
# — Plot total abundances N_i(t) for each species —

plt.figure(figsize=(7, 4))
for i in range(k):
    plt.plot(t, N_t[:, i], label=f"Species {i+1}")

plt.plot(t,N_eq*np.ones(len(t)), '--', label='Analytic $N_{eq}$')#only for one spec
plt.xlabel("time $t$")
plt.ylabel("Total abundance $N_i(t)$")
plt.legend(loc="best")
plt.title("Total abundance vs. time")
plt.tight_layout()
plt.savefig("debug_total_abundance.png")


# — Plot final age‐distribution n_i(a, t_max) for each species —

plt.figure(figsize=(7, 4))
for i in range(k):
    n_final_i = sol[-1, i, :]     # sol has shape (Nt, k, Na1)
    plt.plot(a, n_final_i, label=f"Species {i+1}")
plt.plot(a, n_eq, '--', label='Analytic $n_{eq}(a)$') 
plt.xlabel("age $a$")
plt.ylabel(r"$n_i(a, t_{\max})$")
plt.legend(loc="best")
plt.title("Age distribution at final time")
plt.tight_layout()
plt.savefig("debug_age_dist.png")

error = abs(n_final_i - n_eq)
plt.figure()
plt.plot(a, error, label='Error $n_{approx}-n_{eq}$')
plt.xlabel('age $a$')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.savefig("ptwise_error_age_dist.png")

#put results into outputs directory
# output_dir = os.path.join("outputs", run_id)
# os.makedirs(output_dir, exist_ok=True)

# # Gather parameters into a dict
# params = {
#     "tau": [float(x) for x in tau],
#     "birth_rate": [float(x) for x in birth_rate],
#     "death_rate": [float(x) for x in death_rate],
#     "gamma": float(gamma),
#     "alpha": [float(x) for x in alpha]
# }
# # Write to outputs/<run_id>/params.json
# with open(os.path.join(output_dir, "params.json"), "w") as f:
#     json.dump(params, f, indent=2)


# np.savez(
#     os.path.join(output_dir, "results.npz"),
#     multi_sol_matrix=multi_sol_matrix,
#     N_t=N_t,
#     t=t,
#     a=a
# )   

#read data back in with 
#data = np.load("outputs/<run_id>/results.npz")
#multi_sol_matrix = data["multi_sol_matrix"]
#N_t              = data["N_t"]
#t                = data["t"]
#a                = data["a"]