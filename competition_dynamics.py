# %% 
# # Patch Dynamics DDE Solver


# https://pypi.org/project/ddeint/

# %% imports and configuration
import numpy as np
import matplotlib.pyplot as plt

# ensure: pip install ddeint
from ddeint import ddeint
from config import a, da, rho, tau, tau_idx, tmax, birth, death_rate, alpha1, init_history
from demographic_funcs import reproduction, death, flux

# %% Define RHS, history function, and DDE solver
def rhs(Y, t):
    """
    RHS for delay‐PDE:
      n_t = -∂_a flux + reproduction - death
    Y: history+solution function; Y(t) gives n(a) at time t.
    """
    y     = Y(t)
    y_lag = Y(t - tau)

    F    = flux(y)
    Repo = reproduction(a, y_lag, rho, birth, alpha1)
    Death= death(y, death_rate)

    dndt = np.zeros_like(y)

    # ages 0 < a < tau: pure advection
    dndt[1:tau_idx] = -(F[2:tau_idx+1] - F[1:tau_idx]) / da

    # ages a >= tau: advection + birth - death
    adv = -(F[tau_idx+1:] - F[tau_idx:-1]) / da
    dndt[tau_idx+1:] = Repo[tau_idx+1:] - Death[tau_idx+1:]*y[tau_idx+1:] + adv

    # boundary
    dndt[0] = 0.0

    return dndt


def history_function(time):
    # time is ignored, we always return the same initial vector
    return init_history(a)




def solve_dde():
    # match MATLAB t_span = 0:0.001:tmax
    t = np.arange(0.0, tmax + 1e-8, 0.001)
    sol = ddeint(rhs, history_function, t)
    return t, sol


# %% Solve and plot
t = np.arange(0, tmax +1e-8,.001)

#Call DDE solver 
sol = ddeint(rhs, history_function,t)

# Final (calculated) age distribution
n_eq = sol[-1,:]

#Total abundance 
N_t = np.trapezoid(rho*sol,a,axis =1)

#plot total stable age distribution
plt.figure()
plt.plot(a, n_eq, label="Numeric n(a, t_final)")
plt.xlabel("age a")
plt.ylabel("n(a)")
plt.legend()
plt.show()

#plot total abundance
plt.figure()
plt.plot(t, N_t)
plt.xlabel("time")
plt.ylabel("N(t)")
plt.title("Total abundance over time")
plt.show()

# %% Guard
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, sol = solve_dde()
    # Total abundance N(t) = ∫ rho(a) n(a,t) da
    N_t = np.trapezoid(rho * sol, a, axis=1)

    plt.plot(t, N_t, label="N(t)")
    plt.xlabel("time")
    plt.ylabel("Total abundance")
    plt.legend()
    plt.tight_layout()


    # 2) plot final n(a, t_final) vs analytic equilibrium —
    n_final = sol[-1, :]   # last row is n(a) at t_max
    plt.figure()
    plt.plot(a, n_final, label="n(a, t_final)")
    plt.xlabel("age a")
    plt.ylabel("n(a)")
    plt.legend()
    plt.tight_layout()
    plt.show()