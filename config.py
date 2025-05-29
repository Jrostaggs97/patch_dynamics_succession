# config.py

import numpy as np

# ─── Grid & parameters ────────────────────────────────────────────────────────
Na = 2000              # number of age grid cells
amax = 20.0            # maximum age
tmax = 2.0            # maximum integration time
birth = 2.0          # recruitment rate
death_rate = 1.0      # adult mortality rate
tau = 10              # time to first reproduction (delay)
gamma = 0.5            # disturbance rate
alpha1 = .001          # intraspecies competition coefficient

# ─── Derived quantities ───────────────────────────────────────────────────────
da = amax / Na
a = np.linspace(0, amax, Na + 1)     # age grid (Na+1 points)
tau_idx = int(tau / da)              # delay index (in grid steps)
rho = gamma * np.exp(-gamma * a)     # age‐density function

def init_history(a_grid):
    """
    History n(a,t) for t <= 0:
      - Gaussian‐like initial shape for a > tau
      - Zero for a <= tau (no adults before delay)
    """
    n0 = (1/np.sqrt(2 * np.pi)) * np.exp(-0.5 * (a_grid - 3)**4)
    n0[a_grid <= tau] = 0.0
    return n0
