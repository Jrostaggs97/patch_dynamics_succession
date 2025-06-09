# multispec_config.py

import numpy as np

# ─── Grid & parameters ────────────────────────────────────────────────────────
k = 2                   # number of species
Na = 2000              # number of age grid cells
amax = 20.0            # maximum age
tmax = 5.0            # maximum integration time
birth_rate = [2.0,1.4]          # recruitment rate
death_rate = [.5,.5]      # adult mortality rate
tau = [1,.5]            # time to first reproduction (delay)
alpha = [.001,.001]         # intraspecies competition coefficient
gamma = 0.5            # disturbance rate
  

# ─── Derived quantities ───────────────────────────────────────────────────────
da = amax / Na
a = np.linspace(0, amax, Na + 1)     # age grid (Na+1 points)
tau_idx = [int(t / da) for t in tau]              # delay index (in grid steps)
rho = gamma * np.exp(-gamma * a)     # age‐density function

def init_history(a_grid):
    """
    History n(a,t) for t <= 0:
      - Gaussian‐like initial shape for a > tau
      - Zero for a <= tau (no adults before delay)
      
      - NEED TO MAKE IT EASIER TO UPDATE INITIAL CONDITIONS FOR K SPECIES
      -- first pass would be to do a random shift and scale for gaussians for each species
    """
    base = (1/np.sqrt(2 * np.pi)) * np.exp(-0.5 * (a_grid - 3)**4)
    hist_list = []
    for i, T in enumerate(tau):
        n0 = base.copy()
        n0[a_grid <= T] = 0.0
        hist_list.append(n0)
    # Now hist_list is a list of length k, each of shape (Na+1,)


    return np.concatenate(hist_list)