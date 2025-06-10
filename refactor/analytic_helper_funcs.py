# File to hold onto some functions? used in analytical analysis of coexistence
# Keeping in file for cleanliness

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def coex_check(k, b_vec, mu_vec, tau_vec, alpha_vec, gamma):
    # we restrict to the two species case 
    k=2
    da = 0.005
    a_grid = np.arange(0, 1000 + da, da)
    n_tilde_mat = np.zeros((len(a_grid), k))
    rho = gamma * np.exp(-gamma * a_grid)  # shape: (len(a_grid),)
    tau_idx_vec = (tau_vec / da).astype(int)

    interaction_mat = np.zeros((k, k))
    persist_vec = np.zeros(k)
    persist_check = np.zeros(k)

    for i in range(k):
        n_tilde = (b_vec[i] / mu_vec[i]) * (1 - np.exp(-mu_vec[i] * (a_grid - tau_vec[i])))
        n_tilde[:tau_idx_vec[i]] = 0
        n_tilde_mat[:, i] = n_tilde
        persist_vec[i] = (1 / alpha_vec[i]) * (np.trapz(rho * n_tilde, a_grid) - 1)
        persist_check[i] = (1 / alpha_vec[i]) * ((b_vec[i] * np.exp(-gamma * tau_vec[i])) / (gamma + mu_vec[i]) - 1)

    for i in range(k):
        interaction_mat[i, i] = np.trapz(rho * n_tilde_mat[:, i] * n_tilde_mat[:, i], a_grid)
        for j in range(i + 1, k):
            interaction = np.trapz(rho * n_tilde_mat[:, i] * n_tilde_mat[:, j], a_grid)
            interaction_mat[i, j] = interaction
            interaction_mat[j, i] = interaction

    S_vec = np.zeros((2,1))
    S_vec[1] = alpha_vec[1]*(persist_vec[1]*interaction_mat[2,2]* - persist_vec[2]*interaction_mat[1,2])
    S_vec[2] = alpha_vec[2]*(persist_vec[2]*interaction_mat[1,1]* - persist_vec[1]*interaction_mat[2,1])

    return S_vec



## ------- need to actually turn the following into functions ------- ## 
# can probaobly turn the heat map coexstience code intoa function as well 
# Parameters
alpha = 1
mu = 20
gamma = 0.6

tau_j = 0.6
b_j = 60


def generate_coex_mat(trait_1, trait_2,):
    # again I think we want to pass in like the cohorted table basically
    # Ranges for tau_i and b_i

    #Want to grab column corresponding to focal trait 1 and focal trait 2 from our data table ---- no no. We need to just take in the vectors

    tau_i_vec = np.arange(tau_j - 0.01, tau_j + 1.001, 0.001)
    b_i_vec = np.arange(b_j - 0.05, b_j + 30.001, 0.001)
    tau_i_grid, b_i_grid = np.meshgrid(tau_i_vec, b_i_vec)

    # Persistences
    P_i = b_i_grid * np.exp(-gamma * tau_i_grid) / (gamma + mu) - 1
    P_j = b_j * np.exp(-gamma * tau_j) / (gamma + mu) - 1

    # Intraspecific competition
    A_i = alpha * 2 * (b_i_grid**2) * np.exp(-gamma * tau_i_grid) / ((gamma + mu) * (gamma + 2 * mu))
    A_j = alpha * 2 * (b_j**2) * np.exp(-gamma * tau_j) / ((gamma + mu) * (gamma + 2 * mu))

    # Interspecific competition
    B1 = np.exp(mu * tau_i_grid) / ((gamma + mu) * mu)
    B2 = gamma * np.exp(mu * tau_j) / (mu * (gamma + mu) * (gamma + 2 * mu))
    B_coeff = b_j * b_i_grid * np.exp(-(gamma + mu) * tau_i_grid)
    B = alpha * B_coeff * (B1 - B2)

    # Feasibility conditions
    S_ij = P_i * A_j - B * P_j
    S_ji = P_j * A_i - B * P_i

    # Region codes
    S_ij = (3/4) * (S_ij > 0).astype(float)
    S_ji = (1/2) * (S_ji > 0).astype(float)
    Coex = S_ij + S_ji

    # Anchor for colormap scale
    Coex[-1, -1] = 0

    return Coex



#### --- Plotting the heatmap --- ####

def coexistence_region_plotter(x_vec, y_vec, param_dict, x_label, y_label, title):
    #want an input for the alternates
    # like pass in a parameter dictionary that has the label and the value 
    # or we pass in like a "full table" with first row as headers then the numeric values
    # and then you specify which column is x and y? and all other columns are "fixed" so just 
    # grab the first numeric value. 

    #Basically pass in the cohorted table and the coex mat
    
    # Custom colormap
    cmap = ListedColormap([
        [1, 1, 1],   # Neither wins
        [0, 0, 1],   # Spec j wins
        [1, 0, 0],   # Spec i wins
        [0, 0, 0]    # Coexist
    ])

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(Coex, extent=[tau_i_vec[0], tau_i_vec[1], b_i_vec[0], b_i_vec[1]],
            origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
    plt.axhline(b_j, color='k', linestyle='--')
    plt.axvline(tau_j, color='k', linestyle='--')

    # Labels and title
    plt.xlabel(r"$\tau_i$", fontsize=20)
    plt.ylabel(r"$b_i$", fontsize=20)
    plt.title(fr"$b - \tau$ trade off feasibility plot for $b_j={b_j}, \tau_j={tau_j}$" +
            fr" $(\alpha, \mu, \gamma) = ({alpha}, {mu}, {gamma})$", fontsize=18)

    # Custom legend
    legend_patches = [
        mpatches.Patch(color='white', label='Neither'),
        mpatches.Patch(color='blue', label='Spec. j wins'),
        mpatches.Patch(color='red', label='Spec. i wins'),
        mpatches.Patch(color='black', label='Coex.')
    ]
    plt.legend(handles=legend_patches, title="Outcome", loc='upper right')

    plt.tight_layout()
    plt.show()


# Parameters
b = 55
alpha = 1
gamma = 0.6

tau_j = 0.7
mu_j = 25.0

# Vary tau_i and mu_i
tau_i_vec = np.arange(tau_j - 0.025, tau_j + 2.001, 0.001)
mu_i_vec = np.arange(mu_j - 20, mu_j + 0.501, 0.001)
tau_i_grid, mu_i_grid = np.meshgrid(tau_i_vec, mu_i_vec)

# Compute feasibility terms
P_i = b * np.exp(-gamma * tau_i_grid) / (gamma + mu_i_grid) - 1
P_j = b * np.exp(-gamma * tau_j) / (gamma + mu_j) - 1

A_i = alpha * 2 * (b ** 2) * np.exp(-gamma * tau_i_grid) / ((gamma + mu_i_grid) * (gamma + 2 * mu_i_grid))
A_j = alpha * 2 * (b ** 2) * np.exp(-gamma * tau_j) / ((gamma + mu_j) * (gamma + 2 * mu_j))

B1 = np.exp(mu_j * tau_i_grid) / ((gamma + mu_i_grid) * mu_j)
B2 = gamma * np.exp(mu_j * tau_j) / (mu_j * (gamma + mu_j) * (gamma + mu_j + mu_i_grid))
B_coeff = (b ** 2) * np.exp(-(gamma + mu_j) * tau_i_grid)
B = alpha * B_coeff * (B1 - B2)

# Invasion feasibility conditions
S_ij = P_i * A_j - B * P_j
S_ji = P_j * A_i - B * P_i

S_ij = (3/4) * (S_ij > 0).astype(float)
S_ji = (1/2) * (S_ji > 0).astype(float)
Coex = S_ij + S_ji

# Anchor the color scale
Coex[-1, -1] = 0
# Coex[0, 0] = 2  # Optional

# Colormap definition
cmap = ListedColormap([
    [1, 1, 1],   # Neither wins
    [0, 0, 1],   # Spec j wins
    [1, 0, 0],   # Spec i wins
    [0, 0, 0]    # Coexist
])

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(Coex, extent=[tau_i_vec[0], tau_i_vec[-1], mu_i_vec[0], mu_i_vec[-1]],
           origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
plt.axhline(mu_j, color='k', linestyle='--')
plt.axvline(tau_j, color='k', linestyle='--')

# Labels and title
plt.xlabel(r"$\tau_i$", fontsize=20)
plt.ylabel(r"$\mu_i$", fontsize=20)
plt.title(fr"$\mu - \tau$ trade off feasibility plot for $\mu_j={mu_j}, \tau_j={tau_j}$" +
          fr" $(b, \alpha, \gamma) = ({b}, {alpha}, {gamma})$", fontsize=18)

# Legend
legend_patches = [
    mpatches.Patch(color='white', label='Neither'),
    mpatches.Patch(color='blue', label='Spec. j wins'),
    mpatches.Patch(color='red', label='Spec. i wins'),
    mpatches.Patch(color='black', label='Coex.')
]
plt.legend(handles=legend_patches, title="Outcome", loc='upper right')

plt.tight_layout()
plt.show()