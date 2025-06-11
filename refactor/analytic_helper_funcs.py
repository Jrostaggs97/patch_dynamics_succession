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
    tau_idx_vec = [int(t / da) for t in tau_vec]  

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
    S_vec[0] = alpha_vec[0]*(persist_vec[0]*interaction_mat[1,1]* - persist_vec[1]*interaction_mat[0,1])
    S_vec[1] = alpha_vec[1]*(persist_vec[1]*interaction_mat[0,0]* - persist_vec[0]*interaction_mat[1,0])

    return S_vec



## ------- need to actually turn the following into functions ------- ## 
# can probaobly turn the heat map coexstience code intoa function as well 
# Parameters



# I think what we want to do is make an updated species data table that has alpha and gamma 
# now you input the header string of the two traits you want to see the trade off for
# e.g. tau, birth
# We then access the minimum value/row for trait 1 header
# this is our trait_1_val

# spec_data is a table with a header column 
def generate_coex_mat(spec_data, trait_1_header, trait_2_header, trait_1_max, trait_2_max,
                      alt_1_header, alt_1_val, alt_2_header, alt_2_val, alt_3_header, alt_3_val):
    
    # go into table and find trait_1_header column then find minimum and that's trait 1 val
    # in that same row find the column associated with trait_2_header and that's trait 2 val

    # now we need to have a way to use the header info to assign the variables
    # like a header to variable association kind of thing. 

    #Get the species j (focal) row with the lowest value of trait_1
    trait_1_val = spec_data[trait_1_header].min()
    focal_row = spec_data[spec_data[trait_1_header] == trait_1_val].iloc[0]
    trait_2_val = focal_row[trait_2_header]

    gamma = focal_row["gamma"]

    # Grid for species i over the trait ranges
    trait_1_vec = np.arange(trait_1_val - .01, trait_1_max, trait_1_val*.02)
    trait_2_vec = np.arange(trait_2_val - .01, trait_2_max, trait_2_val*.02)
    trait_1_grid, trait_2_grid = np.meshgrid(trait_1_vec, trait_2_vec)

    #Store fixed values for species j in a dictionary
    trait_dict_j = {
        trait_1_header: trait_1_val,
        trait_2_header: trait_2_val,
        alt_1_header: alt_1_val,
        alt_2_header: alt_2_val,
        alt_3_header: alt_3_val,
        "gamma": gamma
    }

    # Intialize coexistence matrix ---- #have indexing/sizing problem around here
    Coex = np.zeros_like(trait_1_grid)


    # Yeah I think dispatching to coex_check will just be the easiest thing to do even though 
    # we are paying a for loop price (we can parallelize if needed)

    # We want to loop over the trait_1_vec and trait_2_vec and call coex_check with the correct assignment to 
    # b_vec, mu_vec, tau_vec, alpha_vec, gamma that go into coex_check we then want to fill out a Coexistence matrix which 
    # encodes for what trait_1 and trait_2 values yield coexistence S1 and S2>0, spec i wins S1>0 and S2<0, spec j wins S2<0 and S1>0, or neither wins both S1 and S2<0.
    for i in range(len(trait_1_vec)-1):
        for j in range(len(trait_2_vec)-1):
            # Step 1: Copy the fixed trait values from species j
            trait_dict_i = trait_dict_j.copy()

            # Step 2: Overwrite the two traits being varied
            trait_dict_i[trait_1_header] = trait_1_vec[i]
            trait_dict_i[trait_2_header] = trait_2_vec[j]

            # Step 3: Extract coex_check inputs from trait_dict_i and trait_dict_j
            b_vec = [trait_dict_i["birth"], trait_dict_j["birth"]]
            mu_vec = [trait_dict_i["death"], trait_dict_j["death"]]
            tau_vec = [trait_dict_i["tau"], trait_dict_j["tau"]]
            alpha_vec = [trait_dict_i["alpha"], trait_dict_j["alpha"]]
            gamma = trait_dict_j["gamma"]  # shared
    
            S_vec = coex_check(2, b_vec, mu_vec, tau_vec, alpha_vec, gamma)

            # Step 5: Encode outcome without if-statements
            s1 = S_vec[0] > 0
            s2 = S_vec[1] > 0
            Coex[i, j] = (3/4) * s1 + (1/2) * s2
    # Anchor for colormap scale
    Coex[-1, -1] = 0

    return Coex



#### --- Plotting the heatmap --- ####

def coexistence_region_plotter(x_vec, y_vec, coex_mat, x_label, y_label, title,
                               x_ref=None, y_ref=None):
    """
    Plots the coexistence region heatmap using a categorical colormap.

    Parameters:
    -----------
    x_vec : array
        Values along the x-axis (e.g., tau_i).
    y_vec : array
        Values along the y-axis (e.g., b_i).
    coex_mat : 2D array
        Matrix of encoded coexistence outcomes.
    x_label : str
        LaTeX-formatted string for x-axis label (e.g., r"$\\tau_i$").
    y_label : str
        LaTeX-formatted string for y-axis label.
    title : str
        LaTeX-formatted title string.
    x_ref : float, optional
        Optional vertical reference line (e.g., focal species trait value).
    y_ref : float, optional
        Optional horizontal reference line.
    """

    # Define custom colormap
    cmap = ListedColormap([
        [1, 1, 1],   # Neither wins
        [0, 0, 1],   # Species j wins
        [1, 0, 0],   # Species i wins
        [0, 0, 0]    # Coexistence
    ])

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(coex_mat, extent=[x_vec[0], x_vec[-1], y_vec[0], y_vec[-1]],
               origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)

    # Optional reference lines
    if x_ref is not None:
        plt.axvline(x_ref, color='k', linestyle='--')
    if y_ref is not None:
        plt.axhline(y_ref, color='k', linestyle='--')

    # Labels and title
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.title(title, fontsize=18)

    # Custom legend
    legend_patches = [
        mpatches.Patch(color='white', label='Neither'),
        mpatches.Patch(color='blue', label='Spec. j wins'),
        mpatches.Patch(color='red', label='Spec. i wins'),
        mpatches.Patch(color='black', label='Coex.')
    ]
    plt.legend(handles=legend_patches, title="Outcome", loc='upper right')

    plt.tight_layout()
    plt.savefig("coexistence_region_plot.png")