

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multispec_config_refactored import (  # replace with actual module name if needed
    load_species_params,
    cohortize_species,
)
from analytic_helper_funcs import (
    coex_check,
    generate_coex_mat,
    coexistence_region_plotter
)

# --- Step 1: Read species data ---
file = "species_params.csv"
scale = 1.0  # or whatever scale you need
num_cohorts = 2

params = load_species_params(file, scale)
cohort_table = cohortize_species(params, num_cohorts)

# --- Step 2: Add alpha and gamma ---
cohort_table["alpha"] = 1.0        # species-specific alpha
cohort_table["gamma"] = 0.6        # same gamma across all cohorts

# --- Step 3: Define variable and fixed traits ---
trait_1_header = "long tau"
trait_2_header = "birth"
trait_1_max = cohort_table[trait_1_header].max() + 1.0
trait_2_max = cohort_table[trait_2_header].max() + 20.0

# Choose other fixed traits (must cover "tau", "birth", "death", "alpha")
# Suppose we're fixing "tau", "death", "alpha" here
alt_1_header, alt_1_val = "tau", cohort_table["tau"].iloc[0]
alt_2_header, alt_2_val = "death", cohort_table["death"].iloc[0]
alt_3_header, alt_3_val = "alpha", cohort_table["alpha"].iloc[0]

# --- Step 4: Generate the coexistence matrix ---
Coex = generate_coex_mat(
    spec_data=cohort_table,
    trait_1_header=trait_1_header,
    trait_2_header=trait_2_header,
    trait_1_max=trait_1_max,
    trait_2_max=trait_2_max,
    alt_1_header=alt_1_header,
    alt_1_val=alt_1_val,
    alt_2_header=alt_2_header,
    alt_2_val=alt_2_val,
    alt_3_header=alt_3_header,
    alt_3_val=alt_3_val
)

# Reconstruct trait vectors from generate_coex_mat input ----- This is dumb. Just have the function return them
trait_1_val = cohort_table[trait_1_header].min()
trait_2_val = cohort_table[trait_2_header].iloc[
    cohort_table[trait_1_header].idxmin()
]
eps1 = trait_1_val * 0.02
eps2 = trait_2_val * 0.02
trait_1_vec = np.arange(trait_1_val - .01, trait_1_max, eps1)
trait_2_vec = np.arange(trait_2_val - .01, trait_2_max, eps2)

# --- Step 5: Plot the result ---
coexistence_region_plotter(
    x_vec=trait_1_vec,
    y_vec=trait_2_vec,
    coex_mat=Coex,
    x_label=r"$\tau_i$",
    y_label=r"$b_i$",
    title=rf"Coexistence map: varying {trait_1_header} vs. {trait_2_header}",
    x_ref=trait_1_val,
    y_ref=trait_2_val
)
