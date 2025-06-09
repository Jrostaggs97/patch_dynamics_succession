# multispec_config.py

#Imports
import csv
import numpy as np
import pandas as pd
import random


#Read in CSV file with species information
file = "species_params.csv"

# File parsing function
def load_species_params(csv_path):
    """
    Reads species parameters from a CSV with header. Assumes:
      - First column: species name (ignored for numeric arrays, but could be stored)
      - Third column: tau (for sorting)
      - Subsequent columns: birth_rate, death_rate, alpha, etc.
    Returns (species_names, birth_rate, death_rate, tau, alpha).
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)   # skip header row
        for line in reader:
            rows.append(line)

    # Now rows is a list of lists (each sublist is a row of strings)

    # Convert numeric columns and sort by tau (column index 2) descending:
    #   rows[i][2] is the tau for species i.
    rows_sorted = sorted(rows, key=lambda r: float(r[4]), reverse=True)

    # Extract into separate lists
    species_names = []
    birth_list    = []
    death_list    = []
    tau_list      = []
    long_tau_list = []


    for r in rows_sorted:
        species_names.append(r[0])
        tau_list.append(float(r[1]))   # assuming tau is column 2 (index 1)
        long_tau_list.append(float(r[4])) #assuming long tau is column 5 (index 4)
        birth_list.append(float(r[2]))     # birth is column 3 (index 2)
        death_list.append(float(r[3]))   #  death is column 4 (index 3)
        

    return species_names, long_tau_list, tau_list, birth_list, death_list, 

def cohortize_species(params, n=4, split_on="tau"):
    """
    Convert the tuple returned by `load_species_params` into an n-tile
    (“cohort”) table whose rows are the mean of every numeric column
    inside each cohort.

    Parameters
    ----------
    params : tuple
        The (species_names, birth_list, death_list, tau_list) tuple from
        `load_species_params`.
    n : int, optional
        Number of cohorts (default 4 → quartiles).
    split_on : {"tau", "birth", "death"}, optional
        Which numeric column drives the n-tile split.  Default "tau".

    Returns
    -------
    pd.DataFrame
        Columns: cohort label plus the cohort-level means of tau, birth,
        death and a count of species in each cohort.
    """
    species_names, birth, death, tau = params

    # --- 1. Build a tidy DataFrame ----------------------------------
    df = pd.DataFrame({
        "species": species_names,
        "birth"  : birth,
        "death"  : death,
        "tau"    : tau
    })

    # --- 2. Tag rows with cohort labels using qcut ------------------
    labels = [f"C{i+1}" for i in range(n)]
    df["cohort"] = pd.qcut(df[split_on], q=n, labels=labels,
                           duplicates="drop")  # handles ties gracefully

    # --- 3. Aggregate numeric columns within each cohort ------------
    numeric_cols = ["tau", "birth", "death"]
    cohort_table = (df
        .groupby("cohort", observed=True)[numeric_cols]
        .mean()
        .reset_index())

    # Optional: keep a row count
    cohort_table["n_species"] = df.groupby("cohort",
                                           observed=True).size().values

    return cohort_table

# ─── Grid & parameters ────────────────────────────────────────────────────────
#Read in from file

params = load_species_params(file)
 #I think something is wrong... things might be in wrong order..

two_tile = cohortize_species(params, n=2)  
k = len(species_names)
# Convert lists to NumPy arrays 
birth_rate = birth_list
death_rate = death_list
tau        = tau_list
alpha      = 0.001 * np.ones(k)

cohort_birth_rate = cohort_table["birth"].to_numpy() 
cohort_death_rate




#Manual inputs
Na = 1000              # number of age grid cells
amax = 20.0            # maximum age
tmax = 10            # maximum integration time
gamma = 0.5            # disturbance rate
  

# ─── Derived quantities ───────────────────────────────────────────────────────
da = amax / Na
a = np.linspace(0, amax, Na + 1)     # age grid (Na+1 points)
tau_idx = [int(t / da) for t in tau]              # delay index (in grid steps)
rho = gamma * np.exp(-gamma * a)     # age‐density function



def generate_initial_profiles(a_grid, k):
    """
    Returns a list of k arrays (each length len(a_grid)) where
    each array is a Gaussian bump with:
      - shift ~ N(mean=amax/2,  std=amax/10)
      - width  ~ |N(mean=amax/10, std=amax/20)|
    """
    profiles = []
    for i in range(k):
        # random shift and width
        shift = random.gauss(mu=amax/2, sigma=amax/20)
        width = abs(random.gauss(mu=amax/10, sigma=amax/20))
        # compute Gaussian: exp(-((a - shift)^2) / (2*width^2))
        gauss = np.exp(-((a_grid - shift) ** 2) / (2 * width**2))
        # zero out ages below tau[i]
        gauss[a_grid <= tau[i]] = 0.0
        profiles.append(gauss)
    return profiles

def init_history(a_grid):
    """
    For t ≤ 0, return the flattened initial profiles generated by
    generate_initial_profiles(a_grid, k).
    """
    profiles = generate_initial_profiles(a_grid, k)  # list of k arrays
    return np.concatenate(profiles)                  # flat vector length k*(Na+1)



