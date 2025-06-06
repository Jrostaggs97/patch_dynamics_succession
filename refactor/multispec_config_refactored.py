# multispec_config.py

#Imports
import csv
import numpy as np
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
    rows_sorted = sorted(rows, key=lambda r: float(r[1]), reverse=True)

    # Extract into separate lists
    species_names = []
    birth_list    = []
    death_list    = []
    tau_list      = []


    for r in rows_sorted:
        species_names.append(r[0])
        tau_list.append(float(r[1])/100)   # assuming birth is column 2 (index 1)
        birth_list.append(float(r[2]))     # tau is column 3 (index 2)
        death_list.append(float(r[3]))   # death is column 4 (index 3)

    return species_names, birth_list, death_list, tau_list


# ─── Grid & parameters ────────────────────────────────────────────────────────
#Read in from file
species_names, birth_list, death_list, tau_list = load_species_params(file)
k = len(species_names)
# Convert lists to NumPy arrays 
birth_rate = birth_list
death_rate = death_list
tau        = tau_list
alpha      = 0.001 * np.ones(k)


#Manual inputs
Na = 100              # number of age grid cells
amax = 10.0            # maximum age
tmax = 0.5            # maximum integration time
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
        shift = random.gauss(mu=amax/2, sigma=amax/10)
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



