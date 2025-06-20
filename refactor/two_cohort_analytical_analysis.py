# Here we want to do the condit cohort analtycial analysis
# read in condit data and do cohort analysis (use code from config file)
# rewrite coex check function in python``
# Analysis:
# alpha-gamma heat map
# for b vs tau (hold avg mu constant) we loop over alpha and gamma and do the coex check and plot 
# 1. agnostic coexistence and 2. who wins (like mutual invasion)
# repeat for mu vs tau (hold avg b constant)
# For some of the alpha- gamma values generate the coexstennce plot 

#Imports
import csv
import numpy as np
import pandas as pd
import random

##Read in species files and "cohortize them" functions

#Read in CSV file with species information
file = "species_params.csv"

# File parsing function
def load_species_params(csv_path, scale):
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

    # Convert numeric columns and sort by long tau (column index 4) descending:
    #   rows[i][4] is the tau for species i.
    rows_sorted = sorted(rows, key=lambda r: float(r[4]), reverse=True)

    # Extract into separate lists
    species_names = []
    birth_list    = []
    death_list    = []
    tau_list      = []
    long_tau_list = []


    for r in rows_sorted:
        species_names.append(r[0])
        tau_list.append(float(r[1])/scale)   # assuming tau is column 2 (index 1)
        long_tau_list.append(float(r[4])/scale) #assuming long tau is column 5 (index 4)
        birth_list.append(float(r[2])*scale)     # birth is column 3 (index 2)
        death_list.append(float(r[3])*scale)   #  death is column 4 (index 3)
        

    return species_names, long_tau_list, tau_list, birth_list, death_list

def cohortize_species(params, n, split_on="long tau"):
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
    species_names, long_tau, tau, birth, death = params

    # --- 1. Build a tidy DataFrame ----------------------------------
    df = pd.DataFrame({
        "species": species_names,
        "birth"  : birth,
        "death"  : death,
        "tau"    : tau,
        "long tau" : long_tau
    })

    # --- 2. Tag rows with cohort labels using qcut ------------------
    labels = [f"C{i+1}" for i in range(n)]
    df["cohort"] = pd.qcut(df[split_on], q=n, labels=labels,
                           duplicates="drop")  # handles ties gracefully

    # --- 3. Aggregate numeric columns within each cohort ------------
    numeric_cols = ["long tau", "tau", "birth", "death"]
    cohort_table = (df
        .groupby("cohort", observed=True)[numeric_cols]
        .mean()
        .reset_index())

    # Optional: keep a row count
    cohort_table["n_species"] = df.groupby("cohort",
                                           observed=True).size().values

    return cohort_table


#function that takes in parameters and determines coexistence.  


## The section below is for looping over alphas and gammas and providing information on coexistence. 

## These are mutual invasion graphs built using analytical condition. 