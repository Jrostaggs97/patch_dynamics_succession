# imports
import numpy as np

# holds one and two species analytical calculations


## one species 
# Stable age distribution
def one_spec_analytic_eq(tau, b, mu, alpha, gamma, a_grid):
    S = (gamma + 2*mu)*(b - np.exp(gamma*tau)*(gamma+mu))/(2*alpha*(b**2))
    n_eq = (S*b/mu)*(1-np.exp(-mu*(a_grid - tau)))
    n_eq[a_grid <= tau] = 0.0
    return n_eq


def one_spec_analytic_total_density_eq(n_eq, a_grid, gamma):
    rho = gamma*np.exp(-gamma*a_grid)
    N = np.trapezoid(rho*n_eq,x=a_grid)
    return N


def two_spec_analytic_eq(tau, b, mu, alpha, gamma, a_grid):

    n_tilde = np.zeros((2,len(a_grid)))
    n = np.zeros((2,len(a_grid)))
    rho = gamma*np.exp(-gamma*a_grid)

    #set up the tilde formulation then to do linear solve then assemble
    for i in range(2):
        n_tilde[i,:] = (b[i]/mu[i])*(1-np.exp(-mu[i]*(a_grid-tau[i])))
        n_tilde[i, a_grid<=tau[i]] = 0.0
    
    #intraction matrix, M. M_ij = integral (rho*n_tilde_i*n_tilde_j) da
    M = np.zeros((2, 2))
    for i in range(2):
        for j in range(i + 1):               # j = 0..i
            val = np.trapezoid(rho * n_tilde[i, :] * n_tilde[j, :], x=a_grid)
            M[i, j] = val
            M[j, i] = val  

        #RHS, persistence vector P, P_i = (integral (rho*n_tilde_i) da -1 )/alpha_i
        P = np.zeros(2)
        for i in range(2):
            integral_i = np.trapezoid(rho * n_tilde[i, :], x=a_grid)
            P[i] = (integral_i - 1.0) / alpha[i]

        #Solve linear system for S (M*S = P)
        S = np.linalg.solve(M, P)  # length-2 vector of scalars

        for i in range(2):
            n[i, :] = S[i] * n_tilde[i, :]

    return n

def two_spec_analytica_totaldensity_eq(n, gamma, a_grid):
    rho = gamma*np.exp(-gamma*a_grid)
    N = [0,0]
    for i in range(2):
        N[i] = np.trapezoid(rho*n[i,:],x=a_grid)

    return N
    