# solve SIR model
import numpy as np
import scipy 
from scipy.optimize import fsolve

# Given parameters
A = 32.25  # Insert the value of A
K = 0.56  # Insert the value of K
B = 2.5  # Insert the value of B
N = 20  # Insert the value of N
S0 = 19 # Insert the value of S0

# Define the equations to solve
def equations(vars):
    alpha, gamma, rho = vars
    eq1 = alpha - np.sqrt(((S0 / rho - 1)**2 + 2 * S0 * (N - S0) / rho**2))
    eq2 = B - np.tanh**(-1)((S0 / rho - 1) / alpha)
    eq3 = A - (gamma * alpha * rho**2 / (2 * S0))
    eq4 = K - (1 / (2 * alpha * gamma))
    return [eq1, eq2, eq3, eq4]

# Initial guess for the variables
initial_guess = [1, 1, 1]

# Solve the equations
alpha, gamma, rho = fsolve(equations, initial_guess)

print(f"alpha: {alpha}")
print(f"gamma: {gamma}")
print(f"rho: {rho}")
