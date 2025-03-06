##############################################################################
# solving for lambda = 1/r, consistent with other code but more messy so changed 
# to just r in the report. 

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.constants as con
plt.rcParams.update({'axes.labelsize': 'large', 'axes.titlesize': 'large', 'xtick.labelsize': 'medium', 'ytick.labelsize': 'large'})

R = 1
u = 2
mu = 2
l = 0.1e-1
m = 1
E = 10
lambda_0 = m/(l*1.6)

def U(r, u, R, mu):
    return -2 * u * R**2 / (r**2 * ((r/R)**mu + (R/r)**mu)**2)

def ang(l, m, r):
    return l**2 / (2*m*r**2)
r_values = np.linspace(0.01, 3, 1000)

mu_values = [0.8, 0.9, 1]
# Plot U_eff for different mu values
plt.figure(figsize=(8, 6))
plt.grid()

for mu in mu_values:
    U_values = U(r_values, u, R, mu)
    ang_values = ang(l, m, r_values)
    U_eff_values = U_values + ang_values
    
    plt.plot(r_values, U_eff_values, label=f"$\mu={mu}$")

plt.legend()
plt.xlabel("r")
plt.ylabel("$U_{eff}(r)$")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("Effective Potential for Different $\mu$")
plt.show()


### sympy code for r_0, code breaks when including ang term?
# r, u, R, mu = sp.symbols('r u R mu')
# U = -2 * u * R**2 / (r**2 * ((r / R)**mu + (R / r)**mu)**2)
# dU_dr = sp.diff(U, r)
# solutions = sp.solve(dU_dr, r)

#############################################################
# This code is broke now, see the other file for solutions. these gave same results initally
#############################################################


# def U_lambda(lambda_, u, R, l, m, mu):
#     term1 = (m / (l * R * lambda_))**mu
#     term2 = (m / l * R * lambda_)**mu
#     return -2 * u * R * (l / m)**2 * lambda_**2 / ( (term1 + term2)**2 )

# def dlam_dtheta(theta, lambda_, u, R, l, m, mu, E):
#     value = -lambda_**2 + (2/m) * (-U_lambda(lambda_, u, R, l, m, mu) + E)
#     return np.sqrt(value) if value >= 0 else 0  # Ensure real values

# def runge_kutta4(func, theta_0, theta_f, n, lambda_0, u, R, l, m, mu, E):
#     h = (theta_f - theta_0) / n
#     theta = theta_0
#     lambda_ = lambda_0
#     theta_out = [theta]
#     lambda_out = [lambda_]
    
#     for i in range(n):
#         k1 = h * func(theta, lambda_, u, R, l, m, mu, E)
#         k2 = h * func(theta + h/2, lambda_ + k1/2, u, R, l, m, mu, E)
#         k3 = h * func(theta + h/2, lambda_ + k2/2, u, R, l, m, mu, E)
#         k4 = h * func(theta + h, lambda_ + k3, u, R, l, m, mu, E)
#         lambda_ += (k1 + 2*k2 + 2*k3 + k4) / 6
#         theta += h
#         theta_out.append(theta)
#         lambda_out.append(lambda_)
    
#     return np.array(theta_out), np.array(lambda_out)

# # Integration range and steps
# theta_0 = 0
# theta_f = 10*np.pi  # Integrate over multiple orbits
# n_steps = 1000

# theta_vals, lambda_vals = runge_kutta4(dlam_dtheta, theta_0, theta_f, n_steps, lambda_0, u, R, l, m, mu, E)

# # Convert back to r = m / (l * lambda)
# r_vals = np.where(lambda_vals != 0, m / (l * lambda_vals), np.inf)  # Avoid division by zero

# # Convert to Cartesian coordinates
# x_vals = r_vals * np.cos(theta_vals)
# y_vals = r_vals * np.sin(theta_vals)

# # Debugging prints
# print("Theta values:", theta_vals[:10])
# print("Lambda values:", lambda_vals[:10])
# print("r values:", r_vals[:10])

# # Plot the results
# plt.figure(figsize=(8, 8))
# plt.plot(x_vals, y_vals, label="Orbit")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.grid(True, linestyle="--", linewidth=0.5)
# plt.legend()
# plt.title("Orbit in Cartesian Coordinates")
# plt.show()

# # fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# # axs[0].grid()
# # axs[0].plot(r_values, U_eff_values_pos, label='$U_{eff}$', linestyle='-.', color='red')
# # axs[0].legend()

# # axs[1].plot(x_vals, y_vals, label="Orbit")
# # axs[1].set_xlabel("x")
# # axs[1].set_ylabel("y")
# # axs[1].axhline(0, color='black', linewidth=0.5)
# # axs[1].axvline(0, color='black', linewidth=0.5)
# # axs[1].grid(True, linestyle="--", linewidth=0.5)
# # axs[1].legend()
# # axs[1].set_title("Orbit in Cartesian Coordinates")

# # plt.show()
