import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams.update({'axes.labelsize': 'large', 'axes.titlesize': 'large', 'xtick.labelsize': 'medium', 'ytick.labelsize': 'large'})


# Constants
R = 2
u = 1
mu = 3
L = 0.1e-1
m = 1
E = 0

# Lenz pot
def U(r, u, R, mu):
    return -2 * u * R**2 / (r**2 * ((r / R)**mu + (R / r)**mu)**2)

# ode for dr/dθ
def dr_dtheta(theta, r):
    expr = (m**2 / L**2) * (r**4) * (E - U(r, u, R, mu)) - 1
    return np.sqrt(np.maximum(expr, 0))  # Avoid complex values

# Initial conditions
r0 = 10  # Initial radius
theta_span = np.linspace(0, 6*np.pi, 1000) # change pi value to determine number of orbits

sol = solve_ivp(dr_dtheta, [0, 6*np.pi], [r0], t_eval=theta_span, method='RK45')

# Cartesian coordinates
x = sol.y[0] * np.cos(theta_span)
y = sol.y[0] * np.sin(theta_span)

# Plot
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbital Path")
plt.axis("equal")
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
mu_values = [0.9, 1]
for mu in mu_values:
    sol = solve_ivp(dr_dtheta, [0, 20*np.pi], [r0], t_eval=theta_span, method='RK45')
    
    # Convert to Cartesian coordinates
    x = sol.y[0] * np.cos(theta_span)
    y = sol.y[0] * np.sin(theta_span)
    
    plt.plot(x, y, label=f"$\mu={mu}$")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbital Paths for Different $\mu$")
plt.axis("equal")
plt.grid()
plt.legend()
plt.show()