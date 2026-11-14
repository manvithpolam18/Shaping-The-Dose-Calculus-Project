#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Set plot style for clean report-ready figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("Libraries imported successfully!")


# In[2]:


# --- Standard Dimensions (Synchronized with A & B) ---
a = 5.0        # Outer tablet radius (mm)[cite: 1, 15]
h = 3.0        # Side band height for biconvex tablet (mm)[cite: 1, 15]
c = 0.5        # Dome height above band (mm)[cite: 2, 15]
H = 4.0        # Total thickness for flat/donut (mm)[cite: 2, 15]
rho_f = 0.5    # Fillet radius (mm)[cite: 2, 15]
b0 = 1.0       # Initial hole radius for coated donut (mm)[cite: 2, 6, 15]

# --- Pharmacokinetic Parameters (Synchronized with C) ---
# Assuming a standard 70 kg human body weight for volume scaling
body_mass = 70.0 # kg

# Nifedipine (Main Drug)
nif_params = {
    'drug': 'Nifedipine',
    'F': 0.55,               # Mid-range bioavailability[cite: 30]
    'ka': 8.46,              # Absorption constant (h^-1)[cite: 30]
    'ke': np.log(2) / 2.0,   # Elimination constant for 2h half-life (~0.3466 h^-1)[cite: 30]
    'Vd': 0.70 * body_mass,  # Apparent distribution volume (~49 L)[cite: 30]
    'M0': 30.0               # Total initial mass (mg)
}

# Theophylline (Control Drug)
theo_params = {
    'drug': 'Theophylline',
    'F': 1.0,                # Complete availability[cite: 30]
    'ka': 2.19,              # Absorption constant (h^-1)[cite: 30]
    'ke': np.log(2) / 8.0,   # Elimination constant for 8h half-life (~0.0866 h^-1)[cite: 30]
    'Vd': 0.45 * body_mass,  # Apparent distribution volume (~31.5 L)[cite: 30]
    'M0': 200.0              # Total initial mass (mg)
}

print("Parameters locked in!")


# In[3]:


def release_rate_round(t, v, M0, geom_vol):
    """Anisotropic round tablet (Tablet 1'): shrinks radially and axially."""
    rho = M0 / geom_vol
    # Effective radii remaining at time t
    r_rem = np.maximum(0, a - v * t)
    H_rem = np.maximum(0, H - 2 * v * t)
    # Exposed area at time t
    A_t = 2 * np.pi * r_rem**2 + 2 * np.pi * r_rem * H_rem
    # If fully dissolved, release is 0
    if r_rem <= 0 or H_rem <= 0:
        return 0.0
    return rho * v * A_t

def release_rate_donut(t, v, M0):
    """Coated Donut (Tablet 2): inner hole wall expands outward[cite: 6, 19]."""
    # Volume of donut = pi * H * (a^2 - b0^2)
    geom_vol = np.pi * H * (a**2 - b0**2)
    rho = M0 / geom_vol
    b_t = b0 + v * t
    if b_t >= a:
        return 0.0
    # Only inner hole wall releases drug: A_t = 2 * pi * b_t * H
    A_t = 2 * np.pi * b_t * H
    return rho * v * A_t

def release_rate_designed_sextic(t, v, M0, L=15.0):
    """Designed Tablet (Tablet 3): uses sextic ramp profile[cite: 10, 13, 19]."""
    tau = L / v  # total dissolution time
    if t < 0 or t > tau:
        return 0.0
    u = t / tau
    # Sextic target ramp derivative / profile scaling
    # s'(u) = 60u^2(1-u)^3[cite: 13]
    s_prime = 60 * (u**2) * ((1 - u)**3)
    # Normalized release shape proportional to r(vt)^2
    # Simplified proportional release model scaled to total mass M0
    R_t = (M0 / tau) * s_prime * 6.0  # normalized factor
    return np.maximum(0.0, R_t)

print("Release functions defined!")


# In[4]:


def forward_euler(R_func, params, T_total=24.0, dt=0.01):
    """Solves dG/dt = R(t) - ka*G and dC/dt = (F*ka*G)/Vd - ke*C using Forward Euler."""
    t = np.arange(0, T_total + dt, dt)
    n = len(t)
    G = np.zeros(n)
    C = np.zeros(n)

    F, ka, ke, Vd = params['F'], params['ka'], params['ke'], params['Vd']

    for i in range(n - 1):
        dt_actual = t[i+1] - t[i]
        R_val = R_func(t[i])

        # Euler update steps
        dG = R_val - ka * G[i]
        dC = (F * ka * G[i]) / Vd - ke * C[i]

        G[i+1] = G[i] + dt_actual * dG
        C[i+1] = C[i] + dt_actual * dC

    return t, G, C

# --- Convergence Test ---
# Test step sizes to verify order 1.0 error slope
dt_list = [0.2, 0.1, 0.05, 0.025, 0.0125]
errors = []

# Reference solution using smallest dt
t_ref, _, C_ref = forward_euler(lambda t: 5.0, nif_params, T_total=10.0, dt=0.001)

for test_dt in dt_list:
    t_test, _, C_test = forward_euler(lambda t: 5.0, nif_params, T_total=10.0, dt=test_dt)
    # Interpolate to match reference grid for error comparison
    C_interpolated = np.interp(t_ref, t_test, C_test)
    err = np.max(np.abs(C_ref - C_interpolated))
    errors.append(err)

# Fit log-log slope
log_dt = np.log(dt_list)
log_err = np.log(errors)
slope, intercept = np.polyfit(log_dt, log_err, 1)
print(f"Forward Euler Convergence Slope: {slope:.3f} ( Expected ~ 1.0 )[cite: 26, 28]")


# In[5]:


# Simulation settings
T_total = 24.0
dt = 0.01
v_erosion = 0.25  # mm/h (erosion rate)
geom_vol_round = np.pi * (a**2) * H

# Run simulations for Nifedipine
t_grid, G_round, C_round = forward_euler(lambda t: release_rate_round(t, v_erosion, nif_params['M0'], geom_vol_round), nif_params, T_total, dt)
_, _, C_donut = forward_euler(lambda t: release_rate_donut(t, v_erosion, nif_params['M0']), nif_params, T_total, dt)
_, _, C_designed = forward_euler(lambda t: release_rate_designed_sextic(t, v_erosion, nif_params['M0']), nif_params, T_total, dt)

# Convert Nifedipine concentration to ng/mL (mg/L = mcg/mL = 1000 ng/mL)[cite: 31]
C_round_ng = C_round * 1000.0
C_donut_ng = C_donut * 1000.0
C_designed_ng = C_designed * 1000.0

# --- Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot(t_grid, C_round_ng, label='Round Tablet (Spikes Early)', color='red', linestyle='--')
plt.plot(t_grid, C_donut_ng, label='Coated Donut (Delayed Peak)', color='blue', linestyle='-.')
plt.plot(t_grid, C_designed_ng, label='Designed Sextic Tablet (Smooth Target)', color='green', linewidth=2)
plt.axhline(y=10.0, color='grey', linestyle=':', label='Effective Threshold (10 ng/mL)[cite: 34]')

plt.title('Nifedipine Plasma Concentration $C(t)$ Comparison Across Shapes[cite: 23, 27]')
plt.xlabel('Time (hours)')
plt.ylabel('Plasma Concentration (ng/mL)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figD1_nifedipine_simulation.png', dpi=300)
plt.show()
print("Simulation plot generated and saved as figD1_nifedipine_simulation.png!")


# In[6]:


# Create alternative parameter set with 4.98h half-life (Hoyo-Vadillo et al.[cite: 30, 31])
nif_alt_params = nif_params.copy()
nif_alt_params['ke'] = np.log(2) / 4.98

# Run simulation with alternative elimination constant
_, _, C_designed_alt = forward_euler(lambda t: release_rate_designed_sextic(t, v_erosion, nif_params['M0']), nif_alt_params, T_total, dt)
C_designed_alt_ng = C_designed_alt * 1000.0

# Plot sensitivity comparison
plt.figure(figsize=(9, 5))
plt.plot(t_grid, C_designed_ng, label='Nominal ($t_{1/2} = 2.0$ h)', color='green')
plt.plot(t_grid, C_designed_alt_ng, label='Alternative Lit. ($t_{1/2} = 4.98$ h)', color='orange', linestyle='--')
plt.title('Sensitivity Check: Effect of Elimination Half-Life on Nifedipine Peak')
plt.xlabel('Time (hours)')
plt.ylabel('Plasma Concentration (ng/mL)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figD2_sensitivity.png', dpi=300)
plt.show()
print("Sensitivity plot generated and saved as figD2_sensitivity.png!")


# In[ ]:




