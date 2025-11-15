import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ISA import ISA  # International Standard Atmosphere model

# =============================================================================
# AIRCRAFT PARAMETERS
# =============================================================================
m = 750  # Mass (kg)
S = 12.47  # Wing area (m^2)
b = 10.47  # Wing span (m)
c = 1.211  # Mean aerodynamic chord (m)
AR = 8.8   # Aspect ratio
Ixx = 873  # Moment of inertia (kg-m^2)
Ixx = 873  # Moment of inertia (kg-m^2)
Iyy = 907  # Moment of inertia (kg-m^2)
Izz = 1680  # Moment of inertia (kg-m^2)
Ixz = 1144  # Moment of inertia (kg-m^2)
T_SL = 3500  # Max engine thrust at sea level (N)

# Flight Conditions
V_inf = 50  # Free stream velocity (m/s)
H = 3000  # Altitude (m)
rho_SL = 1.225  # Sea level air density (kg/m^3)
rho = (ISA(H/1000))[2]  # Air density at altitude (kg/m^3)
sigma = rho / rho_SL  # Density ratio
g = 9.81  # Gravity (m/s^2)

# =============================================================================
# AERODYNAMIC DERIVATIVES
# =============================================================================
# Longitudinal derivatives
CD0 = 0.036  # Zero-lift drag coefficient
CL0 = 0.365  # Zero-angle-of-attack lift coefficient
Cm0 = 0.05   # Zero-moment coefficient
CD_alpha = 0.041  # Drag due to angle of attack
CL_alpha = 4.2    # Lift curve slope
Cm_alpha = -0.59  # Pitch stability derivative
e = 0.9  # Oswald efficiency factor
CL_q = 27.3  # Lift due to pitch rate
Cm_q = -9.3  # Pitch damping derivative
CD_delta_e = 0.026  # Drag due to elevator deflection
CL_delta_e = 0.26   # Lift due to elevator deflection
Cm_delta_e = -1.008  # Pitch moment due to elevator

# Lateral-directional derivatives
CY0 = -0.013  # Zero sideslip side force
Cl0 = 0.0015  # Zero roll moment
Cn0 = 0.001   # Zero yaw moment
CY_beta = -0.431  # Side force due to sideslip
Cl_beta = -0.051  # Roll due to sideslip (dihedral effect)
Cn_beta = 0.071   # Yaw due to sideslip (weathercock stability)
CY_p = 0.269   # Side force due to roll rate
Cl_p = -0.251  # Roll damping
Cn_p = -0.045  # Yaw due to roll rate
CY_r = 0.433   # Side force due to yaw rate
Cl_r = 0.36    # Roll due to yaw rate
Cn_r = -0.091  # Yaw damping
CY_delta_r = 0.15    # Side force due to rudder
Cl_delta_r = 0.005   # Roll due to rudder
Cn_delta_r = -0.049  # Yaw due to rudder
CY_delta_a = 0       # Side force due to aileron
Cl_delta_a = -0.153  # Roll due to aileron
Cn_delta_a = 0       # Yaw due to aileron

def equations_of_motion(t, state, delta_e, delta_a, delta_r, T):
    """
    6-DOF Aircraft Equations of Motion
    Implements the full nonlinear aircraft dynamics
    """
    # Unpack state variables: [u, v, w, p, q, r, phi, theta, psi, x, y, z]
    u, v, w, p, q, r, phi, theta, psi, x, y, z = state

    # Compute aerodynamic angles
    alpha = np.arctan2(w, u + 1e-6)  # Angle of attack (avoid division by zero)
    V = np.sqrt(u**2 + v**2 + w**2 + 1e-6)  # Total velocity
    beta = np.arcsin(v / V)  # Sideslip angle

    # =========================================================================
    # AERODYNAMIC COEFFICIENTS
    # =========================================================================
    # Longitudinal coefficients
    CL = CL0 + CL_alpha * alpha + CL_q * (q * c) / (2 * V) + CL_delta_e * delta_e
    CD = CD0 + (CL**2) / (np.pi * e * AR)  # Drag polar
    Cm = Cm0 + Cm_alpha * alpha + Cm_q * (q * c) / (2 * V) + Cm_delta_e * delta_e

    # Lateral-directional coefficients
    CY = (CY0 + CY_beta * beta + CY_p * (p * b) / (2 * V) + 
          CY_r * (r * b) / (2 * V) + CY_delta_r * delta_r)
    Cl = (Cl0 + Cl_beta * beta + Cl_p * (p * b) / (2 * V) + 
          Cl_r * (r * b) / (2 * V) + Cl_delta_a * delta_a + Cl_delta_r * delta_r)
    Cn = (Cn0 + Cn_beta * beta + Cn_p * (p * b) / (2 * V) + 
          Cn_r * (r * b) / (2 * V) + Cn_delta_r * delta_r)

    # =========================================================================
    # FORCES AND MOMENTS
    # =========================================================================
    L = 0.5 * rho * V**2 * S * CL  # Lift force
    D = 0.5 * rho * V**2 * S * CD  # Drag force
    Y = 0.5 * rho * V**2 * S * CY  # Side force
    M = 0.5 * rho * V**2 * S * c * Cm  # Pitch moment
    l = 0.5 * rho * V**2 * S * b * Cl  # Roll moment
    n = 0.5 * rho * V**2 * S * b * Cn  # Yaw moment

    # =========================================================================
    # TRANSLATIONAL EQUATIONS OF MOTION (Body axes)
    # =========================================================================
    dudt = (T - D * np.cos(alpha) + L * np.sin(alpha)) / m - q * w + r * v - g * np.sin(theta)
    dvdt = Y / m - r * u + p * w + g * np.sin(phi) * np.cos(theta)
    dwdt = (-D * np.sin(alpha) - L * np.cos(alpha)) / m - p * v + q * u + g * np.cos(phi) * np.cos(theta)

    # =========================================================================
    # ROTATIONAL EQUATIONS OF MOTION (Euler's equations)
    # =========================================================================
    # Note: Simplified by assuming Ixz = 0, would need cross terms for complete solution
    dpdt = (l - (Izz - Iyy) * q * r) / Ixx
    dqdt = (M - (Ixx - Izz) * p * r) / Iyy
    drdt = (n - (Iyy - Ixx) * p * q) / Izz

    # =========================================================================
    # KINEMATIC EQUATIONS (Euler angles)
    # =========================================================================
    dphidt = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    dthetadt = q * np.cos(phi) - r * np.sin(phi)
    dpsidt = (q * np.sin(phi) / np.cos(theta)) - (r * np.cos(phi) / np.cos(theta))

    # =========================================================================
    # NAVIGATION EQUATIONS (Earth-fixed position)
    # =========================================================================
    # Direction cosine matrix transformation from body to earth axes
    dxdt = (u * np.cos(theta) * np.cos(psi) + 
            v * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) + 
            w * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)))
    
    dydt = (u * np.cos(theta) * np.sin(psi) + 
            v * (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) + 
            w * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)))
    
    dzdt = (-u * np.sin(theta) + v * np.sin(phi) * np.cos(theta) + 
            w * np.cos(phi) * np.cos(theta))

    # Round to avoid numerical issues (optional)
    j = 2
    return [round(dudt,j), round(dvdt,j), round(dwdt,j), round(dpdt,j), 
            round(dqdt,j), round(drdt,j), round(dphidt,j), round(dthetadt,j), 
            round(dpsidt,j), round(dxdt,j), round(dydt,j), round(dzdt,j)]

# =============================================================================
# INITIAL CONDITIONS AND SIMULATION SETUP
# =============================================================================
# Initial state: [u, v, w, p, q, r, phi, theta, psi, x, y, z]
state0 = [49.916, -2.325, 1.723, 0, 0, 0, 0, np.deg2rad(1.974), 0, 0, 0, -H]

# Time span for simulation
t_span = (0, 200)
t_eval = np.linspace(0, 200, 1000)  # Evaluation points

def control_inputs(t):
    """
    Define control inputs as function of time
    Simulates a control input change between 11-16 seconds
    """
    if 16 >= t >= 11:
        # Control deflection during maneuver
        return (np.deg2rad(1.685) + np.deg2rad(2),  # Elevator
                np.deg2rad(1.362),                   # Aileron  
                np.deg2rad(-2.693),                  # Rudder
                662.73)                              # Thrust (N)
    else:
        # Trim controls
        return (np.deg2rad(1.685),   # Elevator
                np.deg2rad(1.362),   # Aileron
                np.deg2rad(-2.693),  # Rudder
                662.73)              # Thrust (N)

def ode_func(t, state):
    """
    ODE function for solve_ivp - wraps equations of motion with control inputs
    """
    delta_e, delta_a, delta_r, T = control_inputs(t)
    return equations_of_motion(t, state, delta_e, delta_a, delta_r, T)

# =============================================================================
# RUN SIMULATION
# =============================================================================
print("Running aircraft simulation...")
sol = solve_ivp(ode_func, t_span, state0, t_eval=t_eval, method='RK45')

# Extract results
t = sol.t
u, v, w, p, q, r, phi, theta, psi, x, y, z = sol.y

# =============================================================================
# POST-PROCESSING AND PLOTTING
# =============================================================================
# Compute aerodynamic angles from states
alpha = np.arctan2(w, u)
beta = np.arcsin(v / np.sqrt(u**2 + v**2 + w**2))

# Plot 1: Angle of Attack and Sideslip
plt.figure(figsize=(10, 6))
plt.plot(t, np.degrees(alpha), label='Angle of Attack (α)', linewidth=2)
plt.plot(t, np.degrees(beta), label='Angle of Sideslip (β)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.title('Aerodynamic Angles vs Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Euler Angles
plt.figure(figsize=(10, 6))
plt.plot(t, np.degrees(phi), label='Roll (φ)', linewidth=2)
plt.plot(t, np.degrees(theta), label='Pitch (θ)', linewidth=2)
plt.plot(t, np.degrees(psi), label='Yaw (ψ)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.title('Euler Angles vs Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 3: Body-Axis Velocities
V = np.sqrt(u**2 + v**2 + w**2)
plt.figure(figsize=(10, 6))
plt.plot(t, u, label='u (X-velocity)', linewidth=2)
plt.plot(t, v, label='v (Y-velocity)', linewidth=2)
plt.plot(t, w, label='w (Z-velocity)', linewidth=2)
plt.plot(t, V, label='V (Total velocity)', linewidth=2, linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.title('Body-Axis Velocities vs Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 4: Angular Rates
plt.figure(figsize=(10, 6))
plt.plot(t, p, label='Roll Rate (p)', linewidth=2)
plt.plot(t, q, label='Pitch Rate (q)', linewidth=2)
plt.plot(t, r, label='Yaw Rate (r)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.title('Body-Axis Angular Rates vs Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 5: 2D Trajectory (X vs Altitude)
plt.figure(figsize=(10, 6))
plt.plot(x, -z, linewidth=2)
plt.xlabel('X Position (m)')
plt.ylabel('Altitude (m)')
plt.title('Aircraft Trajectory - X vs Altitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 6: 3D Trajectory
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, -z, linewidth=2)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Altitude (m)')
ax.set_title('3D Aircraft Trajectory')
plt.tight_layout()
plt.show()

print("Simulation completed successfully!")