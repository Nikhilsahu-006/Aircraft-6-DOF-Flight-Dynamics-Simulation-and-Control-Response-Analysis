# Aircraft-6-DOF-Flight-Dynamics-Simulation-and-Control-Response-Analysis


## Project Overview
A comprehensive **Six Degree of Freedom (6-DOF) flight dynamics simulator** implementing nonlinear aircraft equations of motion with complete aerodynamic modeling. This project analyzes aircraft response to control surface deflections during steady level flight conditions.

##  Project Description
This simulation models a light aircraft (750 kg) flying at 3000m altitude with 50 m/s airspeed. The implementation includes:
- **Complete 6-DOF rigid body dynamics**
- **Longitudinal and lateral-directional aerodynamic models**
- **International Standard Atmosphere (ISA) model**
- **Control surface deflection analysis**
- **Comprehensive visualization of all aircraft states**

##  Key Features
- **Realistic Aerodynamic Modeling** with stability derivatives
- **Control Response Analysis** for elevator, aileron,rudder , thrust
- **Multiple Visualization Plots** including 3D trajectories
- **Professional-Grade Simulation** suitable for aerospace applications
- **Numerical Integration** using SciPy's solve_ivp with RK45 method

##  Technical Specifications
### Aircraft Parameters
- **Mass**: 750 kg
- **Wing Area**: 12.47 m²
- **Wing Span**: 10.47 m
- **Moments of Inertia**: Ixx=873, Iyy=907, Izz=1680 kg-m²
- **Max Thrust**: 3500 N at sea level

### Flight Conditions
- **Altitude**: 3000 m
- **Velocity**: 50 m/s
- **Duration**: 200 seconds simulation
- **Control Input**: 2° deflections at 11 seconds

##  Output Analysis
The simulation generates:
- Angle of Attack and Sideslip angles
- Euler Angles (Roll, Pitch, Yaw)
- Body-axis velocities (u, v, w)
- Angular rates (p, q, r)
- 2D and 3D trajectory plots
- Time-domain response analysis

##  Installation & Requirements

1. Python
2. pip install numpy scipy matplotlib
