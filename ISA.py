import numpy as np
import math as m
import matplotlib.pyplot as plt

# Constants
g0 = 9.80665      # gravitational acceleration (m/s^2)
R = 287.05        # specific gas constant for air (J/kg·K)
H_scale = 34e3    # Approximate scale height above 105 km (m)

# ISA definition up to 105 km
a = [-6.5e-3, 0, 3e-3, 0, -4.5e-3, 0, 4e-3]  # Temperature lapse rates (K/m)
h = [0, 11, 25, 47, 53, 79, 90, 105]        # Layer boundaries in km

def ISA(h0_km):
    if h0_km > 500:
        raise ValueError("Altitude out of supported range. Maximum is 500 km.")

    h0 = h0_km * 1000  # convert to meters
    T0 = 288.16
    P0 = 101325
    rho0 = 1.225
    state = [T0, P0, rho0]

    # Traverse standard ISA layers up to 105 km
    for i in range(len(a)):
        h_start = h[i] * 1000
        h_end = h[i+1] * 1000
        if h0 > h_end:
            if a[i] == 0:
                state = isothermal_layer(state[0], state[1], state[2], h_start, h_end)
            else:
                state = gradient_layer(state[0], state[1], state[2], a[i], h_start, h_end)
        else:
            if a[i] == 0:
                state = isothermal_layer(state[0], state[1], state[2], h_start, h0)
            else:
                state = gradient_layer(state[0], state[1], state[2], a[i], h_start, h0)
            return state

    # If above 105 km, use exponential decay
    h_start = 105e3
    T = state[0]  # Assume constant T beyond 105 km
    P = state[1] * np.exp(-(h0 - h_start) / H_scale)
    rho = state[2] * np.exp(-(h0 - h_start) / H_scale)
    return [T, P, rho]

def gradient_layer(T1, P1, rho1, lapse_rate, h_start, h_end):
    T2 = T1 + lapse_rate * (h_end - h_start)
    T_ratio = T2 / T1
    exponent = -g0 / (lapse_rate * R)
    P2 = P1 * (T_ratio ** exponent)
    rho2 = rho1 * (T_ratio ** (exponent - 1))
    return [T2, P2, rho2]

def isothermal_layer(T, P1, rho1, h_start, h_end):
    exp_term = m.exp((-g0 * (h_end - h_start)) / (R * T))
    P2 = P1 * exp_term
    rho2 = rho1 * exp_term
    return [T, P2, rho2]


def h_geo_potential(hg):
    r = 6400

    return (r /( r + hg)) * hg



