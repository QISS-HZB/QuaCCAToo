# TODO: expand

"""
Fit functions to be used in the Analysis class as part of the QuaCAAToo package.
"""

import numpy as np

def fit_rabi(t, A, Tpi, C, phi):
    """
    Cosine function to fit Rabi oscillations

    Parameters
    ----------
    - t (array): time values
    - A (float): amplitude of the cosine function
    - Tpi (float): pi-pulse duration (half the period of the cosine function)
    - C (float): offset of the cosine function
    - phi (float): phase of the cosine function
    """
    return A*np.cos(np.pi*t/Tpi + phi) + C

def fit_rabi_decay(t, A, Tpi, phi, C, Tc, n):
    """
    Cosine function with exponential decay to fit Rabi oscillations

    Parameters
    ----------
    - t (array): time values
    - A (float): amplitude of the cosine function
    - Tpi (float): pi-pulse duration (half the period of the cosine function)
    - phi (float): phase of the cosine function
    - C (float): offset of the cosine function
    - Tc (float): decay time constant
    - n (float): power of the exponential decay
    """
    return A*np.cos(np.pi*t/Tpi + phi)*np.exp(-(t/Tc)**n) + C

def fit_exp_decay(t, A, C, Tc):
    """
    Simple exponential decay function

    Parameters
    ----------
    - t (array): time values
    - A (float): amplitude of the exponential decay
    - C (float): offset of the exponential decay
    - Tc (float): decay time constant
    """
    return A*np.exp(-t/Tc) + C

def fit_exp_decay_n(t, A, C, Tc, n):
    """
    Exponential decay function with power n

    Parameters
    ----------
    - t (array): time values
    - A (float): amplitude of the exponential decay
    - C (float): offset of the exponential decay
    - Tc (float): decay time constant
    """
    return A*np.exp(-(t/Tc)**n) + C

def fit_hahn_mod(t, A, B, C, f1, f2):
    """
    Hahn echo with modulation function with 2 frequencies

    Parameters
    ----------
    - t (array): time values
    - A (float): amplitude of the echo
    - B (float): amplitude of the modulation
    - C (float): offset of the echo
    - f1 (float): first modulation frequency
    - f2 (float): second modulation frequency
    """
    return ( A - B*np.sin(2*np.pi*f1*t/2)**2*np.sin(2*np.pi*f2*t/2)**2 ) + C

def fit_hahn_mod_decay(t, A, B, C, f1, f2, Tc, n):
    """
    Hahn echo with modulation function with 2 frequencies and exponential decay

    Parameters
    ----------
    - t (array): time values
    - A (float): amplitude of the echo
    - B (float): amplitude of the modulation
    - C (float): offset of the echo
    - f1 (float): first modulation frequency
    - f2 (float): second modulation frequency
    - Tc (float): decay time constant
    - n (float): power of the exponential decay
    """
    return np.exp(- (t/Tc)**n)*( A - B*np.sin(2*np.pi*f1*t/2)**2*np.sin(2*np.pi*f2*t/2)**2 ) + C

def fit_lorentz(t, A, gamma, f, C):
    """
    Lorentzian peak

    Parameters
    ----------
    - t (array): time values
    - A (float): amplitude of the peak
    - gamma (float): width of the peak
    - f (float): frequency of the peak
    - C (float): offset of the peak
    """
    return C - A*(gamma**2)/((t-f)**2 + gamma**2)

def fit_two_lorentz(f, A1, A2, gamma1, gamma2, f1, f2, C):
    """
    Two symmetric Lorentzian peaks

    Parameters
    ----------
    - f (array): time values
    - A1 (float): amplitude of the first peak
    - A2 (float): amplitude of the second peak
    - gamma1 (float): width of the first peak
    - gamma2 (float): width of the second peak
    - f1 (float): frequency of the first peak
    - f2 (float): frequency of the second peak
    - C (float): offset of the peaks
    """
    return C - A1*(gamma1**2)/((f-f1)**2 + gamma1**2) - A2*(gamma2**2)/((f-f2)**2 + gamma2**2)

def fit_two_lorentz_sym(f, A, gamma, f1, f2, C):
    """
    Two symmetric Lorentzian peaks

    Parameters
    ----------
    - f (array): time values
    - A (float): amplitude of the peaks
    - gamma (float): width of the peaks
    - f1 (float): frequency of the first peak
    - f2 (float): frequency of the second peak
    - C (float): offset of the peaks
    """
    return C - A*(gamma**2)/((f-f1)**2 + gamma**2) - A*(gamma**2)/((f-f2)**2 + gamma**2)