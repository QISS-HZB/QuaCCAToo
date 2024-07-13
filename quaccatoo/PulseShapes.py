"""
This module contains pulse shape functions to be used in the pulsed experiment pulses.
"""

import numpy as np

def square_pulse(t, **pulse_params):
    """
    Square pulse envelope modulation

    Parameters
    ----------
    t (float): time parameter
    omega_pulse (float): frequency of the pulse
    phi_t (float): phase of the pulse
    """
    return np.cos(pulse_params['omega_pulse']*t + pulse_params['phi_t'])

def gaussian_pulse(t, **pulse_params):
    """
    Gaussian pulse envelope modulation
    
    Parameters
    ----------
    t (float): time parameter
    tmid (float): middle point of the pulse
    sigma (float): width of the pulse
    omega_pulse (float): frequency of the pulse
    phi_t (float): phase of the pulse
    """
    return np.exp(-((t - pulse_params['tmid'])**2)/(2*pulse_params['sigma']**2))*np.cos(pulse_params['omega_pulse']*t + pulse_params['phi_t'])

def lorentzian_pulse(t, **pulse_params):
    """
    Lorentzian pulse envelope modulation

    Parameters
    ----------
    t (float): time parameter
    tmid (float): middle point of the pulse
    gamma (float): width of the pulse
    omega_pulse (float): frequency of the pulse
    phi_t (float): phase of the pulse
    """
    return 1/(1 + ((t - pulse_params['tmid'])/['gamma'])**2)*np.cos(pulse_params['omega_pulse']*t+ pulse_params['phi_t'])