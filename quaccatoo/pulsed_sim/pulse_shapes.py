# TODO: expand and test

"""
This module contains pulse shape functions to be used in the PulsedSim classes as part of the QuaCAAToo package.
"""

import numpy as np

def square_pulse(t, **pulse_params):
    """
    Square pulse envelope modulation

    Parameters
    ----------
    t : float
        time parameter
    f_pulse : float
        frequency of the pulse
    phi_t : float
        phase of the pulse
    """
    return np.cos(pulse_params['f_pulse']*t + pulse_params['phi_t'])


def gaussian_pulse(t, **pulse_params):
    """
    Gaussian pulse envelope modulation

    Parameters
    ----------
    t : float
        time parameter
    t_mid : float
        middle point of the pulse
    sigma : float
        width of the pulse
    f_pulse float): frequency of the pulse
    phi_t (float): phase of the pulse
    """
    return np.exp(-((t - pulse_params['t_mid'])**2)/(2*pulse_params['sigma']**2))*np.cos(pulse_params['f_pulse']*t + pulse_params['phi_t'])


def lorentzian_pulse(t, **pulse_params):
    """
    Lorentzian pulse envelope modulation

    Parameters
    ----------
    t : float
        time parameter
    t_mid : float
        middle point of the pulse
    gamma : float
        width of the pulse
    f_pulse : float
        frequency of the pulse
    phi_t : float
        phase of the pulse
    """
    return 1/(1 + ((t - pulse_params['t_mid'])/['gamma'])**2)*np.cos(pulse_params['f_pulse']*t+ pulse_params['phi_t'])
