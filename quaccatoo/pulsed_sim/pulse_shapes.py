# TODO: expand and test

"""
This module contains pulse shape functions to be used in the PulsedSim classes as part of the QuaCAAToo package.
"""

import numpy as np


def square_pulse(t: np.ndarray, **pulse_params: dict[str, float]) -> np.ndarray:
    """
    Square pulse envelope modulation

    Parameters
    ----------
    t : array_like
        Time parameter
    f_pulse : float
        Frequency of the pulse
    phi_t : float
        Phase of the pulse

    Returns
    -------
    float
        Value of the square pulse at time t
    """
    return np.cos(pulse_params["f_pulse"] * t + pulse_params["phi_t"])


def gaussian_pulse(t: np.ndarray, **pulse_params: dict[str, float]) -> np.ndarray:
    """
    Gaussian pulse envelope modulation

    Parameters
    ----------
    t : array_like
        Time parameter
    t_mid : float
        Middle point of the pulse
    sigma : float
        Width of the pulse
    f_pulse : float
        Frequency of the pulse
    phi_t : float
        Phase of the pulse

    Returns
    -------
    float
        Value of the Gaussian pulse at time t
    """
    return np.exp(
        -((t - pulse_params["t_mid"]) ** 2) / (2 * pulse_params["sigma"] ** 2.0)
    ) * np.cos(  # satisfy the python typing demon (2.)
        pulse_params["f_pulse"] * t + pulse_params["phi_t"]
    )


def lorentzian_pulse(t: np.ndarray, **pulse_params: dict[str, float]) -> np.ndarray:
    """
    Lorentzian pulse envelope modulation

    Parameters
    ----------
    t : array_like
        Time parameter
    t_mid : float
        Middle point of the pulse
    gamma : float
        Width of the pulse
    f_pulse : float
        Frequency of the pulse
    phi_t : float
        Phase of the pulse

    Returns
    -------
    float
        Value of the Lorentzian pulse at time t
    """
    return (
        1
        / (1 + ((t - pulse_params["t_mid"]) / ["gamma"]) ** 2)
        * np.cos(pulse_params["f_pulse"] * t + pulse_params["phi_t"])
    )
