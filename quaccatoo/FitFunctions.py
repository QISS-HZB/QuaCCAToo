# TODO: expand

"""
Fit functions to be used in the Analysis class as part of the QuaCAAToo package.
"""

import numpy as np


def fit_rabi(t, A, Tpi, C, phi):
    """
    Fit a cosine function to Rabi oscillations.

    Parameters
    ----------
    t : array_like
        Time values.
    A : float
        Amplitude of the cosine function.
    Tpi : float
        Pi-pulse duration (half the period of the cosine function).
    C : float
        Offset of the cosine function.
    phi : float
        Phase of the cosine function.
    """
    return A * np.cos(np.pi * t / Tpi + phi) + C


def fit_rabi_decay(t, A, Tpi, phi, C, Tc, n):
    """
    Fit a cosine function with exponential decay to Rabi oscillations.

    Parameters
    ----------
    t : array_like
        Time values.
    A : float
        Amplitude of the cosine function.
    Tpi : float
        Pi-pulse duration (half the period of the cosine function).
    phi : float
        Phase of the cosine function.
    C : float
        Offset of the cosine function.
    Tc : float
        Decay time constant.
    n : float
        Power of the exponential decay.
    """
    return A * np.cos(np.pi * t / Tpi + phi) * np.exp(-((t / Tc) ** n)) + C


def fit_exp_decay(t, A, C, Tc):
    """
    Fit a simple exponential decay function.

    Parameters
    ----------
    t : array_like
        Time values.
    A : float
        Amplitude of the exponential decay.
    C : float
        Offset of the exponential decay.
    Tc : float
        Decay time constant.
    """
    return A * np.exp(-t / Tc) + C


def fit_exp_decay_n(t, A, C, Tc, n):
    """
    Fit an exponential decay function with power n.

    Parameters
    ----------
    t : array_like
        Time values.
    A : float
        Amplitude of the exponential decay.
    C : float
        Offset of the exponential decay.
    Tc : float
        Decay time constant.
    n : float
        Power of the exponential decay.
    """
    return A * np.exp(-((t / Tc) ** n)) + C


def fit_hahn_mod(t, A, B, C, f1, f2):
    """
    Fit a Hahn echo with modulation function with 2 frequencies.

    Parameters
    ----------
    t : array_like
        Time values.
    A : float
        Amplitude of the echo.
    B : float
        Amplitude of the modulation.
    C : float
        Offset of the echo.
    f1 : float
        First modulation frequency.
    f2 : float
        Second modulation frequency.
    """
    return (A - B * np.sin(2 * np.pi * f1 * t / 2) ** 2 * np.sin(2 * np.pi * f2 * t / 2) ** 2) + C


def fit_hahn_mod_decay(t, A, B, C, f1, f2, Tc, n):
    """
    Fit a Hahn echo with modulation function with 2 frequencies and exponential decay.

    Parameters
    ----------
    t : array_like
        Time values.
    A : float
        Amplitude of the echo.
    B : float
        Amplitude of the modulation.
    C : float
        Offset of the echo.
    f1 : float
        First modulation frequency.
    f2 : float
        Second modulation frequency.
    Tc : float
        Decay time constant.
    n : float
        Power of the exponential decay.
    """
    return np.exp(-((t / Tc) ** n)) * (A - B * np.sin(2 * np.pi * f1 * t / 2) ** 2 * np.sin(2 * np.pi * f2 * t / 2) ** 2) + C


def fit_lorentz(t, A, gamma, f, C):
    """
    Fit a Lorentzian peak.

    Parameters
    ----------
    t : array_like
        Time values.
    A : float
        Amplitude of the peak.
    gamma : float
        Width of the peak.
    f : float
        Frequency of the peak.
    C : float
        Offset of the peak.
    """
    return C - A * (gamma**2) / ((t - f) ** 2 + gamma**2)


def fit_two_lorentz(f, A1, A2, gamma1, gamma2, f1, f2, C):
    """
    Fit two symmetric Lorentzian peaks.

    Parameters
    ----------
    f : array_like
        Time values.
    A1 : float
        Amplitude of the first peak.
    A2 : float
        Amplitude of the second peak.
    gamma1 : float
        Width of the first peak.
    gamma2 : float
        Width of the second peak.
    f1 : float
        Frequency of the first peak.
    f2 : float
        Frequency of the second peak.
    C : float
        Offset of the peaks.
    """
    return C - A1 * (gamma1**2) / ((f - f1) ** 2 + gamma1**2) - A2 * (gamma2**2) / ((f - f2) ** 2 + gamma2**2)


def fit_two_lorentz_sym(f, A, gamma, f1, f2, C):
    """
    Fit two symmetric Lorentzian peaks.

    Parameters
    ----------
    f : array_like
        Time values.
    A : float
        Amplitude of the peaks.
    gamma : float
        Width of the peaks.
    f1 : float
        Frequency of the first peak.
    f2 : float
        Frequency of the second peak.
    C : float
        Offset of the peaks.
    """
    return C - A * (gamma**2) / ((f - f1) ** 2 + gamma**2) - A * (gamma**2) / ((f - f2) ** 2 + gamma**2)
