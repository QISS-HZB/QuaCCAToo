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


def fit_lorentz(f, A, gamma, f0, C):
    """
    Fit a Lorentzian peak.

    Parameters
    ----------
    f : array_like
        Frequency values.
    A : float
        Amplitude of the peak.
    gamma : float
        Width of the peak.
    f0 : float
        Central requency of the peak.
    C : float
        Offset of the peak.
    """
    return C - A * (gamma**2) / ((f - f0) ** 2 + gamma**2)


def fit_two_lorentz(f, A1, A2, gamma1, gamma2, f01, f02, C):
    """
    Fit two symmetric Lorentzian peaks.

    Parameters
    ----------
    f : array_like
        Frequency values.
    A1 : float
        Amplitude of the first peak.
    A2 : float
        Amplitude of the second peak.
    gamma1 : float
        Width of the first peak.
    gamma2 : float
        Width of the second peak.
    f01 : float
        Central frequency of the first peak.
    f02 : float
        Central frequency of the second peak.
    C : float
        Offset of the peaks.
    """
    return C + fit_lorentz(f, A1, gamma1, f01, 0) + fit_lorentz(f, A2, gamma2, f02, 0)


def fit_two_lorentz_sym(f, A, gamma, f_mean, f_delta, C):
    """
    Fit two symmetric Lorentzian peaks.

    Parameters
    ----------
    f : array_like
        Frquency values.
    A : float
        Amplitude of the peaks.
    gamma : float
        Width of the peaks.
    f_mean : float
        Mean frequency of the peaks.
    f_delta : float
        Frequency difference between the peaks.
    C : float
        Offset of the peaks.
    """
    return C + fit_lorentz(f, A, gamma, f_mean - f_delta/2, 0) + fit_lorentz(f, A, gamma, f_mean + f_delta/2, 0)

def fit_sinc2(f, A, gamma, f0, C):
    """
    Fit a sinc function.

    Parameters
    ----------
    f : array_like
        Frequency values.
    A : float
        Amplitude of the sinc function.
    gamma : float
        Width of the sinc function.
    f0 : float
        Central frequency of the sinc function.
    C : float
        Offset of the sinc function.
    """
    return C- A*gamma**2/(gamma**2 + (f-f0)**2) * np.sin((gamma**2 + (f-f0)**2)**.5/gamma/2 * np.pi )**2

def fit_two_sinc2_sym(f, A, gamma, f_mean, f_delta, C):
    """
    Fit two symmetric sinc functions.

    Parameters
    ----------
    f : array_like
        Frequency values.
    A : float
        Amplitude of the sinc functions.
    gamma : float
        Width of the sinc functions.
    f_mean : float
        Mean frequency of the sinc functions.
    f_delta : float
        Frequency difference between the sinc functions.
    C : float
        Offset of the sinc functions.
    """
    return C + fit_sinc2(f, A, gamma, f_mean - f_delta/2, 0) + fit_sinc2(f, A, gamma, f_mean + f_delta/2, 0)

def fit_five_sinc2(f, A1, A2, A3, A4, A5, gamma1 , gamma2, gamma3, gamma4, gamma5, f01, f02, f03, f04, f05, C):
    """
    Fit two symmetric sinc functions.

    Parameters
    ----------
    f : array_like
        Frequency values.
    A : float
        Amplitude of the sinc functions.
    gamma : float
        Width of the sinc functions.
    f_mean : float
        Mean frequency of the sinc functions.
    f_delta : float
        Frequency difference between the sinc functions.
    C : float
        Offset of the sinc functions.
    """
    return C + fit_sinc2(f, A1, gamma1, f01, 0) + fit_sinc2(f, A2, gamma2, f02, 0) + fit_sinc2(f, A3, gamma3, f03, 0) + fit_sinc2(f, A4, gamma4, f04, 0) + fit_sinc2(f, A5, gamma5, f05, 0)


def fit_gaussian(f, A, sigma, f0, C):
    """
    Fit a Gaussian peak.

    Parameters
    ----------
    f : array_like
        Frequency values.
    A : float
        Amplitude of the peak.
    sigma : float
        Standard deviation (width) of the Gaussian.
    f0 : float
        Central frequency of the peak.
    C : float
        Offset of the peak.
    """
    return C + A * np.exp(- ( (f - f0)**2 ) / (2 * sigma**2) )
