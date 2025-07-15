"""
Collection of in built fit models and functions.

The predefined model classes take a (fit) function and implement a guess subroutine for intelligent initial values.

The general mechanism is the same as that of lmfit. So a lot of functions have been taken from there.
To create additional fit models, simply instantiate an lmfit model with the target fit function.

Refer to the documentation of analysis.run_fit for usage details.
"""

# ruff: noqa: F401
import numpy as np
from lmfit import Model
from lmfit.models import (
    ConstantModel,
    ExponentialModel,
    GaussianModel,
    LinearModel,
    LorentzianModel,
    SineModel,
    update_param_vals,
)

############################################## Rabi - Periodic Oscillation ##############################################


def _guess_sin(data, x):
    """
    Internal helper function to provide decent first guesses for sinusoidal functions.

    Parameters
    ----------
    data: array_like
        The dependent variable over which fit is to be performed
    x: array_like
        The independent variable in the fitting

    Returns
    -------
    amp: float
        Guessed amplitude
    frequency: float
        Guessed frequency
    """
    y = data - data.mean()
    frequencies = np.fft.fftfreq(len(x), abs(x[-1] - x[0]) / (len(x) - 1))
    fft = abs(np.fft.fft(y))
    argmax = abs(fft).argmax()
    amp = 2.0 * fft[argmax] / len(fft)
    frequency = 2 * np.pi * abs(frequencies[argmax])

    return amp, frequency


def fit_rabi(x, amp=1, Tpi=10, phi=0, offset=0):
    """
    Fit a cosine function to Rabi oscillations.

    Parameters
    ----------
    x : array_like
        Time values.
    amp : float
        Amplitude of the cosine function.
    Tpi : float
        Pi-pulse duration (half the period of the cosine function).
    phi : float
        Phase of the cosine function.
    offset : float
        Offset of the cosine function.
    """
    return amp * np.cos(np.pi * x / Tpi + phi) + offset


class RabiModel(Model):
    """
    Modified from lmfit's SineModel to include an offset as well

    Takes the same parameters as the fit_rabi function.
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update({"prefix": prefix, "nan_policy": nan_policy, "independent_vars": independent_vars})
        super().__init__(fit_rabi, **kwargs)
        self._set_paramhints_prefix()

    def guess(self, data, x, **kwargs):
        offset = data.mean()
        amp, frequency = _guess_sin(data, x)
        data = data - data.mean()

        # try shifts in the range [0, 2*pi) and take the one with best residual
        shift_guesses = np.linspace(0, 2 * np.pi, 11, endpoint=False)
        errors = [
            np.linalg.norm(
                self.eval(x=x, amp=amp, Tpi=np.pi / frequency, phi=shift_guess, offset=offset) - data
            )
            for shift_guess in shift_guesses
        ]
        phi = shift_guesses[np.argmin(errors)]
        pars = self.make_params(amp=amp, Tpi=np.pi / frequency, phi=phi, offset=offset)
        return update_param_vals(pars, self.prefix, **kwargs)


################################################ Exponential Decay ####################################################


def _guess_exp(data, x):
    """
    Internal helper function to provide decent first guesses for exponential functions.

    Parameters
    ----------
    data: array_like
        The dependent variable over which fit is to be performed
    x: array_like
        The independent variable in the fitting

    Returns
    -------
    coeff: array
        An array with amplitude and decay time as its elements.
    """
    y = np.log(np.abs(data))
    result = np.polynomial.Polynomial.fit(x, y, 1)
    coeff = result.convert().coef
    coeff[0] = np.exp(coeff[0])
    coeff[1] = -1 / coeff[1]
    return coeff


def fit_exp_decay(x, amp=1, Tc=1, offset=0):
    """
    Fit a simple exponential decay function.

    Parameters
    ----------
    x : array_like
        Time values.
    amp : float
        Amplitude of the exponential decay.
    Tc : float
        Decay time constant.
    offset : float
        Offset of the exponential decay.
    """
    return amp * np.exp(-x / Tc) + offset


class ExpDecayModel(Model):
    """
    Modified from lmfit's ExponentialModel to include an offset as well

    Takes the same parameters as the fit_exp_decay function.
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update({"prefix": prefix, "nan_policy": nan_policy, "independent_vars": independent_vars})
        super().__init__(fit_exp_decay, **kwargs)

    def guess(self, data, x, **kwargs):
        coeff = _guess_exp(data, x)
        pars = self.make_params(amp=coeff[0], Tc=coeff[1], offset=data[-1])
        return update_param_vals(pars, self.prefix, **kwargs)


######################################################### Rabi with Exp Decay ####################################################


def fit_rabi_decay(x, amp=1, Tpi=10, phi=0, offset=0, Tc=1):
    """
    Fit a cosine function with exponential decay to Rabi oscillations.

    Parameters
    ----------
    x : array_like
        Time values.
    amp : float
        Amplitude of the cosine function.
    Tpi : float
        Pi-pulse duration (half the period of the cosine function).
    phi : float
        Phase of the cosine function.
    offset : float
        Offset of the cosine function.
    Tc : float
        Decay time constant.
    """
    return amp * np.cos(np.pi * x / Tpi + phi) * np.exp(-x / Tc) + offset


class RabiDecayModel(Model):
    """
    Analogous to RabiModel, modulated with an exponential decay.

    Takes the same parameters as the fit_rabi_decay function.
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update({"prefix": prefix, "nan_policy": nan_policy, "independent_vars": independent_vars})
        super().__init__(fit_rabi_decay, **kwargs)

    def guess(self, data, x, **kwargs):
        offset = data.mean()
        amp, frequency = _guess_sin(data, x)
        data = data - data.mean()

        coeff = _guess_exp(data, x)

        shift_guesses = np.linspace(0, 2 * np.pi, 11, endpoint=False)
        errors = [
            np.linalg.norm(
                self.eval(x=x, amp=amp, Tpi=np.pi / frequency, phi=shift_guess, offset=offset, Tc=coeff[1])
                - data
            )
            for shift_guess in shift_guesses
        ]
        phi = shift_guesses[np.argmin(errors)]

        pars = self.make_params(amp=amp, Tpi=np.pi / frequency, phi=phi, offset=offset, Tc=coeff[1])

        return update_param_vals(pars, self.prefix, **kwargs)


def fit_exp_decay_n(x, A=1, C=0, Tc=1, n=1):
    """
    Fit an exponential decay function with power n.

    Parameters
    ----------
    x : array_like
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
    return A * np.exp(-((x / Tc) ** n)) + C


####################################################### Hahn Modulation ##############################################################


def fit_hahn_mod(x, A, B, C, f1, f2):
    """
    Fit a Hahn echo with modulation function with 2 frequencies.

    Parameters
    ----------
    x : array_like
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
    return (A - B * np.sin(2 * np.pi * f1 * x / 2) ** 2 * np.sin(2 * np.pi * f2 * x / 2) ** 2) + C


def fit_hahn_mod_decay(x, A, B, C, f1, f2, Tc, n):
    """
    Fit a Hahn echo with modulation function with 2 frequencies and exponential decay.

    Parameters
    ----------
    x : array_like
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
    return (
        np.exp(-((x / Tc) ** n))
        * (A - B * np.sin(2 * np.pi * f1 * x / 2) ** 2 * np.sin(2 * np.pi * f2 * x / 2) ** 2)
        + C
    )


####################################################### Lorentzian and sinc ##############################################################


def fit_lorentz(x, A, gamma, f0, C):
    """
    Fit a Lorentzian peak.

    Parameters
    ----------
    x : array_like
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
    return C - A * (gamma**2) / ((x - f0) ** 2 + gamma**2)


def fit_two_lorentz(x, A1=1, A2=1, gamma1=0.1, gamma2=0.1, f01=2.87e3, f02=2.87e3, C=0):
    """
    Fit two symmetric Lorentzian peaks.

    Parameters
    ----------
    x : array_like
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
    return C + fit_lorentz(x, A1, gamma1, f01, 0) + fit_lorentz(x, A2, gamma2, f02, 0)


def fit_two_lorentz_sym(x, A=1, gamma=1, f_mean=2.87e3, f_delta=1, C=0):
    """
    Fit two symmetric Lorentzian peaks.

    Parameters
    ----------
    x : array_like
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
    return (
        C
        + fit_lorentz(x, A, gamma, f_mean - f_delta / 2, 0)
        + fit_lorentz(x, A, gamma, f_mean + f_delta / 2, 0)
    )


def fit_sinc2(x, A, gamma, f0, C):
    """
    Fit a sinc function.

    Parameters
    ----------
    x : array_like
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
    return (
        C
        - A
        * gamma**2
        / (gamma**2 + (x - f0) ** 2)
        * np.sin((gamma**2 + (x - f0) ** 2) ** 0.5 / gamma / 2 * np.pi) ** 2
    )


def fit_two_sinc2_sym(x, A, gamma, f_mean, f_delta, C):
    """
    Fit two symmetric sinc functions.

    Parameters
    ----------
    x : array_like
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
    return (
        C + fit_sinc2(x, A, gamma, f_mean - f_delta / 2, 0) + fit_sinc2(x, A, gamma, f_mean + f_delta / 2, 0)
    )


def fit_five_sinc2(
    x, A1, A2, A3, A4, A5, gamma1, gamma2, gamma3, gamma4, gamma5, f01, f02, f03, f04, f05, C
):
    """
    Fit two symmetric sinc functions.

    Parameters
    ----------
    x : array_like
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
    return (
        C
        + fit_sinc2(x, A1, gamma1, f01, 0)
        + fit_sinc2(x, A2, gamma2, f02, 0)
        + fit_sinc2(x, A3, gamma3, f03, 0)
        + fit_sinc2(x, A4, gamma4, f04, 0)
        + fit_sinc2(x, A5, gamma5, f05, 0)
    )
