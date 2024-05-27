import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj
from scipy.optimize import curve_fit
from quaccatoo.analysis import *

def plot_energy_levels(H0, units='MHz', energy_min=None, energy_max=None):
    """
    Returns the energy levels of the system in the given units.

    Parameters
    ----------
    H0 : qutip.Qobj
        The Hamiltonian of the system.
    units : str

    Returns
    -------
    H0_eig : numpy.ndarray
        The eigenenergies of the Hamiltonian.

    Raises
    ------
    ValueError
        If H0 is not a Qobj.
        If energy_min is not a number.
        If energy_max is not a number.
    """
    # if not isinstance(H0, Qobj):
    #     raise ValueError("H0 must be Qobj")
    
    # if not isinstance(energy_min, (int, float)) and energy_min is not None:
    #     raise ValueError("energy_min must be a number")
    
    # if not isinstance(energy_max, (int, float)) and energy_max is not None:
    #     raise ValueError("energy_max must be a number")

    H0_eig = H0.eigenenergies()
    H0_eig = H0_eig - H0_eig[0]

    fig, ax = plt.subplots(figsize=(2, 6))

    for itr in range(H0_eig.size):
        ax.axhline(y=H0_eig[itr])

    ax.set_ylabel(f'Energy ({units})')
    ax.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if energy_min is not None and energy_max is not None:
        ax.set_ylim([energy_min, energy_max])

    return H0_eig

def plot_rabi(p, t, exp=None, units='us', fit_function=None, fit_guess=None):
    """
    Plot the Rabi oscillation.

    Parameters
    ----------
    p : numpy.ndarray
        expectation value of the observable
    t : numpy.ndarray
        pulse length
    exp : numpy.ndarray
        experimental data with first column being the time and the second the expectation value
    units : str
        units of the time
    fit_function : function
        function to fit the data
    fit_guess : numpy.ndarray
        initial guess for the fit
    
    Returns
    -------
    params : numpy.ndarray
        parameters of the fit
    cov : numpy.ndarray
        covariance of the fit    
    """
    # if not isinstance(p, np.ndarray):
    #     raise ValueError("p must be a numpy.ndarray")
    # if not isinstance(t, np.ndarray):
    #     raise ValueError("t must be a numpy.ndarray")
    # if not isinstance(exp, type(None)) or not isinstance(exp[0], (np.ndarray, type(None))) or not isinstance(exp[1], (np.ndarray, type(None))):
    #     raise ValueError("exp must be a numpy.ndarray with first column being the time and the second the expectation value")
    
    fig, ax = plt.subplots()
    ax.plot(t, p, label='Sim')

    if fit_function is not None:
        params, cov = curve_fit(fit_function, t, p, p0=fit_guess, maxfev=1000000)
        ax.plot(t, fit_function(t, *params), label='Fit')
    
    if exp is not None:
        ax.scatter(exp[0], exp[1], label='Exp')

    ax.set_xlabel(f'Pulse Length t {units}')
    ax.set_ylabel('Expectation Value p')
    ax.legend(loc='best')
    ax.set_title('Rabi Oscillation')

    if fit_function:
        return params, cov
    else:
        return None
    
def plot_hahn(p, tau, exp=None, units='us', fit_function=None, fit_guess=None):
    """
    Plot the Hahn echo.

    Parameters
    ----------
    p : numpy.ndarray
        expectation value of the observable
    tau : numpy.ndarray
        pulse separation
    exp : numpy.ndarray
        experimental data with first column being the time and the second the expectation value
    units : str
        units of the time
    fit_function : function
        function to fit the data
    fit_guess : numpy.ndarray
        initial guess for the fit

    Returns
    -------
    params : numpy.ndarray
        parameters of the fit
    cov : numpy.ndarray
        covariance of the fit   
    """
    fig, ax = plt.subplots()
    ax.plot(tau, p, label='Sim')

    if fit_function is not None:
        params, cov = curve_fit(fit_function, tau, p, p0=fit_guess, maxfev=1000000)
        ax.plot(tau, fit_function(tau, *params), label='Fit')
    
    if exp is not None:
        ax.scatter(exp[0], exp[1], label='Exp')

    ax.set_xlabel(f'Pulse separation tau ({units})')
    ax.set_ylabel('Expectation Value p')
    ax.legend(loc='best')
    ax.set_title('Hahn Echo')

    if fit_function:
        return params, cov
    else:
        return None
    
def plot_multipulse(p, tau, exp=None, units='us', sequence=None):
    """
    Plot the XY8 pulse sequence.

    Parameters
    ----------
    p : numpy.ndarray
        expectation value of the observable
    tau : numpy.ndarray
        pulse separation
    exp : numpy.ndarray
        experimental data with first column being the time and the second the expectation value
    units : str
        units of the time
    sequence : str
        name of the pulse sequence

    Returns
    -------
    None
    
    """
    fig, ax = plt.subplots()
    ax.plot(tau, p, label='Sim')
   
    if exp is not None:
        ax.scatter(exp[0], exp[1], label='Exp')

    ax.set_xlabel(f'Pulse separation tau ({units})')
    ax.set_ylabel('Expectation Value p')
    ax.legend(loc='best')
    ax.set_title(f'{sequence} Pulse Sequence')

    return None