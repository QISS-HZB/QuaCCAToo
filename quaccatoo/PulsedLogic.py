import numpy as np
from qutip import mesolve

def square_pulse(t, **pulse_params):
    """
    Square pulse envelope modulation

    Returns
    -------

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
    
    Returns
    -------
    calculated gaussian pulse function (float)
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

    Returns
    -------
    calculated lorentzian pulse function (float)    
    """
    return 1/(1 + ((t - pulse_params['tmid'])/['gamma'])**2)*np.cos(pulse_params['omega_pulse']*t+ pulse_params['phi_t'])

def pulse(Ht, rho0, tarray, c_ops, options, pulse_params):
    """
    Performs a pulse for the specified time-dependent Hamiltonian, initial density matrix, time step, number of steps, collapse operators, options and pulse parameters. Returns the final state of the system using the mesolve method from Qutip.
    
    Parameters
    ----------
    Ht (Qobj): time-dependent Hamiltonian composed of H0, H1 and python function represeting the time depedency of the pulse
    rho0 (Qobj): initial density matrix
    tarray (array): array of time steps
    c_ops (Qobj, list(Qobj)): list of collapse operators
    options (dict): options for the Qutip solver
    pulse_params (dict): dictionary of parameters for the pulse function

    Returns
    -------
    final state of the system (Qobj)
    """
    return mesolve(Ht, rho0, tarray, c_ops, [], options = options, args = pulse_params).states[-1]

def free_evolution(H0, rho0, duration):
    """
    Performs a free evolution for the specified time-independent Hamiltonian, initial density matrix and duration. Returns the final state of the system by applying the time-evolution operator to the initial density matrix.
    
    Parameters
    ----------
    H0 (Qobj): time-independent Hamiltonian
    rho0 (Qobj): initial density matrix
    duration (float, int): duration of the free evolution

    Returns
    -------
    final state of the system (Qobj)
    """
    return (-1j*H0*duration).expm() * rho0 * ((-1j*H0*duration).expm()).dag()
