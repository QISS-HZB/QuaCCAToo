import numpy as np
from qutip import Qobj, Options, mesolve

def B1x(t, args): #TODO in this case i think we need to leave only t as variable and pass the other parameters as a dictionary because of qutip (not sure, please check)
    """
    Time dependent pulse in the x direction of the lab frame

    Parameters
    ----------
    t (float): time parameter
    w_pulse (float): frequency of the pulse
    phi_t (float): phase of the pulse

    Returns
    -------
    calculated H1x function (float)
    """
    return np.cos(2*np.pi*args['w_pulse']*t + args['phi_t'])

def B1y(t, args):
    """
    Time dependent pulse in the y direction of the lab frame

    Parameters
    ----------
    t (float): time parameter
    w_pulse (float): frequency of the pulse
    phi_t (float): phase of the pulse

    Returns
    -------
    calculated H1y function (float)
    """
    return np.cos(2*np.pi*args['w_pulse']*t + args['phi_t'] + np.pi/2)

def pulse(Ht, rho0, t, w_pulse, phi_t, observable, c_ops, options):
    """
    Takes an initial state and evolves it with the given Hamiltonian for the given time array

    Parameters
    ---------- 
    t (array): time array to calculate
    rho_i (Qobj): initial state of the system
    Ht (Qobj): total dependent Hamiltonian of the system
    phi_t (float): phase of the pulse
    w_pulse (float): frequency of the pulse

    Returns
    -------
    rho_t (Qobj): density matrix at every time step or the expectation value of the observable if given
    """
    rho = mesolve(Ht, rho0, t, c_ops, observable, options = options, args={'phi_t':phi_t, 'w_pulse':w_pulse})

    if observable:
        return rho.expect[0]
    else:
        return rho

def free_evolution(H0, tau, rho0):
    """
    Takes an initial state and does a free evolution it with the given time independent Hamiltonian for the given time duration

    Parameters
    ----------
    H0 (Qobj): time independent Hamiltonian of the system
    tau (float): time duration of the free evolution
    rho_i (Qobj): initial state of the system
    
    Returns
    -------
    rho_f (Qobj): final state of the system
    """
    rho = (-1j*2*np.pi*H0*tau).expm()*rho0*((-1j*2*np.pi*H0*tau).expm()).dag()
    return rho