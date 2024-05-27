import matplotlib.pyplot as plt
import numpy as np
from qutip import tensor, qeye, jmat, Qobj, Options, parallel_map
from quaccatoo.pulsed_logic import B1x, B1y, pulse, free_evolution

def sim_rabi(t, H0, H1, w_pulse, rho0, observable, options = Options(nsteps=1e6), phi_t=0, c_ops=None):
    """
    Simulate a Rabi oscillation.

    Parameters
    ----------
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    w_pulse : float
        Frequency of the pulse
    rho_i : Qobj
        Initial state of the system
    options : Options
        Options for the solver in QuTip
    phi_t : float
        Phase of the pulse
    t : numpy.ndarray
        Time array to calculate
    observable : Qobj
        Observable to calculate the expectation value
    c_ops : list of Qobj
        List of collapse operators
    
    Returns
    -------
    rho_t : Qobj, float
        Density matrix at every time step or the expectation value of the observable if given
    """
    # if not isinstance(H0, Qobj):
    #     raise ValueError("H0 must be Qobj")
    # if not isinstance(H1, Qobj):
    #     raise ValueError("H1 must be Qobj")
    # if not isinstance(rho0, Qobj):
    #     raise ValueError("rho_i must be Qobj")
    # if not isinstance(options, Options):
    #     raise ValueError("options must be Options from QuTip")
    # if not isinstance(phi_t, (int, float)):
    #     raise ValueError("phi_t must be a number")
    # if not isinstance(t, np.ndarray):
    #     raise ValueError("t must be a numpy array")
    # if not isinstance(observable, (Qobj, None)):
    #     raise ValueError("observable must be a Qobj or None")
    
    # if not isinstance(c_ops, (type(None), list)):
    #         raise ValueError("c_ops must be a list of Qobj or None")
    # elif isinstance(c_ops, list):
    #     if not c_ops:
    #         print("c_ops is an empty list")
    #     else:
    #         for op in c_ops:
    #             if not isinstance(op, Qobj):
    #                 raise ValueError("All elements in c_ops must be Qobj instances")
                
    Ht = [2*np.pi*H0, [2*np.pi*H1, B1x]]

    return pulse(Ht, rho0, t, w_pulse, phi_t, observable, c_ops, options)

def PODMR_single_f(x, args):
    """
    Simualtes PODMR for a single values of x, in general should only be called by sim_PODMR function

    Parameters
    ----------
    x : float
        Frequency of the pulse
    args : dict
        Dictionary as in sim_PODMR function

    Returns
    -------
    p : float
        Expectation value of the observable for a single x value
    """
    rho = pulse(args['Htx'], args['rho0'], args['t_pi']/2, x, args['phi_t'], args['observable'], args['c_ops'], args['options'])

    return rho[-1]

def sim_PODMR(f, H0, H1, rho0, t_pi, observable, options = Options(nsteps=1e6), phi_t=0, c_ops=[], num_cpus=None, time_steps=100):
    """
    Simulates a PODMR experiment by calling PODMR_single_f function for x values in parellel

    Parameters
    ----------
    f : numpy.ndarray
        Array of frequency values
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    rho0 : Qobj
        Initial state of the system
    t_pi : float
        Pi pulse duration
    observable : Qobj
        Observable to calculate the expectation value
    options : Options
        Options for the solver in QuTip
    phi_t : float
        Initial phase of the pulse
    c_ops : list of Qobj
        List of collapse operators
    num_cpus : int
        Number of cpus to use for the parallel calculation, in None give the number of cpus in the system
    time_steps : int
        Number of time steps to calculate the pulses
    
    Returns
    -------
    p : numpy.ndarray
        Expectation value of the observable for every frequency value
    """
    Htx = [2*np.pi*H0, [2*np.pi*H1, B1x]] 

    args = {
        'Htx': Htx,
        'rho0': rho0,
        't_pi': np.linspace(0, t_pi, time_steps),
        'observable': observable,
        'options': options,
        'phi_t': phi_t,
        'c_ops': c_ops
    }

    p = parallel_map(PODMR_single_f, f, task_args=(args,), num_cpus=num_cpus)

    return np.array(p)

def ramsey_single_tau(x, args):
    """
    Simulate a Ramsey experiment for a single tau value, in general should only be called by sim_ramsey function

    Parameters
    ----------
    x : float
        Time parameter
    args : dict
        Dictionary as in sim_ramsey function

    Returns
    -------
    p : float
        Expectation value of the observable    
    """
    rho = pulse(args['Ht'], args['rho0'], args['t_pi']/2, args['w_pulse'], args['phi_t'], None, args['c_ops'], args['options'])

    rho = free_evolution(args['H0'], x, rho.states[-1])
    phi_t = 2*np.pi*args['w_pulse']*(x + args['t_pi'][-1]/2) + args['phi_t']

    rho = pulse(args['Ht'], rho, args['t_pi']/2, args['w_pulse'], phi_t, args['observable'], args['c_ops'], args['options'])

    return rho[-1]

def sim_ramsey(tau, H0, H1, w_pulse, rho0, t_pi, observable, options = Options(nsteps=1e6), phi_t=0, c_ops=[], num_cpus=None, time_steps=100):
    """
    Simulate a Ramsey experiment by calling ramsey_single_tau function for tau values in parellel

    Parameters
    ----------
    tau : numpy.ndarray
        Array of tau values
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    w_pulse : float
        Frequency of the pulse
    rho_i : Qobj
        Initial state of the system
    t_pi : float
        Pi pulse duration
    options : Options
        Options for the solver in QuTip
    phi_t : float
        Phase of the pulse
    observable : Qobj
        Observable to calculate the expectation value
    c_ops : list of Qobj
        List of collapse operators
    num_cpus : int
        Number of cpus to use for the parallel calculation, in None give the number of cpus in the system
    time_steps : int
        Number of time steps to calculate the pulses
    
    Returns
    -------
    p : numpy.ndarray
        Expectation value of the observable for every tau value
    
    """
    Ht = [2*np.pi*H0, [2*np.pi*H1, B1x]]

    args = {
        'H0': H0,
        'Ht': Ht,
        't_pi': np.linspace(0, t_pi, time_steps),
        'rho0': rho0,
        'w_pulse':w_pulse,
        'phi_t':phi_t,
        'observable':observable,
        'c_ops':c_ops,
        'options':options
        }
    
    p = parallel_map(ramsey_single_tau, tau, task_args=(args,), num_cpus=num_cpus)

    return np.array(p)

def hahn_single_tau(x, args):
    """
    Simulate a Hahn echo experiment for a single tau value, in general should only be called by sim_hahn function

    Parameters
    ----------
    x : float
        Time parameter
    args : dict
        Dictionary as in sim_hahn function

    Returns
    -------
    p : float
        Expectation value of the observable    
    """
    tau = x - args['t_pi'][-1]/2

    rho = pulse(args['Ht'], args['rho0'], args['t_pi']/2, args['w_pulse'], args['phi_t'], None, args['c_ops'], args['options'])

    rho = free_evolution(args['H0'], tau, rho.states[-1])
    phi_t = 2*np.pi*args['w_pulse']*x + args['phi_t']

    rho = pulse(args['Ht'], rho, args['t_pi'], args['w_pulse'], phi_t, None, args['c_ops'], args['options'])

    rho = free_evolution(args['H0'], tau, rho.states[-1])
    phi_t += 2*np.pi*args['w_pulse']*(tau + args['t_pi'][-1])

    rho = pulse(args['Ht'], rho, args['t_pi']/2, args['w_pulse'], phi_t, args['observable'], args['c_ops'], args['options'])

    return rho[-1]

def sim_hahn(tau, H0, H1, w_pulse, rho0, t_pi, observable, options = Options(nsteps=1e6), phi_t=0, c_ops=[], num_cpus=None, time_steps=100):
    """
    Simulate a Hahn echo experiment by calling hahn_single_tau function for tau values in parellel

    Parameters
    ----------
    tau : numpy.ndarray
        Array of tau values
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    w_pulse : float
        Frequency of the pulse
    rho_i : Qobj
        Initial state of the system
    t_pi : float
        Pi pulse duration
    options : Options
        Options for the solver in QuTip
    phi_t : float
        Phase of the pulse
    observable : Qobj
        Observable to calculate the expectation value
    c_ops : list of Qobj
        List of collapse operators
    num_cpus : int
        Number of cpus to use for the parallel calculation, in None give the number of cpus in the system
    time_steps : int
        Number of time steps to calculate the pulses
    
    Returns
    -------
    p : numpy.ndarray
        Expectation value of the observable for every tau value
    
    """
    # if not isinstance(tau, np.ndarray):
    #     raise ValueError("tau must be a numpy array")
    # if not isinstance(H0, Qobj):
    #     raise ValueError("H0 must be Qobj")
    # if not isinstance(H1, Qobj):
    #     raise ValueError("H1 must be Qobj")
    # if not isinstance(rho0, Qobj):
    #     raise ValueError("rho0 must be Qobj")
    # if not isinstance(t_pi, (int, float)):
    #     raise ValueError("t_pi must be a number")
    # if not isinstance(w_pulse, (int, float)):
    #     raise ValueError("w_pulse must be a number")
    # if not isinstance(phi_t, (int, float)):
    #     raise ValueError("phi_t must be a number")
    # if not isinstance(observable, Qobj):
    #     raise ValueError("observable must be a Qobj")
    # if not isinstance(options, Options):
    #     raise ValueError("options must be Options from QuTip")
    # if not isinstance(c_ops, (type(None), list)):
    #     raise ValueError("c_ops must be a list of Qobj or None")
    # elif isinstance(c_ops, list):
    #     if not c_ops:
    #         print("c_ops is an empty list")
    #     else:
    #         for op in c_ops:
    #             if not isinstance(op, Qobj):
    #                 raise ValueError("All elements in c_ops must be Qobj instances")

    Ht = [2*np.pi*H0, [2*np.pi*H1, B1x]]

    args = {
        'H0': H0,
        'Ht': Ht,
        't_pi': np.linspace(0, t_pi, time_steps),
        'rho0': rho0,
        'w_pulse':w_pulse,
        'phi_t':phi_t,
        'observable':observable,
        'c_ops':c_ops,
        'options':options
        }
    
    p = parallel_map(hahn_single_tau, tau, task_args=(args,), num_cpus=num_cpus)

    return np.array(p)

def XY_block(rho0, tau, H0, Htx, Hty, w_pulse, t_pi, phi_t, c_ops, options = Options(nsteps=1e6)):
    """
    Simulates a XY block to be called by XY_single_tau function

    Parameters
    ----------
    rho0 : Qobj
        Initial state of the system
    tau : float
        Pulse separation
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    w_pulse : float
        Frequency of the pulse
    t_pi : numpy.ndarray
        Pi pulse duration
    phi_t : float
        Phase of the pulse
    c_ops : list of Qobj
        List of collapse operators
    options : Options
        Options for the solver in QuTip
    
    Returns
    -------
    rho : Qobj
        Density matrix at the end of the XY block
    phi_t : float
        Phase of the pulse at the end of the XY block
    """
    rho = free_evolution(H0, tau/2, rho0)   
    phi_t += 2*np.pi*w_pulse*tau/2
    rho = pulse(Htx, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau + t_pi[-1])
    rho = pulse(Hty, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau/2, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau/2 + t_pi[-1])

    return rho, phi_t

def XY_single_tau(x, args):
    """
    Simulates a XY-M sequence for a single tau value, in general should only be called by sim_XY function

    Parameters
    ----------
    x : float
        Time parameter
    args : dict
        Dictionary as in sim_XY function
    
    Returns
    -------
    p : float
        Expectation value of the observable for a single tau
    """
    tau = x - args['t_pi'][-1]

    rho = pulse(args['Htx'], args['rho0'], args['t_pi']/2, args['w_pulse'], args['phi_t'], None, args['c_ops'], args['options'])
    phi_t = np.pi*args['w_pulse']*args['t_pi'][-1] + args['phi_t']
    rho = rho.states[-1]

    for itr in range(args['M']):
        rho, phi_t = XY_block(rho, tau, args['H0'], args['Htx'], args['Hty'], args['w_pulse'], args['t_pi'], phi_t, args['c_ops'], args['options'])

    rho = pulse(args['Htx'], rho, args['t_pi']/2, args['w_pulse'], phi_t, args['observable'], args['c_ops'], args['options'])
    return rho[-1]

def sim_XY(tau, H0, H1, w_pulse, rho0, t_pi, observable, M, options = Options(nsteps=1e6), phi_t=0, c_ops=[], num_cpus=None, time_steps=100):
    """
    Simulates a XY-M sequence by calling XY_single_tau function for tau values in parellel

    Parameters
    ----------
    tau : numpy.ndarray
        Array of tau values
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    w_pulse : float
        Frequency of the pulse
    rho0 : Qobj
        Initial state of the system
    t_pi : float
        Pi pulse duration
    observable : Qobj
        Observable to calculate the expectation value
    M : int
        Number of XY blocks
    options : Options
        Options for the solver in QuTip
    phi_t : float
        Initial phase of the pulse
    c_ops : list of Qobj
        List of collapse operators
    num_cpus : int
        Number of cpus to use for the parallel calculation, in None give the number of cpus in the system
    time_steps : int
        Number of time steps to calculate the pulses
    
    Returns
    -------
    p : numpy.ndarray
        Expectation value of the observable for every tau value
    """
    Htx = [2*np.pi*H0, [2*np.pi*H1, B1x]]
    Hty = [2*np.pi*H0, [2*np.pi*H1, B1y]]

    args = {
        'H0': H0,
        'Htx': Htx,
        'Hty': Hty,
        'w_pulse': w_pulse,
        'rho0': rho0,
        't_pi': np.linspace(0, t_pi, time_steps),
        'observable': observable,
        'M': M,
        'options': options,
        'phi_t': phi_t,
        'c_ops': c_ops
    }

    p = parallel_map(XY_single_tau, tau, task_args=(args,), num_cpus=num_cpus)

    return np.array(p)

def CPMG_single_tau(x, args):
    """
    Simulates a CPMG-M sequence for a single tau value, in general should only be called by sim_CPMG function

    Parameters
    ----------
    x : float
        Time parameter
    args : dict
        Dictionary as in sim_CPMG function
    
    Returns
    -------
    p : float
        Expectation value of the observable for a single tau
    """
    tau = x - args['t_pi'][-1]

    rho = pulse(args['Htx'], args['rho0'], args['t_pi']/2, args['w_pulse'], args['phi_t'], None, args['c_ops'], args['options'])
    phi_t = np.pi*args['w_pulse']*args['t_pi'][-1] + args['phi_t']

    for itr in range(args['M']):
        rho = free_evolution(args['H0'], tau/2, rho.states[-1])   
        phi_t += 2*np.pi*args['w_pulse']*tau/2
        rho = pulse(args['Htx'], rho, args['t_pi'], args['w_pulse'], phi_t, None, args['c_ops'], args['options'])
        rho = free_evolution(args['H0'], tau/2, rho.states[-1])
        phi_t += 2*np.pi*args['w_pulse']*(tau/2 + args['t_pi'][-1])

    rho = pulse(args['Htx'], rho, args['t_pi']/2, args['w_pulse'], phi_t, args['observable'], args['c_ops'], args['options'])
    return rho[-1]

def sim_CPMG(tau, H0, H1, w_pulse, rho0, t_pi, observable, M, options = Options(nsteps=1e6), phi_t=0, c_ops=[], num_cpus=None, time_steps=100):
    """
    Simulates a CPMG-M sequence by calling CPMG_single_tau function for tau values in parellel

    Parameters
    ----------
    tau : numpy.ndarray
        Array of tau values
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    w_pulse : float
        Frequency of the pulse
    rho0 : Qobj
        Initial state of the system
    t_pi : float
        Pi pulse duration
    observable : Qobj
        Observable to calculate the expectation value
    M : int
        Number of XY blocks
    options : Options
        Options for the solver in QuTip
    phi_t : float
        Initial phase of the pulse
    c_ops : list of Qobj
        List of collapse operators
    num_cpus : int
        Number of cpus to use for the parallel calculation, in None give the number of cpus in the system
    time_steps : int
        Number of time steps to calculate the pulses
    
    Returns
    -------
    p : numpy.ndarray
        Expectation value of the observable for every tau value
    """
    Htx = [2*np.pi*H0, [2*np.pi*H1, B1x]]

    args = {
        'H0': H0,
        'Htx': Htx,
        'w_pulse': w_pulse,
        'rho0': rho0,
        't_pi': np.linspace(0, t_pi, time_steps),
        'observable': observable,
        'M': M,
        'options': options,
        'phi_t': phi_t,
        'c_ops': c_ops
    }

    p = parallel_map(CPMG_single_tau, tau, task_args=(args,), num_cpus=num_cpus)

    return np.array(p)

def XY8_block(rho0, tau, H0, Htx, Hty, w_pulse, t_pi, phi_t, c_ops, options = Options(nsteps=1e6)):
    """
    Simulates a XY8 block to be called by XY8_single_tau function

    Parameters
    ----------
    rho0 : Qobj
        Initial state of the system
    tau : float
        Pulse separation
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    w_pulse : float
        Frequency of the pulse
    t_pi : numpy.ndarray
        Pi pulse duration
    phi_t : float
        Phase of the pulse
    c_ops : list of Qobj
        List of collapse operators
    options : Options
        Options for the solver in QuTip
    
    Returns
    -------
    rho : Qobj
        Density matrix at the end of the XY8 block
    """
    rho = free_evolution(H0, tau/2, rho0)
    phi_t += 2*np.pi*w_pulse*tau/2
    rho = pulse(Htx, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau + t_pi[-1])
    rho = pulse(Hty, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau + t_pi[-1])
    rho = pulse(Htx, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau + t_pi[-1])
    rho = pulse(Hty, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau + t_pi[-1])

    rho = pulse(Hty, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau + t_pi[-1])
    rho = pulse(Htx, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau + t_pi[-1])
    rho = pulse(Hty, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau + t_pi[-1])
    rho = pulse(Htx, rho, t_pi, w_pulse, phi_t, None, c_ops, options)
    rho = free_evolution(H0, tau/2, rho.states[-1])
    phi_t += 2*np.pi*w_pulse*(tau/2 + t_pi[-1])

    return rho, phi_t

def XY8_single_tau(x, args):
    """
    Simulates a XY8-M sequence for a single tau value, in general should only be called by sim_XY8 function

    Parameters
    ----------
    x : float
        Time parameter
    args : dict
        Dictionary as in sim_XY8 function
    
    Returns
    -------
    p : float
        Expectation value of the observable for a single tau
    """

    tau = x - args['t_pi'][-1]

    rho = pulse(args['Htx'], args['rho0'], args['t_pi']/2, args['w_pulse'], args['phi_t'], None, args['c_ops'], args['options'])
    phi_t = np.pi*args['w_pulse']*args['t_pi'][-1] + args['phi_t']
    rho = rho.states[-1]

    for itr in range(args['M']):
        rho, phi_t = XY8_block(rho, tau, args['H0'], args['Htx'], args['Hty'], args['w_pulse'], args['t_pi'], phi_t, args['c_ops'], args['options'])

    rho = pulse(args['Htx'], rho, args['t_pi']/2, args['w_pulse'], phi_t, args['observable'], args['c_ops'], args['options'])
    return rho[-1]

def sim_XY8(tau, H0, H1, w_pulse, rho0, t_pi, observable, M, options = Options(nsteps=1e6), phi_t=0, c_ops=[], num_cpus=None, time_steps=100):
    """
    Simulates a XY8-M sequence by calling XY8_single_tau function for tau values in parellel

    Parameters
    ----------
    tau : numpy.ndarray
        Array of tau values
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    w_pulse : float
        Frequency of the pulse
    rho0 : Qobj
        Initial state of the system
    t_pi : float
        Pi pulse duration
    observable : Qobj
        Observable to calculate the expectation value
    M : int
        Number of XY8 blocks
    options : Options
        Options for the solver in QuTip
    phi_t : float
        Initial phase of the pulse
    c_ops : list of Qobj
        List of collapse operators
    num_cpus : int
        Number of cpus to use for the parallel calculation, in None give the number of cpus in the system
    time_steps : int
        Number of time steps to calculate the pulses

    Returns
    -------
    p : numpy.ndarray
        Expectation value of the observable for every tau value

    """
    Htx = [2*np.pi*H0, [2*np.pi*H1, B1x]]
    Hty = [2*np.pi*H0, [2*np.pi*H1, B1y]]

    args = {
        'H0': H0,
        'Htx': Htx,
        'Hty': Hty,
        'w_pulse': w_pulse,
        'rho0': rho0,
        't_pi': np.linspace(0, t_pi, time_steps),
        'observable': observable,
        'M': M,
        'options': options,
        'phi_t': phi_t,
        'c_ops': c_ops
    }

    p = parallel_map(XY8_single_tau, tau, task_args=(args,), num_cpus=num_cpus)

    return np.array(p)

def RXY8_single_tau(x, args):
    """
    Simulates a XY8-M sequence for a single tau value, in general should only be called by sim_XY8 function

    Parameters
    ----------
    x : float
        Time parameter
    args : dict
        Dictionary as in sim_XY8 function
    
    Returns
    -------
    p : float
        Expectation value of the observable for a single tau
    """

    tau = x - args['t_pi'][-1]

    Htx = [2*np.pi*args['H0'], [2*np.pi*args['H1'], B1x]] #Total Hamiltonian for the x pulse

    rho = pulse(Htx, args['rho0'], args['t_pi']/2, args['w_pulse'], args['phi_t'], None, args['c_ops'], args['options'])
    phi_t = np.pi*args['w_pulse']*args['t_pi'][-1] + args['phi_t']
    rho = rho.states[-1]

    for itr in range(args['M']):
        random_phase = np.random.uniform(0, 2*np.pi)
        rho, phi_t = XY8_block(rho, tau, args['H0'], args['H1'], args['w_pulse'], args['t_pi'], phi_t + random_phase, args['c_ops'], args['options'])
        phi_t -= random_phase

    rho = pulse(Htx, rho, args['t_pi']/2, args['w_pulse'], phi_t, args['observable'], args['c_ops'], args['options'])
    return rho[-1]

def sim_RXY8(tau, H0, H1, w_pulse, rho0, t_pi, observable, M, options = Options(nsteps=1e6), phi_t=0, c_ops=[], num_cpus=None, time_steps=100):
    """
    Simulates a RXY8-M sequence by calling RXY8_single_tau function for tau values in parellel

    Parameters
    ----------
    tau : numpy.ndarray
        Array of tau values
    H0 : Qobj
        Time independent Hamiltonian of the system
    H1 : Qobj
        Time dependent Hamiltonian of the system
    w_pulse : float
        Frequency of the pulse
    rho0 : Qobj
        Initial state of the system
    t_pi : float
        Pi pulse duration
    observable : Qobj
        Observable to calculate the expectation value
    M : int
        Number of XY8 blocks
    options : Options
        Options for the solver in QuTip
    phi_t : float
        Initial phase of the pulse
    c_ops : list of Qobj
        List of collapse operators
    num_cpus : int
        Number of cpus to use for the parallel calculation, in None give the number of cpus in the system
    time_steps : int
        Number of time steps to calculate the pulses

    Returns
    -------
    p : numpy.ndarray
        Expectation value of the observable for every tau value

    """
    Htx = [2*np.pi*H0, [2*np.pi*H1, B1x]]
    Hty = [2*np.pi*H0, [2*np.pi*H1, B1y]]

    args = {
        'H0': H0,
        'Htx': Htx,
        'Hty': Hty,
        'w_pulse': w_pulse,
        'rho0': rho0,
        't_pi': np.linspace(0, t_pi, time_steps),
        'observable': observable,
        'M': M,
        'options': options,
        'phi_t': phi_t,
        'c_ops': c_ops
    }

    p = parallel_map(RXY8_single_tau, tau, task_args=(args,), num_cpus=num_cpus)

    return np.array(p)