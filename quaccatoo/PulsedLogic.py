import numpy as np

def square_pulse(t, **pulse_params):
    """
    Square pulse envelope modulation

    Returns
    -------

    """
    return np.cos(pulse_params['w_pulse']*t + pulse_params['phi_t'])

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
    return np.exp(-((t - pulse_params['tmid'])**2)/(2*pulse_params['sigma']**2))*np.cos(pulse_params['w_pulse']*t + pulse_params['phi_t'])

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
    return 1/(1 + ((t - pulse_params['tmid'])/['gamma'])**2)*np.cos(pulse_params['w_pulse']*t+ pulse_params['phi_t'])