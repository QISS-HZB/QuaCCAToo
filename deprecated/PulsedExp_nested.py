import numpy as np

def CPMG_sequence(self, tau):
    """
    Defines the CPMG sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a pi pulse and free evolution time tau repeated M times. The sequence is to be called by the parallel_map method of QuTip.

    Parameters
    ----------
    tau (float): free evolution time

    Returns
    -------
    results attribute  
    """
    # calculate the pulse spacing
    ps = tau - self.pi_pulse_duration/2
    # set the total time to 0
    self.total_time = 0
    # initialize the density matrix to the initial density matrix
    self.rho = self.rho0.copy()

    # repeat M times the pi pulse and free evolution of tau
    for itr_M in range(self.M):
        # perform free evolution of tau
        self.free_evolution(ps/2)
        # perform pi pulse
        self.pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params, 0)
        # perform free evolution of tau
        self.free_evolution(ps/2)
    
    # if no observable is given, return the final density matrix
    if self.observable == None:
        return self.rho
    else:
        return np.abs( (self.rho * self.observable).tr() )


def CPMG_sequence_proj(self, tau):
    """
    Defines the CPMG sequence, but with an initial pi/2 pulse and a final pi/2 pulse in order to project the measurement in the Sz basis. The sequence is to be called by the parallel_map method of QuTip.

    Parameters
    ----------
    tau (float): free evolution time

    Returns
    -------
    results attribute     
    """
    # calculate the pulse spacing
    ps = tau - self.pi_pulse_duration/2       
    # set the total time to 0
    self.total_time = 0
    # initialize the density matrix to the initial density matrix
    self.rho = self.rho0.copy()

    # perform the first pi/2 pulse
    self.pulse(self.Ht, self.pi_pulse_duration/2, self.options, self.pulse_params, 0)

    # repeat M times the pi pulse and free evolution of tau
    for itr_M in range(self.M):
        # perform free evolution of tau
        self.free_evolution((tau - self.pi_pulse_duration)/2)
        # perform pi pulse
        self.pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params, 0)
        # perform free evolution of tau
        self.free_evolution((tau - self.pi_pulse_duration)/2)
    
    # perform the last pi/2 pulse
    self.pulse(self.Ht, self.pi_pulse_duration/2, self.options, self.pulse_params, 0)
    
    # if no observable is given, return the final density matrix
    if self.observable == None:
        return self.rho
    else:
        return np.abs( (self.rho * self.observable).tr() )

def XY_sequence(self, tau):
    """
    Defines the XY-M composed of intercalated pi pulses on X and Y axis with free evolutions of time tau repeated M times. The sequence is to be called by the parallel_map method of QuTip.

    Parameters
    ----------
    tau (float): free evolution time

    Returns
    -------
    rho (Qobj): final density matrix        
    """
    # calculate the pulse spacing
    ps = tau - self.pi_pulse_duration
    # initialize the total time and state
    self.total_time = 0
    self.rho = self.rho0.copy()

    # perform half ps evolution
    self.free_evolution(ps/2)

    # repeat 2*M-1 times alternated pi pulses on X and Y axis and free evolutions of tau
    for itr_M in range(2*self.M - 1):
        # perform pi pulse
        self.pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params, self.phi_t[itr_M%2])
        # perform free evolution of tau
        self.free_evolution(ps)

    # perform pi pulse on Y axis
    self.pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params, np.pi/2)
    # perform free evolution of ps/2
    self.free_evolution(ps/2)

    # if no observable is given, return the final density matrix
    if self.observable == None:
        return self.rho
    else:
        return np.abs( (self.rho * self.observable).tr() )

def XY_sequence_proj(self, tau):
    """
    Defines the XY-M sequence with an initial pi/2 pulse and a final pi/2 pulse in order to project the measurement in the Sz basis. The sequence is to be called by the parallel_map method of QuTip.

    Parameters
    ----------
    tau (float): free evolution time

    Returns
    -------
    rho (Qobj): final density matrix        
    """
    # calculate the pulse spacing
    ps = tau - self.pi_pulse_duration
    # initialize the total time and state
    self.total_time = 0
    self.rho = self.rho0.copy()

    # perform the first pi/2 pulse
    self.pulse(self.Ht, self.pi_pulse_duration/2, self.options, self.pulse_params, 0)

    # perform half ps evolution
    self.free_evolution(ps/2)

    # repeat 2*M-1 times alternated pi pulses on X and Y axis and free evolutions of tau
    for itr_M in range(2*self.M - 1):
        # perform pi pulse
        self.pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params, self.phi_t[itr_M%2])
        # perform free evolution of tau
        self.free_evolution(ps)

    # perform pi pulse on Y axis
    self.pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params, np.pi/2)
    # perform free evolution of ps/2
    self.free_evolution(ps/2)

    # perform the final pi/2 pulse
    self.pulse(self.Ht, self.pi_pulse_duration/2, self.options, self.pulse_params, 0)

    # if no observable is given, return the final density matrix
    if self.observable == None:
        return self.rho
    else:
        return np.abs( (self.rho * self.observable).tr() )
    
        # def hahn_sequence(self, tau):
    #     """
    #     Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of two free evolutions with a pi pulse between them. The sequence is to be called by the parallel_map method of QuTip.

    #     Parameters
    #     ----------
    #     tau (float): free evolution time

    #     Returns
    #     -------
    #     rho (Qobj): final density matrix        
    #     """
    #     # calculate pulse separation time
    #     ps = tau - self.pi_pulse_duration/2
    #     # set the total time to 0
    #     self.total_time = 0
    #     # initialize the density matrix to the initial density matrix
    #     self.rho = self.rho0.copy()

    #     # perform the first free evolution
    #     self.free_evolution(ps)

    #     # perform the pi pulse
    #     self.pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params, 0)

    #     # perform the second free evolution
    #     self.free_evolution(ps)

    #     # if no observable is given, return the final density matrix
    #     if self.observable == None:
    #         return self.rho
    #     # if an observable is given, return the expectation value of the observable
    #     else:
    #         return np.real( (self.rho * self.observable).tr() )
    
    # def hahn_sequence_proj(self, tau):
    #     """
    #     Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a pi/2 pulse, a free evolution time tau, a pi pulse and another free evolution time tau followed by a pi/2 pulse. The sequence is to be called by the parallel_map method of QuTip.

    #     Parameters
    #     ----------
    #     tau (float): free evolution time

    #     Returns
    #     -------
    #     rho (Qobj): final density matrix        
    #     """
    #     # calculate pulse separation time
    #     ps = tau - self.pi_pulse_duration
    #     # set the total time to 0
    #     self.total_time = 0
    #     # initialize the density matrix to the initial density matrix
    #     self.rho = self.rho0.copy()

    #     # perform the initial pi/2 pulse
    #     self.pulse(self.Ht, self.pi_pulse_duration/2, self.options, self.pulse_params, 0)
    #     # perform the first free evolution
    #     self.free_evolution(ps)
    #     # perform the pi pulse
    #     self.pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params, 0)
    #     # perform the second free evolution
    #     self.free_evolution(ps)
    #     # perform the final pi/2 pulse
    #     self.pulse(self.Ht, self.pi_pulse_duration/2, self.options, self.pulse_params, 0)

    #     # if no observable is given, return the final density matrix
    #     if self.observable == None:
    #         return self.rho
    #     else:
    #         return np.real( (self.rho * self.observable).tr() )

        # def ramsey_sequence(self, tau):
    #     """
    #     Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a single free evolution. The sequence is to be called by the parallel_map method of QuTip.

    #     Parameters
    #     ----------
    #     tau (float): free evolution time

    #     Returns
    #     -------
    #     rho (Qobj): final density matrix        
    #     """
    #     # initialize the density matrix to the initial density matrix
    #     self.rho = self.rho0.copy()

    #     # perform the free evolution
    #     self.free_evolution(tau)

    #     # if no observable is given, return the final density matrix
    #     if self.observable == None:
    #         return self.rho
    #     # if an observable is given, return the expectation value of the observable
    #     else:
    #         return np.real( (self.rho * self.observable).tr() )

    # def ramsey_sequence_proj(self, tau):
    #     """
    #     Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a single free evolution plus an initial and final pi/2 pulses to project into the Sz basis. The sequence is to be called by the parallel_map method of QuTip.

    #     Parameters
    #     ----------
    #     tau (float): free evolution time

    #     Returns
    #     -------
    #     rho (Qobj): final density matrix        
    #     """
    #     # calculate the pulse separation time 
    #     ps = tau - self.pi_pulse_duration
    #     # set the total time to 0
    #     self.total_time = 0
    #     # initialize the density matrix to the initial density matrix
    #     self.rho = self.rho0.copy()
        
    #     # perform initial pi/2 pulse
    #     self.pulse(self.Ht, self.pi_pulse_duration/2, self.options, self.pulse_params, 0)
    #     # perform the free evolution
    #     self.free_evolution(ps)
    #     # perform final pi/2 pulse
    #     self.pulse(self.Ht, self.pi_pulse_duration/2, self.options, self.pulse_params, 0)

    #     # if no observable is given, return the final density matrix
    #     if self.observable == None:
    #         return self.rho
    #     # if an observable is given, return the expectation value of the observable
    #     else:
    #         return np.real( (self.rho * self.observable).tr() )

        # def PODMR_sequence(self, f):
    #     """
    #     Defines the the Pulsed Optically Detected Magnetic Resonance (PODMR) sequence for a given frequency of the pulse. To be called by the parallel_map in run method.
    #     Parameters
    #     ----------
    #     f (float): free evolution time

    #     Returns
    #     -------
    #     rho (Qobj): final density matrix   
    #     """
    #     self.rho = self.rho0.copy()
    #     self.pulse_params['omega_pulse'] = f

    #     # run the simulation and return the final density matrix
    #     self.pulse(self.Ht, self.pulse_duration, self.options, self.pulse_params, self.pulse_params['phi_t'])

    #     if self.observable == None:
    #         return self.rho
    #     else:
    #         return np.real( (self.rho * self.observable).tr() )