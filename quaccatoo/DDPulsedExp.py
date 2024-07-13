"""
This module contains dynamical decoupling pulse sequences, used in quantum sensing and for extending coherence of quantum systems.
"""

import numpy as np
from qutip import Qobj, mesolve
from types import FunctionType
from.PulsedExp import PulsedExp
from.PulseShapes import square_pulse

####################################################################################################

class CPMG(PulsedExp):
    """
    This class contains a Carr-Purcell-Meiboom-Gill sequence used in quantum sensing experiments, inheriting from the PulsedExp class. The CPMG sequence consists of a series of pi pulses and free evolution times, such that these periodicals inversions will cancel out oscillating noises except for frequencies corresponding to the pulse separation.

    Class Attributes
    ----------------
    M (int): order of the XY sequence
    free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
    pi_pulse_duration (float, int): duration of the pi pulse
    projection_pulses (Boolean): boolean to determine if an initial pi/2 and final pi/2 pulses are to be included in order to project the measurement in the Sz basis
    += PulsedExp

    Class Methods
    -------------
    CPMG_sequence(tau): defines the Carr-Purcell-Meiboom-Gill sequence for a given free evolution time tau and the set of attributes defined in the generator, returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip.
    CPMG_sequence_proj(tau): defines the Carr-Purcell-Meiboom-Gill sequence for a given free evolution time tau and the set of attributes defined in the generator, returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip. An initial pi/2 pulse and final pi/2 pulse are included, in order to perform the measurement in the Sz basis.
    get_pulse_profiles(tau): generates the pulse profiles for the CPMG sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
    += PulsedExp
    """
    def __init__(self, M, free_duration, pi_pulse_duration, system, H1, H2=None, projection_pulses=True, pulse_shape = square_pulse, pulse_params = {}, options = {}, time_steps = 100):
        """
        Class generator for the Carr-Purcell-Meiboom-Gill sequence    

        Parameters
        ----------
        M (int): order of the XY sequence
        free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
        system (QSys): quantum system object containing the initial density matrix, internal time independent Hamiltonian and collapse operators
        H1 (Qobj, list(Qobj)): control Hamiltonian of the system
        pi_pulse_duration (float, int): duration of the pi pulse
        H2 (Qobj, list(Qobj)): time dependent sensing Hamiltonian of the system
        projection_pulses (Boolean): boolean to determine if an initial pi/2 and final pi/2 pulses are to be included in order to project the measurement in the Sz basis
        pulse_shape (FunctionType, list(FunctionType)): pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params (dict): dictionary of parameters for the pulse_shape functions
        time_steps (int): number of time steps in the pulses for the simulation
        """
        # call the parent class constructor
        super().__init__(system, H2)

        # check weather free_duration is a numpy array of real and positive elements and if it is, assign it to the object
        if not isinstance(free_duration, (np.ndarray, list)) or not (np.all(np.isreal(free_duration)) and np.all(np.greater_equal(free_duration, 0))):
            raise ValueError("free_duration must be a numpy array with real positive elements")
        else:
            self.variable = free_duration
            
        if not isinstance(M, int) or M <= 0:
            raise ValueError("M must be a positive integer")
        else:
            self.M = M

        # check weather pi_pulse_duration is a positive real number and if it is, assign it to the object
        if not isinstance(pi_pulse_duration, (int, float)) or pi_pulse_duration <= 0 or pi_pulse_duration > free_duration[0]:
            raise ValueError("pulse_duration must be a positive real number and pi_pulse_duration/2 must be smaller than the free evolution time, otherwise pulses will overlap")
        else:
            self.pi_pulse_duration = pi_pulse_duration

        # check weather time_steps is a positive integer and if it is, assign it to the object
        if not isinstance(time_steps, int) or time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        else:
            self.time_steps = time_steps       

        # check weather pulse_shape is a python function or a list of python functions and if it is, assign it to the object
        if isinstance(pulse_shape, FunctionType) or (isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape)):
            self.pulse_shape = pulse_shape
        else: 
            raise ValueError("pulse_shape must be a python function or a list of python functions")
        
        # check weather H1 is a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list and if it is, assign it to the object
        if isinstance(H1, Qobj) and H1.shape == self.system.rho0.shape:
            self.H1 = H1
            self.Ht = [self.system.H0, [self.H1, pulse_shape]]

        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.system.rho0.shape for op in H1) and len(H1) == len(pulse_shape):
            self.H1 = H1       
            self.Ht = [self.system.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            
        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")

        # check weather pulse_params is a dictionary and if it is, assign it to the object
        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            # initialize the pulse_params attribute as a list of dictionaries for X and Y pulses
            self.pulse_params = np.empty(2, dtype=dict)
            # X pulse parameters are the first element in the list and Y pulses are the second element
            self.pulse_params[0] = {**pulse_params, **{'phi_t': 0}}
            self.pulse_params[1] = {**pulse_params, **{'phi_t': -np.pi/2}}
        
        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        # If projection_pulses is True, the sequence is set to the CPMG_sequence_proj method with the intial and final projection pulses into the Sz basis, otherwise it is set to the CPMG_sequence method without the projection pulses
        if projection_pulses:
            if H2 != None or self.system.c_ops != None:
                self.sequence = self.CPMG_sequence_proj_H2
            else:
                self.sequence = self.CPMG_sequence_proj
        elif not projection_pulses:
            if H2 != None or self.system.c_ops != None:
                self.sequence = self.CPMG_sequence_H2
            else:
                self.sequence = self.CPMG_sequence
        else:
            raise ValueError("projection_pulses must be a boolean")
        
        self.projection_pulses = projection_pulses

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
        ps = tau - self.pi_pulse_duration

        # perform free evolution of ps/2
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * self.system.rho0 * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        t0 = ps/2

        # repeat M times the pi pulse and free evolution of ps
        for itr_M in range(self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
            # perform free evolution of ps
            rho = (-1j*2*np.pi*self.system.H0*ps).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps).expm()).dag()
            t0 += tau

        # perform pi pulse
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
        # perform free evolution of ps/2
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        
        return rho
    
    
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
        ps = tau - self.pi_pulse_duration

        # initial pi/2 pulse on x 
        rho = mesolve(self.Ht, self.system.rho0, 2*np.pi*np.linspace(0, self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[1]).states[-1]

        # perform free evolution of tau/2
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        t0 = tau/2

        # repeat M times the pi pulse and free evolution of tau
        for itr_M in range(self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
            # perform free evolution of tau
            rho = (-1j*2*np.pi*self.system.H0*ps).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps).expm()).dag()
            t0 += tau

        # perform pi pulse
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
        # perform free evolution of tau/2
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        t0 += self.pi_pulse_duration + ps/2

        # final pi/2 pulse on x
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[1]).states[-1]

        return rho

    def CPMG_sequence_H2(self, tau):
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
        ps = tau - self.pi_pulse_duration

        # perform free evolution of ps/2
        rho = mesolve(self.system.H0, self.system.rho0, 2*np.pi*np.linspace(0, ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        t0 = ps/2

        # repeat M times the pi pulse and free evolution of ps
        for itr_M in range(self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
            t0 += self.pi_pulse_duration
            # perform free evolution of ps
            rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
            t0 += ps

        # perform pi pulse
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
        t0 += self.pi_pulse_duration
        # perform free evolution of ps/2
        rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        
        return rho
    
    
    def CPMG_sequence_proj_H2(self, tau):
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
        ps = tau - self.pi_pulse_duration

        # initial pi/2 pulse on x 
        rho = mesolve(self.Ht, self.system.rho0, 2*np.pi*np.linspace(0, self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[1]).states[-1]
        t0 = self.pi_pulse_duration/2

        # perform free evolution of tau/2
        rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        t0 += ps/2

        # repeat M times the pi pulse and free evolution of ps
        for itr_M in range(self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
            t0 += self.pi_pulse_duration
            # perform free evolution of ps
            rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
            t0 += ps

        # perform pi pulse
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
        t0 += self.pi_pulse_duration
        # perform free evolution of ps/2
        rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        t0 += ps/2

        # final pi/2 pulse on x
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[1]).states[-1]

        return rho
    
    def get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the CPMG sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau (float): free evolution variable or pulse spacing for the Hahn echo sequence
        """
        if tau == None:
            tau = self.variable[-1]
        # check weather tau is a positive real number and if it is, assign it to the object
        elif not isinstance(tau, (int, float)) or tau <= 0 or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration/2")

        # initialize the pulse_profiles attribute and total time
        self.pulse_profiles = []
        t0 = 0
        # calculate the pulse spacing
        ps = tau - self.pi_pulse_duration

        # if the projection_pulses is True, add the first pi/2 pulse
        if self.projection_pulses:
            if isinstance(self.H1, Qobj):
                # add the first pi/2 pulse
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params[1]] )
            elif isinstance(self.H1, list):
                # add the first pi/2 pulse
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params[1]] for i in range(len(self.H1))] )

            t0 += self.pi_pulse_duration/2

        # add the first free evolution of ps/2
        self.pulse_profiles.append( [None, [t0, t0 + ps/2], None, None] )
        t0 += ps/2

        # add pulses and free evolution M-1 times
        for itr_M in range(self.M-1):
            # add a pi pulse
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[0]] )
            elif isinstance(self.H1, list):
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))] )
            t0 += self.pi_pulse_duration
            # add a free evolution of ps
            self.pulse_profiles.append( [None, [t0, t0 + ps], None, None] )
            t0 += ps

        # add another pi pulse
        if isinstance(self.H1, Qobj):
            self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[0]] )
        elif isinstance(self.H1, list):
            self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))] )
        t0 += self.pi_pulse_duration
        # add half ps free evolution
        self.pulse_profiles.append( [None, [t0, t0 + ps/2], None, None] )
        t0 += ps/2        

        if self.projection_pulses:
            if isinstance(self.H1, Qobj):
                # add the last pi/2 pulse
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params[1]] )
            elif isinstance(self.H1, list):
                # add the first pi/2 pulse
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params[1]] for i in range(len(self.H1))] )

            t0 += self.pi_pulse_duration/2

        self.total_time = t0
    
    def plot_pulses(self, tau=None, figsize=(6, 6), xlabel='Pulse Separation', ylabel='Expectation Value', title='Pulse Profiles'):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the CPMG sequence for a given tau and then plot them.

        Parameters
        ----------
        tau (float): free evolution time for the Hahn echo sequence. Contrary to the run method, the free evoluiton must be a single number in order to plot the pulse profiles.
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        xlabel (str): label of the x-axis
        ylabel (str): label of the y-axis
        title (str): title of the plot
        """
        
        self.get_pulse_profiles(tau)

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)

####################################################################################################

class XY(PulsedExp):
    """
    This class contains the XY-M pulse sequence, inhereting from PulsedExp class. The sequence is composed of intercalated X and Y pi pulses and free evolutions repeated M times. It acts similar to the CPMG sequence, but the alternation of the pulse improves noise suppression on different axis.

    Class Attributes
    ----------------
    M (int): order of the XY sequence
    free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
    pi_pulse_duration (float, int): duration of the pi pulse
    projection_pulses (Boolean): boolean to determine if an initial pi/2 and final pi/2 pulses are to be included in order to project the measurement in the Sz basis
    += PulsedExp

    Class Methods
    -------------
    XY_sequence(tau): defines the XY sequence for a given free evolution time tau and the set of attributes defined in the generator, returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip.
    XY_sequence_proj(tau): defines the XY sequence for a given free evolution time tau and the set of attributes defined in the generator, returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip. An initial pi/2 pulse and final pi/2 pulse are included, in order to perform the measurement in the Sz basis.
    get_pulse_profiles(tau): generates the pulse profiles for the XY-M sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
    += PulsedExp
    """
    def __init__(self, M, free_duration, pi_pulse_duration, system, H1, H2=None, c_ops=None, projection_pulses=True, pulse_shape = square_pulse, pulse_params = {}, options = {}, time_steps = 100):
        """
        Class generator for the XY sequence

        Parameters
        ----------
        M (int): order of the XY sequence
        free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
        system (QSys): quantum system object containing the initial density matrix, internal time independent Hamiltonian and collapse operators
        H1 (Qobj, list(Qobj)): control Hamiltonian of the system
        pi_pulse_duration (float, int): duration of the pi pulse
        H2 (Qobj, list(Qobj)): time dependent sensing Hamiltonian of the system
        pulse_shape (FunctionType, list(FunctionType)): pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params (dict): dictionary of parameters for the pulse_shape functions
        time_steps (int): number of time steps in the pulses for the simulation
        """
        # call the parent class constructor
        super().__init__(system, H2)

        # check weather free_duration is a numpy array of real and positive elements and if it is, assign it to the object
        if not isinstance(free_duration, (np.ndarray, list)) or not np.all(np.isreal(free_duration)) or not np.all(np.greater_equal(free_duration, 0)):
            raise ValueError("free_duration must be a numpy array with real positive elements")
        else:
            self.variable = free_duration
            
        if not isinstance(M, int) or M <= 0:
            raise ValueError("M must be a positive integer")
        else:
            self.M = M

        # check weather pi_pulse_duration is a positive real number and if it is, assign it to the object
        if not isinstance(pi_pulse_duration, (int, float)) or pi_pulse_duration <= 0 or pi_pulse_duration > free_duration[0]:
            raise ValueError("pulse_duration must be a positive real number and pi_pulse_duration/2 must be smaller than the free evolution time, otherwise pulses will overlap")
        else:
            self.pi_pulse_duration = pi_pulse_duration

        # check weather time_steps is a positive integer and if it is, assign it to the object
        if not isinstance(time_steps, int) and time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        else:
            self.time_steps = time_steps       

        # check weather pulse_shape is a python function or a list of python functions and if it is, assign it to the object
        if isinstance(pulse_shape, FunctionType) or (isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape)):
            self.pulse_shape = pulse_shape
        else: 
            raise ValueError("pulse_shape must be a python function or a list of python functions")
        
        # check weather H1 is a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list and if it is, assign it to the object
        if isinstance(H1, Qobj) and H1.shape == self.system.rho0.shape:
            self.H1 = H1
            self.Ht = [self.system.H0, [self.H1, pulse_shape]]

        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.system.rho0.shape for op in H1) and len(H1) == len(pulse_shape):
            self.H1 = H1       
            self.Ht = [self.system.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            
        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")

        # check weather pulse_params is a dictionary and if it is, assign it to the object
        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            # initialize the pulse_params attribute as a list of dictionaries for X and Y pulses
            self.pulse_params = np.empty(2, dtype=dict)
            # X pulse parameters are the first element in the list and Y pulses are the second element
            self.pulse_params[0] = {**pulse_params, **{'phi_t': 0}}
            self.pulse_params[1] = {**pulse_params, **{'phi_t': -np.pi/2}}
        
        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        # If projection_pulses is True, the sequence is set to the XY_sequence_proj method with the intial and final projection pulses into the Sz basis, otherwise it is set to the XY_sequence method without the projection pulses
        if projection_pulses:
            if H2 != None or self.system.c_ops != None:
                self.sequence = self.XY_sequence_proj_H2
            else:
                self.sequence = self.XY_sequence_proj
        elif not projection_pulses:
            if H2 != None or self.system.c_ops != None:
                self.sequence = self.XY_sequence_H2
            else:
                self.sequence = self.XY_sequence
        else:
            raise ValueError("projection_pulses must be a boolean")

        self.projection_pulses = projection_pulses   

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

        # perform half free evolution
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * self.system.rho0 * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        t0 = ps/2

        # repeat M times the pi X pulse, free evolution of tau pi Y pulse and free evolution of tau
        for itr_M in range(2*self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[itr_M%2]).states[-1]
            # perform free evolution of tau
            rho = (-1j*2*np.pi*self.system.H0*ps).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps).expm()).dag()
            t0 += tau

        # perform pi pulse on Y axis
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[1]).states[-1]

        # perform free evolution of tau/2
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()

        return rho

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

        # initial pi/2 pulse
        rho = mesolve(self.Ht, self.system.rho0, 2*np.pi*np.linspace(0, self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]

        # perform half free evolution
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        t0 = tau/2

        # repeat M times the pi X pulse, free evolution of tau pi Y pulse and free evolution of tau
        for itr_M in range(2*self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[itr_M%2]).states[-1]
            # perform free evolution of tau
            rho = (-1j*2*np.pi*self.system.H0*ps).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps).expm()).dag()
            t0 += tau

        # perform pi pulse on Y axis
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[1]).states[-1]
        # perform free evolution of tau/2
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        t0 += self.pi_pulse_duration + ps/2

        # final pi/2 pulse
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]

        return rho

    def XY_sequence_H2(self, tau):
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

        # perform half free evolution
        rho = mesolve(self.system.H0, self.system.rho0, 2*np.pi*np.linspace(0, ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        t0 = ps/2

        # repeat M times the pi X pulse, free evolution of tau pi Y pulse and free evolution of tau
        for itr_M in range(2*self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[itr_M%2]).states[-1]
            t0 += self.pi_pulse_duration
            # perform free evolution of tau
            rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
            t0 += ps

        # perform pi pulse on Y axis
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[1]).states[-1]
        t0 += self.pi_pulse_duration

        # perform free evolution of tau/2
        rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]

        return rho

    def XY_sequence_proj_H2(self, tau):
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

        # initial pi/2 pulse
        rho = mesolve(self.Ht, self.system.rho0, 2*np.pi*np.linspace(0, self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
        t0 = self.pi_pulse_duration/2

        # perform half free evolution
        rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        t0 += ps/2

        # repeat M times the pi X pulse, free evolution of tau pi Y pulse and free evolution of tau
        for itr_M in range(2*self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[itr_M%2]).states[-1]
            t0 += self.pi_pulse_duration
            # perform free evolution of tau
            rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
            t0 += ps

        # perform pi pulse on Y axis
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[1]).states[-1]
        t0 += self.pi_pulse_duration

        # perform free evolution of tau/2
        rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        t0 += ps/2

        # final pi/2 pulse
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]

        return rho
    
    def get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the XY-M sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
        
        Parameters
        ----------
        tau (float): free evolution variable or pulse spacing for the Hahn echo sequence
        """
        if tau == None:
            tau = self.variable[-1]
        # check weather tau is a positive real number and if it is, assign it to the object
        elif not isinstance(tau, (int, float)) or tau <= 0 or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration/2")

        # initialize the pulse_profiles attribute and total time
        self.pulse_profiles = []
        t0 = 0
        # calculate the pulse spacing
        ps = tau - self.pi_pulse_duration

        # if the projection_pulses is True, add the first pi/2 pulse
        if self.projection_pulses:
            if isinstance(self.H1, Qobj):
                # add the first pi/2 pulse
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params[0]] )
            elif isinstance(self.H1, list):
                # add the first pi/2 pulse
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))] )

            t0 += self.pi_pulse_duration/2

        # add the first free evolution of ps/2
        self.pulse_profiles.append( [None, [t0, t0 + ps/2], None, None] )
        t0 += ps/2
        
        # add pulses and free evolution M-1 times
        for itr_M in range(2*self.M-1):
            # add a pi pulse
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[itr_M%2]] )
            elif isinstance(self.H1, list):
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[itr_M%2]] for i in range(len(self.H1))] )
            t0 += self.pi_pulse_duration
            # add a free evolution of ps
            self.pulse_profiles.append( [None, [t0, t0 + ps], None, None] )
            t0 += ps

        # add another pi pulse
        if isinstance(self.H1, Qobj):
            self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[1]] )
        elif isinstance(self.H1, list):
            self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[1]] for i in range(len(self.H1))] )
        t0 += self.pi_pulse_duration
        # add half ps free evolution
        self.pulse_profiles.append( [None, [t0, t0 + ps/2], None, None] )
        t0 += ps/2        

        if self.projection_pulses:
            if isinstance(self.H1, Qobj):
                # add the last pi/2 pulse
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params[0]] )
            elif isinstance(self.H1, list):
                # add the first pi/2 pulse
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))] )

            t0 += self.pi_pulse_duration/2

        self.total_time = t0
    
    def plot_pulses(self, tau=None, figsize=(6, 6), xlabel='Pulse Separation', ylabel='Expectation Value', title='Pulse Profiles'):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the XY-M sequence for a given tau and then plot them.

        Parameters
        ----------
        tau (float): free evolution time for the Hahn echo sequence. Contrary to the run method, the free evoluiton must be a single number in order to plot the pulse profiles.
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        xlabel (str): label of the x-axis
        ylabel (str): label of the y-axis
        title (str): title of the plot
        """
        # generate the pulse profiles for the given tau       
        self.get_pulse_profiles(tau)

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)

####################################################################################################

class XY8(PulsedExp):
    """
    This contains the XY8-M sequence, inhereting from Pulsed Experiment. The XY8-M is a further improvement from the XY-M sequence, where the X and Y pulses are group antisymmetrically in pairs of 4 as X-Y-X-Y-Y-X-Y-X, in order to improve noise suppression and pulse errors.

    Class Attributes
    ----------------
    M (int): order of the XY sequence
    free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
    pi_pulse_duration (float, int): duration of the pi pulse
    projection_pulses (Boolean): boolean to determine if an initial pi/2 and final pi/2 pulses are to be included in order to project the measurement in the Sz basis
    += PulsedExp

    Class Methods
    -------------
    XY8_sequence(tau): defines the XY8 sequence for a given free evolution time tau and the set of attributes defined in the generator, returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip.
    XY8_sequence_proj(tau): defines the XY8 sequence for a given free evolution time tau and the set of attributes defined in the generator, returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip. An initial pi/2 pulse and final pi/2 pulse are included, in order to perform the measurement in the Sz basis.
    get_pulse_profiles(tau): generates the pulse profiles for the XY8-M sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
    += PulsedExp
    """
    def __init__(self, M, free_duration, pi_pulse_duration, system, H1, H2=None, c_ops=None, projection_pulses=True, pulse_shape = square_pulse, pulse_params = {}, options = {}, time_steps = 100):
        """
        Class generator for the XY8 sequence

        Parameters
        ----------
        M (int): order of the XY sequence
        free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
        system (QSys): quantum system object containing the initial density matrix, internal time independent Hamiltonian and collapse operators
        H1 (Qobj, list(Qobj)): control Hamiltonian of the system
        pi_pulse_duration (float, int): duration of the pi pulse
        H2 (Qobj, list(Qobj)): time dependent sensing Hamiltonian of the system
        pulse_shape (FunctionType, list(FunctionType)): pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params (dict): dictionary of parameters for the pulse_shape functions
        time_steps (int): number of time steps in the pulses for the simulation
        """
        # call the parent class constructor
        super().__init__(system, H2)

        # check weather free_duration is a numpy array of real and positive elements and if it is, assign it to the object
        if not isinstance(free_duration, (np.ndarray, list)) or not np.all(np.isreal(free_duration)) or not np.all(np.greater_equal(free_duration, 0)):
            raise ValueError("free_duration must be a numpy array with real positive elements")
        else:
            self.variable = free_duration
            
        if not isinstance(M, int) or M <= 0:
            raise ValueError("M must be a positive integer")
        else:
            self.M = M

        # check weather pi_pulse_duration is a positive real number and if it is, assign it to the object
        if not isinstance(pi_pulse_duration, (int, float)) or pi_pulse_duration <= 0 or pi_pulse_duration > free_duration[0]:
            raise ValueError("pulse_duration must be a positive real number and pi_pulse_duration/2 must be smaller than the free evolution time, otherwise pulses will overlap")
        else:
            self.pi_pulse_duration = pi_pulse_duration

        # check weather time_steps is a positive integer and if it is, assign it to the object
        if not isinstance(time_steps, int) and time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        else:
            self.time_steps = time_steps       

        # check weather pulse_shape is a python function or a list of python functions and if it is, assign it to the object
        if isinstance(pulse_shape, FunctionType) or (isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape)):
            self.pulse_shape = pulse_shape
        else: 
            raise ValueError("pulse_shape must be a python function or a list of python functions")
        
        # check weather H1 is a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list and if it is, assign it to the object
        if isinstance(H1, Qobj) and H1.shape == self.system.rho0.shape:
            self.H1 = H1
            self.Ht = [self.system.H0, [self.H1, pulse_shape]]

        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.system.rho0.shape for op in H1) and len(H1) == len(pulse_shape):
            self.H1 = H1       
            self.Ht = [self.system.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            
        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")

        # check weather pulse_params is a dictionary and if it is, assign it to the object
        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            self.pulse_params = pulse_params
            # initialize the pulse_params attribute as a list of dictionaries for X and Y pulses
            self.pulse_params = np.empty(8, dtype=dict)
            # define XY8 pulse parameters for the sequence: X-Y-X-Y-Y-X-Y-X
            self.pulse_params[0] = {**pulse_params, **{'phi_t': 0}}
            self.pulse_params[1] = {**pulse_params, **{'phi_t': -np.pi/2}}
            self.pulse_params[2] = self.pulse_params[0]
            self.pulse_params[3] = self.pulse_params[1]
            self.pulse_params[4] = self.pulse_params[1]
            self.pulse_params[5] = self.pulse_params[0]
            self.pulse_params[6] = self.pulse_params[1]
            self.pulse_params[7] = self.pulse_params[0]
        
        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        # If projection_pulses is True, the sequence is set to the XY8_sequence_proj method with the intial and final projection pulses into the Sz basis, otherwise it is set to the XY8_sequence method without the projection pulses
        if projection_pulses:
            if H2 != None or self.system.c_ops != None:
                self.sequence = self.XY8_sequence_proj_H2
            else:
                self.sequence = self.XY8_sequence_proj
        elif not projection_pulses:
            if H2 != None or self.system.c_ops != None:
                self.sequence = self.XY8_sequence_H2
            else:
                self.sequence = self.XY8_sequence
        else:
            raise ValueError("projection_pulses must be a boolean")

        self.projection_pulses = projection_pulses
    
    def XY8_sequence(self, tau):
        """
        Defines the XY8-M composed of 8 intercalated pi pulses on X and Y axis with free evolutions of time tau repeated M times. The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix        
        """
        # calculate the pulse spacing
        ps = tau - self.pi_pulse_duration

        # perform half free evolution
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * self.system.rho0 * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        t0 = ps/2

        # repeat 8*M-1 times alternated pi pulses on X and Y axis and free evolutions of tau
        for itr_M in range(8*self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[itr_M%8]).states[-1]
            # perform free evolution of tau
            rho = (-1j*2*np.pi*self.system.H0*ps).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps).expm()).dag()
            t0 += tau

        # perform pi pulse on X axis
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]

        # perform free evolution of tau/2
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()

        return rho
        
    def XY8_sequence_proj(self, tau):
        """
        Defines the XY8-M composed of 8 intercalated pi pulses on X and Y axis with free evolutions of time tau repeated M times. The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix        
        """
        # calculate the pulse spacing
        ps = tau - self.pi_pulse_duration

        # perform pi/2 pulse
        rho = mesolve(self.Ht, self.system.rho0, 2*np.pi*np.linspace(0, self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]

        # perform half free evolution
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        t0 = tau/2 

        # repeat 8*M-1 times alternated pi pulses on X and Y axis and free evolutions of tau
        for itr_M in range(8*self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[itr_M%8]).states[-1]
            # perform free evolution of tau
            rho = (-1j*2*np.pi*self.system.H0*ps).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps).expm()).dag()
            t0 += tau

        # perform pi pulse on Y axis
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]

        # perform free evolution of tau/2
        rho = (-1j*2*np.pi*self.system.H0*ps/2).expm() * rho * ((-1j*2*np.pi*self.system.H0*ps/2).expm()).dag()
        t0 += self.pi_pulse_duration + ps/2

        # perform pi/2 pulse
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]

        return rho   
    
    def XY8_sequence_H2(self, tau):
        """
        Defines the XY8-M composed of 8 intercalated pi pulses on X and Y axis with free evolutions of time tau repeated M times. The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix        
        """
        # calculate the pulse spacing
        ps = tau - self.pi_pulse_duration

        # perform half free evolution
        rho = mesolve(self.system.H0, self.system.rho0, 2*np.pi*np.linspace(0, ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        t0 = ps/2

        # repeat 8*M-1 times alternated pi pulses on X and Y axis and free evolutions of tau
        for itr_M in range(8*self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[itr_M%8]).states[-1]
            t0 += self.pi_pulse_duration
            # perform free evolution of tau
            rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
            t0 += ps

        # perform pi pulse on X axis
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
        t0 += self.pi_pulse_duration

        # perform free evolution of tau/2
        rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]

        return rho
        
    def XY8_sequence_proj_H2(self, tau):
        """
        Defines the XY8-M composed of 8 intercalated pi pulses on X and Y axis with free evolutions of time tau repeated M times. The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix        
        """
        # calculate the pulse spacing
        ps = tau - self.pi_pulse_duration

        # perform pi/2 pulse
        rho = mesolve(self.Ht, self.system.rho0, 2*np.pi*np.linspace(0, self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
        t0 = self.pi_pulse_duration/2

        # perform half free evolution
        rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(0, ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        t0 += ps/2

        # repeat 8*M-1 times alternated pi pulses on X and Y axis and free evolutions of tau
        for itr_M in range(8*self.M-1):
            # perform pi pulse
            rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[itr_M%8]).states[-1]
            t0 += self.pi_pulse_duration
            # perform free evolution of tau
            rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
            t0 += ps

        # perform pi pulse on X axis
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]
        t0 += self.pi_pulse_duration

        # perform free evolution of tau/2
        rho = mesolve(self.system.H0, rho, 2*np.pi*np.linspace(t0, t0 + ps/2, self.time_steps) , self.system.c_ops, [], options = self.options).states[-1]
        t0 += ps/2

        # perform pi/2 pulse
        rho = mesolve(self.Ht, rho, 2*np.pi*np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps) , self.system.c_ops, [], options = self.options, args = self.pulse_params[0]).states[-1]

        return rho
    
    def get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the XY-M sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
        
        Parameters
        ----------
        tau (float): free evolution variable or pulse spacing for the Hahn echo sequence
        """
        if tau == None:
            tau = self.variable[-1]
        # check weather tau is a positive real number and if it is, assign it to the object
        elif not isinstance(tau, (int, float)) or tau <= 0 or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration/2")

        # initialize the pulse_profiles attribute and total time
        self.pulse_profiles = []
        t0 = 0
        # calculate the pulse spacing
        ps = tau - self.pi_pulse_duration

        # if the projection_pulses is True, add the first pi/2 pulse
        if self.projection_pulses:
            if isinstance(self.H1, Qobj):
                # add the first pi/2 pulse
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params[0]] )
            elif isinstance(self.H1, list):
                # add the first pi/2 pulse
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))] )

            t0 += self.pi_pulse_duration/2

        # add the first free evolution of ps/2
        self.pulse_profiles.append( [None, [t0, t0 + ps/2], None, None] )
        t0 += ps/2
        
        # add pulses and free evolution M-1 times
        for itr_M in range(8*self.M-1):
            # add a pi pulse
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[itr_M%8]] )
            elif isinstance(self.H1, list):
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[itr_M%8]] for i in range(len(self.H1))] )
            t0 += self.pi_pulse_duration
            # add a free evolution of ps
            self.pulse_profiles.append( [None, [t0, t0 + ps], None, None] )
            t0 += ps

        # add another pi pulse
        if isinstance(self.H1, Qobj):
            self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[0]] )
        elif isinstance(self.H1, list):
            self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))] )
        t0 += self.pi_pulse_duration
        # add half ps free evolution
        self.pulse_profiles.append( [None, [t0, t0 + ps/2], None, None] )
        t0 += ps/2        

        if self.projection_pulses:
            if isinstance(self.H1, Qobj):
                # add the last pi/2 pulse
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params[0]] )
            elif isinstance(self.H1, list):
                # add the first pi/2 pulse
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))] )

            t0 += self.pi_pulse_duration/2

        self.total_time = t0

        # set the total_time attribute to the total time of the pulse sequence
        self.total_time = t0
    
    def plot_pulses(self, tau=None, figsize=(6, 6), xlabel=None, ylabel='Expectation Value', title='Pulse Profiles'):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the XY-M sequence for a given tau and then plot them.

        Parameters
        ----------
        tau (float): free evolution time for the Hahn echo sequence. Contrary to the run method, the free evoluiton must be a single number in order to plot the pulse profiles.
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        self.xlabel = 'Free Evolution Time'
        xlabel (str): label of the x-axis
        ylabel (str): label of the y-axis
        title (str): title of the plot
        """
        # check weather tau is a positive real number and if it is, assign it to the object
        if not isinstance(tau, (int, float)) and tau <= 0 and tau > self.pi_pulse_duration/2:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration/2")
        
        self.get_pulse_profiles(tau)

        # if no xlabel is given, set the xlabel to the default value
        if xlabel == None:
            pass
        elif isinstance(xlabel, str):
            self.xlabel = xlabel
        else:
            raise ValueError("xlabel must be a string or None")

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)