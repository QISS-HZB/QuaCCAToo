"""
This module contains predefined basic pulsed experiments inheriting from the PulsedExp class.

Classes
-------
- Rabi: resonant pulse of varying duration, such that the quantum system will undergo periodical transitions between the excited and ground states.
- PODMR: Pulsed Optically Detected Magnetic Resonance (PODMR) experiment, composed by a single pulse where the frequency is changed such that when it correponds to a transition in the Hamiltonian of the system, the observable will be affected.
- Ramsey: Ramsey experiment, consiting of a free evolution that causes a phase accumulation between states in the sytem which can be used for interferometry.
- Hahn: Hahn echo experiment, consisting of two free evolutions with a pi pulse in the middle, in order to cancel out dephasings. The Hahn echo is usually used to measure the coherence time of a quantum system, however it can also be used to sense coupled spins.
"""

import numpy as np
from qutip import Qobj, mesolve
from types import FunctionType
from.PulsedExp import PulsedExp
from.PulseShapes import square_pulse

class Rabi(PulsedExp):
    """
    This class contains a Rabi experiments, inheriting from the PulsedExperiment class. A Rabi sequences is composed of a resonant pulse of varying duration, such that the quantum system will undergo periodical transitions between the excited and ground states.

    Class Attributes
    ----------------
    pulse_duration (numpy array): time array for the simulation representing the pulse duration to be used as the variable for the simulation
    += PulsedExperiment

    Methods
    -------
    = PulsedExperiment    
    """
    def __init__(self, pulse_duration, system, H1, H2=None, pulse_shape = square_pulse, pulse_params = {}, options={}):
        """
        Generator for the Rabi pulsed experiment class, taking a specific pulse_duration to run the simulation.

        Parameters
        ----------
        pulse_duration (numpy array):time array for the simulation representing to be used as the variable for the simulation
        system (QSys): quantum system object containing the initial density matrix, internal Hamiltonian and collapse operators
        H1 (Qobj, list(Qobj)): control Hamiltonian of the system
        H2 (Qobj, list(Qobj)): time dependent sensing Hamiltonian of the system
        pulse_shape (FunctionType, list(FunctionType)): pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params (dict): dictionary of parameters for the pulse_shape functions
        options (dict): dictionary of solver options from Qutip
        """
        # call the parent class constructor
        super().__init__(system, H2)

        # check weather pulse_duration is a numpy array and if it is, assign it to the object
        if not isinstance(pulse_duration, (np.ndarray, list)) or not np.all(np.isreal(pulse_duration)) or not np.all(np.greater_equal(pulse_duration, 0)):
            raise ValueError("pulse_duration must be a numpy array of real positive elements")
        else:
            self.total_time = pulse_duration[-1]
            self.variable = pulse_duration

        # check weather pulse_shape is a python function or a list of python functions and if it is, assign it to the object
        if isinstance(pulse_shape, FunctionType) or (isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape) ):
            self.pulse_shape = pulse_shape
        else: 
            raise ValueError("pulse_shape must be a python function or a list of python functions")
        
        # check weather H1 is a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list and if it is, assign it to the object
        if isinstance(H1, Qobj) and H1.shape == self.rho0.shape:
            self.Ht = [self.H0, [H1, pulse_shape]]
            self.pulse_profiles = [[H1, pulse_duration, pulse_shape, pulse_params]]
            
        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.rho0.shape for op in H1) and len(H1) == len(pulse_shape):       
            self.Ht = [self.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            self.pulse_profiles = [[H1[i], pulse_duration, pulse_shape[i], pulse_params] for i in range(len(H1))]

        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")
        
        # check weather pulse_params is a dictionary and if it is, assign it to the object
        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            self.pulse_params = pulse_params
            # if phi_t is not in the pulse_params dictionary, assign it as 0
            if 'phi_t' not in pulse_params:
                self.pulse_params['phi_t'] = 0

        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options
                
    def run(self):
        """
        Overwrites the run method of the parent class. Runs the simulation and stores the results in the results attribute. If an observable is given, the expectation values are stored in the results attribute. For the Rabi sequence, the calculation is optimally performed sequentially instead of in parallel over the pulse lengths, thus the run method from the parent class is overwritten.      
        """
        # if no observable is given in QSys, run the simulation and store the the calculated density matrices in the results attribute
        if self.observable == None:
            self.results = mesolve(self.Ht, self.rho0, 2*np.pi*self.variable, self.c_ops, [], options = self.options, args = self.pulse_params).states
        
        # if one observable is given in QSys, run the simulation and store the expectation values in the results attribute
        elif isinstance(self.observable, Qobj):
            self.results = mesolve(self.Ht, self.rho0, 2*np.pi*self.variable, self.c_ops, self.observable, options = self.options, args = self.pulse_params).expect[0]
        
        # if more than one observable is given in QSys, run the simulation and store the expectation values in the results attribute
        elif isinstance(self.observable, list):
            self.results =  mesolve(self.Ht, self.rho0, 2*np.pi*self.variable, self.c_ops, self.observable, options = self.options, args = self.pulse_params).expect
        # otherwise raise an error
        else:
            raise ValueError("observable must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1.")

class PODMR(PulsedExp):
    """
    This class contains a Pulsed Optically Detected Magnetic Resonance (PODMR) experiments where the frequency is the variable being changed, inheriting from the PulsedExp class. The PODMR consists of a single pulse of fixed length and changing frequency. If the frequency matches a resonance of the system, it will go some transition which will affect the observable. This way, the differences between energy levels can be determined with the linewidht usually limited by the pulse lenght. Here we make reference to optical detection as it is the most common detection scheme of pulsed magnetic resonance in color centers, however the method can be more general. 

    Class Attributes
    ----------------
    frequencies (numpy array): array of frequencies to run the simulation
    pulsed_duration (float, int): duration of the pulse
    += PulsedExperiment

    Class Methods
    -------------
    PODMR_sequence(f): defines the the Pulsed Optically Detected Magnetic Resonance (PODMR) sequence for a given frequency of the pulse. To be called by the parallel_map in run method.
    += PulsedExperiment
    """
    def __init__(self, frequencies, pulse_duration, system, H1, H2=None, pulse_shape = square_pulse, pulse_params = {}, time_steps = 100, options={}):
        """
        Generator for the PODMR pulsed experiment class

        Parameters
        ----------
        frequencies (numpy array): array of frequencies to run the simulation
        pulse_duration (float, int): duration of the pulse
        system (QSys): quantum system object containing the initial density matrix, internal Hamiltonian and collapse operators
        H1 (Qobj, list(Qobj)): control Hamiltonian of the system
        H2 (Qobj, list(Qobj)): time dependent sensing Hamiltonian of the system
        pulse_shape (FunctionType, list(FunctionType)): pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params (dict): dictionary of parameters for the pulse_shape functions
        time_steps (int): number of time steps in the pulses for the simulation
        options (dict): dictionary of solver options from Qutip
        """
        # call the parent class constructor
        super().__init__(system, H2)

        # check weather frequencies is a numpy array or list and if it is, assign it to the object
        if not isinstance(frequencies, (np.ndarray, list)) or not np.all(np.isreal(frequencies)) or not np.all(np.greater_equal(frequencies, 0)):
            raise ValueError("frequencies must be a numpy array or list of real positive elements")
        else:
            self.variable = frequencies

        # check weather pulse_duration is a numpy array and if it is, assign it to the object
        if not isinstance(pulse_duration, (float, int)) or pulse_duration <= 0:
            raise ValueError("pulse_duration must be a positive real number")
        else:
            self.pulse_duration = pulse_duration

        # check weather pulse_shape is a python function or a list of python functions and if it is, assign it to the object
        if isinstance(pulse_shape, FunctionType) or (isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape)):
            self.pulse_shape = pulse_shape
        else: 
            raise ValueError("pulse_shape must be a python function or a list of python functions")
        
        # check weather time_steps is a positive integer and if it is, assign it to the object
        if not isinstance(time_steps, int) or time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        else:
            self.time_steps = time_steps       
        
        # check weather pulse_params is a dictionary and if it is, assign it to the object
        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            self.pulse_params = pulse_params
            # check weather phi_t is in the pulse_params dictionary and if it is not, assign it to the object as 0
            if 'phi_t' not in pulse_params:
                self.pulse_params['phi_t'] = 0
        
        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options
        
        # check if H1 is a Qobj or a list of Qobj with the same dimensions as H0 and rho0
        if isinstance(H1, Qobj) and H1.shape == self.rho0.shape:
            # create the time independent + time dependent Hamiltonian
            self.Ht = [self.H0, [H1, pulse_shape]]
            # append it to the pulse_profiles list
            self.pulse_profiles.append( [H1, np.linspace(0, self.pulse_duration, self.time_steps), pulse_shape, pulse_params] )
        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.rho0.shape for op in H1) and len(H1) == len(pulse_shape):
            self.Ht = [self.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            self.pulse_profiles.append( [[H1[i], np.linspace(0, self.pulse_duration, self.time_steps), pulse_shape[i], pulse_params] for i in range(len(H1))] )
        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")

        # set the sequence attribute to the PODMR_sequence method
        self.sequence = self.PODMR_sequence

    def PODMR_sequence(self, f):
        """
        Defines the the Pulsed Optically Detected Magnetic Resonance (PODMR) sequence for a given frequency of the pulse. To be called by the parallel_map in run method.
        Parameters
        ----------
        f (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix   
        """
        self.rho = self.rho0.copy()
        self.pulse_params['omega_pulse'] = f

        # run the simulation and return the final density matrix
        self.pulse(self.Ht, np.linspace(0, self.pulse_duration, self.time_steps), self.options, self.pulse_params, self.pulse_params['phi_t'])

        if self.observable == None:
            return self.rho
        else:
            return np.abs( (self.rho * self.observable).tr() )
        
    def plot_pulses(self, omega_pulse=None, figsize=(6, 4), xlabel='Time', ylabel='Pulse Intensity', title='Pulse Profiles'):
        """
        Overwrites the plot_pulses method of the parent class in order to first define a pulse frequency to be plotted.

        Parameters
        ----------
        omega_pulse (float, int): frequency of the pulse to be plotted
        += PulsedExperiment.plot_pulses
        """
        # if omega_pulse is None, assign the first element of the variable attribute to the pulse_params dictionary
        if omega_pulse == None:
            self.pulse_params['omega_pulse'] = self.variable[0]
        # if omega_pulse is a float or an integer, assign it to the pulse_params dictionary
        elif isinstance(omega_pulse, (int, float)):
            self.pulse_params['omega_pulse'] = omega_pulse
        else:
            raise ValueError("omega_pulse must be a float or an integer")
        
        self.total_time = self.pulse_duration

        super().plot_pulses(figsize, xlabel, ylabel, title)

class Ramsey(PulsedExp):
    """
    This class contains a Ramsey experiments, inheriting from the PulsedExperiment class.

    Class Attributes
    ----------------
    free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
    pi_pulse_duration (float, int): duration of the pi pulse
    projection_pulses (Boolean): boolean to determine if an initial pi/2 and final pi/2 pulses are to be included in order to project the measurement in the Sz basis
    += PulsedExperiment

    Class Methods
    -------------
    ramsey_sequence(tau): defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a single free evolution. The sequence is to be called by the parallel_map method of QuTip.
    ramsey_sequence_proj(tau): defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a single free evolution plus an initial and final pi/2 pulses to project into the Sz basis. The sequence is to be called by the parallel_map method of QuTip.
    get_pulse_profiles(tau): generates the pulse profiles for the Ramsey sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
    += PulsedExperiment
    """
    def __init__(self, free_duration, pi_pulse_duration, system, H1, H2=None, projection_pulses = True, pulse_shape = square_pulse, pulse_params = {}, options={}, time_steps = 100):
        """
        Class generator for the Ramsey pulsed experiment class

        Parameters
        ----------
        free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
        system (QSys): quantum system object containing the initial density matrix, internal Hamiltonian and collapse operators
        H1 (Qobj, list(Qobj)): control Hamiltonian of the system
        pi_pulse_duration (float, int): duration of the pi pulse
        H2 (Qobj, list(Qobj)): time dependent sensing Hamiltonian of the system
        projection_pulses (Boolean): boolean to determine if the measurement is to be performed in the Sz basis or not. If True, a initial pi/2 pulse and a final pi/2 pulse are included in order to project into the Sz basis, as in most color centers.
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

        # check weather pi_pulse_duration is a positive real number and if it is, assign it to the object
        if not isinstance(pi_pulse_duration, (int, float)) or pi_pulse_duration <= 0 or pi_pulse_duration/2 > free_duration[-1]:
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

        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            self.pulse_params = pulse_params
            if 'phi_t' not in pulse_params:
                self.pulse_params['phi_t'] = 0

        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        # check if H1 is a Qobj or a list of Qobj with the same dimensions as H0 and rho0
        if isinstance(H1, Qobj) and H1.shape == self.rho0.shape:
            # create the time independent + time dependent Hamiltonian
            self.H1 = H1
            self.Ht = [self.H0, [H1, pulse_shape]]
        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.rho0.shape for op in H1) and len(H1) == len(pulse_shape):
            self.H1 = H1
            self.Ht = [self.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")

        # If projection_pulses is True, the sequence is set to the ramsey_sequence_proj method with the intial and final projection pulses into the Sz basis, otherwise it is set to the ramsey_sequence method without the projection pulses
        if projection_pulses:
            self.sequence = self.ramsey_sequence_proj
        elif not projection_pulses:
            self.sequence = self.ramsey_sequence
        else:
            raise ValueError("projection_pulses must be a boolean")
        
        self.projection_pulses = projection_pulses

    def ramsey_sequence(self, tau):
        """
        Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a single free evolution. The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix        
        """
        # set the total time to 0
        self.total_time = 0
        # initialize the density matrix to the initial density matrix
        self.rho = self.rho0.copy()

        # perform the free evolution
        self.free_evolution(tau)

        # if no observable is given, return the final density matrix
        if self.observable == None:
            return self.rho
        # if an observable is given, return the expectation value of the observable
        else:
            return np.abs( (self.rho * self.observable).tr() )

    def ramsey_sequence_proj(self, tau):
        """
        Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a single free evolution plus an initial and final pi/2 pulses to project into the Sz basis. The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix        
        """
        # set the total time to 0
        self.total_time = 0
        # initialize the density matrix to the initial density matrix
        self.rho = self.rho0.copy()
        
        # perform initial pi/2 pulse
        self.pulse(self.Ht, np.linspace(self.total_time, self.total_time + self.pi_pulse_duration/2, self.time_steps), self.options, self.pulse_params, 0)
        # perform the free evolution
        self.free_evolution(tau)
        # perform final pi/2 pulse
        self.pulse(self.Ht, np.linspace(self.total_time, self.total_time + self.pi_pulse_duration/2, self.time_steps), self.options, self.pulse_params, 0)

        # if no observable is given, return the final density matrix
        if self.observable == None:
            return self.rho
        # if an observable is given, return the expectation value of the observable
        else:
            return np.abs( (self.rho * self.observable).tr() )
        
    def get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the Ramsey sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau (float): free evolution variable or pulse spacing for the Hahn echo sequence
        """
        # check if tau is None and if it is, assign the first element of the variable attribute to tau
        if tau == None:
            tau = self.variable[0]
        # else if it is not a float or an integer, raise an error
        elif not isinstance(tau, (int, float)):
            raise ValueError("tau must be a float or an integer")
        
        # initialize the pulse_profiles attribute to an empty list
        self.pulse_profiles = []

        # if tau is None, assign the first element of the variable attribute to tau
        if tau == None:
            tau = self.variable[0]
        # if tau is not a float or an integer, raise an error
        elif not isinstance(tau, (int, float)):
            raise ValueError("tau must be a float or an integer")
        
        # if projection_pulses is True, include initial and final pi/2 pulses in the pulse_profiles
        if self.projection_pulses:
            # if only one control Hamiltonian is given, append the pulse_profiles with the Ramsey sequence
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append([self.H1, np.linspace(0, self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 = self.pi_pulse_duration/2
                self.pulse_profiles.append( [None, [t0, tau + t0], None, None] )
                t0 += tau
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 += self.pi_pulse_duration/2

            # otherwise if a list of control Hamiltonians is given, it sums over all H1 and appends to the pulse_profiles
            elif isinstance(self.H1, list):
                self.pulse_profiles.append( [[self.H1[i], np.linspace(0, self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))] )
                t0 = self.pi_pulse_duration/2
                self.pulse_profiles.append( [None, [t0, tau + t0], None, None] )
                t0 += tau
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))] )
                t0 += self.pi_pulse_duration/2

        # if projection_pulses is false, do not include initial and final pi/2 pulses in the pulse_profiles
        else:
            self.pulse_profiles.append( [None, [0, tau], None, None] )
            t0 = tau

        # set the total time to t0
        self.total_time = t0
    
    def plot_pulses(self, tau=None, figsize=(6, 4), xlabel='Free Evolution Time', ylabel='Pulse Intensity', title='Pulse Profiles of Ramsey Sequence'):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the Ramsey sequence for a given tau and then plot them.

        Parameters
        ----------
        tau (float): free evolution time for the Hahn echo sequence. Contrary to the run method, the free evoluiton must be a single number in order to plot the pulse profiles.
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        xlabel (str): label of the x-axis
        ylabel (str): label of the y-axis
        title (str): title of the plot
        """
        # generate the pulse profiles for the Ramsey sequence for a given tau
        self.get_pulse_profiles(tau)

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)
    
class Hahn(PulsedExp):
    """
    This class contains a Hahn echo experiment, inheriting from the PulsedExperiment class. The Hahn echo sequence consists of two free evolutions with a pi pulse in the middle, in order to cancel out dephasings. The Hahn echo is usually used to measure the coherence time of a quantum system, however it can also be used to sense coupled spins.

    Class Attributes
    ----------------
    free_duration (numpy array): time array of the free evolution times to run the simulation
    pi_pulse_duration (float, int): duration of the pi pulse
    projection_pulses (Boolean): boolean to determine if an initial pi/2 and final pi/2 pulses are to be included in order to project the measurement in the Sz basis
    += PulsedExperiment

    Class Methods
    -------------
    hahn_sequence(tau): defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the generator, returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip.
    hahn_sequence_proj(tau): defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the generator, returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip. An initial pi/2 pulse and final pi/2 pulse are included, in order to perform the measurement in the Sz basis.
    += PulsedExperiment    
    """

    def __init__(self, free_duration, pi_pulse_duration, system, H1,  H2=None, projection_pulses = True, pulse_shape = square_pulse, pulse_params = {}, options={}, time_steps = 100):
        """
        Generator for the Hahn echo pulsed experiment class, taking a specific free_duration to run the simulation and the pi_pulse_duration.

        Parameters
        ----------
        free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
        system (QSys): quantum system object containing the initial density matrix, internal Hamiltonian and collapse operators
        H1 (Qobj, list(Qobj)): control Hamiltonian of the system
        pi_pulse_duration (float, int): duration of the pi pulse
        H2 (Qobj, list(Qobj)): time dependent sensing Hamiltonian of the system
        projection_pulses (Boolean): boolean to determine if the measurement is to be performed in the Sz basis or not. If True, a initial pi/2 pulse and a final pi/2 pulse are included in order to project into the Sz basis, as in most color centers.
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

        # check weather pi_pulse_duration is a positive real number and if it is, assign it to the object
        if not isinstance(pi_pulse_duration, (int, float)) or pi_pulse_duration <= 0 or pi_pulse_duration/2 > free_duration[-1]:
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
        if isinstance(H1, Qobj) and H1.shape == self.rho0.shape:
            self.H1 = H1
            self.Ht = [self.H0, [self.H1, pulse_shape]]
        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.rho0.shape for op in H1) and len(H1) == len(pulse_shape):
            self.H1 = H1       
            self.Ht = [self.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")

        # check weather pulse_params is a dictionary and if it is, assign it to the object
        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            self.pulse_params = pulse_params
            # if phi_t is not in the pulse_params dictionary, assign it as 0
            if 'phi_t' not in pulse_params:
                self.pulse_params['phi_t'] = 0

        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        # If projection_pulses is True, the sequence is set to the hahn_sequence_proj method with the intial and final projection pulses into the Sz basis, otherwise it is set to the hahn_sequence method without the projection pulses
        if projection_pulses:
            self.sequence = self.hahn_sequence_proj
        elif not projection_pulses:
            self.sequence = self.hahn_sequence
        else:
            raise ValueError("projection_pulses must be a boolean")
        
        self.projection_pulses = projection_pulses
        
    def hahn_sequence(self, tau):
        """
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of two free evolutions with a pi pulse between them. The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix        
        """
        # set the total time to 0
        self.total_time = 0
        # initialize the density matrix to the initial density matrix
        self.rho = self.rho0.copy()

        # perform the first free evolution
        self.free_evolution(tau - self.pi_pulse_duration/2)

        # perform the pi pulse
        self.pulse(self.Ht, np.linspace(self.total_time, self.total_time + self.pi_pulse_duration, self.time_steps), self.options, self.pulse_params, 0)

        # perform the second free evolution
        self.free_evolution(tau - self.pi_pulse_duration/2)

        # if no observable is given, return the final density matrix
        if self.observable == None:
            return self.rho
        # if an observable is given, return the expectation value of the observable
        else:
            return np.abs( (self.rho * self.observable).tr() )
    
    def hahn_sequence_proj(self, tau):
        """
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a pi/2 pulse, a free evolution time tau, a pi pulse and another free evolution time tau followed by a pi/2 pulse. The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix        
        """
        # set the total time to 0
        self.total_time = 0
        # initialize the density matrix to the initial density matrix
        self.rho = self.rho0.copy()

        # perform the initial pi/2 pulse
        self.pulse(self.Ht, np.linspace(self.total_time, self.total_time + self.pi_pulse_duration/2, self.time_steps), self.options, self.pulse_params, 0)
        # perform the first free evolution
        self.free_evolution(tau - self.pi_pulse_duration/2)
        # perform the pi pulse
        self.pulse(self.Ht, np.linspace(self.total_time, self.total_time + self.pi_pulse_duration, self.time_steps), self.options, self.pulse_params, 0)
        # perform the second free evolution
        self.free_evolution(tau - self.pi_pulse_duration/2)
        # perform the final pi/2 pulse
        self.pulse(self.Ht, np.linspace(self.total_time, self.total_time + self.pi_pulse_duration/2, self.time_steps), self.options, self.pulse_params, 0)

        # if no observable is given, return the final density matrix
        if self.observable == None:
            return self.rho
        else:
            return np.abs( (self.rho * self.observable).tr() )
        
    def get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the Hahn echo sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau (float): free evolution variable or pulse spacing for the Hahn echo sequence 
        """
        # check if tau is None and if it is, assign the first element of the variable attribute to tau
        if tau == None:
            tau = self.variable[0]
        # else if it is not a float or an integer, raise an error
        elif not isinstance(tau, (int, float)) and tau <= 0 and tau > self.pi_pulse_duration/2:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration/2")
        
        # initialize the pulse_profiles attribute to an empty list and t0 to 0
        self.pulse_profiles = []
        t0 = 0

        # if projection_pulses is True, include initial and final pi/2 pulses in the pulse_profiles
        if self.projection_pulses:
            # if only one control Hamiltonian is given, append the pulse_profiles with the Hahn echo sequence as in the hahn_sequence method
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params] )
                t0 += self.pi_pulse_duration/2
                self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
                t0 += tau - self.pi_pulse_duration/2
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params] )
                t0 += self.pi_pulse_duration
                self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
                t0 += tau - self.pi_pulse_duration/2
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params] )
                t0 += self.pi_pulse_duration/2

            # otherwise if a list of control Hamiltonians is given, it sums over all H1 and appends to the pulse_profiles the Hahn echo sequence as in the hahn_sequence method
            elif isinstance(self.H1, list):
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))] )
                t0 += self.pi_pulse_duration/2
                self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
                t0 += tau - self.pi_pulse_duration/2
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))] )
                t0 += self.pi_pulse_duration
                self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
                t0 += tau - self.pi_pulse_duration/2
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))] )
                t0 += self.pi_pulse_duration/2

        # if projection_pulses is False, do not include initial and final pi/2 pulses in the pulse_profiles
        else:
            # if only one control Hamiltonian is given, append the pulse_profiles with the Hahn echo sequence as in the hahn_sequence method
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
                t0 += tau - self.pi_pulse_duration/2
                self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params] )
                t0 += self.pi_pulse_duration
                self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
                t0 += tau - self.pi_pulse_duration/2

            # otherwise if a list of control Hamiltonians is given, it sums over all H1 and appends to the pulse_profiles the Hahn echo sequence as in the hahn_sequence method
            elif isinstance(self.H1, list):
                self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
                t0 += tau - self.pi_pulse_duration/2
                self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))] )
                t0 += self.pi_pulse_duration
                self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
                t0 += tau - self.pi_pulse_duration/2
        
        # set the total_time attribute to the total time of the pulse sequence
        self.total_time = t0
        
    def plot_pulses(self, tau=None, figsize=(6, 6), xlabel='Free Evolution Time', ylabel='Expectation Value', title='Pulse Profiles'):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the Hahn echo sequence for a given tau and then plot them.

        Parameters
        ----------
        tau (float): free evolution time for the Hahn echo sequence. Contrary to the run method, the free evoluiton must be a single number in order to plot the pulse profiles.
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        xlabel (str): label of the x-axis
        ylabel (str): label of the y-axis
        title (str): title of the plot
        """
        # generate the pulse profiles for the Hahn echo sequence for a given tau
        self.get_pulse_profiles(tau)

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)