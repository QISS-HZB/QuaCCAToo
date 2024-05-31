import numpy as np
from qutip import Qobj, mesolve, parallel_map
from types import FunctionType

from.PulsedExperiment import PulsedExperiment
from.PulsedLogic import square_pulse

class Rabi(PulsedExperiment):
    """
    This class contains a Rabi experiments, inheriting from the PulsedExperiment class.

    Class Attributes
    ----------------
    = PulsedExperiment

    Methods
    -------
    = PulsedExperiment    
    """
    def __init__(self, pulse_duration, rho0, H0, H1, H2=None, c_ops=None, pulse_shape = square_pulse, pulse_params = {}):
        """
        Generator for the Rabi pulsed experiment class, taking a specific pulse_duration to run the simulation.

        Parameters
        ----------
        pulse_duration (numpy array):time array for the simulation representing to be used as the variable for the simulation
        rho0 (Qobj): initial density matrix
        H0 (Qobj): internal time independent Hamiltonian of the system
        H1 (Qobj): control Hamiltonian of the system
        H2 (Qobj): time dependent sensing Hamiltonian of the system
        c_ops (list): list of collapse operators
        pulse_shape (FunctionType, list(FunctionType)): pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params (dict): dictionary of parameters for the pulse_shape functions
        """
        # call the parent class constructor
        super().__init__(rho0, H0, H2 = H2, c_ops = c_ops)

        # check weather pulse_duration is a numpy array and if it is, assign it to the object
        if not isinstance(pulse_duration, np.ndarray):
            raise ValueError("pulse_duration must be a numpy array")
        else:
            # Check if all elements are real and positive
            if np.all(np.isreal(pulse_duration)) and np.all(np.greater_equal(pulse_duration, 0)):
                self.variable = pulse_duration
                self.total_time = pulse_duration[-1]
            else:
                raise ValueError("All elements in pulse_duration must be real and positive.")

        # check weather pulse_shape is a python function or a list of python functions and if it is, assign it to the object
        if isinstance(pulse_shape, FunctionType):
            self.pulse_shape = pulse_shape
        elif isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape):
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
            if 'phi_t' not in pulse_params:
                self.pulse_params['phi_t'] = 0

        # set the default xlabel attribute
        self.xlabel = 'Pulse Duration'
                
    def run(self, observable=None, options={}):
        """
        Overwrites the run method of the parent class. Runs the simulation and stores the results in the results attribute. If an observable is given, the expectation values are stored in the results attribute.

        Parameters
        ----------
        observable (Qobj, list(Qobj)): observable to calculate the expectation value of. If none are given, the density matrices are stored in the results attribute
        options (dict): dictionary of solver options from Qutip        
        """
        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        # if no observable is given, run the simulation and store the the calculated density matrices in the results attribute
        if observable == None:
            self.results = mesolve(self.Ht, self.rho0, self.variable, self.c_ops, [], options = self.options, args = self.pulse_params).states
        
        # if an observable is given, check if it is a Qobj of the same shape as rho0, H0 and H1 and run the simulation storing the expectation values in the results attribute
        elif isinstance(observable, Qobj) and observable.shape == self.rho0.shape:
            self.observable = observable
            self.results = mesolve(self.Ht, self.rho0, self.variable, self.c_ops, observable, options = self.options, args = self.pulse_params).expect[0]
        
        # if the observable i a list of Qobjs of the same shape as rho0, H0 and H1, run the simulation storing the expectation values in the results attribute
        elif isinstance(observable, list) and all(isinstance(q, Qobj) and q.shape == self.rho0.shape for q in observable):
            self.observable = observable
            self.results =  mesolve(self.Ht, self.rho0, self.variable, self.c_ops, observable, options = self.options, args = self.pulse_params).expect
        
        else:
            raise ValueError("observable must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1.")
        
class Hahn(PulsedExperiment):

    """
    This class contains a Hahn echo experiments, inheriting from the PulsedExperiment class. The Hahn echo sequence consists of a pi/2 pulse, a free evolution time tau, a pi pulse and another free evolution time tau followed by a pi/2 pulse.

    Class Attributes
    ----------------
    free_duration (numpy array): time array of the free evolution times to run the simulation
    pi_pulse_duration (float, int): duration of the pi pulse
    += PulsedExperiment

    Class Methods
    -------------
    hahn_sequence(tau): defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the generator, returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip.
    += PulsedExperiment    
    """

    def __init__(self, free_duration, pi_pulse_duration, rho0, H0, H1,  H2=None, c_ops=None, pulse_shape = square_pulse, pulse_params = {}, time_steps = 100):
        """
        Generator for the Hahn echo pulsed experiment class, taking a specific free_duration to run the simulation and the pi_pulse_duration.

        Parameters
        ----------
        free_duration (numpy array): time array for the simulation representing the free evolution time to be used as the variable atritbute for the simulation
        rho0 (Qobj): initial density matrix
        H0 (Qobj): internal time independent Hamiltonian of the system
        H1 (Qobj, list(Qobj)): control Hamiltonian of the system
        pi_pulse_duration (float, int): duration of the pi pulse
        H2 (Qobj): time dependent sensing Hamiltonian of the system
        c_ops (list): list of collapse operators
        pulse_shape (FunctionType, list(FunctionType)): pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params (dict): dictionary of parameters for the pulse_shape functions
        time_steps (int): number of time steps in the pulses for the simulation
        """
        # call the parent class constructor
        super().__init__(rho0, H0, H2, c_ops)

        # check weather free_duration is a numpy array of real and positive elements and if it is, assign it to the object
        if not isinstance(free_duration, np.ndarray):
            raise ValueError("free_duration must be a numpy array")
        else:
            # Check if all elements are real and positive
            if np.all(np.isreal(free_duration)) and np.all(np.greater_equal(free_duration, 0)):
                self.variable = free_duration
            else:
                raise ValueError("All elements in free_duration must be real and positive.")

        # check weather pi_pulse_duration is a positive real number and if it is, assign it to the object
        if not isinstance(pi_pulse_duration, (int, float)) and pi_pulse_duration <= 0:
            raise ValueError("pulse_duration must be a positive real number")
        else:
            if pi_pulse_duration/2 > free_duration[-1]:
                # check if the pulse duration over 2 is larger than the free evolution time
                raise ValueError("the pulse interval must be larger than pi_pulse_duration/2, othwerwise pulses will overlap")
            else:
                self.pi_pulse_duration = pi_pulse_duration

        # check weather time_steps is a positive integer and if it is, assign it to the object
        if not isinstance(time_steps, int) and time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        else:
            self.time_steps = time_steps       

        # check weather pulse_shape is a python function or a list of python functions and if it is, assign it to the object
        if isinstance(pulse_shape, FunctionType):
            self.pulse_shape = pulse_shape
        elif isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape):
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

        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            self.pulse_params = pulse_params
            if 'phi_t' not in pulse_params:
                self.pulse_params['phi_t'] = 0
        
        # set the default xlabel attribute
        self.xlabel = 'Free Evolution Time'
        
    def hahn_sequence(self, tau):
        """
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the generator. The sequence consists of a pi/2 pulse, a free evolution time tau, a pi pulse and another free evolution time tau followed by a pi/2 pulse. The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau (float): free evolution time

        Returns
        -------
        rho (Qobj): final density matrix        
        """
        # set the initial time to 0
        t0 = 0
        # perform the first pi/2 pulse
        rho = mesolve(self.Ht, self.rho0, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.c_ops, [], options = self.options, args = self.pulse_params).states[-1]
        # perform the first free evolution
        rho = (-1j*self.H0*(tau - self.pi_pulse_duration/2)).expm() * rho * ((-1j*self.H0*(tau - self.pi_pulse_duration/2)).expm()).dag()
        # append the time to the initial time for the next pulse 
        t0 += tau
        # perform the second pi pulse
        rho = mesolve(self.Ht, rho, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.c_ops, [], options = self.options, args = self.pulse_params).states[-1]
        # perform the second free evolution
        rho = (-1j*self.H0*(tau - self.pi_pulse_duration/2)).expm() * rho * ((-1j*self.H0*(tau - self.pi_pulse_duration/2)).expm()).dag()
        # append the time to the initial time for the next pulse
        t0 += tau
        # perform the third pi/2 pulse
        rho = mesolve(self.Ht, rho, np.linspace(t0, t0 + 3*self.pi_pulse_duration/2, self.time_steps), self.c_ops, [], options = self.options, args = self.pulse_params).states[-1]
        # return the final density matrix
        return rho

    def run(self, observable=None, options={}):
        """
        Overwrites the run method of the parent class. Runs the simulation and stores the results in the results attribute.

        Parameters
        ----------
        observable (Qobj, list(Qobj)): observable to calculate the expectation value of. If none are given, the density matrices are stored in the results attribute
        options (dict): dictionary of solver options from Qutip        
        """
        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        self.rho = self.rho0.copy()
        # run the simulation in parallel using the parallel_map method of QuTip to calculate the density matrices for the given free evolution times
        self.rho = parallel_map(self.hahn_sequence, self.variable)
        
        # if no observable is given, store the calculated density matrices in the results attribute
        if observable == None:
            self.results = self.rho

        # if an observable is given, check if it is a Qobj of the same shape as rho0, H0 and H1 and store the expectation values in the results attribute
        elif isinstance(observable, Qobj) and observable.shape == self.rho0.shape:
            self.observable = observable
            for rho in self.rho:
                self.results.append( np.abs(( observable * rho).tr() ) )
        
        # if the observable is a list of Qobjs of the same shape as rho0, H0 and H1, store the expectation values in the results attribute
        elif isinstance(observable, list) and all(isinstance(q, Qobj) and q.shape == self.rho0.shape for q in observable):
            self.observable = observable

            for itr_observable in range(len(observable)):
                results_array = np.empty(len(self.rho))

                for itr_rho in range(len(self.rho)):
                    results_array[itr_rho] = np.abs(( observable[itr_observable] * self.rho[itr_rho]).tr() )

                self.results = results_array       
        else:
            raise ValueError("observable must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1.")
        
    def plot_pulses(self, tau, figsize=(6, 6), xlabel=None, ylabel='Expectation Value', title='Pulse Profiles'):
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
        # check weather tau is a positive real number and if it is, assign it to the object
        if not isinstance(tau, (int, float)) and tau <= 0 and tau > self.pi_pulse_duration/2:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration/2")
        
        # if only one control Hamiltonian is given, append the pulse_profiles with the Hahn echo sequence as in the hahn_sequence method
        if isinstance(self.H1, Qobj):
            t0 = 0
            self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params] )
            t0 += self.pi_pulse_duration/2
            self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
            t0 += tau - self.pi_pulse_duration/2
            self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params] )
            t0 += self.pi_pulse_duration
            self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
            t0 += tau - self.pi_pulse_duration/2
            self.pulse_profiles.append( [self.H1, np.linspace(t0, t0 + 3*self.pi_pulse_duration/2, self.time_steps), self.pulse_shape, self.pulse_params] )
            t0 += 3*self.pi_pulse_duration/2

        # otherwise if a list of control Hamiltonians is given, it sums over all H1 and appends to the pulse_profiles the Hahn echo sequence as in the hahn_sequence method
        elif isinstance(self.H1, list):
            t0 = 0
            self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))] )
            t0 += self.pi_pulse_duration/2
            self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
            t0 += tau - self.pi_pulse_duration/2
            self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))] )
            t0 += self.pi_pulse_duration
            self.pulse_profiles.append( [None, [t0, t0 + tau - self.pi_pulse_duration/2], None, None] )
            t0 += tau - self.pi_pulse_duration/2
            self.pulse_profiles.append( [[self.H1[i], np.linspace(t0, t0 + 3*self.pi_pulse_duration/2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))] )
            t0 += 3*self.pi_pulse_duration/2
        
        # set the total_time attribute to the total time of the pulse sequence
        self.total_time = t0

        # if no xlabel is given, set the xlabel to the default value
        if xlabel == None:
            pass
        elif isinstance(xlabel, str):
            self.xlabel = xlabel
        else:
            raise ValueError("xlabel must be a string or None")

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)

class PODMR(PulsedExperiment):
    """
    
    """
    def __init__(self, frequencies, pulse_duration, rho0, H0, H1,  H2=None, c_ops=None, pulse_shape = square_pulse, pulse_params = {}, time_steps = 100, freqs_rad = True):

        super().__init__(rho0, H0, H2, c_ops)

        if not isinstance(frequencies, np.ndarray):
            raise ValueError("frequencies must be a numpy array")
        else:
            # Check if all elements are real and positive
            if np.all(np.isreal(frequencies)) and np.all(np.greater_equal(frequencies, 0)):
                self.variable = frequencies
            else:
                raise ValueError("All elements in frequencies must be real and positive.")
            
        if not isinstance(freqs_rad, bool):
            raise ValueError("freqs_rad must be a boolean")
        else:
            self.freqs_rad = freqs_rad
            self.xlabel = 'Frequency'

        # check weather pulse_duration is a numpy array and if it is, assign it to the object
        if not isinstance(pulse_duration, (float, int)) and pulse_duration <= 0:
            raise ValueError("pulse_duration must be a positive real number")
        else:
            self.total_time = pulse_duration
            self.pulse_duration = pulse_duration

        # check weather pulse_shape is a python function or a list of python functions and if it is, assign it to the object
        if isinstance(pulse_shape, FunctionType):
            self.pulse_shape = pulse_shape
        elif isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape):
            self.pulse_shape = pulse_shape
        else: 
            raise ValueError("pulse_shape must be a python function or a list of python functions")
        
        # check weather time_steps is a positive integer and if it is, assign it to the object
        if not isinstance(time_steps, int) and time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        else:
            self.time_steps = time_steps       
        
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
            if 'phi_t' not in pulse_params:
                self.pulse_params['phi_t'] = 0

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
        # check the units of the frequencies and assign the pulse_params dictionary accordingly
        if self.freqs_rad:
            self.pulse_params['omega_pulse'] = f
        else:
            self.pulse_params['omega_pulse'] = 2*np.pi*f

        # return the final density matrix after pulse
        return mesolve(self.Ht, self.rho, np.linspace(0, self.pulse_duration, self.time_steps), self.c_ops, [], options = self.options, args = self.pulse_params).states[-1]

    def run(self, observable=None, options={}):
        """
        Overwrites the run method of the parent class. Runs the simulation and stores the results in the results attribute.

        Parameters
        ----------
        observable (Qobj, list(Qobj)): observable to calculate the expectation value of. If none are given, the density matrices are stored in the results attribute
        options (dict): dictionary of solver options from Qutip        
        """
        # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        self.rho = self.rho0.copy()
        # run the simulation in parallel using the parallel_map method of QuTip to calculate the density matrices for the given free evolution times
        self.rho = parallel_map(self.PODMR_sequence, self.variable)
        
        # if no observable is given, store the calculated density matrices in the results attribute
        if observable == None:
            self.results = self.rho

        # if an observable is given, check if it is a Qobj of the same shape as rho0, H0 and H1 and store the expectation values in the results attribute
        elif isinstance(observable, Qobj) and observable.shape == self.rho0.shape:
            self.observable = observable
            for rho in self.rho:
                self.results.append( np.abs(( observable * rho).tr() ) )
        
        # if the observable is a list of Qobjs of the same shape as rho0, H0 and H1, store the expectation values in the results attribute
        elif isinstance(observable, list) and all(isinstance(q, Qobj) and q.shape == self.rho0.shape for q in observable):
            self.observable = observable

            for itr_observable in range(len(observable)):
                results_array = np.empty(len(self.rho))

                for itr_rho in range(len(self.rho)):
                    results_array[itr_rho] = np.abs(( observable[itr_observable] * self.rho[itr_rho]).tr() )

                self.results = results_array       
        else:
            raise ValueError("observable must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1.")