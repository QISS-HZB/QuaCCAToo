import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, mesolve, parallel_map
from types import FunctionType
from scipy.optimize import curve_fit

import quaccatoo.pulsed_logic as pulsed_logic

class PulsedExperiment:
    def __init__(self, rho0, H0, H2 = None, c_ops = None):
        """
        
        """
        if not isinstance(rho0, Qobj) or not isinstance(H0, Qobj):
            raise ValueError("H0 and rho0 must be a Qobj")
        else:
            if H0.shape != rho0.shape:
                raise ValueError("H0 and rho0 must have the same dimensions")
            else:
                self.H0 = H0
                self.rho0 = rho0
        
        if c_ops == None:
            self.c_ops = c_ops

        elif isinstance(c_ops, list):
            if not all(isinstance(op, Qobj) and op.shape == H0.shape for op in c_ops):
                raise ValueError("All items in c_ops must be Qobj with the same dimensions as H0")
            else:
                self.c_ops = c_ops
        else:
            raise ValueError("c_ops must be a list of Qobj or None")

        self.phi_t = 0
        self.total_time = 0
        self.sequence = []
        self.pulse_profiles = []

    def add_pulse(self, duration, H1, pulse_shape = pulsed_logic.square_pulse, pulse_params = {}, time_steps = 100):
        """
        
        """
        if not isinstance(duration, (int, float)) and duration <= 0:
            raise ValueError("duration must be a positive real number")
        else:
            tarray = np.linspace(self.total_time, self.total_time + duration, time_steps)
            self.total_time += duration

        if isinstance(pulse_shape, FunctionType):
            pass
        elif isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape):
            pass
        else: 
            raise ValueError("pulse_shape must be a python function or a list of python functions")
        
        if isinstance(H1, Qobj) and H1.shape == self.rho0.shape:
            Ht = [self.H0, [H1, pulse_shape]]
            self.pulse_profiles.append( [H1, tarray, pulse_shape, pulse_params] )

        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.rho0.shape for op in H1) and len(H1) == len(pulse_shape):       
            Ht = [self.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            self.pulse_profiles = [[H1[i], tarray, pulse_shape[i], pulse_params] for i in range(len(H1))]
        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")

        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')

        def pulse():
            return mesolve(Ht, self.rho, tarray, self.c_ops, [], options = self.options, args = pulse_params).states[-1]
        
        self.sequence.append( pulse )
        
    def add_free_evolution(self, duration):
        """
        
        """
        if not isinstance(duration, (int, float)) and duration <= 0:
            raise ValueError("duration must be a positive real number")

        def free_evolution():
            return (-1j*self.H0*duration).expm() * self.rho * ((-1j*self.H0*duration).expm()).dag()
        
        self.total_time += duration
        self.sequence.append( free_evolution )
        self.pulse_profiles.append( [None, duration, None, None] )

    def run(self, observable=None, options={}):
        """
        
        """
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        self.rho = self.rho0.copy()

        if observable == None:
            for operation in self.sequence:
                self.rho = operation()
            self.results = self.rho.copy()
        
        if isinstance(observable, Qobj) and observable.shape == self.rho0.shape:
            for operation in self.sequence:
                self.rho = operation()
            self.results = np.abs(( observable * self.rho ).tr() )
        
        if isinstance(observable, list) and all(isinstance(q, Qobj) and q.shape == self.rho0.shape for q in observable):
            for operation in self.sequence:
                self.rho = operation()
            self.results = np.abs(( observable * self.rho ).tr() )
        
        else:
            raise ValueError("observable must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1.")
            
    def plot_pulses(self, figsize=(6, 6), xlabel='Time', ylabel='Expectation Value', title='Pulse Profiles'):
        """
        
        """
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for itr in range(len(self.pulse_profiles)):
            ax.plot(self.pulse_profiles[itr][1], self.pulse_profiles[itr][2](self.pulse_profiles[itr][1], **self.pulse_profiles[itr][3]), label = f'H1 {itr}', lw=2, alpha=0.7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.2, 1))
            ax.set_title(title)

    def plot_result(self, figsize=(6, 6), fit_function=None, fit_guess=None, xlabel='Time', ylabel='Expectation Value', title='Pulsed Experiment Result'):
        """
        
        """
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if len(self.results) == 1:
            ax.plot(self.tarray, self.results, lw=2, alpha=0.7)
        
        else:
            for itr in range(len(self.results)):
                ax.plot(self.tarray, self.results[itr], label = f'Observable {itr}', lw=2, alpha=0.7)

        if isinstance(fit_function, FunctionType):
            params, cov = curve_fit(self.results, self.tarray, fit_function=fit_function, fit_guess=fit_guess)
            ax.plot(self.tarray, fit_function(self.tarray, *params), linestyle='--', lw=2, alpha=0.7)
        elif fit_function == None:
            pass
        else:
            raise ValueError("fit_function must be a python function or None")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.2, 1))
        ax.set_title(title)

class rabi(PulsedExperiment):
    """
    
    """
    def __init__(self, tarray, rho0, H0, H1, H2=None, c_ops=None, pulse_shape = pulsed_logic.square_pulse, pulse_params = {}, time_steps = 100):
        """
        
        """
        if not isinstance(tarray, np.ndarray):
            raise ValueError("tarray must be a numpy array")
        else:
            self.tarray = tarray

        super().__init__(H0, rho0, H2, c_ops)

        if isinstance(pulse_shape, FunctionType):
            self.pulse_shape = pulse_shape
        elif isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape):
            self.pulse_shape = pulse_shape
        else: 
            raise ValueError("pulse_shape must be a python function or a list of python functions")
        
        if isinstance(H1, Qobj) and H1.shape == self.rho0.shape:
            self.Ht = [self.H0, [H1, pulse_shape]]
            self.pulse_profiles = [[H1, tarray, pulse_shape, pulse_params]]
            
        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.rho0.shape for op in H1) and len(H1) == len(pulse_shape):       
            self.Ht = [self.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            self.pulse_profiles = [[H1[i], tarray, pulse_shape[i], pulse_params] for i in range(len(H1))]

        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")
        
        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            self.pulse_params = pulse_params
            if 'phi_t' not in pulse_params:
                self.pulse_params['phi_t'] = 0
                
    def run(self, observable=None, options={}):
        """
        
        """
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        if observable == None:
            self.results = mesolve(self.Ht, self.rho0, self.tarray, self.c_ops, [], options = self.options, args = self.pulse_params).states
        
        if isinstance(observable, Qobj) and observable.shape == self.rho0.shape:
            self.results = mesolve(self.Ht, self.rho0, self.tarray, self.c_ops, [observable], options = self.options, args = self.pulse_params).expect[0]
        
        if isinstance(observable, list) and all(isinstance(q, Qobj) and q.shape == self.rho0.shape for q in observable):
            self.results =  mesolve(self.Ht, self.rho0, self.tarray, self.c_ops, observable, options = self.options, args = self.pulse_params).expect
        
        else:
            raise ValueError("observable must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1.")
        
class hahn(PulsedExperiment):
    """
    
    """
    def __init__(self, tarray, rho0, H0, H1, pi_pulse_duration, H2=None, c_ops=None, pulse_shape = pulsed_logic.square_pulse, pulse_params = {}):
        """
        
        """
        if not isinstance(tarray, np.ndarray):
            raise ValueError("tarray must be a numpy array")
        else:
            self.tarray = tarray

        if not isinstance(pi_pulse_duration, (int, float)) and pi_pulse_duration <= 0:
            raise ValueError("pulse_duration must be a positive real number")
        else:
            self.pi_pulse_duration = pi_pulse_duration
        
        super().__init__(H0, rho0, H2, c_ops)

        if isinstance(pulse_shape, FunctionType):
            self.pulse_shape = pulse_shape
        elif isinstance(pulse_shape, list) and all(isinstance(pulse_shape, FunctionType) for pulse_shape in pulse_shape):
            self.pulse_shape = pulse_shape
        else: 
            raise ValueError("pulse_shape must be a python function or a list of python functions")
        
        if isinstance(H1, Qobj) and H1.shape == self.rho0.shape:
            self.H1 = H1

        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.rho0.shape for op in H1) and len(H1) == len(pulse_shape):       
            self.H1 = H1

        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")
        
        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        else:
            self.pulse_params = pulse_params
            if 'phi_t' not in pulse_params:
                self.pulse_params['phi_t'] = 0

        self.hahn_sequence = self._hahn_sequence
        
    def _hahn_sequence(self, tau):
        self.add_pulse(self.pi_pulse_duration/2, self.H1, self.pulse_shape, self.pulse_params)
        self.add_free_evolution(tau - self.pi_pulse_duration/2)
        self.add_pulse(self.pi_pulse_duration, self.H1, self.pulse_shape, self.pulse_params)
        self.add_free_evolution(tau - self.pi_pulse_duration/2)
        self.add_pulse(self.pi_pulse_duration/2, self.H1, self.pulse_shape, self.pulse_params)

    def run(self, observable=None, options={}):
        """
        
        """
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        else:
            self.options = options

        self.rho = self.rho0.copy()
        self.rho = parallel_map(self.hahn_sequence, self.tarray)
 
        if observable == None:
            self.results = self.rho

        if isinstance(observable, Qobj) and observable.shape == self.rho0.shape:
            print(self.rho)
            self.results = np.abs(( observable * self.rho ).tr() )
        
        if isinstance(observable, list) and all(isinstance(q, Qobj) and q.shape == self.rho0.shape for q in observable):
            self.results = np.abs(( observable * self.rho ).tr() )
        
        else:
            raise ValueError("observable must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1.")