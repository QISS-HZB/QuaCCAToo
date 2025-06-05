# TODO: units in plot_pulses

"""
This module contains the PulsedSim class that is used to define a general pulsed experiment with a sequence of pulses and free evolution operations, part of the QuaCAAToo package.
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings
from qutip import Qobj, mesolve, parallel_map, measurement
from .pulse_shapes import square_pulse
from ..qsys.qsys import QSys

class PulsedSim:
    """
    The PulsedSim class is used to define a general pulsed experiment with a sequence of pulses and free evolution operations.
    The class contains methods to add pulses and free evolution operations to the sequence of operations,
    run the experiment, plot the pulse profiles and results of the experiment.
    By default the Hamiltonian is in frequency units and the time is in time units.

    Attributes
    ----------
    system : QSys
        quantum system object representing the quantum system
    H2 : list(Qobj, function)
        time dependent sensing Hamiltonian of the system in the form [Qobj, function]
    H0_H2 : list
        list of Hamiltonians for the pulse operation in the form [H0, H2]
    total_time : float
        total time of the experiment
    variable : np.array
        variable of the experiment which the results depend on
    variable_name : str
        name of the variable
    pulse_profiles :list
        list of pulse profiles for plotting purposes, where each element is a list [H1, tarray, pulse_shape, pulse_params]
    results : list
        results of the experiment to be later generated in the run method
    sequence : callable
        parallel sequence of operations to be overwritten in PredefSeqs and PredefDDSeqs, or defined by the user
    time_steps : int
        number of time steps for the pulses

    Methods
    -------
    add_pulse :
        adds a pulse operation to the sequence of operations of the experiment
    pulse :
        updates the total time of the experiment, sets the phase for the pulse and calls mesolve from QuTip to perform the pulse operation
    add_free_evolution :
        adds a free evolution operation to the sequence of operations of the experiment
    _free_evolution :
        updates the total time of the experiment and applies the time-evolution operator to perform the free evolution operation with the exponential operator
    run :
        runs the pulsed experiment by calling the parallel_map function from QuTip over the variable attribute
    _get_results :
        gets the results of the experiment from the calculated rho, based on the observable of the system
    measure_qsys :
        measures the observable over the system, storing the measurement outcome in the results attribute and collapsing rho in the corresponding eigenstate of the observable
    plot_pulses :
        plots the pulse profiles of the experiment by iterating over the pulse_profiles list and plotting each pulse profile and free evolution
    _check_attr_predef_seqs :
        checks the common attributes of the PulsedSim object for the predefined sequences and sets them accordingly
    """
    def __init__(self, system, H2=None):
        """
        Initializes a general PulsedSim object with a quantum system, time dependent Hamiltonian and collapse operators.

        Parameters
        ----------
        system : QSys
            quantum system object representing the quantum system
        H2 : Qobj
            time dependent sensing Hamiltonian of the system
        """
        if not isinstance(system, QSys):
            raise ValueError("system must be a QSys object")

        self.system = system

        if system.rho0 is not None:
            self.rho = system.rho0.copy()

        # if collapse operators are given, the H0_H2 attributed needs to be set with H0 for the mesolve function
        if self.system.c_ops is not None:
            self.H0_H2 = self.system.H0

        if H2 is None:
            self.H2 = None
        elif H2[0].shape != self.system.H0.shape or not callable(H2[1]):
            raise ValueError("H2 must be a list where the first element is a Qobj of the same shape as H0 and the second element is a time dependent function")
        else:
            self.H2 = H2
            self.H0_H2 = [self.system.H0, self.H2]

        # initialize the rest of the attributes
        self.total_time = 0
        self.variable = None
        self.variable_name = None
        self.pulse_profiles = []
        self.results = []
        self.sequence = None
        self.time_steps = None

    def add_pulse(self, duration, H1, pulse_shape=square_pulse, pulse_params=None, time_steps=100, options=None):
        """
        Perform variables checks and adds a pulse operation to the sequence of operations of the experiment for a given duration of the pulse,
        control Hamiltonian H1, pulse phase, pulse shape function, pulse parameters and time steps by calling the pulse method.

        Parameters
        ----------
        duration : float or int
            duration of the pulse
        H1 : Qobj or list(Qobj)
            control Hamiltonian of the system
        pulse_shape : callable or list(callable)
            pulse shape function or list of pulse shape functions representing the time modulation of t H1
        pulse_params : dict
            dictionary of parameters for the pulse_shape functions
        time_steps : int
            number of time steps for the pulses
        options : dict
            options for the Qutip solver
        """
        # check all the parameters
        if options is None:
            options = {}
        elif not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")

        if not isinstance(time_steps, int) or time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        else:
            self.time_steps = time_steps

        if not isinstance(duration, (int, float)) and duration <= 0:
            raise ValueError("duration must be a positive real number")

        if not (callable(pulse_shape) or (isinstance(pulse_shape, list) and all(callable(p) for p in pulse_shape))):
            raise ValueError("pulse_shape must be a python function or a list of python functions")

        if pulse_params is None:
            pulse_params = {}
        elif not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        
        # if the user doesn't provide a phi_t, set it to 0
        if 'phi_t' not in pulse_params:
            pulse_params['phi_t'] = 0

        # check if H1 is a Qobj or a list of Qobj with the same dimensions as H0
        if isinstance(H1, Qobj) and H1.shape == self.system.H0.shape:
            # append it to the pulse_profiles list
            self.pulse_profiles.append([H1, np.linspace(self.total_time, self.total_time + duration, self.time_steps), pulse_shape, pulse_params])
            if self.H2 is None:
                Ht = [self.system.H0, [H1, pulse_shape]]
            else:
                Ht = [self.system.H0, [H1, pulse_shape], self.H2]

        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.system.H0.shape for op in H1) and len(H1) == len(pulse_shape):
            self.pulse_profiles = [[H1[i], np.linspace(self.total_time, self.total_time + duration, self.time_steps), pulse_shape[i], pulse_params] for i in range(len(H1))]
            if self.H2 is None:
                Ht = [self.system.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            else:
                Ht = [self.system.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))] + self.H2

        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as H0 and with the same length as the pulse_shape list")

        # add the pulse operation to the sequence of operations by calling the pulse method
        self._pulse(Ht, duration, options, pulse_params)

    def _pulse(self, Ht, duration, options, core_pulse_params):
        """
        Calls the mesolve function from QuTip to perform the pulse operation with the given Hamiltonian, time array and options,
        by updating the rho attribute of the class with the result of the operation.
        Adds the pulse duration to the total time.
        This method should be used internally by other methods, as it does not perform any checks on the input parameters for better performance.

        Parameters
        ----------
        Ht : list
            list of Hamiltonians for the pulse operation in the form [H0, [H1, pulse_shape]]
        tarray : np.array
            time array for the pulse operation
        options : dict
            options for the Qutip solver
        pulse_params : dict
            dictionary of parameters for the pulse_shape functions
        """
        # perform the pulse operation. The time array is multiplied by 2*pi so that [H*t] has units of radians
        self.rho = mesolve(Ht, self.rho, 2*np.pi*np.linspace(self.total_time, self.total_time + duration, self.time_steps) , self.system.c_ops, e_ops=[], options = options, args = core_pulse_params).states[-1]

        self.total_time += duration

    def add_free_evolution(self, duration, options=None):
        """
        Adds a free evolution operation to the sequence of operations of the experiment for a given duration of the free evolution by calling the _free_evolution method.

        Parameters
        ----------
        duration : float or int
            duration of the free evolution
        options : dict
            options for the Qutip solver
        """
        # check if duration of the pulse is a positive real number
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError("duration must be a positive real number")
        
        if options is None:
            options = {}
        elif not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")

        # add the free evolution to the pulse_profiles list
        self.pulse_profiles.append([None, [self.total_time, duration + self.total_time], None, None])

        self._free_evolution(duration, options)

    def _free_evolution(self, duration, options):
        """
        Updates the total time of the experiment and applies the time-evolution operator to the initial state.
        This method should be used internally by other methods, as it does not perform any checks on the input parameters for better performance.
        If the system has collapse operators or time dependent Hamiltonian H2, mesolve is used to perform the free evolution operation.
        Otherwise, the time-evolution operator is applied directlyby the exponential operator.

        Parameters
        ----------
        duration : float or int
            duration of the free evolution
        """
        if self.system.c_ops is not None or self.H2 is not None:
            self.rho = mesolve(self.H0_H2, self.rho, 2*np.pi*np.linspace(self.total_time, self.total_time + duration, self.time_steps) , self.system.c_ops, e_ops=[], options=options).states[-1]
        else:
            if self.rho.isket:
                self.rho = (-1j*2*np.pi*self.system.H0*duration).expm() * self.rho
            else:
                self.rho = (-1j*2*np.pi*self.system.H0*duration).expm() * self.rho * ((-1j*2*np.pi*self.system.H0*duration).expm()).dag()

        self.total_time += duration

    def measure_qsys(self, observable=None):
        """
        Measures the observable over the system, storing the measurent outcome in the results attribute and collapsing rho in the corresponding eigenstate of the observable.
        If no observable is given, the observable of the qsys is used.

        Parameters
        ----------
        observable : Qobj
            observable to be measured after the sequence of operations

        Returns
        -------
        results : float or list
            measurement outcome of the observable, which can be a float or a list of floats if the observable is a list of Qobjs
        """
        if isinstance(observable, Qobj) and observable.shape == self.system.H0.shape:
            if not observable.isherm:
                warnings.warn("Passed observable is not hermitian.")
            self.results, self.rho = measurement.measure_observable(self.rho, observable)

        elif observable is None and (isinstance(self.system.observable, Qobj) and self.system.observable.shape == self.system.H0.shape):
            self.results, self.rho = measurement.measure_observable(self.rho, self.system.observable)

        else:
            raise ValueError("observable must be a Qobj of the same shape as rho0, H0 and H1.")
        
        return self.results.copy()

    def run(self, variable=None, sequence=None, sequence_kwargs=None, map_kw=None):
        """
        Runs the pulsed experiment by calling the parallel_map function from QuTip over the variable attribute.
        The rho attribute is updated.

        Parameters
        ----------
        variable : np.array
            xaxis variable of the plot representing the parameter being changed in the experiment
        sequence : callable
            sequence of operations to be performed in the experiment
        sequence_kwargs : dict
            dictionary of arguments to be passed to the sequence function
        map_kw : dict
            dictionary of options for the parallel_map function from QuTip
        """
        # if no sequence is passed but the PulsedSim has one, uses the attribute sequence
        if sequence is None and self.sequence is not None:
            pass
        # if a sequence is passed, checks if it is a python function and overwrites the attribute
        elif callable(sequence):
            self.sequence = sequence
        else:
            raise ValueError("sequence must be a python function with a list operations returning a number")

        # check if a variable was passed by the user, if it is numpy array overwrite the variable attribute
        if isinstance(variable, np.ndarray):
            self.variable = variable
        elif variable is None and len(self.variable) != 0:
            pass
        else:
            raise ValueError("variable must be a numpy array")
        
        # check if map_kw and sequence_kwargs are None or dictionaries
        if map_kw is None:
            map_kw = {'num_cpus': None}
        elif not isinstance(map_kw, dict):
            raise ValueError("map_kw must be a dictionary of options for the parallel_map function from QuTip")

        if sequence_kwargs is None:
            sequence_kwargs = {}
        elif not isinstance(sequence_kwargs, dict):
            raise ValueError("sequence_args must be a dictionary of arguments to be passed to the sequence function")
        
        # the rho attribute needs to be reset to the initial state, so it doesnt run over the previous simulation
        self.rho = self.system.rho0.copy()

        # run the experiment by calling the parallel_map function from QuTip over the variable attribute
        self.rho = parallel_map(self.sequence, self.variable, task_kwargs=sequence_kwargs, map_kw=map_kw)

        self._get_results()

    def _get_results(self):
        """
        Gets the results of the experiment from the calculated rho, based on the observable of the system.
        The results are stored in the results attribute of the class.
        """
        if self.rho[0].isket:
            if isinstance(self.system.observable, Qobj):
                # np.real is used to ensure no imaginary components will be attributed to results
                self.results = np.array([np.real(rho.dag()*self.system.observable*rho) for rho in self.rho])  
            elif isinstance(self.system.observable, list):
                self.results = [np.array([np.real(rho.dag()*observable*rho) for rho in self.rho]) for observable in self.system.observable]
        else:
            if isinstance(self.system.observable, Qobj):
                self.results = np.array([np.real((rho*self.system.observable).tr()) for rho in self.rho])  
            elif isinstance(self.system.observable, list):
                self.results = [np.array([np.real((rho*observable).tr()) for rho in self.rho]) for observable in self.system.observable]

    def plot_pulses(self, figsize=(6, 4), xlabel=None, ylabel='Pulse Intensity', title='Pulse Profiles'):
        """
        Plots the pulse profiles of the experiment by iterating over the pulse_profiles list and plotting each pulse profile and free evolution.

        Parameters
        ----------
        figsize : tuple
            size of the figure to be passed to matplotlib.pyplot
        xlabel : str
            label of the x-axis
        ylabel : str
            label of the y-axis
        title : str
            title of the plot
        """
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        if xlabel is None:
            xlabel = self.variable_name
        elif not isinstance(xlabel, str):
            raise ValueError("xlabel must be a string")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # iterate over all operations in the sequence
        for itr_pulses in range(len(self.pulse_profiles)):

            # if the pulse profile is a free evolution, plot a horizontal line at 0
            if self.pulse_profiles[itr_pulses][0] is None:
                ax.plot(self.pulse_profiles[itr_pulses][1], [0, 0], label='Free Evolution', lw=2, alpha=0.7, color='C0')

            # if the pulse has operators, plot the pulse profiles
            elif isinstance(self.pulse_profiles[itr_pulses][0], Qobj):
                ax.plot(self.pulse_profiles[itr_pulses][1], self.pulse_profiles[itr_pulses][2](2*np.pi*self.pulse_profiles[itr_pulses][1], **self.pulse_profiles[itr_pulses][3]), label='H1', lw=2, alpha=0.7, color='C1')

            elif isinstance(self.pulse_profiles[itr_pulses][0], list):
                for itr_op in range(len(self.pulse_profiles[itr_pulses])):
                    ax.plot(self.pulse_profiles[itr_pulses][itr_op][1], self.pulse_profiles[itr_pulses][itr_op][2](2*np.pi*self.pulse_profiles[itr_pulses][itr_op][1], **self.pulse_profiles[itr_pulses][itr_op][3]), label=f'H1_{itr_op}', lw=2, alpha=0.7, color=f'C{2+itr_op}')

        ax.set_xlim(0, self.total_time)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # make sure that the legend only shows unique labels.
        # Adapted from user Julien J in https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib/40870637#40870637
        handles, labels = ax.get_legend_handles_labels()
        unique_legend = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique_legend), loc='upper right', bbox_to_anchor=(1.2, 1))

    def _check_attr_predef_seqs(self, H1, pulse_shape, pulse_params, options, time_steps, free_duration, pi_pulse_duration, M):
        """
        Checks the commom attributes of the PulsedSim object for the predefined sequences and sets them accordingly.

        Parameters
        ----------
        H1 : Qobj or list(Qobj)
            control Hamiltonian of the system
        pulse_shape : callable or list(callable)
            pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params : dict
            dictionary of parameters for the pulse_shape functions
        options : dict
            options for the Qutip solver
        time_steps : int
            number of time steps for the pulses, if applicable
        free_duration : np.array
            free evolution times of the sequence, if applicable
        pi_pulse_duration : float
            duration of the pi pulse, if applicable
        M : int
            order of the sequence, if applicable 
        """
        # check whether pulse_shape is a python function or a list of python functions and if it is, assign it to the object
        if callable(pulse_shape) or (isinstance(pulse_shape, list) and all(callable(pulse_shape) for pulse_shape in pulse_shape)):
            self.pulse_shape = pulse_shape
        else:
            raise ValueError("pulse_shape must be a python function or a list of python functions")

        # check whether pulse_params is a dictionary and if it is, assign it to the object
        if pulse_params is None:
            self.pulse_params = {}
        elif isinstance(pulse_params, dict):
            self.pulse_params = pulse_params
        else:
            raise ValueError("pulse_params must be a dictionary or a list of dictionaries of parameters for the pulse function")
        
        # if phi_t is not in the pulse_params dictionary, assign it as 0
        if "phi_t" not in self.pulse_params:
            self.pulse_params["phi_t"] = 0

        # check whether options is a dictionary of solver options from Qutip and if it is, assign it to the object
        if options is None:
            self.options = {}
        elif isinstance(options, dict):
            self.options = options
        else:
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")
        
        # check whether H1 is a Qobj or a list of Qobjs of the same shape as H0 and with the same length as the pulse_shape list and if it is, assign it to the object
        if isinstance(H1, Qobj) and H1.shape == self.system.H0.shape:
            self.H1 = H1
            if self.H2 is None:
                self.Ht = [self.system.H0, [H1, pulse_shape]]
            else:
                self.Ht = [self.system.H0, [H1, pulse_shape], self.H2]
                self.H0_H2 = [self.system.H0, self.H2]

        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.system.H0.shape for op in H1) and len(H1) == len(pulse_shape):
            self.H1 = H1
            if self.H2 is None:
                self.Ht = [self.system.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            else:
                self.Ht = [self.system.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))] + self.H2
                self.H0_H2 = [self.system.H0, self.H2]

        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as H0 with the same length as the pulse_shape list")
        
        # check whether time_steps is a positive integer and if it is, assign it to the object
        if not isinstance(time_steps, int) or time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        else:
            self.time_steps = time_steps

        # check whether free_duration is a numpy array of real and positive elements and if it is, assign it to the object
        if free_duration is None:
            pass
        elif not isinstance(free_duration, (np.ndarray, list)) or not np.all(np.isreal(free_duration)) or not np.all(np.greater_equal(free_duration, 0)):
            raise ValueError("free_duration must be a numpy array with real positive elements")
        else:
            self.variable = free_duration
            self.variable_name = f"Tau (1/{self.system.units_H0})"
        
        # check whether pi_pulse_duration is a positive real number and if it is, assign it to the object
        if pi_pulse_duration is None:
            pass
        elif not isinstance(pi_pulse_duration, (int, float)) or pi_pulse_duration <= 0 or pi_pulse_duration > free_duration[0]:
            warnings.warn("pulse_duration must be a positive real number and pi_pulse_duration must be smaller than the free evolution time, otherwise pulses will overlap")
            self.pi_pulse_duration = pi_pulse_duration
        else:
            self.pi_pulse_duration = pi_pulse_duration

        # check whether M is a positive integer and if it is, assign it to the object
        if M is None:
            pass
        elif not isinstance(M, int) or M <= 0:
            raise ValueError("M must be a positive integer")
        else:
            self.M = M