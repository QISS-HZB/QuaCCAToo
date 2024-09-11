# TODO: units in plot_pulses
# TODO: implement save method

"""
This module contains the PulsedSim class that is used to define a general pulsed experiment with a sequence of pulses and free evolution operations, part of the QuaCAAToo package.
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, mesolve, parallel_map
from .PulseShapes import square_pulse
from .QSys import QSys


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
        parallel sequence of operations to be overwritten in BasicPulsedSim and DDPulsedSim, or defined by the user
    time_steps :

    Methods
    -------
    add_pulse
        adds a pulse operation to the sequence of operations of the experiment
    pulse
        updates the total time of the experiment, sets the phase for the pulse and calls mesolve from QuTip to perform the pulse operation
    add_free_evolution
        adds a free evolution operation to the sequence of operations of the experiment
    free_evolution
        updates the total time of the experiment and applies the time-evolution operator to perform the free evolution operation with the exponential operator
    free_evolution_H2
        same as free_evolution but using mesolve for the time dependent Hamiltonian or collapse operators
    run
        runs the pulsed experiment by calling the parallel_map function from QuTip over the variable attribute
    measure
        measures the observable after the sequence of operations and returns the expectation value of the observable
    plot_pulses
        plots the pulse profiles of the experiment by iterating over the pulse_profiles list and plotting each pulse profile and free evolution
    save
        saves the experiment
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
        # check if system is a QSys object
        if not isinstance(system, QSys):
            raise ValueError("system must be a QSys object")

        self.system = system

        # get the attributes of the system
        self.rho = system.rho0.copy()

        # if collapse operators are given, the H0_H2 attributed needs to be set with H0 for the mesolve function
        if self.system.c_ops is not None:
            self.H0_H2 = self.system.H0

        if H2 is None:
            self.H2 = None
        elif H2[0].shape != self.system.rho0.shape or not callable(H2[1]):
            raise ValueError("H2 must be None or a list of one Qobj of the same shape as rho0 and one time dependent function")
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

    def add_pulse(self, duration, H1, phi_t=0, pulse_shape=square_pulse, pulse_params={}, time_steps=100, options={}):
        """
        Perform variables checks and adds a pulse operation to the sequence of operations of the experiment for a given duration of the pulse,
        control Hamiltonian H1, pulse phase, pulse shape function, pulse parameters and time steps by calling the pulse method.

        Parameters
        ----------
        duration : float or int
            duration of the pulse
        H1 : Qobj or list(Qobj)
            control Hamiltonian of the system
        phi_t : float
            time phase of the pulse representing the rotation axis in the rotating frame
        pulse_shape : callable or list(callable)
            pulse shape function or list of pulse shape functions representing the time modulation of t H1
        pulse_params : dict
            dictionary of parameters for the pulse_shape functions
        time_steps : int
            number of time steps for the pulses
        options : dict
            options for the Qutip solver
        """
        # check if options is a dictionary of dynamic solver options from Qutip
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")

        # check if time_steps is a positive integer
        if not isinstance(time_steps, int) or time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        else:
            self.time_steps = time_steps

        # check if duration of the pulse is a positive real number
        if not isinstance(duration, (int, float)) and duration <= 0:
            raise ValueError("duration must be a positive real number")

        # Check if pulse_shape is a single function or a list of functions
        if not (callable(pulse_shape) or (isinstance(pulse_shape, list) and all(callable(p) for p in pulse_shape))):
            raise ValueError("pulse_shape must be a python function or a list of python functions")

        # check if pulse_params is a dictionary to be passed to the pulse_shape function
        if not isinstance(pulse_params, dict):
            raise ValueError('pulse_params must be a dictionary of parameters for the pulse function')
        # if the user doesn't provide a phi_t, set it to 0
        if 'phi_t' not in pulse_params:
            pulse_params['phi_t'] = 0

        # check if phi_t is a real number
        if not isinstance(phi_t, (int, float)):
            raise ValueError("phi_t must be a real number")

        # check if H1 is a Qobj or a list of Qobj with the same dimensions as H0 and rho0
        if isinstance(H1, Qobj) and H1.shape == self.system.rho0.shape:
            # append it to the pulse_profiles list
            self.pulse_profiles.append([H1, np.linspace(self.total_time, self.total_time + duration, self.time_steps), pulse_shape, pulse_params])
            if self.H2 is None:
                Ht = [self.system.H0, [H1, pulse_shape]]
            else:
                Ht = [self.system.H0, [H1, pulse_shape], self.H2]

        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.system.rho0.shape for op in H1) and len(H1) == len(pulse_shape):
            self.pulse_profiles = [[H1[i], np.linspace(self.total_time, self.total_time + duration, self.time_steps), pulse_shape[i], pulse_params] for i in range(len(H1))]
            if self.H2 is None:
                Ht = [self.system.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            else:
                Ht = [self.system.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))] + self.H2

        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")

        # add the pulse operation to the sequence of operations by calling the pulse method
        self._pulse(Ht, duration, options, pulse_params, phi_t)

    def _pulse(self, Ht, duration, options, core_pulse_params, phi_t):
        """
        Updates the total time of the experiment, sets the phase for the pulse and calls the pulse_operation function to perform the pulse operation.
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
        phi_t : float
            time phase of the pulse representing the rotation axis in the rotating frame
        """
        # update the phase of the pulse
        core_pulse_params['phi_t'] += phi_t

        # perform the pulse operation. The time array is multiplied by 2*pi so that [H*t] has units of radians
        self.rho = mesolve(Ht, self.rho, 2*np.pi*np.linspace(self.total_time, self.total_time + duration, self.time_steps) , self.system.c_ops, [], options = options, args = core_pulse_params).states[-1]

        # update the total time
        self.total_time += duration

    def add_free_evolution(self, duration, options={}):
        """
        Adds a free evolution operation to the sequence of operations of the experiment for a given duration of the free evolution by calling the free_evolution method.

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

        # add the free evolution to the pulse_profiles list
        self.pulse_profiles.append([None, [self.total_time, duration + self.total_time], None, None])

        # if a H2 or collapse operators are given, use free_evolution_H2 method, otherwise use free_evolution method
        if self.H2 is not None or self.system.c_ops is not None:
            # check weather options is a dictionary of solver options from Qutip and if it is, assign it to the object
            if not isinstance(options, dict):
                raise ValueError("options must be a dictionary of dynamic solver options from Qutip")

            self.free_evolution_H2(duration, options)
        else:
            self.free_evolution(duration)

    def free_evolution(self, duration):
        """
        Updates the total time of the experiment and applies the time-evolution operator to the initial density matrix to perform the free evolution operation with the exponential operator.
        This method should be used internally by other methods, as it does not perform any checks on the input parameters for better performance.

        Parameters
        ----------
        duration : float or int
            duration of the free evolution
        """
        self.rho = (-1j*2*np.pi*self.system.H0*duration).expm() * self.rho * ((-1j*2*np.pi*self.system.H0*duration).expm()).dag()

        # update the total time
        self.total_time += duration

    def free_evolution_H2(self, duration, options={}):
        """
        Same as free_evolution but using mesolve for the time dependent Hamiltonian or collapse operators.

        Parameters
        ----------
        duration : float or int
            duration of the free evolution
        options : dict
            options for the Qutip solver
        """
        self.rho = mesolve(self.H0_H2, self.rho, 2*np.pi*np.linspace(self.total_time, self.total_time + duration, self.time_steps) , self.system.c_ops, [], options=options).states[-1]

        # update the total time
        self.total_time += duration

    def measure(self, observable=None):
        """
        Measures the observable after the sequence of operations and returns the expectation value of the observable.

        Parameters
        ----------
        observable : Qobj
            observable to be measured after the sequence of operations

        Returns
        -------
        results of the experiment
        """
        # if no observable is passed and the QSys doesn't have one, returns the final density matrix
        if observable is None and self.system.observable is None:
            return self.rho.copy()
        # if no observable is passed but the QSys has one, returns the expectation value of the observable from QSys
        elif observable is None and self.system.observable is not None:
            self.results = np.real((self.system.observable * self.rho).tr())
        # if an observable is passed, checks the dimensions of the observable and returns the expectation value of the observable
        elif observable is None and (isinstance(observable, Qobj) and observable.shape == self.system.rho0.shape):
            self.system.observable = observable
            self.results = np.real((observable * self.rho).tr())
        # else raises an error
        else:
            raise ValueError("observable must be a Qobj of the same shape as rho0, H0 and H1.")

        # return the results of the experiment
        return self.results

    def run(self, variable=None, sequence=None):
        """
        Runs the pulsed experiment by calling the parallel_map function from QuTip over the variable attribute.

        Parameters
        ----------
        variable : np.array
            xaxis variable of the plot representing the parameter being changed in the experiment
        sequence : callable
            sequence of operations to be performed in the experiment
        """
        # if no sequence is passed but the PulsedSim has one, uses the attribute sequence
        if sequence is None and self.sequence is not None:
            pass
        # if a sequence is passed, checks if it is a python function and overwrites the attribute
        elif callable(sequence):
            self.sequence = sequence
        # else raises an error
        else:
            raise ValueError("sequence must be a python function with a list operations returning a number")

        # check if a variable was passed by the user, if it is numpy array overwrite the variable attribute
        if isinstance(variable, np.ndarray):
            self.variable = variable
        elif variable is None and len(self.variable) != 0:
            pass
        else:
            raise ValueError("variable must be a numpy array")

        # run the experiment by calling the parallel_map function from QuTip over the variable attribute
        self.rho = parallel_map(self.sequence, self.variable)

        # if an observable is given, calculate the expectation values
        if isinstance(self.system.observable, Qobj):
            self.results = np.array([np.real((rho*self.system.observable).tr()) for rho in self.rho])  # np.real is used to ensure no imaginary components will be attributed to results
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
        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        if xlabel is None:
            xlabel = self.variable_name
        elif not isinstance(xlabel, str):
            raise ValueError("xlabel must be a string")

        # initialize the figure and axis for the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # iterate over all operations in the sequence
        for itr_pulses in range(len(self.pulse_profiles)):

            # if the pulse profile is a free evolution, plot a horizontal line at 0
            if self.pulse_profiles[itr_pulses][0] is None:
                ax.plot(self.pulse_profiles[itr_pulses][1], [0, 0], label='Free Evolution', lw=2, alpha=0.7, color='C0')

            # if the pulse has only one operator, plot the pulse profile
            elif isinstance(self.pulse_profiles[itr_pulses][0], Qobj):
                ax.plot(self.pulse_profiles[itr_pulses][1], self.pulse_profiles[itr_pulses][2](2*np.pi*self.pulse_profiles[itr_pulses][1], **self.pulse_profiles[itr_pulses][3]), label='H1', lw=2, alpha=0.7, color='C1')

            # if the pulse has multiple operators, plot each pulse profile
            elif isinstance(self.pulse_profiles[itr_pulses][0], list):
                for itr_op in range(len(self.pulse_profiles[itr_pulses])):
                    ax.plot(self.pulse_profiles[itr_pulses][itr_op][1], self.pulse_profiles[itr_pulses][itr_op][2](2*np.pi*self.pulse_profiles[itr_pulses][itr_op][1], **self.pulse_profiles[itr_pulses][itr_op][3]), label=f'H1_{itr_op}', lw=2, alpha=0.7, color=f'C{2+itr_op}')

        # set the x-axis limits to the total time of the experiment
        ax.set_xlim(0, self.total_time)
        # set the axes labels according to the parameters
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # make sure that the legend only shows unique labels.
        # Adapted from user Julien J in https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib/40870637#40870637
        handles, labels = ax.get_legend_handles_labels()
        unique_legend = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique_legend), loc='upper right', bbox_to_anchor=(1.2, 1))

    def save():
        pass
