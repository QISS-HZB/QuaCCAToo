"""
This files contains the PulsedExperiment class that is used to define a general pulsed experiment with a sequence of pulse and free evolution operations.

Class Attributes
----------------
- rho0: initial state density matrix
- H0: time independent Hamiltonian of the system
- H1: control Hamiltonian of the system
- H2: time dependent Hamiltonian of the system
- c_ops: list of collapse operators
- total_time: total time of the experiment
- variable: variable of the experiment which the results depend on
- sequence: list of pulse and free evolution operations as functions
- pulse_profiles: list of pulse profiles for plotting purposes. Eeach element is a list [H1, tarray, pulse_shape, pulse_params], where H1 is the control Hamiltonian, tarray is the time array of the pulse, pulse_shape is the pulse time modulation function and pulse_params is the dictionary of parameters for the pulse_shape function
- results: results of the experiment from the run method
- options: dictionary of dynamic solver options from Qutip
- observable: observable to be measured after the sequence of operations


Class Methods
-------------
- add_pulse: adds a pulse operation to the sequence of operations of the experiment
- add_free_evolution: adds a free evolution operation to the sequence of operations of the experiment
- run: runs the pulsed experiment by performing each operation in the sequence list and saves the results in the results attribute
- plot_pulses: plots the pulse profiles of the experiment by iterating over the pulse_profiles list and plotting each pulse profile and free evolution
- plot_results: plots the results of the experiment and fits the results with predefined or user defined functions
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, mesolve, parallel_map
from types import FunctionType, MethodType
from scipy.optimize import curve_fit
from.PulseShapes import square_pulse
from.QSys import QSys

class PulsedExp:
    def __init__(self, system, H2 = None, c_ops = None):
        """
        Initializes a general PulsedExperiment object with the initial state density matrix, the time independent Hamiltonian, the time dependent Hamiltonian and the collapse operators.

        Parameters
        ----------
        rho0 (Qobj): initial state density matrix
        H0 (Qobj): time independent Hamiltonian of the system 
        H2 (Qobj): time dependent sensing Hamiltonian of the system
        c_ops (list(Qobj)): list of collapse operators
        """
        if not isinstance(system, QSys):
            raise ValueError("system must be a QSys object")
        
        self.rho0 = system.rho0.copy()
        self.rho = system.rho0.copy()
        self.H0 = system.H0.copy()
        self.observable = system.observable.copy()
        
        # check if c_ops is a list of Qobj with the same dimensions as H0
        if c_ops == None:
            self.c_ops = c_ops

        elif isinstance(c_ops, list):
            if not all(isinstance(op, Qobj) and op.shape == self.H0.shape for op in c_ops):
                raise ValueError("All items in c_ops must be Qobj with the same dimensions as H0")
            else:
                self.c_ops = c_ops
        else:
            raise ValueError("c_ops must be a list of Qobj or None")

        # initialize the rest of the variables and attributes
        self.total_time = 0 # total time of the experiment
        self.variable = None # variable of the experiment which the results depend on
        self.pulse_profiles = [] # list of pulse profiles for plotting purposes, where each element is a list [H1, tarray, pulse_shape, pulse_params]
        self.results = [] # results of the experiment to be later generated in the run method
        self.sequence = None # parallel sequence of operations to be overwritten in PredefinedPulsedExperiments or defined by the user
    
    def add_pulse(self, duration, H1, phi_t=0, pulse_shape = square_pulse, pulse_params = {}, time_steps = 100, options={}):
        """
        Perform variables checks and adds a pulse operation to the sequence of operations of the experiment for a given duration of the pulse, control Hamiltonian H1, pulse shape function and pulse parameters by calling the pulse method.

        Parameters
        ----------
        duration (float, int): duration of the pulse
        H1 (Qobj, list(Qobj)): control Hamiltonian of the system
        phi_t (float): time phase of the pulse representing the rotation axis in the rotating frame
        pulse_shape (FunctionType, list(FunctionType)): pulse shape function or list of pulse shape functions representing the time modulation of t H1
        pulse_params (dict): dictionary of parameters for the pulse_shape functions
        time_steps (int): number of time steps for the pulses
        options (dict): options for the Qutip solver
        """
        # check if options is a dictionary of dynamic solver options from Qutip
        if not isinstance(options, dict):
            raise ValueError("options must be a dictionary of dynamic solver options from Qutip")

        # check if time_steps is a positive integer
        if not isinstance(time_steps, int) or time_steps <= 0:
            raise ValueError("time_steps must be a positive integer")
        
        # check if duration of the pulse is a positive real number
        if not isinstance(duration, (int, float)) and duration <= 0:
            raise ValueError("duration must be a positive real number")
        else:
            # create a time array for the pulse, starting from the total time of the experiment up to the beginning of the pulse
            tarray = np.linspace(self.total_time, self.total_time + duration, time_steps) 

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
        if isinstance(H1, Qobj) and H1.shape == self.rho0.shape:
            # create the time independent + time dependent Hamiltonian
            Ht = [self.H0, [H1, pulse_shape]]
            # append it to the pulse_profiles list
            self.pulse_profiles.append( [H1, tarray, pulse_shape, pulse_params] )
        elif isinstance(H1, list) and all(isinstance(op, Qobj) and op.shape == self.rho0.shape for op in H1) and len(H1) == len(pulse_shape):
            Ht = [self.H0] + [[H1[i], pulse_shape[i]] for i in range(len(H1))]
            self.pulse_profiles = [[H1[i], tarray, pulse_shape[i], pulse_params] for i in range(len(H1))]
        else:
            raise ValueError("H1 must be a Qobj or a list of Qobjs of the same shape as rho0, H0 and H1 with the same length as the pulse_shape list")
        
        # add the pulse operation to the sequence of operations
        self.pulse(Ht, tarray, options, pulse_params, phi_t)

    def pulse(self, Ht, tarray, options, pulse_params, phi_t):
        """
        Updates the total time of the experiment, adds H1 to the pulse profile for plotting, sets the phase for the pulse and calls the pulse_operation function to perform the pulse operation.

        Parameters
        ----------
        Ht (list): list of Hamiltonians for the pulse operation
        tarray (np.array): time array for the pulse operation
        pulse_params (dict): dictionary of parameters for the pulse_shape functions
        phi_t (float): time phase of the pulse representing the rotation axis in the rotating frame

        Returns
        -------
        final state of the system (Qobj)
        """
        # update the total time
        self.total_time = tarray[-1]
       
        # update the phase of the pulse
        pulse_params['phi_t'] = pulse_params['phi_t'] + phi_t
        # perform the pulse operation
        self.rho = mesolve(Ht, self.rho, 2*np.pi*tarray, self.c_ops, [], options = options, args = pulse_params).states[-1]
        # return the final state of the system
        return self.rho
        
    def add_free_evolution(self, duration):
        """
        Adds a free evolution operation to the sequence of operations of the experiment for a given duration of the free evolution by calling the free_evolution method.

        Parameters
        ----------
        duration (float, int): duration of the free evolution
        """
        # check if duration of the pulse is a positive real number
        if not isinstance(duration, (int, float)) and duration <= 0:
            raise ValueError("duration must be a positive real number")
        
        # add the free evolution to the pulse_profiles list
        self.pulse_profiles.append( [None, [self.total_time, duration + self.total_time], None, None] )

        # add the free evolution operation to the sequence of operations
        self.free_evolution(duration)
    
    def free_evolution(self, duration):
        """
        Updates the total time of the experiment and applies the time-evolution operator to the initial density matrix to perform the free evolution operation.

        Parameters
        ----------
        duration (float, int): duration of the free evolution

        Returns
        -------
        final state of the system (Qobj)
        """
        # update the total time
        self.total_time += duration
        # perform the free evolution operation
        self.rho = (-1j*2*np.pi*self.H0*duration).expm() * self.rho * ((-1j*2*np.pi*self.H0*duration).expm()).dag()
        # return the final state of the system
        return self.rho
    
    def measure(self, observable=None):
        """
        """
        if observable == None and self.observable == None:
            self.results = self.rho.copy()
        elif observable == None and self.observable != None:
            self.results = np.abs(( self.observable * self.rho ).tr() )
        elif observable != None and (isinstance(observable, Qobj) and observable.shape == self.rho0.shape):
            self.observable = observable
            self.results = np.abs(( observable * self.rho ).tr() )   
        else:
            raise ValueError("observable must be a Qobj of the same shape as rho0, H0 and H1.")

        return self.results

    def run(self, variable=None, sequence=None):
        """
        """
        if sequence == None and self.sequence != None:
            pass
        elif isinstance(sequence, FunctionType):
            self.sequence = sequence
        else:
            raise ValueError("sequence must be a python function with a list operations returning a number")

        # check if a variable was passed by the user, if it is numpy array overwrite the variable attribute
        if isinstance(variable, np.ndarray):
            self.variable = variable
        elif variable == None and len(self.variable) !=0:
            pass
        else:
            raise ValueError("variable must be a numpy array")            

        self.results = parallel_map(self.sequence, self.variable)
            
    def plot_pulses(self, figsize=(6, 4), xlabel='Time', ylabel='Pulse Intensity', title='Pulse Profiles'):
        """
        Plots the pulse profiles of the experiment by iterating over the pulse_profiles list and plotting each pulse profile and free evoltution.

        Parameters
        ----------
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        xlabel (str): label of the x-axis
        ylabel (str): label of the y-axis
        title (str): title of the plot
        """
        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        # initialize the figure and axis for the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # iterate over all operations in the sequence
        for itr_pulses in range(len(self.pulse_profiles)):

            # if the pulse profile is a free evolution, plot a horizontal line at 0
            if self.pulse_profiles[itr_pulses][0] == None:
                ax.plot(self.pulse_profiles[itr_pulses][1], [0,0], label = 'Free Evolution', lw=2, alpha=0.7, color = 'C0')

            # if the pulse has only one operator, plot the pulse profile
            elif isinstance(self.pulse_profiles[itr_pulses][0], Qobj):
                ax.plot(self.pulse_profiles[itr_pulses][1], self.pulse_profiles[itr_pulses][2](self.pulse_profiles[itr_pulses][1], **self.pulse_profiles[itr_pulses][3]), label = f'H1', lw=2, alpha=0.7, color = 'C1')

            # if the pulse has multiple operators, plot each pulse profile
            elif isinstance(self.pulse_profiles[itr_pulses][0], list):
                for itr_op in range(len(self.pulse_profiles[itr_pulses])):
                    ax.plot(self.pulse_profiles[itr_pulses][itr_op][1], self.pulse_profiles[itr_pulses][itr_op][2](self.pulse_profiles[itr_pulses][itr_op][1], **self.pulse_profiles[itr_pulses][itr_op][3]), label = f'H1_{itr_op}', lw=2, alpha=0.7, color = f'C{2+itr_op}')

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

    def plot_results(self, figsize=(6, 4), fit_function=None, fit_guess=None, xlabel='Time', ylabel='Expectation Value', title='Pulsed Experiment Result'):
        """
        Plots the results of the experiment and fits the results with predefined or user defined functions.

        Parameters
        ----------
        variable (np.array): xaxis variable of the plot representing the parameter being changed in the experiment
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        fit_function (FunctionType): function or list of functions to fit the results of the experiment, if None, no fit is performed
        fit_guess (list): initial guess or list of initial guesses for the fit_function
        xlabel (str): label of the x-axis
        ylabel (str): label of the y-axis
        title (str): title of the plot        
        """
        
        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        # initialize the figure and axis for the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # check if the observable is a Qobj or a list of Qobj
        if isinstance(self.observable, Qobj):
            ax.plot(self.variable, self.results, lw=2, alpha=0.7, label = 'Observable')

            # check if fit_function is a python function or None
            if fit_function == None: # if no function is given, fit is not performed
                pass
            elif isinstance(fit_function, FunctionType): # if a function is given, performs a fit to the results with the predefined or user defined function.
                #@TODO predefine some fit functions
                params, cov = curve_fit(fit_function, self.variable, self.results, p0=fit_guess, maxfev=10000000) # perform the fit using scipy.optimize.curve_fit and the fit_guess if provided
                ax.plot(self.variable, fit_function(self.variable, *params), linestyle='--', lw=2, alpha=0.7, label = 'Fit')

                print(f'Fit parameters: {params}')
                print(f'Covariance matrix: {cov}')

            else:
                raise ValueError("fit_function must be a python function or None")
                    
        else:
            # if it is a list, iterate over the observables and plot each one
            for itr in range(len(self.observable)):
                # plot all observables in the results
                ax.plot(self.variable, self.results[itr], label = f'Observable {itr}', lw=2, alpha=0.7)
            
            if fit_function == None:
                pass

            elif isinstance(fit_function, list):
                # check if the fit_function is a list of python functions and has the same length as the observables
                if all(isinstance(fit, FunctionType) for fit in fit_function) and len(fit_function) == len(self.observable) and len(fit_guess) == len(self.observable):
                    for itr in range(len(fit_function)):
                        # perform each fit and plot
                        params, cov = curve_fit(fit_function[itr], self.variable, self.results[itr], p0=fit_guess[itr], maxfev=10000000)
                        ax.plot(self.variable, fit_function[itr](self.variable, *params), linestyle='--', lw=2, alpha=0.7, label = f'Fit {itr}')

                        print(f'Fit parameters {itr}: {params}')
                        print(f'Covariance matrix {itr}: {cov}')
                else:
                    raise ValueError("fit_function must be a list of python functions or None")
            else:
                raise ValueError("fit_function must be a list of python functions or None")
            
        # set the x-axis limits to the variable of the experiment
        ax.set_xlim(self.variable[0], self.variable[-1])
        # set the axes labels according to the parameters
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        ax.set_title(title)