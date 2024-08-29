# TODO: add baseline correction
# TODO: improve/rework the fit method with easier guess and bounds inputs
# TODO: add comparisions with experiments
# TODO: add option for more than one observable in the quantum system

"""
This the Analysis class to process and analyze the results of a PulsedSimulation object.
"""

import numpy as np
from.PulsedSim import PulsedSim
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from qutip import Bloch

class Analysis:
    """
    The Analysis class contains several methods to be applied with the results of a PulsedSimulation object or experimental data.

    Class Attributes
    ----------
    - experiment: PulsedSimulation object to be analyzed
    - FFT_values: tuple with the frequency values and the FFT values
    - FFT_peaks: array with the peaks of the FFT values

    Class Methods
    -------------
    - run_FFT: run the real fast fast Fourier transform for the results and variable attributes of the PulsedSimulation object
    - get_peaks_FFT: find the peaks of the FFT values calculated by the run_FFT method
    - plot_FFT: plot the FFT values calculated by the run_FFT method
    - run_fit: run the curve_fit method from scipy.optimize to fit the results of the experiment with a given fit function
    - plot_fit: plot the results of the experiment with the fitted function
    - plot_bloch: plot the results of the experiment in a Bloch sphere if the quantum system has dimension of two
    """
    def __init__(self, experiment):
        """
        Class generator for Analysis. It takes a PulsedSimulation object as input and checks if the results and variable attributes are not empty and have the same length.

        Parameters
        ----------
        experiment (PulsedSimulation): PulsedSimulation object to be analyzed
        """
        # check weather experiment is a PulsedSimulation object
        if not isinstance(experiment, PulsedSim):
            raise ValueError("PulsedSimulation must be a PulsedSimulation object")
        
        self.experiment = experiment
        
        # check if the results and variable attributes are not empty and have the same length
        if len(experiment.results) == 0:
            raise ValueError("Results attribute of PulsedSimulation object is empty, you must run the experiment first")
        if len(experiment.variable) == 0:
            raise ValueError("Variable attribute of PulsedSimulation object is empty, please define the variable of the experiment")
        if len(experiment.results) != len(experiment.variable):
            raise ValueError("Results and Variable attributes of PulsedSimulation object must have the same length")
         
        self.FFT_values = []
        self.FFT_peaks = []

    def run_FFT(self):
        """
        Run the real fast fast Fourier transform for the results and variable attributes of the PulsedSimulation object. The results are centered around the mean value before the FFT is calculated in order to remove the DC component.

        Returns
        -------
        FFT_values (tuple): tuple with the frequency values and the FFT values
        """
        y = np.abs(np.fft.rfft(self.experiment.results - np.mean(self.experiment.results)))
        freqs = np.fft.rfftfreq(len(self.experiment.variable), self.experiment.variable[1] - self.experiment.variable[0])

        self.FFT_values = (freqs, y)

        return self.FFT_values
    
    def get_peaks_FFT(self, **find_peaks_args):
        """
        Find the peaks of the FFT values calculated by the run_FFT method.

        Parameters
        ----------
        find_peaks_args (dict): dictionary with the arguments to be passed to the scipy.signal.find_peaks function
        """
        if not isinstance(self.FFT_values, tuple):
            raise ValueError("No FFT values to analyze, you must run the FFT first")
        
        self.FFT_peaks_index = find_peaks(self.FFT_values[1], **find_peaks_args)
        self.FFT_peaks = self.FFT_values[0][self.FFT_peaks_index[0]]
        return self.FFT_peaks
    
    def plot_FFT(self, freq_lim = None, figsize=(6, 4), xlabel='Frequencies', ylabel='FFT Intensity', title='FFT of the Results'):
        """
        Plots the pulse profiles of the experiment by iterating over the pulse_profiles list and plotting each pulse profile and free evolution.

        Parameters
        ----------
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        xlabel (str): label of the x-axis
        ylabel (str): label of the y-axis
        title (str): title of the plot
        """
        if self.FFT_values == ():
            raise ValueError("No FFT values to plot, you must run the FFT first")
        
        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        # initialize the figure and axis for the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(self.FFT_values[0], self.FFT_values[1])

        # if peaks have been found, plot them as red points
        if len(self.FFT_peaks) != 0:
            ax.scatter(self.FFT_peaks, self.FFT_values[1][self.FFT_peaks_index[0]], color='red', label='Peaks', s=50)

        # set the x-axis limits to the total time of the experiment
        if freq_lim == None:
            ax.set_xlim(self.FFT_values[0][0], self.FFT_values[0][-1])
        elif len(freq_lim) == 2:
            ax.set_xlim(freq_lim[0], freq_lim[1])
        else:
            raise ValueError("freq_lim must be a tuple of two floats")
        
        # set the axes labels according to the parameters
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    def run_fit(self, fit_function, guess=None, bounds=(-np.inf, np.inf)):
        """
        Run the curve_fit method from scipy.optimize to fit the results of the experiment with a given fit function, guess for the initial parameters and bounds for the parameters.

        Parameters
        ----------
        fit_function (function): function to be used to fit the results
        guess (list): initial guess for the parameters of the fit function
        bounds (list): bounds for the parameters of the fit function

        Returns
        -------
        fit (array): array with the fitted parameters
        fit_cov (array): array with the covariance of the fitted parameters
        """
        if not callable(fit_function):
            raise ValueError("fit_function must be a callable function")
        
        self.fit_function = fit_function
        
        fit, fit_cov = curve_fit(fit_function, self.experiment.variable, self.experiment.results, p0=guess, bounds=bounds, maxfev=100000)

        self.fit = fit
        self.fit_cov = fit_cov

        return self.fit, self.fit_cov
    
    def plot_fit(self, figsize=(6, 4), xlabel='Time', ylabel='Expectation Value', title='Pulsed Result'):
        """
        Plot the results of the experiment with the fitted function.

        Parameters
        ----------
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        xlabel (str): label of the x-axis
        ylabel (str): label of the y-axis
        title (str): title of the plot
        """
        self.experiment.plot_results(figsize, xlabel, ylabel, title)

        plt.plot(self.experiment.variable, self.fit_function(self.experiment.variable, *self.fit), label='Fit')
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1)) # updates the legend adding the fit label

    def plot_bloch(self, figsize=(6, 4)):
        """
        Plot the results of the experiment in a Bloch sphere if the quantum system has dimension of two.

        Parameters
        ----------
        figsize (tuple): size of the figure to be passed to matplotlib.pyplot
        """
        if len(self.experiment.rho) == 1:
            raise ValueError('Density matrices were not calculated, please run experiment first.')
        elif isinstance(self.experiment.rho, list) and all(rho.shape == (2,2) for rho in self.experiment.rho):
            pass
        else:
            raise ValueError('QSys must have dimension of two be able to plot a Bloch sphere')

        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        fig, axs = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': '3d'})

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.experiment.rho))) 

        bloch = Bloch(fig)
        bloch.add_states(self.experiment.rho, kind='point', colors=colors)
        bloch.frame_alpha = 0
        bloch.render()