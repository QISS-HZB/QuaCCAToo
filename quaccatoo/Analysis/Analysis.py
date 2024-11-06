# TODO: improve/rework the fit method with easier guess and bounds inputs inspired on qudi

"""
This module contains the Analysis class as part of the QuaCCAToo package.
"""

import matplotlib.pyplot as plt
import numpy as np
from qutip import Bloch
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress, pearsonr

from ..ExpData.ExpData import ExpData
from ..PulsedSim.PulsedSim import PulsedSim


class Analysis:
    """
    The Analysis class contains several methods for data Analysis, such as FFT, fitting and plotting.

    Attributes
    ----------
    experiment : PulsedSim or ExpData
        experiment object to be analyzed containing the results and variable attributes
    FFT_values : tuple
        tuple with the frequency values and the FFT values
    FFT_peaks : array
        array with the peaks of the FFT values
    fit_function : list
        list with the fit function for each result
    fit : list
        list with the fitted parameters for each result
    fit_cov : list
        list with the covariance of the fitted parameters for each result
    pearson : float
        Pearson correlation coefficient between two experiments
    exp_comparison : PulsedSim or ExpData
        experiment to be compared with the first one

    Methods
    -------
    compare_with :
        loads a second experiment to compare with the first one
    plot_comparison :
        plots the results of the experiment and the comparison experiment
    run_FFT :
        run the real fast fast Fourier transform for the results and variable attributes of the PulsedSimulation object
    get_peaks_FFT :
        find the peaks of the FFT values calculated by the run_FFT method
    plot_FFT :
        plot the FFT values calculated by the run_FFT method
    run_fit :
        run the curve_fit method from scipy.optimize to fit the results of the experiment with a given fit function
    plot_fit :
        plot the results of the experiment with the fitted function
    plot_results :
        plot the results of the experiment
    plot_bloch :
        plot the results of the experiment in a Bloch sphere if the quantum system has dimension of two
    """

    def __init__(self, experiment):
        """
        Class constructor for Analysis. It takes a PulsedSim or ExpData object as input and checks if the results and variable attributes are not empty and have the same length.

        Parameters
        ----------
        experiment : PulsedSim or ExpData
            experiment object to be analyzed containing the results and variable attributes
        """
        if isinstance(experiment, PulsedSim) or isinstance(experiment, ExpData):
            self.experiment = experiment
        else:
            raise ValueError("experiment must be a PulsedSimulation or ExpData object")

        if not isinstance(experiment.results, np.ndarray) and not (isinstance(experiment.results, list) and all(isinstance(res, np.ndarray) for res in experiment.results)):
            raise ValueError("Results attribute of the experiment must be a numpy array or a list of numpy arrays")

        if len(experiment.results) != len(experiment.variable) and any(len(experiment.variable) != len(res) for res in experiment.results):
            raise ValueError("Results and Variable attributes of experiment must have the same length")

        self.FFT_values = []
        self.FFT_peaks = []
        # the fit attributes need to be lists of the same length as the results attribute to avoid index errors
        self.fit_function = [None] * len(self.experiment.results)
        self.fit = [None] * len(self.experiment.results)
        self.fit_cov = [None] * len(self.experiment.results)
        self.pearson = None
        self.exp_comparison = None

    def compare_with(self, exp_comparison, results_index=0, comparison_index=0, linear_fit=True):
        """
        Loads a second experiment to compare with the first one.
        If linear_fit is True, a linear fit is performed between the two data sets, which is common for optical experiments.
        Otherwise the Pearson correlation coefficient without linear fit is calculated between the two data sets.

        Parameters
        ----------
        exp_comparison : PulsedSim or ExpData
            experiment to be compared with the first one
        results_index : int
            index of the results to be compared if the results attribute is a list
        comparison_index : int
            index of the results to be compared if the results attribute of the exp_comparison is a list
        linear_fit : bool
            boolean indicating whether or not to perform a linear fit between the two data sets

        Returns
        -------
        r : float
            Pearson correlation coefficient between the two experiments
        """
        if not isinstance(exp_comparison, PulsedSim) and not isinstance(exp_comparison, ExpData):
            raise ValueError("experiment_2 must be a PulsedSim or ExpData object")

        if not isinstance(results_index, int) or not isinstance(comparison_index, int):
            raise ValueError("results_index and comparison_index must be integers")
        elif results_index > len(self.experiment.results) - 1 or comparison_index > len(exp_comparison.results) - 1:
            raise ValueError("results_index and comparison_index must be less than the number of results")

        if not isinstance(linear_fit, bool):
            raise ValueError("linear_fit must be a boolean indicating whether or not to perform a linear fit between the two data sets.")

        if len(self.experiment.variable) != len(exp_comparison.variable):
            raise ValueError("The variable attributes of the experiments must have the same length")

        self.exp_comparison = exp_comparison

        # if linear_fit is True, perform a linear fit between the two data sets, otherwise calculate the Pearson correlation coefficient
        if linear_fit:
            if isinstance(exp_comparison.results, np.ndarray):
                r = linregress(exp_comparison.results, self.experiment.results)
                self.exp_comparison.results = r[0] * exp_comparison.results + r[1]
            else:
                # if the results are a list, index the results and perform the linear fit
                r = linregress(exp_comparison.results[comparison_index], self.experiment.results[results_index])
                self.exp_comparison.results = r[0] * exp_comparison.results[comparison_index] + r[1]

            self.pearson = r[2]

        else:
            if isinstance(exp_comparison.results, np.ndarray):
                r = pearsonr(exp_comparison.results, self.experiment.results)
            else:
                r = pearsonr(exp_comparison.results[comparison_index], self.experiment.results[results_index])

            self.exp_comparison.results = exp_comparison.results
            self.pearson = r[0]

        self.pearson = r
        return r

    def plot_comparison(self, figsize=(6, 4), xlabel=None, ylabel="Observable", title="Results Comparisions"):
        """
        Plots the results of the experiment and the comparison experiment.

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
        self.plot_results(figsize, xlabel, ylabel, title)

        if self.pearson is None:
            raise ValueError("You must run the compare_with method before plotting the comparison")

        plt.plot(self.exp_comparison.variable, self.exp_comparison.results, label="Compared Experiment", lw=2, alpha=0.7)
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    ######################################################## FFT Methods ########################################################

    def run_FFT(self):
        """
        Run the real fast Fourier transform for the results and variable attributes of the PulsedSimulation object.
        The results are centered around the mean value before the FFT is calculated in order to remove the DC component.

        Returns
        -------
        FFT_values : tuple
            tuple with the frequency values and the FFT values
        """
        if isinstance(self.experiment.results, np.ndarray):
            y = np.abs(np.fft.rfft(self.experiment.results - np.mean(self.experiment.results)))

        elif isinstance(self.experiment.results, list) and all(isinstance(res, np.ndarray) for res in self.experiment.results):
            y = [np.abs(np.fft.rfft(res - np.mean(res))) for res in self.experiment.results]

        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        freqs = np.fft.rfftfreq(len(self.experiment.variable), self.experiment.variable[1] - self.experiment.variable[0])

        self.FFT_values = (freqs, y)

        return self.FFT_values

    def get_peaks_FFT(self, **find_peaks_args):
        """
        Find the peaks of the FFT values calculated by the run_FFT method.

        Parameters
        ----------
        find_peaks_args : dict
            dictionary with the arguments to be passed to the scipy.signal.find_peaks function

        Returns
        -------
        FFT_peaks : array
            array with the peaks of the FFT values
        """
        if len(self.FFT_values) == 0:
            raise ValueError("No FFT values to analyze, you must run the FFT first")

        if isinstance(self.experiment.results, np.ndarray):
            self.FFT_peaks_index = find_peaks(self.FFT_values[1], **find_peaks_args)
            self.FFT_peaks = self.FFT_values[0][self.FFT_peaks_index[0]]

        elif isinstance(self.experiment.results, list):
            self.FFT_peaks_index = [find_peaks(FFT, **find_peaks_args) for FFT in self.FFT_values[1]]
            self.FFT_peaks = [self.FFT_values[0][index[0]] for index in self.FFT_peaks_index]
        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        return self.FFT_peaks

    def plot_FFT(self, freq_lim=None, figsize=(6, 4), xlabel=None, ylabel="FFT Intensity", title="FFT of the Results"):
        """
        Plots the FFT

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
        if len(self.FFT_values) == 0:
            raise ValueError("No FFT values to plot, you must run the FFT first")

        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        if xlabel is None and isinstance(self.experiment, PulsedSim):
            xlabel = f"Frequency ({self.experiment.system.units_H0})"
        elif xlabel is None and isinstance(self.experiment, ExpData):
            xlabel = "Frequency"
        elif not isinstance(xlabel, str):
            raise ValueError("xlabel must be a string")

        # initialize the figure and axis for the plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # if the FFT_values[1] is an array plot it, otherwise if it is a list iterate over the elements and plot each one
        if isinstance(self.FFT_values[1], np.ndarray):
            ax.plot(self.FFT_values[0], self.FFT_values[1])
            if len(self.FFT_peaks) != 0:
                ax.scatter(self.FFT_peaks, self.FFT_values[1][self.FFT_peaks_index[0]], color="red", label="Peaks", s=50)

        elif isinstance(self.FFT_values[1], list):
            # if the FFT_peaks attribute is not empty, then plot them with the FFT
            if len(self.FFT_peaks) != 0:
                for itr in range(len(self.FFT_values[1])):
                    ax.plot(self.FFT_values[0], self.FFT_values[1][itr], label=f"FFT {itr}")
                    ax.scatter(self.FFT_peaks[itr], self.FFT_values[1][itr][self.FFT_peaks_index[itr][0]], color="red", label=f"Peaks {itr}", s=50)
            else:
                for itr in range(len(self.FFT_values[1])):
                    ax.plot(self.FFT_values[0], self.FFT_values[1][itr], label=f"FFT {itr}")

        # set the x-axis limits to the total time of the experiment
        if freq_lim is None:
            ax.set_xlim(self.FFT_values[0][0], self.FFT_values[0][-1])
        elif len(freq_lim) == 2:
            ax.set_xlim(freq_lim[0], freq_lim[1])
        else:
            raise ValueError("freq_lim must be a tuple of two floats")

        # set the axes labels according to the parameters
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    ######################################################## FIT Methods ########################################################

    def run_fit(self, fit_function, results_index=0, guess=None, bounds=(-np.inf, np.inf)):
        """
        Run the curve_fit method from scipy.optimize to fit the results of the experiment with a given fit function,
        guess for the initial parameters and bounds for the parameters.

        Parameters
        ----------
        fit_function : function
            function to be used to fit the results
        results_index : int
            index of the results to be fitted if the results attribute is a list
        guess : list
            initial guess for the parameters of the fit function
        bounds : list
            bounds for the parameters of the fit function

        Returns
        -------
        fit : array
            array with the fitted parameters
        fit_cov : array
            array with the covariance of the fitted parameters
        """
        if not callable(fit_function):
            raise ValueError("fit_function must be a callable function")

        # if there is only one result, just fit the results with the fit_function
        if isinstance(self.experiment.results, np.ndarray):
            self.fit_function = fit_function
            self.fit, self.fit_cov = curve_fit(fit_function, self.experiment.variable, self.experiment.results, p0=guess, bounds=bounds, maxfev=100000)
            return self.fit, self.fit_cov

        # if there are multiple results, check if the results_index is an integer and if it is less than the number of results then fit
        elif isinstance(self.experiment.results, list):
            if not isinstance(results_index, int):
                raise ValueError("results_index must be an integer")
            elif results_index > len(self.experiment.results) - 1:
                raise ValueError("results_index must be less than the number of results")

            self.fit_function[results_index] = fit_function
            self.fit[results_index], self.fit_cov[results_index] = curve_fit(fit_function, self.experiment.variable, self.experiment.results[results_index], p0=guess, bounds=bounds, maxfev=100000)
            return self.fit[results_index], self.fit_cov[results_index]

    def plot_fit(self, figsize=(6, 4), xlabel="Time", ylabel="Expectation Value", title="Pulsed Result"):
        """
        Plot the results of the experiment with the fitted function.

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
        self.plot_results(figsize, xlabel, ylabel, title)

        if isinstance(self.experiment.results, np.ndarray):
            plt.plot(self.experiment.variable, self.fit_function(self.experiment.variable, *self.fit), label="Fit")

        elif isinstance(self.experiment.results, list):
            for itr in range(len(self.experiment.results)):
                if self.fit_function[itr] is not None:
                    plt.plot(self.experiment.variable, self.fit_function[itr](self.experiment.variable, *self.fit[itr]), label=f"Fit {itr}")

        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    ######################################################## Other Plotting Methods ########################################################

    def plot_results(self, figsize=(6, 4), xlabel=None, ylabel="Observable", title="Results"):
        """
        Plots the results of the experiment

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

        if not isinstance(ylabel, str) or not isinstance(title, str):
            raise ValueError("ylabel and title must be strings")

        if xlabel is None:
            xlabel = self.experiment.variable_name
        elif not isinstance(xlabel, str):
            raise ValueError("xlabel must be a string")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # check if the observable is a Qobj or a list of Qobj
        if isinstance(self.experiment.results, np.ndarray):
            ax.plot(self.experiment.variable, self.experiment.results, lw=2, alpha=0.7, label="Observable")

        elif isinstance(self.experiment.results, list):
            # if it is a list, iterate over the observables and plot each one
            for itr in range(len(self.experiment.system.observable)):
                # plot all observables in the results
                ax.plot(self.experiment.variable, self.experiment.results[itr], label=f"Observable {itr}", lw=2, alpha=0.7)

        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        # set the x-axis limits to the variable of the experiment
        ax.set_xlim(self.experiment.variable[0], self.experiment.variable[-1])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        ax.set_title(title)

    def plot_bloch(self, figsize=(6, 4)):
        """
        Plot the results of the experiment in a Bloch sphere if the quantum system has dimension of two.

        Parameters
        ----------
        figsize : tuple
            size of the figure to be passed to matplotlib.pyplot
        """
        if not isinstance(self.experiment, PulsedSim):
            raise ValueError("experiment must be a PulsedSim object")

        if len(self.experiment.rho) == 1:
            raise ValueError("Density matrices were not calculated, please run experiment first.")
        elif isinstance(self.experiment.rho, list) and all(rho.shape == (2, 2) for rho in self.experiment.rho):
            pass
        else:
            raise ValueError("QSys must have dimension of two to be able to plot a Bloch sphere")

        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        fig, axs = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": "3d"})

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.experiment.rho)))

        bloch = Bloch(fig)
        bloch.add_states(self.experiment.rho, kind="point", colors=colors)
        bloch.frame_alpha = 0
        bloch.render()

    def save():
        pass
