import numpy as np
from.PulsedExp import PulsedExp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from qutip import Bloch

def fit_rabi(t, A, Tpi, C, phi):
    return A*np.cos(np.pi*t/Tpi + phi) + C

def fit_rabi_decay(t, A, T, phi, C, Tc, n):
    return A*np.cos(2*np.pi*t/T + phi)*np.exp(-(t/Tc)**n) + C

def fit_exp_decay(t, A, C, Tc):
    return A*np.exp(-t/Tc) + C

def fit_exp_decay_n(t, A, C, Tc, n):
    return A*np.exp(-(t/Tc)**n) + C

def fit_hahn_mod(t, A, B, C, f1, f2):
    return ( A - B*np.sin(2*np.pi*f1*t/2)**2*np.sin(2*np.pi*f2*t/2)**2 ) + C

def fit_hahn_mod_decay(t, A, B, C, f1, f2, Tc, n):
    return np.exp(- (t/Tc)**n)*( A - B*np.sin(2*np.pi*f1*t/2)**2*np.sin(2*np.pi*f2*t/2)**2 ) + C

class Analysis:
    def __init__(self, experiment):
        """
        """
        # check weather PulseExperiment is a PulsedExperiment object
        if not isinstance(experiment, PulsedExp):
            raise ValueError("PulsedExperiment must be a PulsedExperiment object")
        
        self.experiment = experiment
        
        # check if the results and variable attributes are not empty and have the same length
        if len(experiment.results) == 0:
            raise ValueError("Results attribute of PulsedExperiment object is empty, you must run the experiment first")
        if len(experiment.variable) == 0:
            raise ValueError("Variable attribute of PulsedExperiment object is empty, please define the variable of the experiment")
        if len(experiment.results) != len(experiment.variable):
            raise ValueError("Results and Variable attributes of PulsedExperiment object must have the same length")
        else:
            self.results = experiment.results
            self.variable = experiment.variable
            self.rho = experiment.rho
         
        self.FFT_values = []
        self.FFT_peaks = []

    def run_FFT(self):
        """
        """
        
        y = np.abs(np.fft.rfft(self.results - np.mean(self.results)))
        freqs = np.fft.rfftfreq(len(self.variable), self.variable[1] - self.variable[0])

        self.FFT_values = (freqs, y)

        return self.FFT_values
    
    def get_peaks_FFT(self, **find_peaks_args):
        """
        """
        if not isinstance(self.FFT_values, tuple):
            raise ValueError("No FFT values to analyze, you must run the FFT first")
        
        self.FFT_peaks_index = find_peaks(self.FFT_values[1], **find_peaks_args)
        self.FFT_peaks = self.FFT_values[0][self.FFT_peaks_index[0]]
        return self.FFT_peaks
    
    def plot_FFT(self, freq_lim = None, figsize=(6, 4), xlabel='Frequencies', ylabel='FFT Intensity', title='FFT of the Results'):
        """
        Plots the pulse profiles of the experiment by iterating over the pulse_profiles list and plotting each pulse profile and free evoltution.

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
        """
        if not callable(fit_function):
            raise ValueError("fit_function must be a callable function")
        
        self.fit_function = fit_function
        
        fit, fit_cov = curve_fit(fit_function, self.variable, self.results, p0=guess, bounds=bounds, maxfev=100000)

        self.fit = fit
        self.fit_cov = fit_cov

        return self.fit, self.fit_cov
    
    def plot_fit(self, figsize=(6, 4), xlabel='Time', ylabel='Expectation Value', title='Pulsed Experiment Result'):
        """
        """
        self.experiment.plot_results(figsize, xlabel, ylabel, title)

        plt.plot(self.variable, self.fit_function(self.variable, *self.fit), label='Fit')

    def plot_bloch(self, figsize=(6, 4)):
        """
        """
        if len(self.rho) == 1:
            raise ValueError('Density matrices were not calculated, please run experiment first.')
        elif isinstance(self.rho, list) and all(rho.shape == (2,2) for rho in self.rho):
            pass
        else:
            raise ValueError('QSys must have dimesion of two be able to plot a Bloch sphere')

        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        fig, axs = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': '3d'})

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.rho))) 

        bloch = Bloch(fig)
        bloch.add_states(self.rho, kind='point', colors=colors)
        bloch.frame_alpha = 0
        bloch.render()