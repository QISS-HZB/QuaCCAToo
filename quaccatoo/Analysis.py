#TODO FFT

import numpy as np
from.PulsedExperiment import PulsedExperiment
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def fit_rabi(t, A, Tpi, C, phi):
    return A*np.cos(np.pi*t/Tpi + phi) + C

def fit_rabi_decay(t, A, T, phi, C, Tc, n):
    return A*np.cos(2*np.pi*t/T + phi)*np.exp(-(t/Tc)**n) + C

def fit_hahn_mod(t, A, B, C, f1, f2):
    return ( A - B*np.sin(2*np.pi*f1*t/2)**2*np.sin(2*np.pi*f2*t/2)**2 ) + C

def fit_hahn_mod_decay(t, A, B, C, f1, f2, Tc, n):
    return np.exp(- (t/Tc)**n)*( A - B*np.sin(2*np.pi*f1*t/2)**2*np.sin(2*np.pi*f2*t/2)**2 ) + C

class Analysis(PulsedExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.FFT_values = ()
        self.FFT_peaks = ()

    def run_FFT(self):
        """
        """
        if self.results == []:
            raise ValueError("No results to analyze, you must run the experiment first")
        
        y = np.abs(np.fft.rfft(self.results - np.mean(self.results)))
        freqs = np.fft.fftfreq(len(self.variable), self.variable[1] - self.variable[0])

        self.FFT_values = (freqs, y)

        return freqs, y
    
    def get_peaks_FFT(self, height=0.1, distance=3):
        """
        """
        if not isinstance(self.FFT_values, tuple):
            raise ValueError("No FFT values to analyze, you must run the FFT first")
        
        if not isinstance(height, (int, float)):
            raise ValueError("height must be a float or int")
        if not isinstance(distance, (int, float)):
            raise ValueError("distance must be a float or int")
        
        peaks_position = find_peaks(self.FFT_values[1], height=height, distance=distance)
        self.FFT_peaks = self.FFT_values[0][peaks_position[0]]
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

        if self.FFT_peaks != ():
            ax.scatter(self.FFT_peaks, self.FFT_values[1][self.FFT_peaks], color='red', label='Peaks', s=50)

        # set the x-axis limits to the total time of the experiment
        if freq_lim == None:
            ax.set_xlim(self.FFT_values[0], self.FFT_values[-1])
        elif len(freq_lim) == 2:
            ax.set_xlim(freq_lim[0], freq_lim[1])
        else:
            raise ValueError("freq_lim must be a tuple of two floats")
        
        # set the axes labels according to the parameters
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)