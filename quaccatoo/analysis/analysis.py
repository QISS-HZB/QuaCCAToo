"""
This module contains the Analysis class and the plot_histogram method.
"""

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from qutip import Bloch, Qobj, fidelity
from scipy.signal import find_peaks
from scipy.stats import linregress, pearsonr
from typing import Any, Literal, Optional

from ..exp_data.exp_data import ExpData
from ..pulsed_sim.pulsed_sim import PulsedSim

class Analysis:
    """
    The Analysis class contains several methods for data Analysis, such as FFT, fitting and plotting.

    Attributes
    ----------
    experiment : PulsedSim or ExpData
        Experiment object to be analyzed containing the results and variable attributes
    FFT_values : tuple
        Tuple with the frequency values and the FFT values
    FFT_peaks : array
        Array with the peaks of the FFT values
    fit_model : lmfit.Model or list(lmfit.Model)
        List with the fitting model for each result
    fit_params : list
        List with the fitted parameters for each result
    fit_cov : list
        List with the covariance of the fitted parameters for each result
    pearson : float
        Pearson correlation coefficient between two experiments
    exp_comparison : PulsedSim or ExpData
        Experiment to be compared with the first one

    Methods
    -------
    compare_with :
        Load a second experiment to compare with the first one
    plot_comparison :
        Plot the results of the experiment and the comparison experiment
    run_FFT :
        Run the real fast fast Fourier transform for the results and variable attributes of the PulsedSimulation object
    get_peaks_FFT :
        Find the peaks of the FFT values calculated by the run_FFT method
    plot_FFT :
        Plot the FFT values calculated by the run_FFT method
    run_fit :
        Run the fit method from lmfit to fit the results of the experiment with a given model
    plot_fit :
        Plot the results of the experiment with the fitted function
    plot_results :
        Plot the results of the experiment
    plot_bloch :
        Plot the results of the experiment in a Bloch sphere if the quantum system has dimension of two
    """
    def __init__(
        self,
        experiment : ExpData | PulsedSim  
    ) -> None:
        """
        Class constructor for Analysis. It takes a PulsedSim or ExpData object as input and checks if the results and variable attributes are not empty and have the same length.

        Parameters
        ----------
        experiment : PulsedSim or ExpData
            Experiment object to be analyzed containing the results and variable attributes
        """
        if isinstance(experiment, (ExpData, PulsedSim)):
            self.experiment = experiment
        else:
            raise ValueError("experiment must be a PulsedSimulation or ExpData object")

        if not isinstance(experiment.results, np.ndarray) and not (
            isinstance(experiment.results, list)
            and all(isinstance(res, np.ndarray) for res in experiment.results)
        ):
            raise ValueError(
                "Results attribute of the experiment must be a numpy array or a list of numpy arrays"
            )

        if len(experiment.results) != len(experiment.variable) and any(
            len(experiment.variable) != len(res) for res in experiment.results
        ):
            raise ValueError("Results and Variable attributes of experiment must have the same length")

        self.FFT_values = []
        self.FFT_peaks = []
        # the fit attributes need to be lists of the same length as the results attribute to avoid index errors
        self.fit_model = [None] * len(self.experiment.results)
        self.fit_params = [None] * len(self.experiment.results)
        self.fit_cov = [None] * len(self.experiment.results)
        self.pearson = None
        self.exp_comparison = None

    def compare_with(
        self,
        exp_comparison : ExpData | PulsedSim ,
        results_index : int = 0,
        comparison_index : int = 0,
        linear_fit : bool = True
    ) -> float:
        """
        Loads a second experiment to compare with the first one.
        If linear_fit is True, a linear fit is performed between the two data sets, which is common for optical experiments.
        Otherwise the Pearson correlation coefficient without linear fit is calculated between the two data sets.

        Parameters
        ----------
        exp_comparison : PulsedSim or ExpData
            Experiment to be compared with the first one
        results_index : int
            Index of the results to be compared if the results attribute is a list
        comparison_index : int
            Index of the results to be compared if the results attribute of the exp_comparison is a list
        linear_fit : bool
            Boolean indicating whether or not to perform a linear fit between the two data sets

        Returns
        -------
        r : float
            Pearson correlation coefficient between the two experiments
        """
        if not isinstance(exp_comparison, PulsedSim) and not isinstance(exp_comparison, ExpData):
            raise ValueError("experiment_2 must be a PulsedSim or ExpData object")

        if not isinstance(results_index, int) or not isinstance(comparison_index, int):
            raise ValueError("results_index and comparison_index must be integers")
        elif (
            results_index > len(self.experiment.results) - 1
            or comparison_index > len(exp_comparison.results) - 1
        ):
            raise ValueError("results_index and comparison_index must be less than the number of results")

        if not isinstance(linear_fit, bool):
            raise ValueError(
                "linear_fit must be a boolean indicating whether or not to perform a linear fit between the two data sets."
            )

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
                r = linregress(
                    exp_comparison.results[comparison_index], self.experiment.results[results_index]
                )
                self.exp_comparison.results = r[0] * exp_comparison.results[comparison_index] + r[1]

            self.pearson = r[2]

        else:
            if isinstance(exp_comparison.results, np.ndarray):
                r = pearsonr(exp_comparison.results, self.experiment.results)
            else:
                r = pearsonr(
                    exp_comparison.results[comparison_index], self.experiment.results[results_index]
                )

            self.exp_comparison.results = exp_comparison.results
            self.pearson = r[0]

        self.pearson = r
        return r

    def plot_comparison(
        self,
        figsize : tuple[int, int] = (6, 4),
        xlabel : Optional[str] = None,
        ylabel : str = "Observable",
        title : str = "Results Comparisons"
    ) -> None:
        """
        Plots the results of the experiment and the comparison experiment.

        Parameters
        ----------
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot
        xlabel : str
            Label of the x-axis
        ylabel : str
            Label of the y-axis
        title : str
            Title of the plot
        """
        self.plot_results(figsize, xlabel, ylabel, title)

        if self.pearson is None:
            raise ValueError("You must run the compare_with method before plotting the comparison")

        if hasattr(self.exp_comparison, 'yerror'):
            plt.errorbar(
                self.exp_comparison.variable,
                self.exp_comparison.results,
                self.exp_comparison.yerror,
                label="Compared Experiment",
                alpha=0.7,
                fmt='o'
            )
        
        else:
            plt.scatter(
                self.exp_comparison.variable,
                self.exp_comparison.results,
                label="Compared Experiment",
                alpha=0.7,
                s=15,
            )
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    ######################################################## FFT Methods ########################################################

    def run_FFT(
        self
    ) -> None:
        """
        Run the real fast Fourier transform for the results and variable attributes of the PulsedSimulation object.
        The results are centered around the mean value before the FFT is calculated in order to remove the DC component.

        Returns
        -------
        FFT_values : tuple
            Tuple with the frequency values and the FFT values
        """
        if isinstance(self.experiment.results, np.ndarray):
            y = np.abs(np.fft.rfft(self.experiment.results - np.mean(self.experiment.results)))

        elif isinstance(self.experiment.results, list) and all(
            isinstance(res, np.ndarray) for res in self.experiment.results
        ):
            y = [np.abs(np.fft.rfft(res - np.mean(res))) for res in self.experiment.results]

        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        freqs = np.fft.rfftfreq(
            len(self.experiment.variable), self.experiment.variable[1] - self.experiment.variable[0]
        )

        self.FFT_values = (freqs, y)

        return self.FFT_values

    def get_peaks_FFT(
        self,
        **find_peaks_kwargs : Any
    ) -> np.ndarray:
        """
        Find the peaks of the FFT values calculated by the run_FFT method.

        Parameters
        ----------
        find_peaks_kwargs : dict
            Dictionary with the arguments to be passed to the scipy.signal.find_peaks function

        Returns
        -------
        FFT_peaks : array
            Array with the peaks of the FFT values
        """
        if len(self.FFT_values) == 0:
            raise ValueError("No FFT values to analyze, you must run the FFT first")

        if isinstance(self.experiment.results, np.ndarray):
            self.FFT_peaks_index = find_peaks(self.FFT_values[1], **find_peaks_kwargs)
            self.FFT_peaks = self.FFT_values[0][self.FFT_peaks_index[0]]

        elif isinstance(self.experiment.results, list):
            self.FFT_peaks_index = [find_peaks(FFT, **find_peaks_kwargs) for FFT in self.FFT_values[1]]
            self.FFT_peaks = [self.FFT_values[0][index[0]] for index in self.FFT_peaks_index]
        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        return self.FFT_peaks

    def plot_FFT(
        self,
        freq_lim : Optional[tuple[float, float]] = None,
        figsize : tuple[int, int] = (6, 4),
        xlabel : Optional[str] = None,
        ylabel : str = "FFT Intensity",
        title : str = "FFT of the Results"
    ) -> None:
        """
        Plots the FFT

        Parameters
        ----------
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot
        xlabel : str
            Label of the x-axis
        ylabel : str
            Label of the y-axis
        title : str
            Title of the plot
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
                ax.scatter(
                    self.FFT_peaks,
                    self.FFT_values[1][self.FFT_peaks_index[0]],
                    color="red",
                    label="Peaks",
                    s=50,
                )

        elif isinstance(self.FFT_values[1], list):
            for idx_fft, val_fft in enumerate(self.FFT_values[1]):
                ax.plot(self.FFT_values[0], val_fft, label=f"FFT {idx_fft}")

                # if the FFT_peaks attribute is not empty, then plot them with the FFT
                if len(self.FFT_peaks) != 0:
                    ax.scatter(
                        self.FFT_peaks[idx_fft],
                        val_fft[self.FFT_peaks_index[idx_fft][0]],
                        color="red",
                        label=f"Peaks {idx_fft}",
                        s=50,
                    )

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

    def run_fit(
        self,
        fit_model : Model,
        results_index : int= 0,
        guess : Optional[dict] = None
    ) -> dict:
        """
        Run the fit method from lmfit to fit the results of the experiment with a given model,
        guess for the initial parameters.

        Parameters
        ----------
        fit_model : lmfit.Model
            Model to be used to fit the results
        results_index : int
            Index of the results to be fitted if the results attribute is a list
        guess : dict
            Initial guess for the parameters of the model
            Takes a dictionary consisting of parameter names as the keys and their initial guess as the value.
            See the definitions of the models in the fit_functions.py file for details.

        Examples
        --------
        The typical usage scheme is to pass a model and optionally, a dictionary of initial values. The file
        fit_functions.py includes a bunch of predefined models and imports some commonly needed ones from lmfit
        which can be directly passed to the function and used. Alternatively, the Model function from lmfit takes
        a custom Python function and instantiates a model class object with it, which can then be used here.

        >>> from lmfit import Model #needed for custom models
        >>> # my_analysis_obj is an instance of the analysis class
        >>> my_analysis_obj.run_fit(fit_model=RabiModel())
        >>> my_analysis_obj.run_fit(
                fit_model=Model(fit_two_lorentz_sym),
                guess = {'A': 0.5, 'gamma': 0.2, 'f_mean':1749, 'f_delta':3,'C':1}
                )

        In the snippet above, we showcase some ways to perform fits using this method. Using predefined
        models takes care of most of the stuff. One can still pass a guess dictionary to provide better
        initial values. The second example shows the usage with a custom model instantiated from a function
        along with user provided guesses. It's important to note that the first parameter of these custom
        functions should be `x`. Moreover, the keys in the dictionary should correspond to the other
        parameters of the function (this holds true for the predefined models as well). The names of the
        parameters can be referred to from the source file (fit_functions.py).

        Returns
        -------
        fit_params : dict
            best fit parameter values of parameters as a dict with the parameter names as the keys

        """
        if not isinstance(fit_model, Model):
            raise TypeError(
                "fit_model must be an instance of lmfit.Model. Remember to instantiate the class by adding parentheses."
            )

        # if there is only one result, just fit the results with the model
        if isinstance(self.experiment.results, np.ndarray):
            self.fit_model = fit_model
            if guess:
                self.fit_params = fit_model.fit(self.experiment.results, x=self.experiment.variable, **guess)
            else:
                try:
                    params = fit_model.guess(self.experiment.results, x=self.experiment.variable)
                    self.fit_params = fit_model.fit(
                        self.experiment.results, x=self.experiment.variable, params=params
                    )
                except NotImplementedError:
                    params = fit_model.make_params()
                    self.fit_params = fit_model.fit(
                        self.experiment.results, x=self.experiment.variable, params=params
                    )

            return self.fit_params.best_values

        # if there are multiple results, check if the results_index is an integer and if it is less than the number of results then fit
        elif isinstance(self.experiment.results, list):
            if (
                not isinstance(results_index, int)
                or results_index < 0
                or results_index >= len(self.experiment.results)
            ):
                raise ValueError(
                    "results_index must be a non-negative integer less than the number of results"
                )

            self.fit_model[results_index] = fit_model
            if guess:
                self.fit_params[results_index] = fit_model.fit(
                    self.experiment.results[results_index], x=self.experiment.variable, **guess
                )
            else:
                try:
                    params = fit_model.guess(
                        self.experiment.results[results_index], x=self.experiment.variable
                    )
                    self.fit_params[results_index] = fit_model.fit(
                        self.experiment.results[results_index], x=self.experiment.variable, params=params
                    )
                except NotImplementedError:
                    params = fit_model.make_params()
                    self.fit_params[results_index] = fit_model.fit(
                        self.experiment.results[results_index], x=self.experiment.variable, params=params
                    )
            return self.fit_params[results_index].best_values

    def plot_fit(
        self,
        figsize : tuple[int, int] = (6, 4),
        xlabel : Optional[str] = None,
        ylabel : str = "Expectation Value",
        title : str = "Pulsed Result"
    ) -> None:
        """
        Plot the results of the experiment with the fitted function.

        Parameters
        ----------
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot
        xlabel : str
            Label of the x-axis
        ylabel : str
            Label of the y-axis
        title : str
            Title of the plot
        """
        self.plot_results(figsize, xlabel, ylabel, title)

        if isinstance(self.experiment.results, np.ndarray):
            plt.plot(self.experiment.variable, self.fit_params.best_fit, label="Fit")

        elif isinstance(self.experiment.results, list):
            for idx_res, _ in enumerate(self.experiment.results):
                if self.fit_model[idx_res] is not None:
                    plt.plot(self.experiment.variable, self.fit_params[idx_res].best_fit, label=f"Fit {idx_res}")

        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

    ######################################################## Other Plotting Methods ########################################################

    def plot_results(
        self,
        figsize : tuple[int, int] = (6, 4),
        xlabel : Optional[str] = None,
        ylabel : str = "Observable",
        title  : str = "Results"
    ) -> None:
        """
        Plots the results of the experiment

        Parameters
        ----------
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot
        xlabel : str
            Label of the x-axis
        ylabel : str
            Label of the y-axis
        title : str
            Title of the plot
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
            for idx_obs, _ in enumerate(self.experiment.system.observable):
                # plot all observables in the results
                ax.plot(
                    self.experiment.variable,
                    self.experiment.results[idx_obs],
                    label=f"Observable {idx_obs}",
                    lw=2,
                    alpha=0.7,
                )

        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        # set the x-axis limits to the variable of the experiment
        ax.set_xlim(self.experiment.variable[0], self.experiment.variable[-1])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        ax.set_title(title)

    def plot_bloch(
        self,
        figsize : tuple[int, int] = (6, 4)
    ) -> None:
        """
        Plot the results of the experiment in a Bloch sphere if the quantum system has dimension of two.

        Parameters
        ----------
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot
        """
        if not isinstance(self.experiment, PulsedSim):
            raise ValueError("experiment must be a PulsedSim object")

        if len(self.experiment.rho) == 1:
            raise ValueError("Density matrices were not calculated, please run experiment first.")
        elif isinstance(self.experiment.rho, list) and all(rho.shape[0] == 2 for rho in self.experiment.rho):
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

#####################################

def plot_histogram(
    rho : Qobj,
    rho_comparison : Optional[Qobj] = None,
    component : Literal["real", "imag", "abs"]  = "real",
    figsize : tuple[int, int] = (5, 5),
    title : str = "Matrix Histogram"
    ) -> None:
    """
    Plot a 3D histogram of the final density matrix of the simulation.

    Parameters
    ----------
    rho : Qobj
        A Qobj representing a density matrix to be plotted
    rho_comparison : Qobj, optional
        A Qobj representing a density matrix to be compared with
    component : str
        Component of the density matrix to be plotted. Can be 'real', 'imag', or 'abs'.
    figsize : tuple
        Size of the figure to be passed to matplotlib.pyplot
    title : str
        Title of the plot
    """
    # Check all parameters
    if component not in {"real", "imag", "abs"}:
        raise ValueError("component must be 'real', 'imag', or 'abs'")

    if not (isinstance(figsize, tuple) or len(figsize) == 2):
        raise ValueError("figsize must be a tuple of two positive floats")

    if isinstance(rho, Qobj) and rho.shape[0] == rho.shape[1]:
        N = rho.shape[0]
        rho = rho.full()
    else:
        raise ValueError("rho must be a Qobj representing a square density matrix")

    if rho_comparison is None:
        pass
    elif isinstance(rho_comparison, Qobj) and rho_comparison.shape[0] == rho_comparison.shape[1] == N:
        rho_comparison = rho_comparison.full()
    else:
        raise ValueError(
            "rho_comparison must be a Qobj with the same shape as the density matrix to be compared with"
        )

    # Create 3D plot
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(azim=-50, elev=30)

    # Dimensions and positions for the bars
    xpos, ypos = np.meshgrid(np.arange(N), np.arange(N))
    xpos, ypos = xpos.flatten(), ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.5 * np.ones_like(zpos)

    # Prepare the data for the bars depending on the component
    if component == "real":
        dz_sim = np.real(rho).flatten()
        color = "C0"
        if rho_comparison is not None:
            dz_comp = np.real(rho_comparison).flatten()

    elif component == "imag":
        dz_sim = np.imag(rho).flatten()
        color = "C1"
        if rho_comparison is not None:
            dz_comp = np.imag(rho_comparison).flatten()

    elif component == "abs":
        dz_sim = np.abs(rho).flatten()
        color = "C2"
        if rho_comparison is not None:
            dz_comp = np.abs(rho_comparison).flatten()

    # Plot the bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz_sim, color=color, linewidth=0, alpha=0.3, zorder=0)

    if rho_comparison is not None:
        print(f"Fidelity: {fidelity(Qobj(rho), Qobj(rho_comparison))}")
        ax.bar3d(
            xpos,
            ypos,
            zpos,
            dx,
            dy,
            dz_comp,
            edgecolor="k",
            linewidth=0.5,
            alpha=0,
            zorder=1,
        )

    # Aesthetics, labels, and title
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    fig.suptitle(title, fontsize=12, x=0.2, y=0.85)
    ax.set_xticks(np.arange(N) + 0.5)
    ax.set_yticks(np.arange(N) + 0.5)
    ax.set_xticklabels([f"$|{i}\\rangle$" for i in range(N)])
    ax.set_yticklabels([f"$|{i}\\rangle$" for i in range(N)])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1.0])
