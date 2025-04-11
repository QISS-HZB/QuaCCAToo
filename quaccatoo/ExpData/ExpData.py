# TODO: implement the baseline_correction and save methods

"""
This module contains the ExpData class as part of the QuaCCAToo package.
"""

import numpy as np
import matplotlib.pyplot as plt

class ExpData:
    """
    Class to load experimental data and perform basic data processing.

    Attributes
    ----------
    variable : np.ndarray
        array containing the variable data
    results : np.ndarray or list of np.ndarray
        array or list of arrays containing the results data
    variable_name : str
        name of the variable
    result_name : str
        name of the results

    Methods
    -------
    subtract_results_columns:
        subtracts the results of the negative column from the positive column
    offset_correction:
        substracts a background value from the results
    rescale_correction:
        multiplies the results by a rescale value
    poly_baseline_correction:
        performs a polynomial baseline correction to the results
    plot_exp_data:
        plots the experimental data
    """

    def __init__(self, file_path, variable_column=0, results_columns=1, variable_name="Time", result_name="Expectation Value",
                 plot=False, figsize=(6, 4), figtitle='Experimental Data', **loadtxt_args):
        """
        Constructor of the ExpData class.
        It loads experimental data from a file and sets the variable and results attributes according with the specified column arguments.

        Parameters
        ----------
        file_path  : str
            path to the file containing the experimental data
        variable_column : int
            column index of the variable
        results_columns : int or list of int
            column index of the results
        variable_name : str
            name of the variable
        result_name : str
            name of the results
        plot : bool
            if True, plot the experimental data
        figsize : tuple
            size of the figure for the plot
        figtitle : str
            title of the figure for the plot       
        **loadtxt_args : dict
            additional arguments for the np.loadtxt function
        """
        if not isinstance(file_path, str):
            raise ValueError("file_path must be a string")

        if not isinstance(variable_column, int):
            raise ValueError("variable_column must be an integer")

        # the results columns needs to be an integer or a list of integers
        if not isinstance(results_columns, int) and not (isinstance(results_columns, list) and all(isinstance(col, int) for col in results_columns)):
            raise ValueError("results_columns must be an integer or a list of two integers")

        if not isinstance(variable_name, str) or not isinstance(result_name, str):
            raise ValueError("variable_name and result_name must be strings")

        if not isinstance(loadtxt_args, dict):
            raise ValueError("loadtxt_args must be a dictionary for the np.loadtxt function")

        # loads experimental data from a file with the specified arguments
        exp_data = np.loadtxt(file_path, **loadtxt_args)

        # sets the results and variable attributes of the ExpData object
        self.variable = exp_data[:, variable_column]

        if isinstance(results_columns, int):
            self.results = exp_data[:, results_columns]
        else:
            self.results = [exp_data[:, column] for column in results_columns]

        self.variable_name = variable_name
        self.result_name = result_name

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        # plots the experimental data
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def subtract_results_columns(self, pos_col=0, neg_col=1, plot=False, figsize=(6, 4), figtitle='Subtracted Expt. Data'):
        """
        Overwrites the results attribute substracting the results of the negative column from the positive column.

        Parameters
        ----------
        pos_col: int
            index of the positive column
        neg_col: int
            index of the negative column
        plot: bool
            if True, plot the experimental data
        figsize: tuple
            size of the figure for the plot
        figtitle: str
            title of the figure for the plot
        """
        if not isinstance(self.results[pos_col], np.ndarray) or not isinstance(self.results[neg_col], np.ndarray):
            raise ValueError(f"pos_col={pos_col} and neg_col={neg_col} where not found in the results.")
        
        self.results = self.results[pos_col] - self.results[neg_col]  

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def offset_correction(self, background_value, plot=False, figsize=(6, 4), figtitle='Expt. Data with Offset Correction'):
        """
        Overwrites the results attribute substracting the background value from the results.

        Parameters
        ----------
        background_value : int or float
            value to be substracted from the results
        plot : bool
            if True, plot the experimental data
        figsize : tuple
            size of the figure for the plot
        figtitle : str
            title of the figure for the plot
        """
        if not isinstance(background_value, (int, float)):
            raise ValueError("background_value must be a number.")

        if isinstance(self.results, np.ndarray):
            self.results = self.results - background_value
        elif isinstance(self.results, list) and all(isinstance(result, np.ndarray) for result in self.results):
            self.results = [result - background_value for result in self.results]
        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def rescale_correction(self, rescale_value, plot=False, figsize=(6, 4), figtitle='Expt. Data with Rescale Correction'):
        """
        Overwrites the results attribute multiplying the results by the rescale value.

        Parameters
        ----------
        rescale_value : int or float
            value to be multiplied by the results
        plot : bool
            if True, plot the experimental data
        figsize : tuple
            size of the figure for the plot
        figtitle : str
            title of the figure for the plot
        """
        if not isinstance(rescale_value, (int, float)):
            raise ValueError("rescale_value must be a number.")

        if isinstance(self.results, np.ndarray):
            self.results = self.results * rescale_value
        elif isinstance(self.results, list) and all(isinstance(result, np.ndarray) for result in self.results):
            self.results = [result * rescale_value for result in self.results]
        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def poly_base_correction(self, x_start=None, x_end=None, poly_order=2, plot=False, figsize=(6, 4), figtitle='Expt. Data with Polynomial Baseline Correction'):
        """
        Overwrites the results attribute performing a polynomial baseline correction.
        The baseline is fitted to the data between x_start and x_end, representing the start and end of the xaxis index.

        Parameters
        ----------
        x_start : int or list of int
            start index of the x axis for the baseline fit
        x_end : int or list of int
            end index of the x axis for the baseline fit
        poly_order : int
            order of the polynomial to fit the baseline
        plot : bool
            if True, plot the experimental data
        figsize : tuple
            size of the figure for the plot
        figtitle : str
            title of the figure for the plot
                
        """
        # check all variables
        if x_start is None:
            x_start = 0
        elif not isinstance(x_start, int) and not (isinstance(x_start, list) and all(isinstance(x, int) for x in x_start)):
            raise ValueError("x_start must be a integer index or a list of integer indexes.")
        
        if x_end is None:
            x_end = -1
        elif not isinstance(x_end, int) and not (isinstance(x_end, list) and all(isinstance(x, int) for x in x_end)):
            raise ValueError("x_end must be a integer index or a list of integer indexes.")
        
        if not isinstance(poly_order, int):
            raise ValueError("poly_order must be an integer.")

        # crops the x and y axis for performing the baseline fit  
        if isinstance(x_start, int) and isinstance(x_end, int):
            baseline_xaxis = self.variable[x_start:x_end]
            baseline_yaxis = self.results[x_start:x_end]
        elif isinstance(x_start, list) and isinstance(x_end, list) and len(x_start) == len(x_end):
            baseline_xaxis = np.concatenate([self.variable[x_start[i]:x_end[i]] for i in range(len(x_start))])
            baseline_yaxis = np.concatenate([self.results[x_start[i]:x_end[i]] for i in range(len(x_start))])
        else:
            raise ValueError("x_start and x_end must int or a list of the same length.")
        
        print(f"Baseline x-axis: {baseline_xaxis}")
        print(f"Baseline y-axis: {baseline_yaxis}")

        if isinstance(self.results, np.ndarray):
            poly_fit = np.polyfit(baseline_xaxis, baseline_yaxis, poly_order)
            self.results = self.results - np.polyval(poly_fit, self.variable)

        elif isinstance(self.results, list) and all(isinstance(result, np.ndarray) for result in self.results):
            poly_fit = [np.polyfit(baseline_xaxis[i], baseline_yaxis[i], poly_order) for i in range(len(baseline_xaxis))]
            self.results = [self.results[i] - np.polyval(poly_fit[i], self.variable) for i in range(len(self.results))]

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def plot_exp_data(self, figsize=(6, 4), figtitle='Experimental Data'):
        """
        Plots the experimental data.

        Parameters
        ----------
        figsize : tuple
            size of the figure for the plot
        figtitle : str
            title of the figure for the plot        
        """
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")
        
        if not isinstance(figtitle, str):
            raise ValueError("figtitle must be a string")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # check if the results is a list of results or a single result
        if isinstance(self.results, np.ndarray):
            ax.scatter(self.variable, self.results, lw=2, alpha=0.7, label="Observable", s= 15)

        elif isinstance(self.results, list) and all(isinstance(result, np.ndarray) for result in self.results):
            for itr in range(len(self.results)):
                ax.scatter(self.variable, self.results[itr], label=f"Observable {itr}", alpha=0.7, s= 15)

        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        # set the x-axis limits to the variable of the experiment
        ax.set_xlim(self.variable[0], self.variable[-1])

        ax.set_xlabel(self.variable_name)
        ax.set_ylabel(self.result_name)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        ax.set_title(figtitle)


    def save():
        pass