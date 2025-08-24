"""
This module contains the ExpData class as part of the QuaCCAToo package.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Any

class ExpData:
    """
    Class to load experimental data and perform basic data processing.

    Attributes
    ----------
    variable : np.ndarray
        Array containing the variable data
    results : np.ndarray or list of np.ndarray
        Array or list of arrays containing the results data
    yerror: None, np.ndarray or list of np.ndarray
        Array or list of arrays containing the uncertainty of the results
    variable_name : str
        Name of the variable
    result_name : str
        Name of the results

    Methods
    -------
    subtract_results_columns:
        Subtracts the results of the negative column from the positive column
    offset_correction:
        Substracts a background value from the results
    rescale_correction:
        Multiplies the results by a rescale value
    poly_baseline_correction:
        Performs a polynomial baseline correction to the results
    plot_exp_data:
        Plots the experimental data
    """

    def __init__(
        self,
        file_path : str,
        variable_column : int = 0,
        results_columns : int = 1,
        yerr_columns : int | None = None,
        variable_name : str = "Time",
        result_name : str = "Expectation Value",
        plot : bool = False,
        figsize : tuple[int, int] = (6, 4),
        figtitle : str = "Experimental Data",
        **loadtxt_kwargs : Any,
    ) -> None:
        """
        Constructor of the ExpData class.
        It loads experimental data from a file and sets the variable and results attributes according with the specified column arguments.

        Parameters
        ----------
        file_path  : str
            Path to the file containing the experimental data
        variable_column : int
            Column index of the variable
        results_columns : int or list of int
            Column index of the results
        yerr_columns : None, int or list of int
            Column index of the result uncertainty values
        variable_name : str
            Name of the variable
        result_name : str
            Name of the results
        plot : bool
            Boolean indicating whether to plot the experimental data or not
        figsize : tuple
            Size of the figure for the plot
        figtitle : str
            Title of the figure for the plot
        **loadtxt_kwargs : dict
            Additional arguments for the np.loadtxt function
        """
        if not isinstance(file_path, str):
            raise ValueError("file_path must be a string")

        if not isinstance(variable_column, int):
            raise ValueError("variable_column must be an integer")

        # the results columns needs to be an integer or a list of integers
        if not isinstance(results_columns, int) and not (
            isinstance(results_columns, list) and all(isinstance(col, int) for col in results_columns)
        ):
            raise ValueError("results_columns must be an integer or a list of integers")
        
        # the error columns needs to be None, an integer or a list of integers
        if yerr_columns is not None and not isinstance(yerr_columns, int) and not (
            isinstance(yerr_columns, list) and all(isinstance(col, int) for col in yerr_columns)
        ):
            raise ValueError("yerr_columns must be None, an integer or a list of integers")
        elif isinstance(yerr_columns, list) and len(yerr_columns) != len(results_columns):
            raise ValueError("yerr_columns must have the same lenght of the results_columns")

        if not isinstance(variable_name, str) or not isinstance(result_name, str):
            raise ValueError("variable_name and result_name must be strings")

        if not isinstance(loadtxt_kwargs, dict):
            raise ValueError("loadtxt_kwargs must be a dictionary for the np.loadtxt function")

        # loads experimental data from a file with the specified arguments
        exp_data = np.loadtxt(file_path, **loadtxt_kwargs)

        # sets the results and variable attributes of the ExpData object
        self.variable = exp_data[:, variable_column]

        if isinstance(results_columns, int):
            self.results = exp_data[:, results_columns]
        else:
            self.results = [exp_data[:, column] for column in results_columns]

        if isinstance(yerr_columns, int):
            self.yerror = exp_data[:, yerr_columns]
        elif isinstance(yerr_columns, list):
            self.yerror = [exp_data[:, column] for column in yerr_columns]

        self.variable_name = variable_name
        self.result_name = result_name

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        # plots the experimental data
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def subtract_results_columns(
        self,
        pos_col: int = 0,
        neg_col: int = 1,
        plot: bool = False,
        figsize: tuple[int, int] = (6, 4),
        figtitle: str = "Subtracted Expt. Data",
    ) -> None:
        """
        Overwrites the results attribute substracting the results of the negative column from the positive column.

        Parameters
        ----------
        pos_col: int
            Index of the positive column
        neg_col: int
            Index of the negative column
        plot: bool
            Boolean indicating whether to plot the experimental data or not
        figsize: tuple
            Size of the figure for the plot
        figtitle: str
            Title of the figure for the plot
        """
        if not isinstance(self.results[pos_col], np.ndarray) or not isinstance(
            self.results[neg_col], np.ndarray
        ):
            raise ValueError(f"pos_col={pos_col} and neg_col={neg_col} where not found in the results.")

        self.results = self.results[pos_col] - self.results[neg_col]

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def offset_correction(
        self,
        background_value: int | float,
        plot: bool = False,
        figsize: tuple[int, int] = (6, 4),
        figtitle: str = "Expt. Data with Offset Correction",
    ) -> None:
        """
        Overwrites the results attribute substracting the background value from the results.

        Parameters
        ----------
        background_value : int or float
            Value to be substracted from the results
        plot : bool
            Boolean indicating whether to plot the experimental data or not
        figsize : tuple
            Size of the figure for the plot
        figtitle : str
            Title of the figure for the plot
        """
        if not isinstance(background_value, (int, float)):
            raise ValueError("background_value must be a number.")

        if isinstance(self.results, np.ndarray):
            self.results = self.results - background_value
        elif isinstance(self.results, list) and all(
            isinstance(result, np.ndarray) for result in self.results
        ):
            self.results = [result - background_value for result in self.results]
        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def rescale_correction(
        self,
        rescale_value: int | float,
        plot: bool = False,
        figsize: tuple[int, int] = (6, 4),
        figtitle: str = "Expt. Data with Rescale Correction",
    ) -> None:
        """
        Overwrites the results attribute multiplying the results by the rescale value.

        Parameters
        ----------
        rescale_value : int or float
            Value to be multiplied by the results
        plot : bool
            Boolean indicating whether to plot the experimental data or not
        figsize : tuple
            Size of the figure for the plot
        figtitle : str
            Title of the figure for the plot
        """
        if not isinstance(rescale_value, (int, float)):
            raise ValueError("rescale_value must be a number.")

        if isinstance(self.results, np.ndarray):
            self.results = self.results * rescale_value
        elif isinstance(self.results, list) and all(
            isinstance(result, np.ndarray) for result in self.results
        ):
            self.results = [result * rescale_value for result in self.results]
        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def poly_base_correction(
        self,
        x_start: int | list[int] | None = None,
        x_end: int | list[int] | None = None,
        poly_order: int = 2,
        plot: bool = False,
        figsize: tuple[int, int] = (6, 4),
        figtitle: str = "Expt. Data with Polynomial Baseline Correction",
    ) -> None:
        """
        Overwrites the results attribute performing a polynomial baseline correction.
        The baseline is fitted to the data between x_start and x_end, representing the start and end of the xaxis index.

        Parameters
        ----------
        x_start : int or list of int
            Start index of the x axis for the baseline fit
        x_end : int or list of int
            End index of the x axis for the baseline fit
        poly_order : int
            Order of the polynomial to fit the baseline
        plot : bool
            Boolean indicating whether to plot the experimental data or not
        figsize : tuple
            Size of the figure for the plot
        figtitle : str
            Title of the figure for the plot

        """
        # check all variables
        if x_start is None:
            x_start = 0
        elif not isinstance(x_start, int) and not (
            isinstance(x_start, list) and all(isinstance(x, int) for x in x_start)
        ):
            raise ValueError("x_start must be a integer index or a list of integer indexes.")

        if x_end is None:
            x_end = -1
        elif not isinstance(x_end, int) and not (
            isinstance(x_end, list) and all(isinstance(x, int) for x in x_end)
        ):
            raise ValueError("x_end must be a integer index or a list of integer indexes.")

        if not isinstance(poly_order, int):
            raise ValueError("poly_order must be an integer.")

        # crops the x and y axis for performing the baseline fit
        if isinstance(x_start, int) and isinstance(x_end, int):
            baseline_xaxis = self.variable[x_start:x_end]
            baseline_yaxis = self.results[x_start:x_end]
        elif isinstance(x_start, list) and isinstance(x_end, list) and len(x_start) == len(x_end):
            baseline_xaxis = np.concatenate(
                [self.variable[start : end] for start, end in zip(x_start, x_end)]
            )
            baseline_yaxis = np.concatenate(
                [self.results[start : end] for start, end in zip(x_start, x_end)]
            )
        else:
            raise ValueError("x_start and x_end must int or a list of the same length.")

        if isinstance(self.results, np.ndarray):
            poly_fit = np.polyfit(baseline_xaxis, baseline_yaxis, poly_order)
            self.results -= np.polyval(poly_fit, self.variable)

        elif isinstance(self.results, list) and all(
            isinstance(result, np.ndarray) for result in self.results
        ):
            poly_fit = [
                np.polyfit(baseline_xaxis[idx_base], val_base, poly_order)
                for idx_base, val_base in enumerate(baseline_xaxis)
            ]
            self.results = [
                val_res - np.polyval(poly_fit[idx_res], self.variable) for idx_res, val_res in enumerate(self.results)
            ]

        if not isinstance(plot, bool):
            raise ValueError("plot must be a boolean")
        elif plot:
            self.plot_exp_data(figsize=figsize, figtitle=figtitle)

    def plot_exp_data(
        self,
        figsize: tuple[int, int] = (6, 4),
        figtitle: str = "Experimental Data",
    ) -> None:
        """
        Plots the experimental data.

        Parameters
        ----------
        figsize : tuple
            Size of the figure for the plot
        figtitle : str
            Title of the figure for the plot
        """
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        if not isinstance(figtitle, str):
            raise ValueError("figtitle must be a string")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # check if the results is a list of results or a single result
        if isinstance(self.results, np.ndarray):
            if hasattr(self, 'yerror'):
                ax.errorbar(self.variable, self.results, self.yerror, alpha=0.7, label="Observable", fmt='o')
            else:
                ax.scatter(self.variable, self.results, alpha=0.7, label="Observable", s=15)

        elif isinstance(self.results, list) and all(
            isinstance(result, np.ndarray) for result in self.results
        ):
            for idx_res, val_res in enumerate(self.results):
                if hasattr(self, 'yerror'):
                    ax.errorbar(self.variable, val_res, self.yerror[idx_res], alpha=0.7, label="Observable", fmt='o')
                else:
                    ax.scatter(self.variable, val_res, label=f"Observable {idx_res}", alpha=0.7, s=15)

        else:
            raise ValueError("Results must be a numpy array or a list of numpy arrays")

        # set the x-axis limits to the variable of the experiment
        ax.set_xlim(self.variable[0], self.variable[-1])

        ax.set_xlabel(self.variable_name)
        ax.set_ylabel(self.result_name)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        ax.set_title(figtitle)
