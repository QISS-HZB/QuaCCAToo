# TODO: implement the baseline_correction and save methods

"""
This module contains the ExpData class as part of the QuaCCAToo package.
"""

import numpy as np


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
    -------------
    subtract_results_columns(pos_col=0, neg_col=1)
        subtracts the results of the negative column from the positive column
    background_correction(background_value)
        substracts a background value from the results
    rescale_correction(rescale_value)
        multiplies the results by a rescale value
    baseline_correction
    save
    """

    def __init__(self, file_path, variable_column=0, results_columns=1, variable_name="Time", result_name="Expectation value", **loadtxt_args):
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

    def subtract_results_columns(self, pos_col=0, neg_col=1):
        """
        Overwrites the results attribute substracting the results of the negative column from the positive column.

        Parameters
        ----------
        pos_col: int
            index of the positive column
        neg_col: int
            index of the negative column
        """
        if not isinstance(self.results[pos_col], np.ndarray) or not isinstance(self.results[neg_col], np.ndarray):
            raise ValueError(f"pos_col={pos_col} and neg_col={neg_col} where not found in the results.")

        self.results = self.results[pos_col] - self.results[neg_col]

    def background_correction(self, background_value):
        """
        Overwrites the results attribute substracting the background value from the results.

        Parameters
        ----------
        background_value : int or float
            value to be substracted from the results
        """
        if not isinstance(background_value, (int, float)):
            raise ValueError("background_value must be a number.")

        if isinstance(self.results, np.ndarray):
            self.results = self.results - background_value
        else:
            self.results = [result - background_value for result in self.results]

    def rescale_correction(self, rescale_value):
        """
        Overwrites the results attribute multiplying the results by the rescale value.

        Parameters
        ----------
        rescale_value : int or float
            value to be multiplied by the results
        """
        if not isinstance(rescale_value, (int, float)):
            raise ValueError("rescale_value must be a number.")

        if isinstance(self.results, np.ndarray):
            self.results = self.results * rescale_value
        else:
            self.results = [result * rescale_value for result in self.results]

    def baseline_correction(self):
        """ """
        pass

    def save():
        pass
