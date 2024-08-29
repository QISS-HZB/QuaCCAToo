import numpy as np

class ExpData:
    """
    
    """
    def __init__(self, file_path, variable_column=0, results_columns=1, error_column=None, variable_name='Time (s)', result_name='Fluorescence (counts)', **loadtxt_args):
        """
        

        Parameters
        ----------
        

        Returns
        -------

        """
        if not isinstance(file_path, str):
            raise ValueError("file_path must be a string")
        
        if not isinstance(variable_column, int):
            raise ValueError("variable_column must be an integer")
        
        # the results columns needs to be an integer or a list of integers
        if not isinstance(results_columns, int) or not (isinstance(results_columns, list) and all(isinstance(col, int) for col in results_columns)):
            raise ValueError("results_columns must be an integer or a list of two integers")
        
        # check if the error column is None, or a list of integers of the same length as results_columns
        if error_column is not None:
            if isinstance(results_columns, int) and not isinstance(error_column, int):
                raise ValueError("error_column must be an integer")
            elif isinstance(results_columns, list) and not (isinstance(error_column, list) and len(error_column) == len(results_columns)):
                raise ValueError("error_column must be an integer or a list of integers of the same length as results_columns")
        
        if not isinstance(variable_name, str) or not isinstance(result_name, str):
            raise ValueError("variable_name and result_name must be strings")

        if not isinstance(loadtxt_args, dict):
            raise ValueError("loadtxt_args must be a dictionary for the np.loadtxt function")
        
        # loads experimental data from a file with the specified arguments   
        exp_data = np.loadtxt(file_path, **loadtxt_args)

        # sets the results and variable attributes of the PulsedExperiment object
        self.variable = exp_data[:, variable_column]

        if isinstance(results_columns, int):
            self.results = exp_data[:, results_columns]
        else:
            self.results = [exp_data[:, column] for column in results_columns]

        if error_column is not None:
            self.error = exp_data[:, error_column]

    def subtract_results_columns(self, pos_col, neg_col):
        """
        Overwrites the results attribute substracting the results of the negative column from the positive column. If the error attribute is not None, it also calculates the error of the subtracted results.

        Parameters
        ----------
        pos_col (int): index of the positive column
        neg_col (int): index of the negative column
        """
        if not isinstance(self.results[pos_col], np.ndarray) or not isinstance(self.results[neg_col], np.ndarray):
            raise ValueError(f"pos_col={pos_col} and neg_col={neg_col} where not found in the results.")

        self.results = self.results[pos_col] - self.results[neg_col]

        if self.error is not None:
            self.error = np.sqrt(self.error[pos_col]**2 + self.error[neg_col]**2)

    def background_corrections(self, background_value):
        """
        Overwrites the results attribute substracting the background value from the results.

        Parameters
        ----------
        background_value (int, float): value to be substracted from the results
        """

        if not isinstance(background_value , (int, float)):
            raise ValueError(f"background_value must be a number.")
        
        self.results = self.results - background_value
    