"""
This module provides functions to save and load quaccatoo objects, such as instances from QSys, ExpData, Analysis and PulsedSim.
"""

import os
import shutil
import zipfile

import dill
import numpy as np
from qutip import Qobj, fileio

__all__ = ["save_quaccatoo", "load_quaccatoo"]

####################################################################################################


def save_quaccatoo(obj_save, file_path):
    """
    Look for all the attributes of the obj, save Qobj attributes to separate files,
    and save the rest of the attributes to a pickle file. Finally, create a zip file
    containing all the files.

    Parameters
    ----------
    obj_save : quaccatoo obj
        The obj to be saved.
    file_path : str
        Path to the file where the attributes will be saved.
    """
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string")

    tmp_dir = "tmp"

    try:
        # create a temporary directory to store files and get a list of all the attributes defined in the obj
        os.makedirs(tmp_dir, exist_ok=True)
        attributes = list(obj_save.__dict__.keys())
        py_attr = []

        # Separate attributes into Python and Qobj types
        for attr in attributes:
            value = getattr(obj_save, attr)
            if isinstance(value, Qobj) or (
                isinstance(value, (list, np.ndarray))
                and len(value) > 0
                and all(isinstance(item, Qobj) for item in value)
            ):
                # Save Qobj attributes to a file in the temporary directory
                fileio.qsave(value, os.path.join(tmp_dir, str(attr)))
            else:
                py_attr.append(attr)

        # Create a dictionary to store the python attributes names and values
        py_data = {
            "__type__": obj_save.__class__.__name__,
            **{attr: getattr(obj_save, attr) for attr in py_attr},
        }

        # Save the python data to a file in the temporary directory
        with open(os.path.join(tmp_dir, "py_data.pkl"), "wb") as f_pkl:
            dill.dump(py_data, f_pkl)

        # Create a zip file to store all the files
        with zipfile.ZipFile(file_path, "w") as zip_file:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, start=tmp_dir)
                    zip_file.write(file_path, rel_path)

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def load_quaccatoo(file_path):
    """
    Loads the attributes of an obj from a zip file,
    creates an instance of the obj, and sets the attributes.

    Parameters
    ----------
    file_path : str
        Path to the zip file where the attributes are saved.

    Returns
    -------
    obj
        The loaded obj.
    """
    if not isinstance(file_path, str):
        raise ValueError("file_name must be a string")

    tmp_dir = "tmp"

    import quaccatoo  # pylint: disable=import-outside-toplevel,cyclic-import

    try:
        # Extract the zip file to a temporary directory
        os.makedirs(tmp_dir, exist_ok=True)
        with zipfile.ZipFile(file_path, "r") as zip_file:
            zip_file.extractall(tmp_dir)

        # Load the py_data.pkl file
        with open(os.path.join(tmp_dir, "py_data.pkl"), "rb") as f_pkl:
            py_data = dill.load(f_pkl)

        # Get the obj name from the py_data and create new instance
        cls = getattr(quaccatoo, py_data["__type__"])
        obj_load = cls.__new__(cls)

        # Load the attributes from the py_data
        for attr, value in py_data.items():
            if attr != "__type__":
                setattr(obj_load, attr, value)

        # Load the Qobj attributes from the files
        for file in os.listdir(tmp_dir):
            if file != "py_data.pkl":
                attr_name = os.path.splitext(file)[0]
                value = fileio.qload(os.path.join(tmp_dir, attr_name))
                setattr(obj_load, attr_name, value)

        return obj_load

    finally:
        if os.path.exists(tmp_dir):
            # Remove the temporary directory
            shutil.rmtree(tmp_dir)
