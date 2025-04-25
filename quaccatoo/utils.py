import inspect
import shutil
import os
import zipfile
import dill
import quaccatoo

from qutip import Qobj, fileio

def save(object, file_name):
    """
    Look for all the attributes of the object, save Qobj attributes to separate files,
    and save the rest of the attributes to a pickle file. Finally, create a zip file
    containing all the files.

    Parameters
    ----------
    object : quaccatoo object
        The object to be saved.
    file_name : str
        Path to the file where the attributes will be saved.
    """
    if not isinstance(file_name, str):
        raise ValueError("file_name must be a string")
    
    # create a temporary directory to store files
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Get a list of all the attributes defined in the object
        attributes = [attr for attr in dir(object) if not attr.startswith('__') and not (inspect.isfunction(getattr(object, attr)) or inspect.ismethod(getattr(object, attr)))]
        py_attr = []

        # Separate attributes into Python and Qobj types
        for attr in attributes:
            value = getattr(object, attr)
            if isinstance(value, Qobj) or (isinstance(value, list) and all(isinstance(item, Qobj) for item in value)):
                # Save Qobj attributes to a file in the temporary directory
                fileio.qsave(value, os.path.join(tmp_dir, str(attr)))
            else:         
                py_attr.append(attr)
                
        # Create a dictionary to store the python attributes names and values
        py_data = {'__type__': object.__class__.__name__,
                **{attr: getattr(object, attr) for attr in py_attr}
        }
    
        # Save the python data to a file in the temporary directory
        with open(os.path.join(tmp_dir, 'py_data.pkl'), 'wb') as f_pkl:
            dill.dump(py_data, f_pkl)

        # Create a zip file to store all the files
        with zipfile.ZipFile(file_name, 'w') as zip_file:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, start=tmp_dir)
                    zip_file.write(file_path, rel_path)

    finally:
        # Remove the temporary directory
        shutil.rmtree(tmp_dir)

def load(file_name):
    """
    Loads the attributes of an object from a zip file,
    creates an instance of the object, and sets the attributes.

    Parameters
    ----------
    file_name : str
        Path to the zip file where the attributes are saved.

    Returns
    -------
    object
        The loaded object.
    """
    if not isinstance(file_name, str):
        raise ValueError("file_name must be a string")

    # Extract the zip file to a temporary directory
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(file_name, 'r') as zip_file:
            zip_file.extractall(tmp_dir)

        # Load the py_data.pkl file
        with open(os.path.join(tmp_dir, 'py_data.pkl'), 'rb') as f_pkl:
            py_data = dill.load(f_pkl)

        # Get the object name from the py_data
        cls = getattr(quaccatoo, py_data['__type__'])
        
        # Create an empty instance
        obj = cls.__new__(cls)

        # Load the attributes from the py_data
        for attr, value in py_data.items():
            if attr != '__type__':
                setattr(obj, attr, value)

        # Load the Qobj attributes from the files
        for file in os.listdir(tmp_dir):
            if file != 'py_data.pkl':
                attr_name = os.path.splitext(file)[0]
                value = fileio.qload(os.path.join(tmp_dir, attr_name))
                setattr(obj, attr_name, value)         

        return obj
    
    finally:
        # Remove the temporary directory
        shutil.rmtree(tmp_dir)