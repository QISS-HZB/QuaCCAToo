# Quantum Color Centers Analysis Toolbox
![Logo](./docs/QuaCCAToo_logo.svg)

## Installation
We strongly recommend using a virtual environment so that the system Python remains untouched.

You can create a `conda` virtual environment as follows:

```sh
conda create --name quaccatoo-env python
conda activate quaccatoo-env
conda install numpy matplotlib scipy qutip
```
**OR**

You can use a native Python venv (you'll need `pip` as well)

``` sh
python -m venv quaccatoo-env
source quaccatoo-env/bin/activate
```

After the virtual environment has been setup, clone the repository and run (from inside the repo) 

``` sh
pip install .
```

Check the notebooks in the `docs` subdirectory for examples.

## Class Hierarchy

![Class diagram](./docs/class_diagram.svg)


## Contribution guidelines
- To contribute, fork the main branch and make a pull request.
- Properly _document everything_ in details following the `numpy` [docstring format](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
- Test your branch by running notebooks Ex01 and Ex02. Before making the pull request, _clear the outputs_.
- Use _US-English_, not British-English. Eg: analyze instead of analyse, color instead of colour, center instead of centre.

## Documentation
The current documentation is available at https://qiss-hzb.github.io/QuaCCAToo/
