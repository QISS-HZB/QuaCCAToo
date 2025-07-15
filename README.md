# Quantum Color Centers Analysis Toolbox
![Logo](./docs/QuaCCAToo_logo.svg)

QuaCCAToo is a Python library for simulating and analyzing spin dynamics of color centers for quantum
technology applications, without using rotating wave approximations. The software serves as an extension for
QuTip, inheriting its object-oriented framework and the `Qobj` class. This way, the software combines
accessibility from the high level of abstraction and human-readability of Python with the efficiency of
compiled programming languages provided by Qutip's parallelization and the matrix algebra from Scipy
and Numpy. 

Documentation and usage tutorial available at https://qiss-hzb.github.io/QuaCCAToo/

## Installation

We strongly recommend using a virtual environment (use whichever tool like `venv`/`conda`/`uv` that you
prefer) so that the system Python remains untouched.

``` sh
pip install quaccatoo
```

Check [here](https://qiss-hzb.github.io/QuaCCAToo/installation.html) for detailed installation instructions.

## Class Hierarchy

The package is organized as follows:
- `QSys` defines the quantum system of the problem. It has an obligatory intrinsic internal Hamiltonian
  $H_0$, optional initial state, observable and a set of collapse operators. On `QSys`, one can calculate the
  eigenstates and eigenvalues of the system. QuaCCAToo provides `NV`(`NV_Sys`) as a predefined system for nitrogen
  vacancy centers in diamonds, more systems will be provided soon.
- `PulsedSim` contains the logic for performing the simulation of pulsed experiments upon a `QSys` object. It
  has attributes of a pulse sequence containing a set of pulses and free evolutions, control Hamiltonian
  $H_1$, experiment variable and simulation results. Many predefined common pulse sequences are given in
  `PredefSeqs` and `PredefDDSeqs`.
- `ExpData` is a class to load experimental data and perform basic data processing.
- `Analysis` can be used either on simulation or experimental results, with a series of methods like FFT,
  fits, data comparison and plotting.

![Class diagram](./docs/class_diagram.svg)

## Contribution guidelines

Any contribution or bug report are welcome.

- To contribute, fork the main branch and make a pull request.
- We use `hatch/hatchling` as the build backend. The other development dependencies include `pytest` and
  `ruff`. They can be installed by running `pip install -e '.[dev]'` from within the cloned repository. See
  [here](https://qiss-hzb.github.io/QuaCCAToo/installation.html) for details.
- Properly _document everything_ in details following the `numpy` [docstring
  format](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
- Test your branch by running `pytest` and the tutorial notebooks. Feel free to add more tests.
- Please pay attention to linter warnings (`ruff check`) and format your code with `ruff format`.
- Module level refactors require corresponding changes in the `sphinx` setup, too.
- Use US-English, not British-English. Eg: analyze instead of analyse, color instead of colour, center
  instead of centre.
