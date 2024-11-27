# Quantum Color Centers Analysis Toolbox
![Logo](./docs/QuaCCAToo_logo.svg)

QuaCCAToo is a Python software for simulating and analyzing spin dynamics of color centers for quantum technology applications, without using rotating wave approximations. The software serves as an extension for QuTip, inheriting its object-oriented framework and the Qobj class. This way, the software combines accessibility from the high level of abstraction and human-readability of Python with the efficiency of compiled programming languages as C++, provided by Qutip's parallelization and the matrix algebra from Scipy and Numpy. What unifies color centers and defines the framework of QuaCCAToo is the quantum mechanical description of the magnetic spins with density matrices, combined with a semi-classical description of the control and optical fields. Despite the name, the software then is applicable to a much broader set of physical systems as discussed, as long as the quantum states can be expressed in terms of a density matrix, the system has a well-defined internal Hamiltonian $H_0$ and a control Hamiltonian $H_1$ describing the interaction with external fields.

To see examples of utilization, check the notebooks in the `docs` subdirectory. They contain:
- [Ex01](https://github.com/QISS-HZB/QuaCCAToo/blob/main/docs/ExampleNotebooks/Ex01_spin1%E2%81%842_Rabi_Hahn.ipynb): simplest two-level system, where we first define the system and plot the energy levels. Following that, a Rabi oscillation is simulated for two different pulse vectors, with the results being fitted and plotted in the Bloch sphere. Lastly, we simulated a Hahn echo decay for a modeled collapse operator.
- [Ex02](https://github.com/QISS-HZB/QuaCCAToo/blob/main/docs/ExampleNotebooks/Ex02_NV_Ramsey_PODMR.ipynb): simulation of nitrogen vacancy centers in diamond, first calculating the energy levels, then performing Rabi and comparing with experimental data. Ramsey and PODMR are also simulated.
- [Ex03](https://github.com/QISS-HZB/QuaCCAToo/blob/main/docs/ExampleNotebooks/Ex03_NV_ambiguous_resonances.ipynb): simulation of [Ambiguous Resonances in Multipulse Quantum Sensing with Nitrogen Vacancy Centers](https://arxiv.org/abs/2407.09411).

To see the documentation, check [https://qiss-hzb.github.io/QuaCCAToo/](https://qiss-hzb.github.io/QuaCCAToo/).

## Installation

We strongly recommend using a virtual environment so that the system Python remains untouched.

You can create a `conda` virtual environment as follows:

```sh
conda create --name quaccatoo-env python qutip matplotlib
conda activate quaccatoo-env
```

**OR**

You can use a native Python venv (you'll need `pip` as well)

``` sh
python -m venv quaccatoo-env
source quaccatoo-env/bin/activate
```

After the virtual environment has been setup, clone the repository and run from  <u>inside the local cloned folder </u>

``` sh
pip install .
```

To check the installation, you can run

``` sh
pip show quaccatoo
```

Notebooks can be the most convenient way of running QuaCCAToo (but definitely  not the most efficient for long simulations). If you wish to use them, you must also install `jupyter` in you conda environment.

``` sh
pip install jupyter
```

### Note for Windows Users

Qutip and thus QuaCCAToo are best optimized to run on Linux operational systems. More specifically, the `parallel_map` method from Qutip, which QuaCCAToo heavily relies on, does not work properly on Windows. This leads to severe slow downs in simulations and even crashes in some hardware. The developers are currently working to solve this issue. macOS operational systems have not been tested by developers.

## Class Hierarchy

The package is organized as follows:
- `QSys` defines the quantum system of the problem. It has an obligatory intrinsic internal Hamiltonian $H_0$, optional initial state, observable and a set of collapse operators. On QSys, one can calculate the eigenstates and eigenvalues of the system. QuaCCAToo provides NVSys as a predefined system for nitrogen vacancy centers in diamonds, more systems will be provided soon.
- `PulsedSim` contains the logic for perfoming the simulation of pulsed experiments upon a QSys object. It has attributes of a pulse sequence containing a set of pulses and free evolutions, control Hamiltonian $H_1$, experiment variable and simulation results. Many predefined common pulse sequences are given in `PredefSeqs` and `PredefDDSeqs`.
- `ExpData` is a class to load experimental data and perform basic data processing.
- `Analysis` can be used either on simulation or experimental results, with a series of methods like FFT, fits, data comparison and plotting.

![Class diagram](./docs/class_diagram.svg)

## Contribution guidelines

Any contribution or bug report are welcome.

- To contribute, fork the main branch and make a pull request.
- Properly _document everything_ in details following the `numpy` [docstring format](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
- Test your branch by running notebooks Ex01 and Ex02.
- Use US-English, not British-English. Eg: analyze instead of analyse, color instead of colour, center instead of centre.