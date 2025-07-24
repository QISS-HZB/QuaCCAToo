# Quantum Color Centers Analysis Toolbox
![Logo](./docs/QuaCCAToo_logo.svg)

QuaCCAToo is a Python library for simulating and analyzing spin dynamics of color centers for quantum technology applications.
The systems' time evolution under pulsed experiments are calculated through quantum master equations based on the provided Hamiltonian, with realistic pulses in the laboratory frame. 
The software is built on top of QuTip, inheriting its object-oriented framework and the `Qobj` class.

For learning more about the package, we recommend first checking the [tutorials](https://qiss-hzb.github.io/QuaCCAToo/notebooks.html) section.

If you used QuaCCAToo in your work, please cite this repository and [10.1103/PhysRevA.111.022606](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.022606).

## Links
- Repository: https://github.com/QISS-HZB/QuaCCAToo
- Documentation: https://qiss-hzb.github.io/QuaCCAToo/
- PyPI: https://pypi.org/project/QuaCCAToo/

## Installation

We strongly recommend using a virtual environment (use whichever tool like `venv`/`conda`/`uv` that you prefer) so that the system Python remains untouched.

``` sh
pip install quaccatoo
```

Check [here](https://qiss-hzb.github.io/QuaCCAToo/installation.html) for detailed installation instructions.

## Featured In

- L. Tsunaki, A. Singh, K. Volkova, S. Trofimov, T. Pregnolato, T. Schröder, & B. Naydenov. (2025). Ambiguous resonances in multipulse quantum sensing with nitrogen-vacancy centers. Physical Review A, 111(2), 022606. doi: [10.1103/PhysRevA.111.022606](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.022606)
- S. Trofimov, C. Thessalonikios, V. Deinhart, A. Spyrantis, L. Tsunaki, K. Volkova, K. Höflich, & B. Naydenov. (2025). Local nanoscale probing of electron spins using NV centers in diamond. arXiv:2507.13295 quant-ph. [2507.13295](https://arxiv.org/abs/2507.13295).

## Class Hierarchy

QuaCCAToo is an object-oriented package organized with the following classes:
- `QSys` defines the quantum system of the problem. It has an obligatory intrinsic internal Hamiltonian $H_0$, optional initial state, observable and a set of collapse operators.
On `QSys`, calculates the eigenstates and eigenvalues of the system and has methods for truncating the systems and adding other spins.
QuaCCAToo provides `NV` as a predefined system for nitrogen vacancy centers in diamonds, more systems will be provided soon.
- `PulsedSim` contains the logic for performing the simulation of pulsed experiments upon a `QSys` object.
It has attributes of a pulse sequence containing a set of pulses and free evolutions, control Hamiltonian $H_1$, experiment variable and simulation results. Many predefined common pulse sequences are given in `predef_seqs` and `predef_dd_seqs` modules.
Different pulse shapes are predefined in the `pulse_shapes` module.
- `ExpData` is a class to load experimental data and perform basic data processing, such as rescaling, subtracting columns or performing polynomial baseline corrections.
- `Analysis` can be used either on simulation or experimental results, with a series of methods like for fitting (based on `lmfit`), Fourier transforms and data comparison.
The class can also used for plotting the results in multiple forms, including density matrix histograms and Bloch spheres.
Several fit models and functions relevant for analysis of color centers are provided in the `fit_functions` module.

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