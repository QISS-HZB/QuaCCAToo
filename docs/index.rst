.. QuaCCAToo documentation master file, created by
   sphinx-quickstart on Wed Sep 18 12:25:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QuaCCAToo: Quantum Color Centers Analysis Toolbox
=================================================

.. image:: QuaCCAToo_logo.svg

QuaCCAToo is a Python library for simulating and analyzing spin dynamics of color centers for quantum
technology applications. The systems' time evolution under pulsed experiments are calculated through quantum
master equations based on the provided Hamiltonian, with realistic pulses in the laboratory frame. The
software is built on top of QuTip, inheriting its object-oriented framework and the `Qobj` class. This way,
the software provides accessibility from the high level of abstraction and human-readability of Python, but
at the expense of limited performance compared to compiled programming languages.

The documentation for QuaCCAToo is available here. Merge requests welcome at https://github.com/QISS-HZB/QuaCCAToo !

To see usage examples, check the tutorial notebooks linked here. They
contain:

- `01 - Rabi and Hahn of a Spin Half System <https://qiss-hzb.github.io/QuaCCAToo/tutorials/01_spin_half_Rabi_Hahn.html>`__:
  simplest two-level system, where we first define the system and plot
  the energy levels. Following that, a Rabi oscillation is simulated for
  two different pulse vectors, with the results being fitted and plotted
  in the Bloch sphere. Lastly, we simulated a Hahn echo decay for a
  modeled collapse operator.
- `02 - Ramsey and PODMR with NV <https://qiss-hzb.github.io/QuaCCAToo/tutorials/02_NV_Ramsey_PODMR.html>`__:
  simulation of nitrogen vacancy centers in diamond, first calculating
  the energy levels, then performing Rabi and comparing with
  experimental data. Ramsey and PODMR are also simulated.
- `03 - Conditional Gates with NV-13C <https://qiss-hzb.github.io/QuaCCAToo/tutorials/03_NV_conditional_gates.html>`__:
  conditional gates with resonant MW and RF pulses of an NV center strongly coupled to a 13C nuclear spin from
  `Observation of Coherent Oscillation of a Single Nuclear Spin and Realization of a Two-Qubit Conditional Quantum Gate <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.93.130501>`__.
- `04 - Sensing and Control of 13C Spin with NV <https://qiss-hzb.github.io/QuaCCAToo/tutorials/04_NV_sensing_control_by_DD.html>`__:
  dynamical decoupling sequences for sensing and controling coupled 13C nuclear spins from
  `Coherent dynamics of coupled electron and nuclear spin qubits in diamond <https://www.science.org/doi/10.1126/science.1131871>`__
  and
  `Detection and control of individual nuclear spins using a weakly coupled electron spin <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.137602>`__.
- `05 - Ambiguous Resonances in DD with NV <https://qiss-hzb.github.io/QuaCCAToo/tutorials/05_NV_ambiguous_resonances.html>`__:
  simulation of `Ambiguous Resonances in Multipulse Quantum Sensing with
  Nitrogen Vacancy
  Centers <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.022606>`__.
- `06 - Teleportation Protocol with NV Pair <https://qiss-hzb.github.io/QuaCCAToo/tutorials/06_NV_teleportation.html>`__:
  simulation of NV teleportation protocol in
  `Unconditional quantum teleportation between distant solid-state quantum bits
  <https://www.science.org/doi/10.1126/science.1253512>`__.

Class Hierarchy
---------------

QuaCCAToo is an object-oriented package organized with the following classes:

- **QSys** defines the quantum system of the problem. It has an obligatory intrinsic internal Hamiltonian :math:`H_0`, optional initial state, observable and a set of collapse operators.
  On ``QSys``, calculates the eigenstates and eigenvalues of the system and has methods for truncating the systems and adding other spins.
  QuaCCAToo provides ``NV`` (``NV_Sys``) as a predefined system for nitrogen vacancy centers in diamonds, more systems will be provided soon.

- **PulsedSim** contains the logic for performing the simulation of pulsed experiments upon a ``QSys`` object.
  It has attributes of a pulse sequence containing a set of pulses and free evolutions, control Hamiltonian :math:`H_1`, experiment variable and simulation results. Many predefined common pulse sequences are given in ``predef_seqs`` and ``predef_dd_seqs`` modules.
  Different pulse shapes are predefined in the ``pulse_shapes`` module.

- **ExpData** is a class to load experimental data and perform basic data processing, such as rescaling, subtracting columns or performing polynomial baseline corrections.

- **Analysis** can be used either on simulation or experimental results, with a series of methods like for fitting (based on ``lmfit``), Fourier transforms and data comparison.
  The class can also used for plotting the results in multiple forms, including density matrix histograms and Bloch spheres.
  Several fit models and functions relevant for analysis of color centers are provided in the ``fit_functions`` module.

.. image:: class_diagram.svg

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   notebooks
   modules
