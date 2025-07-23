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

- `01 <https://qiss-hzb.github.io/QuaCCAToo/tutorials/01_spin_half_Rabi_Hahn.html>`__:
  simplest two-level system, where we first define the system and plot
  the energy levels. Following that, a Rabi oscillation is simulated for
  two different pulse vectors, with the results being fitted and plotted
  in the Bloch sphere. Lastly, we simulated a Hahn echo decay for a
  modeled collapse operator.
- `02 <https://qiss-hzb.github.io/QuaCCAToo/tutorials/02_NV_Ramsey_PODMR.html>`__:
  simulation of nitrogen vacancy centers in diamond, first calculating
  the energy levels, then performing Rabi and comparing with
  experimental data. Ramsey and PODMR are also simulated.
- `03 <https://qiss-hzb.github.io/QuaCCAToo/tutorials/03_NV_conditional_gates.html>`__:
  conditional gates with resonant MW and RF pulses of an NV center strongly coupled to a 13C nuclear spin from
  `Observation of Coherent Oscillation of a Single Nuclear Spin and Realization of a Two-Qubit Conditional
  Quantum Gate <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.93.130501>`__.
- `04 <https://qiss-hzb.github.io/QuaCCAToo/tutorials/04_NV_sensing_control_by_DD.html>`__:
  `Coherent dynamics of coupled electron and nuclear spin qubits in diamond
  <https://www.science.org/doi/10.1126/science.1131871>`__ and
  `Detection and control of individual nuclear spins using a weakly coupled electron spin
  <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.137602>`__.
- `05 <https://qiss-hzb.github.io/QuaCCAToo/tutorials/05_NV_ambiguous_resonances.html>`__:
  simulation of `Ambiguous Resonances in Multipulse Quantum Sensing with
  Nitrogen Vacancy
  Centers <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.022606>`__.
- `06 <https://qiss-hzb.github.io/QuaCCAToo/tutorials/06_NV_teleportation.html>`__:
  simulation of NV teleportation protocol in
  `Unconditional quantum teleportation between distant solid-state quantum bits
  <https://www.science.org/doi/10.1126/science.1253512>`__.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   notebooks
   modules
