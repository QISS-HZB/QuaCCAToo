.. QuaCCAToo documentation master file, created by
   sphinx-quickstart on Wed Sep 18 12:25:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QuaCCAToo: Quantum Color Centers Analysis Toolbox
=================================================

.. image:: QuaCCAToo_logo.svg

QuaCCAToo is a Python library for simulating and analyzing spin dynamics of color centers for quantum
technology applications, without using rotating wave approximations. The software serves as an extension for
QuTip, inheriting its object-oriented framework and the Qobj class. This way, the software combines
accessibility from the high level of abstraction and human-readability of Python with the efficiency of
compiled programming languages provided by Qutip's parallelization and the matrix algebra from Scipy
and Numpy.

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
- `03 <https://qiss-hzb.github.io/QuaCCAToo/tutorials/03_NV_ambiguous_resonances.html>`__:
  simulation of `Ambiguous Resonances in Multipulse Quantum Sensing with
  Nitrogen Vacancy
  Centers <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.022606>`__.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   notebooks
   modules
