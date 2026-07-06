Installation
============

Normal installation
-------------------

We strongly recommend using a virtual environment using whichever tool like `venv`/`conda`/`uv` that you prefer so that the system Python remains untouched. For example, with conda:

.. code:: sh

   conda create --name quaccatoo-env python
   conda activate quaccatoo-env


The package can then be installed from pip:

.. code:: sh

   pip install quaccatoo


Or from conda-forge:

.. code:: sh

   conda install -c conda-forge quaccatoo


Finally, to check installation:

.. code:: sh

   conda list quaccatoo


In some cases, switching to an alternate ``BLAS`` implementation can provide
speedups (in particular, ``MKL``, ``BLIS``, or ``netlib`` implementations).
Refer `here <https://conda-forge.org/docs/maintainer/knowledge_base/#blas>`__
for more details and instructions.

.. code:: sh

   conda install "libblas=*=*_blis"


.. note::
   QuaCCAToo is best optimized to run on GNU/Linux systems. More specifically, the ``parallel_map`` method from
   Qutip, which QuaCCAToo heavily relies on, does not work properly on Windows. This leads to severe slow downs
   in simulations and even crashes in some hardware. macOS machines have not been tested by us.

Development installation
------------------------

.. note::
   This section is applicable only if you want to hack on QuaCCAToo.

Once again, we strongly recommend using a virtual environment. Please set it up with any tool of your choice
(``uv`` / ``conda``/ ``venv``) before proceeding.

.. code:: sh

   git clone https://github.com/QISS-HZB/QuaCCAToo
   cd QuaCCAToo
   pip install -e '.[dev]'    #not the system pip


Contribution guidelines can be found in the `Github repository <https://github.com/QISS-HZB/QuaCCAToo>`__.