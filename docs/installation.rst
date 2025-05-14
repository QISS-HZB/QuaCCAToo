Installation
============

We strongly recommend using a virtual environment so that the system
Python remains untouched.

You can create a ``conda`` virtual environment as follows:

.. code:: sh

   conda create --name quaccatoo-env python qutip matplotlib lmfit
   conda activate quaccatoo-env

**OR**

You can use a native Python venv (you'll need ``pip`` as well)

.. code:: sh

   python -m venv quaccatoo-env
   source quaccatoo-env/bin/activate

After the virtual environment has been setup, clone the `repository <https://github.com/QISS-HZB/QuaCCAToo>`_ and
run from inside the local cloned folder

.. code:: sh

   pip install .

To check the installation, you can run

.. code:: sh

   pip show quaccatoo

Notebooks can be the most convenient way of running QuaCCAToo (but
definitely not the most efficient for long simulations). If you wish to
use them, you can also install ``jupyter`` in you conda environment.

.. code:: sh

   pip install jupyter
