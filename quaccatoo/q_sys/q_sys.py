# TODO: think on a better strategy for the plot_energy_B0 function
# TODO: implement eV units conversion to frequencies

"""
This module contains the plot_energy_B0 function, the QSys class part of QuaCCAToo package.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, qeye, tensor

def plot_energy_B0(B0, H0, figsize=(6, 4), energy_lim=None, xlabel="Magnetic Field", ylabel="Energy (MHz)"):
    """
    Plots the energy levels of a Hamiltonian as a function of a magnetic field B0.

    Parameters
    ----------
    B0 : list or numpy.ndarray
        List of magnetic fields.
    H0 : list of Qobj
        List of Hamiltonians.
    figsize : tuple
        Size of the figure.
    energy_lim : tuple
        Limits of the energy levels.
    xlabel : str
        Label of the x-axis.
    ylabel : str
        Label of the y-axis.
    """
    # check if figsize is a tuple of two positive floats
    if not (isinstance(figsize, tuple) or len(figsize) == 2):
        raise ValueError("figsize must be a tuple of two positive floats")

    # check if B0 is a numpy array or list and if all elements are real numbers
    if not isinstance(B0, (np.ndarray, list)) and all(isinstance(b, (int, float)) for b in B0):
        raise ValueError("B0 must be a list or a numpy array of real numbers")

    # check if H0 is a list of Qobj of the same size as B0
    if not isinstance(H0, list) or not all(isinstance(h, Qobj) for h in H0) or len(H0) != len(B0):
        raise ValueError("H0 must be a list of Qobj of the same size as B0")

    energy_levels = []

    H0_0 = H0[0].eigenenergies()[0]

    # iterate over all the Hamiltonians and calculate the energy levels
    for itr_B0 in range(len(B0)):
        H0_eig = H0[itr_B0].eigenenergies()
        energy_levels.append(H0_eig - H0_0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for itr_e in range(len(energy_levels[0])):
        ax.plot(B0, [energy_levels[itr_B0][itr_e] for itr_B0 in range(len(B0))])

    if isinstance(energy_lim, tuple) and len(energy_lim) == 2:
        ax.set_ylim(energy_lim)

    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)
    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel)

    fig.suptitle("Energy Levels")

####################################################################################################

class QSys:
    """
    The QSys class defines a general quantum system and contains a method for plotting the energy levels of the Hamiltonian.

    Attributes
    ----------
    H0 : Qobj or numpy.ndarray
        Time-independent internal Hamiltonian of the system.
    rho0 : Qobj, numpy.ndarray, or int
        Initial state of the system.
    c_ops : Qobj or list of Qobj
        List of collapse operators.
    observable : Qobj or list of Qobj
        Observable to be measured.
    units_H0 : str
        Units of the Hamiltonian.
    energy_levels : numpy.ndarray
        Energy levels of the Hamiltonian.
    eigenstates : numpy.ndarray of Qobj
        Eigenstates of the Hamiltonian.
    dim_add_spin : int
        Dimension of the added spin if the add_spin method is used.

    Methods
    -------
    plot_energy
        Plots the energy levels of the Hamiltonian.
    """

    def __init__(self, H0, rho0=None, c_ops=None, observable=None, units_H0=None):
        """
        Construct the QSys class. It initializes the system with the Hamiltonian, the initial state, the collapse operators and the observable.
        Checking all inputs and calculating the energy levels of the Hamiltonian.

        Parameters
        ----------
        H0 : Qobj/array
            time-independent internal Hamiltonian of the system
        rho0 : Qobj/array/int
            initial state of the system. Can be a Qobj, an array or an index number indicating the system eigenstates
        c_ops : list(Qobj)
            list of collapse operators
        observable : Qobj or list(Qobj)
            observable to be measured
        units_H0 : str
            units of the Hamiltonian
        """
        # if the units are in frequency, assign the Hamiltonian as it is
        if units_H0 is None:
            self.units_H0 = "MHz"
            warnings.warn("No units supplied, assuming default value of MHz.")
        elif units_H0 in ["MHz", "GHz", "kHz", "eV"]:
            self.units_H0 = units_H0
        else:
            raise ValueError(f"Invalid value for units_H0. Expected either units of frequencies or 'eV', got {units_H0}. The Hamiltonian will be considered in MHz.")

        if not Qobj(H0).isherm:
            warnings.warn("Passed H0 is not a hermitian object.")

        self.H0 = Qobj(H0)

        # calculate the eigenenergies of the Hamiltonian
        H_eig = H0.eigenenergies()
        # subtracts the ground state energy from all the eigenenergies to get the lowest level at 0
        self.energy_levels = H_eig - H_eig[0]

        # calculate the eigenstates of the Hamiltonian
        self.eigenstates = np.array([psi * psi.dag() for psi in H0.eigenstates()[1]])

        # check if rho0 is None
        if rho0 is None:
            warnings.warn("Initial state not provided.")

        # check if rho0 is a number
        elif rho0 in range(0, len(self.eigenstates)):
            self.rho0 = self.eigenstates[rho0] * self.eigenstates[rho0].dag()  # In this case the initial state is the i-th energy state

        elif Qobj(rho0).isket and Qobj(rho0).shape[0] == H0.shape[0]:
            rho0 = Qobj(rho0)
            self.rho0 = rho0 * rho0.dag()

        elif Qobj(rho0).isherm and Qobj(rho0).shape == H0.shape:
            self.rho0 = Qobj(rho0)

        else:
            raise ValueError("H0 and rho0 must be a Qobj or an index number indicating the system eigenstates")

        # check if observable is not None, or if it is a Qobj of the same dimension as H0 and rho0, or a list of Qobj
        if observable is None:
            self.observable = None
        elif (isinstance(observable, (Qobj, np.ndarray)) and observable.shape == H0.shape) or (
            isinstance(observable, list) and all(isinstance(obs, (Qobj, np.ndarray)) for obs in observable) and all(obs.shape == H0.shape for obs in observable)
        ):
            self.observable = observable
        else:
            raise ValueError("Invalid value for observable. Expected a Qobj or a list of Qobj of the same dimensions as H0 and rho0.")

        # check if c_ops is a list of Qobj with the same dimensions as H0
        if c_ops is None:
            self.c_ops = c_ops
        elif isinstance(c_ops, list):
            if all(isinstance(op, (Qobj, np.ndarray)) and op.shape == self.H0.shape for op in c_ops):
                self.c_ops = c_ops
            else:
                raise ValueError("All items in c_ops must be Qobj with the same dimensions as H0")
        else:
            raise ValueError("c_ops must be a list of Qobj or None")

    def plot_energy(self, figsize=(2, 6), energy_lim=None):
        """
        Plots the energy levels of the Hamiltonian defined in the system.

        Parameters
        ----------
        figsize : tuple
            size of the figure.
        energy_lim : list
            limits of the energy levels.
        """
        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for itr in range(self.energy_levels.size):
            ax.axhline(y=self.energy_levels[itr], lw=2)

        if self.units_H0 is not None:
            ax.set_ylabel(f"Energy ({self.units_H0})")
        else:
            ax.set_ylabel("Energy")

        ax.get_xaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        if energy_lim is None:
            ax.set_ylim(-0.05 * self.energy_levels[-1], 1.05 * self.energy_levels[-1])
        elif len(energy_lim) == 2:
            ax.set_ylim(energy_lim[0], energy_lim[1])
        else:
            raise ValueError("freq_lim must be a tuple of two floats")
        
    def add_spin(self, H_spin):
        """
        Adds another spin to the system's Hamiltonian,
        given a Hamiltonian for the new spin.

        Parameters
        ----------
        H_spin : Qobj
            Hamiltonian of the new spin        
        """
        if not isinstance(H_spin, Qobj):
            raise ValueError("H_spin must be a Qobj in the form of a tensor with the original Hamiltonian H0.")
        
        if not Qobj(H_spin).isherm:
            warnings.warn("Passed H_spin is not a hermitian object.")
        
        self.dim_add_spin = int(H_spin.shape[0]/self.H0.shape[0])
        
        if self.dim_add_spin <= 1:
            raise ValueError("H_spin must be a Qobj with a dimension higher than H0.")
        
        self.H0 = tensor(self.H0, qeye(self.dim_add_spin )) +  H_spin
        
        if self.rho0 is not None:
            self.rho0 = tensor(self.rho0, qeye(self.dim_add_spin )).unit()

        if self.observable is not None:
            self.observable = tensor(self.observable, qeye(self.dim_add_spin ))

        if self.c_ops is not None:
            self.c_ops = [tensor(op, qeye(self.dim_add_spin )) for op in self.c_ops]