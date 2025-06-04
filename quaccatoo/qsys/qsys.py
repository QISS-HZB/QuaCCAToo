# TODO: think on a better strategy for the plot_energy_B0 function
# TODO: implement eV units conversion to frequencies

"""
This module contains the plot_energy_B0 function, compose_sys function and the QSys class part of QuaCCAToo package.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, qeye, tensor, basis

def compose_sys(qsys1, qsys2):
    """
    Takes two quantum systems and returns the composed system by performing tensor products of the two,
    after checking if all parameters are valid.

    Parameters
    ----------
    qsys1 : QSys
        First quantum system.
    qsys2 : QSys
        Second quantum system.

    Returns
    -------
    qsys1 X qsys2 : QSys
        Composed quantum system.
    """
    if not isinstance(qsys1, QSys) or not isinstance(qsys2, QSys):
        raise ValueError("Both qsys1 and qsys2 must be instances of the QSys class.")
    
    if qsys1.units_H0 != qsys2.units_H0:
        warnings.warn("The two systems have different units.")

    if isinstance(qsys1.H0, Qobj) and isinstance(qsys2.H0, Qobj):
        H0 = tensor(qsys1.H0, qsys2.H0)
    else:
        raise ValueError("Both Hamiltonians must be Qobj.")

    if qsys1.rho0.isherm and qsys2.rho0.isherm:
        rho0 = tensor(qsys1.rho0, qsys2.rho0).unit()
    elif qsys1.rho0.isket and qsys2.rho0.isket:
        rho0 = tensor(qsys1.rho0, qsys2.rho0).unit()
    else:
        rho0 = None
        
    if qsys1.observable is not None and qsys2.observable is not None:
        if isinstance(qsys1.observable, Qobj) and isinstance(qsys2.observable, Qobj):
            observable = tensor(qsys1.observable, qsys2.observable)
        elif isinstance(qsys1.observable, list) and isinstance(qsys2.observable, list):
            observable = [tensor(obs1, qeye(qsys2.H0.shape[0])) for obs1 in qsys1.observable] + [tensor(qeye(qsys1.H0.shape[0]), obs2) for obs2 in qsys2.observable]
        else:
            raise ValueError("Both observables must be Qobj or None")
    else:
        observable = None
    
    if qsys1.c_ops is None and qsys2.c_ops is None:
        c_ops = None
    elif qsys1.c_ops is not None and qsys2.c_ops is None:
        c_ops = [tensor(op1, qeye(qsys2.H0.shape[0])) for op1 in qsys1.c_ops]
    elif qsys1.c_ops is None and qsys2.c_ops is not None:
        c_ops = [tensor(qeye(qsys1.H0.shape[0]), op2) for op2 in qsys2.c_ops]
    elif qsys1.c_ops is not None and qsys2.c_ops is not None:
        c_ops = [tensor(op1, qeye(qsys2.H0.shape[0])) for op1 in qsys1.c_ops] + [tensor(qeye(qsys1.H0.shape[0]), op2) for op2 in qsys2.c_ops]
    else:
        raise ValueError("Both collapse operators must be Qobj, a list of Qobj or None")
    
    return QSys(H0, rho0, c_ops, observable, qsys1.units_H0)

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
    if not (isinstance(figsize, tuple) or len(figsize) == 2):
        raise ValueError("figsize must be a tuple of two positive floats")

    if not isinstance(B0, (np.ndarray, list)) and all(isinstance(b, (int, float)) for b in B0):
        raise ValueError("B0 must be a list or a numpy array of real numbers")

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
    _get_energy_levels
        Calculates the eigenenergies of the Hamiltonian and sets the energy_levels and eigenstates attributes.
    plot_energy
        Plots the energy levels of the Hamiltonian.
    add_spin
        Adds another spin to the system's Hamiltonian and updates the system accordingly.
    truncate
        Truncates the quantum system by removing the states specified by the indexes from the system
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
        
        self.H0 = Qobj(H0)

        if not self.H0.isherm:
            warnings.warn("Passed H0 is not a hermitian object.")

        self._get_energy_levels()

        # check if rho0 is correctly defined
        if rho0 is None:
            warnings.warn("Initial state not provided.")
        elif rho0 in range(0, len(self.eigenstates)):
            self.rho0 = self.eigenstates[rho0]  # In this case the initial state is the i-th energy state
        elif Qobj(rho0).isket and Qobj(rho0).shape[0] == H0.shape[0]:
            self.rho0 = Qobj(rho0)
        elif Qobj(rho0).isherm and Qobj(rho0).shape == H0.shape:
            self.rho0 = Qobj(rho0)
        else:
            raise ValueError("rho0 must be a Qobj, None or an index number indicating the system eigenstates")

        # check if observable is not None, or if it is a Qobj of the same dimension as H0 or a list of Qobj
        if observable is None:
            self.observable = None
        elif isinstance(observable, (Qobj, np.ndarray)) and observable.shape == H0.shape:
            self.observable = observable
            if not observable.isherm:
                warnings.warn("Passed observable is not hermitian.")
        elif isinstance(observable, list) and all(isinstance(obs, (Qobj, np.ndarray)) for obs in observable) and all(obs.shape == H0.shape for obs in observable):
            self.observable = observable
            if not all(obs.isherm for obs in observable):
                warnings.warn("Passed observables are not hermitian.")
        else:
            raise ValueError("Invalid value for observable. Expected a Qobj or a list of Qobj of the same dimensions as H0.")

        # check if c_ops is a list of Qobj with the same dimensions as H0
        if c_ops is None:
            self.c_ops = c_ops
        elif isinstance(c_ops, Qobj) and c_ops.shape == self.H0.shape:
            self.c_ops = c_ops
        elif isinstance(c_ops, list):
            if all(isinstance(op, (Qobj, np.ndarray)) and op.shape == self.H0.shape for op in c_ops):
                self.c_ops = c_ops
            else:
                raise ValueError("All items in c_ops must be Qobj with the same dimensions as H0")
        else:
            raise ValueError("c_ops must be a list of Qobj or None")
        
    def _get_energy_levels(self):
        """
        Calculates the eigenenergies of the Hamiltonian and subtract the ground state energy from all of them to get the lowest level at 0.
        Sets the energy_levels and eigenstates attributes of the class.
        """
        H_eig = self.H0.eigenenergies()
        self.energy_levels = H_eig - H_eig[0]

        self.eigenstates = self.H0.eigenstates()[1]


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
        Adds another spin to the system's Hamiltonian, given a Hamiltonian for the new spin.
        Updates the time-independent Hamiltonian, the energy levels, the initial state, the observable and the collapse operators accordingly.

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
        self._get_energy_levels()
        
        if self.rho0.isherm:
            self.rho0 = tensor(self.rho0, qeye(self.dim_add_spin )).unit()
        elif self.rho0.isket:
            self.rho0 = tensor(self.rho0, basis(self.dim_add_spin, 0)).unit()

        if self.observable is not None:
            self.observable = tensor(self.observable, qeye(self.dim_add_spin ))

        if self.c_ops is not None:
            self.c_ops = [tensor(op, qeye(self.dim_add_spin )) for op in self.c_ops]

    def truncate(self, indexes):
        """
        Truncantes the quantum system by removing the states specified by the indexes from the Hamiltonian,
        the initial state, the observable and the collapse operators. For example, if S=1 and the user wants to remove the ms=-1 state,
        the indexes is set to 0 and the qsys is truncated to the ms=0 and ms=+1 subspace. The method uses the numpy.delete function
        to remove the specified indexes from the Hamiltonian and the initial state.

        Parameters
        ----------
        indexes : int or list of int
            Index or list of indexes to be removed from the system.
        """
        if isinstance(indexes, int):
            if indexes < 0 or indexes >= self.H0.shape[0]:
                raise ValueError("sel must be a valid index of the Hamiltonian.")
        if isinstance(indexes, (list, np.array)):
            if not all(isinstance(i, int) for i in indexes) and not all(0 <= i < self.H0.shape[0] for i in indexes):
                raise ValueError("All elements in sel must be valid indices of the Hamiltonian.")
        else:
            raise ValueError("sel must be an integer or a list of integers.")
        
        self.H0 = Qobj( np.delete(np.delete(self.H0.full(), indexes, axis=0), indexes, axis=1) )
        self._get_energy_levels()

        if self.observable is not None:
            if isinstance(self.observable, Qobj):
                self.observable = Qobj(np.delete(np.delete(self.observable.full(), indexes, axis=0), indexes, axis=1))
            elif isinstance(self.observable, list):
                self.observable = [Qobj(np.delete(np.delete(obs.full(), indexes, axis=0), indexes, axis=1)) for obs in self.observable]

        if self.rho0 is not None:
            if self.rho0.isket:
                self.rho0 = Qobj(np.delete(self.rho0.full(), indexes, axis=0)).unit()
            else:
                self.rho0 = Qobj(np.delete(np.delete(self.rho0.full(), indexes, axis=0), indexes, axis=1)).unit()
        
        if self.c_ops is not None:
            if isinstance(self.c_ops, Qobj):
                self.c_ops = Qobj(np.delete(np.delete(self.c_ops.full(), indexes, axis=0), indexes, axis=1))
            elif isinstance(self.c_ops, list):
                self.c_ops = [Qobj(np.delete(np.delete(op.full(), indexes, axis=0), indexes, axis=1)) for op in self.c_ops]