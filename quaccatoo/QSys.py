# TODO: add_spin method for QSys class
# TODO: include new systems
# TODO: think on a better strategy for the plot_energy_B0 function
# TODO: implement eV units conversion to frequencies

"""
This module contains the plot_energy_B0 function, the QSys class and the NV subclass.
"""

from qutip import tensor, jmat, qeye, fock_dm, Qobj, basis
import numpy as np
import scipy.constants as cte
import matplotlib.pyplot as plt
import warnings

def plot_energy_B0(B0, H0, figsize=(6, 4), energy_lim = None, xlabel='Magnetic Field', ylabel='Energy (MHz)'):
    """
    Plots the energy levels of a Hamiltonian as a function of a magnetic field B0.

    Parameters
    ----------
    B0 (list): list of magnetic fields
    H0 (list of Qobj): list of Hamiltonians
    figsize (tuple): size of the figure
    energy_lim (tuple): limits of the energy levels
    xlabel (str): label of the x-axis
    ylabel (str): label of the y-axis
    """
    # check if figsize is a tuple of two positive floats
    if not (isinstance(figsize, tuple) or len(figsize) == 2):
        raise ValueError("figsize must be a tuple of two positive floats")
    
    # check if B0 is a numpy array or list and if all elements are real numbers
    if not isinstance(B0 , (np.ndarray, list)) and all(isinstance(b, (int, float)) for b in B0):
        raise ValueError("B0 must be a list or a numpy array of real numbers")
    
    # check if H0 is a list of Qobj of the same size as B0
    if not isinstance(H0, list) or not all(isinstance(h, Qobj) for h in H0) or len(H0) != len(B0):
        raise ValueError("H0 must be a list of Qobj of the same size as B0")
    
    energy_levels = []

    H0_0 = H0[0].eigenenergies()[0]
    
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

    fig.suptitle('Energy Levels')

####################################################################################################

class QSys:
    """
    The QSys class is a general class to define a quantum system. It contains the Hamiltonian, the initial state, the collapse operators and the observable of the system. It also contains a method to plot the energy levels of the Hamiltonian. It is latter used by pulsed experiment simulations.

    Class Attributes
    ----------------
    - H0 (Qobj): time-independent internal Hamiltonian of the system
    - rho0 (Qobj): initial state of the system
    - c_ops (list of Qobj): list of collapse operators
    - observable (Qobj or list of Qobj): observable to be measured
    - units_H0 (str): units of the Hamiltonian
    - energy_levels (np.array): energy levels of the Hamiltonian
    - eigenstates (np.array of Qobj): eigenstates of the Hamiltonian

    Methods
    -------
    - plot_energy: plots the energy levels of the Hamiltonian
    """
    def __init__(self, H0, rho0=None, c_ops=None, observable=None, units_H0=None):
        """
        Generator for the QSys class. It initializes the system with the Hamiltonian, the initial state, the collapse operators and the observable, checking all inputs and calculating the energy levels of the Hamiltonian.

        Parameters
        ----------
        H0 (Qobj): time-independent internal Hamiltonian of the system
        rho0 (Qobj): initial state of the system
        c_ops (list of Qobj): list of collapse operators
        observable (Qobj or list of Qobj): observable to be measured
        units_H0 (str): units of the Hamiltonian
        """

        # if the units are in frequency, assign the Hamiltonian as it is
        if units_H0 == None:
            pass
        elif not isinstance(units_H0, str) :
            raise ValueError("units_H0 must be a string")
        elif units_H0 == 'MHz' or units_H0 == 'GHz' or units_H0 == 'kHz' or units_H0 == 'Hz' or units_H0 == 'eV':
            pass # !!! Units are used only when plotting energy. It is needed to add the conversion to frequencies in case of eV !!!
        else:
            warnings.warn(f"Invalid value for units_H0. Expected either units of frequencies or 'eV', got {units_H0}. The Hamiltonian will be considered in MHz.")

        self.H0 = H0
        self.units_H0 = units_H0
        
        # calculate the eigenenergies of the Hamiltonian
        H_eig = H0.eigenenergies()
        # subtracts the ground state energy from all the eigenenergies to get the lowest level at 0
        self.energy_levels = H_eig - H_eig[0]

        # calculate the eigenstates of the Hamiltonian
        self.eigenstates = np.array([psi*psi.dag() for psi in H0.eigenstates()[1]])

        # check if rho0 is None
        if rho0 == None:
            self.rho0 = self.eigenstates[0] # In this case the initial state is the lowest energy state
            warnings.warn(f"The initial state rho0 is set to be the lowest energy state")

        # check if rho0 is a number
        elif rho0 in range(0, len(self.eigenstates) + 1):
            self.rho0 = self.eigenstates[i] # In this case the initial state is the i-th energy state
            
        # check if rho0 and H0 are Qobj and if they have the same dimensions
        elif isinstance(rho0, Qobj) and isinstance(H0, Qobj):
            if H0.shape != rho0.shape:
                raise ValueError("H0 and rho0 must have the same dimensions")
            # if they are correct, assign them to the objects
            else:
                self.rho0 = rho0

        else:
            raise ValueError("H0 and rho0 must be a Qobj or an index number indicating the system eigenstates")
        
        # check if observable is not None, or if it is a Qobj of the same dimension as H0 and rho0, or a list of Qobj
        if observable is None:
            self.observable = None
        elif (isinstance(observable, Qobj) and observable.shape == H0.shape) or (isinstance(observable, list) and all(isinstance(obs, Qobj) for obs in observable) and all(obs.shape == H0.shape for obs in observable)):
            self.observable = observable
        else:
            raise ValueError("Invalid value for observable. Expected a Qobj or a list of Qobj of the same dimensions as H0 and rho0.")
        
        # check if c_ops is a list of Qobj with the same dimensions as H0
        if c_ops == None:
            self.c_ops = c_ops
        elif isinstance(c_ops, list):
            if not all(isinstance(op, Qobj) and op.shape == self.H0.shape for op in c_ops):
                raise ValueError("All items in c_ops must be Qobj with the same dimensions as H0")
            else:
                self.c_ops = c_ops
        else:
            raise ValueError("c_ops must be a list of Qobj or None")


    def plot_energy(self, figsize=(2, 6), energy_lim = None):
        """
        Plots the energy levels of the Hamiltonian defined in the system.

        Parameters
        ----------
        figsize (tuple): size of the figure.
        energy_lim (list): limits of the energy levels.
        """
        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        fig, ax = plt.subplots(1,1, figsize=figsize)

        for itr in range(self.energy_levels.size):
            ax.axhline(y=self.energy_levels[itr], lw=2)

        if self.units_H0 != None:
            ax.set_ylabel(f'Energy ({self.units_H0})')
        else:
            ax.set_ylabel('Energy')

        ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if energy_lim == None:
            ax.set_ylim(-.05*self.energy_levels[-1], 1.05*self.energy_levels[-1])
        elif len(energy_lim) == 2:
            ax.set_ylim(energy_lim[0], energy_lim[1])
        else:
            raise ValueError("freq_lim must be a tuple of two floats")
        
    def save():
        pass

####################################################################################################

class NV(QSys):
    def __init__(self, B0, N, c_ops=None, units_B0=None, theta=0, phi_r=0, units_angles='deg'):
        """
        Generator for the NV class. Takes the nitrogen isotope, the magnetic field intensity and angles with the quantization axis as inputs and calculates the energy levels of the Hamiltonian.

        Parameters
        ----------
        B0 (float): magnetic field
        N (int): nitrogen isotope (14 or 15)
        c_ops (list of Qobj): list of collapse operators
        units_B0 (str): units of the magnetic field (T, mT or G)
        theta (float): angle of the magnetic field with respect to the NV axis
        phi_r (float): angle of the magnetic field in the xy plane
        units_angles (str): units of the angles (deg or rad)
        """
        # converts the magnetic field to Gauss
        if not isinstance(B0, (int, float)):
            raise ValueError(f"B0 must be a real number, got {B0}.")

        if units_B0 == None:
            warnings.warn("No units for the magnetic field were given. The magnetic field will be considered in mT.")
        elif units_B0 == 'T':
            B0 = B0*1e3
        elif units_B0 == 'mT':
            pass
        elif units_B0 == 'G':
            B0 = B0*1e-3
        else:
            raise ValueError(f"Invalid value for units_B0. Expected either 'G', 'mT' or 'T', got {units_B0}.")

        # check if c_ops is a list of Qobj with the same dimensions as H0
        if c_ops == None:
            self.c_ops = c_ops
        elif isinstance(c_ops, list):
            if not all(isinstance(op, Qobj) and op.shape == self.H0.shape for op in c_ops):
                raise ValueError("All items in c_ops must be Qobj with the same dimensions as H0")
            else:
                self.c_ops = c_ops
        else:
            raise ValueError("c_ops must be a list of Qobj or None")

        if not isinstance(theta, (int, float)) or not isinstance(phi_r, (int, float)):
            raise ValueError(f"Invalid value for theta or phi_r. Expected a number, got {theta} or {phi_r}.")
        else:
            # converts the angles to radians
            if units_angles == 'deg':
                theta = np.deg2rad(theta)
                phi_r = np.deg2rad(phi_r)
            elif units_angles == 'rad':
                pass
            else:
                raise ValueError(f"Invalid value for units_angles. Expected either 'deg' or 'rad', got {units_angles}.")
            
        # standard units for NV Hamiltonian
        self.units_H0 = 'MHz'
            
        # calculates the Hamiltonian for the given field and nitrogen isotope
        if N == 15:
            self.H0 = (  2.87e3*tensor(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3, qeye(2)) # Zero Field Splitting                
                    + 28.025*B0*tensor( np.sin(theta)*np.cos(phi_r)*jmat(1, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1, 'y') + np.cos(theta)*jmat(1,'z'), qeye(2)) # Electron Zeeman
                    + 3.03*tensor(jmat(1,'z'), jmat(1/2,'z')) + 3.65*(tensor(jmat(1,'x'), jmat(1/2,'x')) + tensor(jmat(1,'y'), jmat(1/2,'y'))) # Hyperfine Coupling
                    + 4.316e-3*B0*tensor(qeye(3),  np.sin(theta)*np.cos(phi_r)*jmat(1/2, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1/2, 'y') + np.cos(theta)*jmat(1/2,'z')) ) # 15N Nuclear Zeeman
            
            self.rho0 = tensor(fock_dm(3,1), qeye(2)).unit()
            self.observable = tensor(fock_dm(3,1), qeye(2))

        elif N == 14:
            self.H0 = (  2.87e3*tensor(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3, qeye(3)) # Zero Field Splitting                
                    + 28.025*B0*tensor( np.sin(theta)*np.cos(phi_r)*jmat(1, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1, 'y') + np.cos(theta)*jmat(1,'z'), qeye(3)) # Electron Zeeman                   
                    - 2.7*tensor(jmat(1,'z'), jmat(1,'z')) - 2.14*(tensor(jmat(1,'x'), jmat(1,'x')) + tensor(jmat(1,'y'), jmat(1,'y'))) # Hyperfine Coupling                                    
                    - 5.01*tensor(qeye(3), jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2) # Quadrupole Interaction
                    - 3.077e-3*B0*tensor(qeye(3), np.sin(theta)*np.cos(phi_r)*jmat(1,'x') + np.sin(theta)*np.sin(phi_r)*jmat(1, 'y') + np.cos(theta)*jmat(1,'z'))) # 14N Nuclear Zeeman
            
            self.rho0 = tensor(fock_dm(3,1), qeye(3)).unit()
            self.observable = tensor(fock_dm(3,1), qeye(3))

        elif N == 0 or N == None:
            self.H0 = (  2.87e3*(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3) # Zero Field Splitting                
                    + 28.025*B0*( np.sin(theta)*np.cos(phi_r)*jmat(1, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1, 'y') + np.cos(theta)*jmat(1,'z')) )# Electron Zeeman
            
            self.rho0 = fock_dm(3,1).unit()
            self.observable = fock_dm(3,1)

        else:
            raise ValueError(f"Invalid value for Nitrogen isotope. Expected either 14 or 15, got {N}.")
        
        self.N = N
        # get the eigenenergies of the Hamiltonian
        H0_eig = self.H0.eigenenergies()
        self.energy_levels = H0_eig - H0_eig[0]
    
    def rho0_lowT(self, T, units_T='K'):
        """
        Calculates the initial state of the system at low temperatures using the Boltzmann distribution. At room temperatures and moderate fields, the initial state of the nuclear spins is simply an identity matrix.

        Parameters
        ----------
        T (float): temperature
        units_T (str): units of the temperature (K or C)

        Returns
        -------
        rho0 (Qobj): initial state of the system
        """
        # check the units and convert the temperature to Kelvin
        if units_T == 'K':
            pass
        elif units_T == 'C':
            T += 273.15
        elif units_T == 'F':
            raise ValueError(f"'F' is not a valid unit for temperature, learn the metric system.")
        else:
            raise ValueError(f"Invalid value for units_T. Expected either 'K' or 'C', got {units_T}.")

        # check if the temperature is a positive real number
        if not isinstance(T, (int, float)) and T > 0:
            raise ValueError(f"T must be a positive real number.")
        
        # a loop to find the |0,1/2> and |0,-1/2> states
        max_1= 0
        max_2= 0
        max_3 = 0
        index_1 = None
        index_2 = None
        index_3 = None
        # iterates over all the eigenstates and find the one closest related to the |0,1/2> and |0,-1/2> states
        for itr in range(len(self.energy_levels)):
            if self.N == 15:
                proj_1 = np.abs(self.energy_levels[itr].overlap(basis(6, 2)))
                proj_2 = np.abs(self.energy_levels[itr].overlap(basis(6, 3)))

            elif self.N == 14:
                proj_1 = np.abs(self.energy_levels[itr].overlap(basis(9, 3)))
                proj_2 = np.abs(self.energy_levels[itr].overlap(basis(9, 4)))
                proj_3 = np.abs(self.energy_levels[itr].overlap(basis(9, 5)))
                if proj_3 > max_3:
                    # if the projection is higher than the previous maximum, update the maximum and the index
                    max_3 = proj_3
                    index_3 = itr

            if proj_1 > max_1:
                max_1 = proj_1
                index_1 = itr
            if proj_2 > max_2:
                max_2 = proj_2
                index_2 = itr

        beta = -1/cte.Boltzmann*T

        if self.N == 15:
            # calculate the partition function based on the Hamiltonian eigenvalues
            Z = np.exp(beta*self.energy_levels[index_1]) + np.exp(beta*self.energy_levels[index_2])
            
            self.rho0 = tensor(fock_dm(3,1), Qobj([ [np.exp(beta*self.energy_levels[index_1]), 0] , [0, np.exp(beta*self.energy_levels[index_2])] ])/ Z )
        
        elif self.N == 14:
            Z = np.exp(beta*self.energy_levels[index_1]) + np.exp(beta*self.energy_levels[index_2]) + np.exp(beta*self.energy_levels[index_3])

            self.rho0 = tensor(fock_dm(3,1), Qobj([ [np.exp(beta*self.energy_levels[index_1]), 0, 0] , [0, np.exp(beta*self.energy_levels[index_2]), 0], [0, 0, np.exp(beta*self.energy_levels[index_3])] ])/ Z ) 

        return self.rho0

    def MW_freqs(self):
        """
        Gets the standard resonant microwave frequencies for the NV center corresponding to the electronic spin transitions.

        Returns
        -------
        f1 (float): first resonant frequency
        f2 (float): second resonant frequency
        """

        if self.N == 15:
            f1 = (np.sum(self.energy_levels[2:4]) - np.sum(self.energy_levels[0:2]))/2
            f2 = (np.sum(self.energy_levels[4:6]) - np.sum(self.energy_levels[0:2]))/2
        elif self.N == 14:
            f1 = (np.sum(self.energy_levels[3:6]) - np.sum(self.energy_levels[0:3]))/3
            f2 = (np.sum(self.energy_levels[6:9]) - np.sum(self.energy_levels[0:3]))/3
        elif self.N == 0 or self.N == None:
            f1 = self.energy_levels[1] - self.energy_levels[0]
            f2 = self.energy_levels[2] - self.energy_levels[0]
        return f1, f2
        
    def RF_freqs(self):
        """
        Gets the standard resonant RF frequencies for the NV center corresponding to the nuclear spin transitions.

        Returns
        -------
        fi (float): resonant frequencies
        """

        if self.N == 15:
            f1 = self.energy_levels[1] - self.energy_levels[0]
            f2 = self.energy_levels[3] - self.energy_levels[2]
            f3 = self.energy_levels[5] - self.energy_levels[4]
            return f1, f2, f3
        elif self.N == 14:
            f11 = self.energy_levels[1] - self.energy_levels[0]
            f12= self.energy_levels[2] - self.energy_levels[1]
            f21 = self.energy_levels[4] - self.energy_levels[3]
            f22 = self.energy_levels[5] - self.energy_levels[4]
            f31 = self.energy_levels[7] - self.energy_levels[6]
            f32 = self.energy_levels[8] - self.energy_levels[7]
            return f11, f12, f21, f22, f31, f32
        
    def MW_H1(self):
        """
        Gets the standard microwave Hamiltonian for the NV center corresponding to the electronic spin transitions.
        """
        if self.N == 15:
            return tensor(jmat(1, 'x'), qeye(2))*2**.5
        elif self.N == 14:
            return tensor(jmat(1, 'x'), qeye(3))*2**.5
        elif self.N == 0 or self.N == None:
            return tensor(jmat(1, 'x'))*2**.5
        
    def RF_H1(self):
        """
        Gets the standard RF Hamiltonian for the NV center corresponding to the nuclear spin transitions.
        """
        if self.N == 15:
            return tensor(qeye(3), jmat(1/2, 'x'))*2
        elif self.N == 14:
            return tensor(qeye(3), jmat(1, 'x'))*2**.5
        elif self.N == 0 or self.N == None:
            return jmat(1, 'x')*2**.5
