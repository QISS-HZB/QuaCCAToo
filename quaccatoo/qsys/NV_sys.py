# TODO: add electric and crystal stress hamiltonians for NV

"""
This module contains NV class, which is a subclass of QSys.
"""

import warnings
import numpy as np
import scipy.constants as cte
from qutip import Qobj, basis, fock_dm, jmat, qeye, tensor
from .qsys import QSys

gamma_e = cte.value("electron gyromag. ratio in MHz/T")*1e-3  # MHz/mT
gamma_N14 = 3.077e-3
gamma_N15 = -4.316e-3

class NV(QSys):
    """
    NN class contains attributes and methods to simulate the nitrogen vacancy center in diamond.

    Attributes
    ----------
    B0 : float
        magnetic field
    N : 15, 14, 0 or None
        nitrogen isotope, or 0 for no nuclear spin
    units_B0 : str
        units of the magnetic field (T, mT or G)
    theta : float
        angle of the magnetic field with respect to the NV axis
    phi_r : float
        azimutal angle of the magnetic field with the NV axis
    units_angles : str
        units of the angles (deg or rad)
    temp : float or None
        temperature
    units_temp : str
        temperature units 'C' or 'K'
    energy_levels : list
        list of energy levels of the Hamiltonian
    MW_freqs : numpy.ndarray
        microwave frequencies
    RF_freqs : numpy.ndarray
        RF frequencies
    MW_H1 : Qobj
        microwave Hamiltonian
    RF_H1 : Qobj
        RF Hamiltonian
    
    Methods
    -------
    rho0_lowT
        calculates the initial state of the system at low temperatures using the Boltzmann distribution
    _set_MW_freqs
        sets the standard resonant microwave frequencies for the NV center corresponding to the electronic spin transitions
    _set_RF_freqs
        sets the standard resonant RF frequencies for the NV center corresponding to the nuclear spin transitions
    _set_MW_H1
        sets the standard microwave Hamiltonian for the NV center corresponding to the electronic spin transitions
    _set_RF_H1
        sets the standard RF Hamiltonian for the NV center corresponding to the nuclear spin transitions
    _ZeroField
        get the NV Hamiltonian term accounting for zero field splitting
    _ElectronZeeman
        get the NV hamiltonian term accounting for the electron Zeeman effect
    _NuclearZeeman
        get the NV hamiltonian term accounting for the nuclear (Nitrogen) Zeeman effect
    _HyperfineN
        get the NV hamiltonian term accounting for the hyperfine coupling with Nitrogen    
    """
    def __init__(self, B0, N, c_ops=None, units_B0=None, theta=0., phi_r=0., units_angles="deg", temp=None, units_temp="K", E=0):
        """
        Constructor for the NV class.
        Takes the nitrogen isotope, the magnetic field intensity and angles with the quantization axis as inputs and calculates the energy levels of the Hamiltonian.

        Parameters
        ----------
        B0 : float
            magnetic field
        N : 15/14/0/None
            nitrogen isotope, or 0 for no nuclear spin
        c_ops : list(Qobj)
            list of collapse operators
        units_B0 : str
            units of the magnetic field (T, mT or G)
        theta : float
            angle of the magnetic field with respect to the NV axis
        phi_r : float
            angle of the magnetic field in the xy plane
        units_angles : str
            units of the angles (deg or rad)
        temp : float
            temperature
        units_temp : str
            temperature units ('C'/'K')
        E : float
            perpedicular component of the zero field splitting
        """
        if not isinstance(B0, (int, float)):
            raise TypeError(f"B0 must be a real number, got {B0}: {type(B0)}.")
        else:
            self.B0 = B0

        if units_B0 is None:
            warnings.warn("No units for the magnetic field were given. The magnetic field will be considered in mT.")
        elif units_B0 == "T":
            self.B0 = B0 * 1e3
        elif units_B0 == "mT":
            pass
        elif units_B0 == "G":
            self.B0 = B0 * 1e-1
        else:
            raise ValueError(f"Invalid value for units_B0. Expected either 'G', 'mT' or 'T', got {units_B0}.")

        if not isinstance(theta, (int, float)) or not isinstance(phi_r, (int, float)):
            raise TypeError(f"Invalid type for theta or phi_r. Expected a float or int, got theta: {type(theta)}, phi_r: {type(phi_r)}.")
        else:
            if units_angles == "deg":
                theta = np.deg2rad(theta)
                phi_r = np.deg2rad(phi_r)
            elif units_angles == "rad":
                pass
            else:
                raise ValueError(f"Invalid value for units_angles. Expected either 'deg' or 'rad', got {units_angles}.")
            
        if not isinstance(E, (int, float)):
            raise TypeError(f"E must be a real number, got {E}: {type(E)}.")
        else:
            self.E = E

        self.theta = theta
        self.phi_r = phi_r
        self.N = N

        # calculates the Hamiltonian for the given field and nitrogen isotope
        if N == 15:

            H0 = self._ZeroField() + self._ElectronZeeman() + self._HyperfineN() + self._NuclearZeeman()
            rho0 = tensor(fock_dm(3, 1), qeye(2)).unit()
            observable = tensor(fock_dm(3, 1), qeye(2))

        elif N == 14:
            
            H0 = self._ZeroField() + self._ElectronZeeman() + self._HyperfineN() + self._NuclearZeeman() + self._Quadrupole()
            rho0 = tensor(fock_dm(3, 1), qeye(3)).unit()
            observable = tensor(fock_dm(3, 1), qeye(3))

        elif N == 0 or N is None:

            H0 = self._ZeroField() + self._ElectronZeeman()
            rho0 = basis(3, 1)
            observable = fock_dm(3, 1)

        else:
            raise ValueError(f"Invalid value for Nitrogen isotope. Expected either 14 or 15, got {N}.")

        super().__init__(H0, rho0, c_ops, observable, units_H0="MHz")

        if temp is not None:
            self.rho0_lowT(temp, units_temp)

        self._set_MW_freqs()
        self._set_RF_freqs()
        self._set_MW_H1()
        self._set_RF_H1()

    def rho0_lowT(self, temp, units_temp="K"):
        """
        Calculates the initial state of the system at low temperatures using the Boltzmann distribution.
        At room temperatures and moderate fields, the initial state of the nuclear spins is simply an identity matrix.

        Parameters
        ----------
        T : float
            temperature
        units_temp : str
            units of the temperature (K or C)

        Returns
        -------
        rho0 : Qobj
            initial state of the system
        """
        if units_temp == "K":
            pass
        elif units_temp == "C":
            temp += 273.15
        elif units_temp == "F":
            raise ValueError("'F' is not a valid unit for temperature, learn the metric system.")
        else:
            raise ValueError(f"Invalid value for units_temp. Expected either 'K' or 'C', got {units_temp}.")

        if not isinstance(temp, (int, float)) and temp > 0:
            raise ValueError("T must be a positive real number.")

        # a loop to find the |0,1/2> and |0,-1/2> states
        max_1 = 0
        max_2 = 0
        max_3 = 0
        index_1 = None
        index_2 = None
        index_3 = None
        # iterates over all the eigenstates and find the one closest related to the |0,1/2> and |0,-1/2> states
        for itr in range(len(self.eigenstates)):
            if self.N == 15:
                proj_1 = np.abs(self.eigenstates[itr].overlap(basis(6, 2)))
                proj_2 = np.abs(self.eigenstates[itr].overlap(basis(6, 3)))

            elif self.N == 14:
                proj_1 = np.abs(self.eigenstates[itr].overlap(basis(9, 3)))
                proj_2 = np.abs(self.eigenstates[itr].overlap(basis(9, 4)))
                proj_3 = np.abs(self.eigenstates[itr].overlap(basis(9, 5)))
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

        beta = -cte.h*1e6 / (cte.Boltzmann * temp)

        if self.N == 15:
            # calculate the partition function based on the Hamiltonian eigenvalues
            Z = np.exp(beta * self.energy_levels[index_1]) + np.exp(beta * self.energy_levels[index_2])

            self.rho0 = tensor(fock_dm(3, 1), Qobj([[np.exp(beta * self.energy_levels[index_1]), 0], [0, np.exp(beta * self.energy_levels[index_2])]]) / Z)

        elif self.N == 14:
            Z = np.exp(beta * self.energy_levels[index_1]) + np.exp(beta * self.energy_levels[index_2]) + np.exp(beta * self.energy_levels[index_3])

            self.rho0 = tensor(
                fock_dm(3, 1), Qobj([[np.exp(beta * self.energy_levels[index_1]), 0, 0],
                                     [0, np.exp(beta * self.energy_levels[index_2]), 0],
                                     [0, 0, np.exp(beta * self.energy_levels[index_3])]]) / Z
            )
        elif self.N ==0 or self.N is None:
            self.rho0 = basis(3, 1)
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _set_MW_freqs(self):
        """
        Sets the standard resonant microwave frequencies for the NV center corresponding to the electronic spin transitions.
        """
        if self.N == 15:
            f1 = (np.sum(self.energy_levels[2:4]) - np.sum(self.energy_levels[1:2])) / 2
            f2 = (np.sum(self.energy_levels[4:6]) - np.sum(self.energy_levels[1:2])) / 2
            self.MW_freqs = np.array([f1, f2])
        elif self.N == 14:
            f1 = (np.sum(self.energy_levels[3:6]) - np.sum(self.energy_levels[1:3])) / 3
            f2 = (np.sum(self.energy_levels[6:9]) - np.sum(self.energy_levels[1:3])) / 3
            self.MW_freqs = np.array([f1, f2])
        elif self.N == 0 or self.N is None:
            f1 = self.energy_levels[1]
            f2 = self.energy_levels[2]
            self.MW_freqs = np.array([f1, f2])
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _set_RF_freqs(self):
        """
        Sets the standard resonant RF frequencies for the NV center corresponding to the nuclear spin transitions.
        """
        if self.N == 15:
            f1 = self.energy_levels[1]
            f2 = self.energy_levels[3] - self.energy_levels[2]
            f3 = self.energy_levels[5] - self.energy_levels[4]
            self.RF_freqs = np.array([f1, f2, f3])
        # for the 14N isotope, the RF frequencies are more complicated as they need to respect the selection rule of Delta mI = +-1
        elif self.N == 14:
            # the order of the ms states changes above the GSLAC
            if self.B0 <= 102.5:
                f1 = self.energy_levels[2] - self.energy_levels[1] # 0 -> -1 at ms=0
                f2 = self.energy_levels[2] # 0 -> +1 at ms=0
                f3 = self.energy_levels[5] - self.energy_levels[3] # 0 -> -1 at ms=-1
                f4 = self.energy_levels[5] - self.energy_levels[4] # 0 -> +1 at ms=-1
                f5 = self.energy_levels[8] - self.energy_levels[7] # 0 -> -1 at ms=+1
                f6 = self.energy_levels[8] - self.energy_levels[6] # 0 -> +1 at ms=-1
            else:
                f1 = self.energy_levels[2] # 0 -> -1 at ms=-1
                f2 = self.energy_levels[2] - self.energy_levels[1] # 0 -> +1 at ms=-1
                f3 = self.energy_levels[5] - self.energy_levels[4] # 0 -> -1 at ms=0
                f4 = self.energy_levels[5] - self.energy_levels[3] # 0 -> +1 at ms=0
                f5 = self.energy_levels[8] - self.energy_levels[7] # 0 -> -1 at ms=+1
                f6 = self.energy_levels[8] - self.energy_levels[6] # 0 -> +1 at ms=-1

            self.RF_freqs = np.array([f1, f2, f3, f4, f5, f6])
        elif self.N == 0 or self.N is None:
            self.RF_freqs = None
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _set_MW_H1(self):
        """
        Sets the standard microwave Hamiltonian for the NV center corresponding to the electronic spin transitions.
        """
        if self.N == 15:
            self.MW_H1 = tensor(jmat(1, "x"), qeye(2)) * 2**0.5
        elif self.N == 14:
            self.MW_H1 = tensor(jmat(1, "x"), qeye(3)) * 2**0.5
        elif self.N == 0 or self.N is None:
            self.MW_H1 = tensor(jmat(1, "x")) * 2**0.5
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _set_RF_H1(self):
        """
        Sets the standard RF Hamiltonian for the NV center corresponding to the nuclear spin transitions.
        """
        if self.N == 15:
            self.RF_H1 = tensor(qeye(3), jmat(1 / 2, "x")) * 2
        elif self.N == 14:
            self.RF_H1 = tensor(qeye(3), jmat(1, "x")) * 2**0.5
        elif self.N == 0 or self.N is None:
            self.RF_H1 = qeye(3)
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _ZeroField(self):
        """Get the NV Hamiltonian term accounting for zero field splitting.

        Parameters
        ----------
        D : float
            Axial component of magnetic dipole-dipole interaction, by default 2.87e3 MHz (NV)
        E : float
            Non axial compononet, by default 0. Usually much (1000x) smaller than `D`

        Returns
        -------
        Zero Field Hamiltonian : Qobj
        """
        if self.N == 14:
            return tensor(2.87e3 * jmat(1, "z") ** 2 + self.E * (jmat(1, "x") ** 2 - jmat(1, "y") ** 2), qeye(3))
        elif self.N == 15:
            return tensor(2.87e3 * jmat(1, "z") ** 2 + self.E * (jmat(1, "x") ** 2 - jmat(1, "y") ** 2), qeye(2))
        elif self.N == 0:
            return 2.87e3 * jmat(1, "z") ** 2  + self.E * (jmat(1, "x") ** 2 - jmat(1, "y") ** 2)
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _ElectronZeeman(self):
        """
        Get the NV hamiltonian term accounting for the electron Zeeman effect.

        Returns
        -------
        Electron Zeeman Hamiltonian : Qobj
        """

        if self.N == 14:
            return tensor(
                gamma_e * self.B0 * (np.cos(self.theta) * jmat(1, "z") + np.sin(self.theta) * np.cos(self.phi_r) * jmat(1, "x") + np.sin(self.theta) * np.sin(self.phi_r) * jmat(1, "y")), qeye(3)
            )
        elif self.N == 15:
            return tensor(
                gamma_e * self.B0 * (np.cos(self.theta) * jmat(1, "z") + np.sin(self.theta) * np.cos(self.phi_r) * jmat(1, "x") + np.sin(self.theta) * np.sin(self.phi_r) * jmat(1, "y")), qeye(2)
            )
        elif self.N == 0 or self.N is None:
            return gamma_e * self.B0 * (np.cos(self.theta) * jmat(1, "z") + np.sin(self.theta) * np.cos(self.theta) * jmat(1, "x") + np.sin(self.theta) * np.sin(self.phi_r) * jmat(1, "y"))
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _NuclearZeeman(self):
        """
        Get the NV hamiltonian term accounting for the nuclear (Nitrogen) Zeeman effect.

        Returns
        -------
        Nuclear Zeeman Hamiltonian : Qobj
        """

        if self.N == 14:
            return -tensor(qeye(3), 
                gamma_N14 * self.B0 * (np.cos(self.theta) * jmat(1, "z") + np.sin(self.theta) * np.cos(self.phi_r) * jmat(1, "x") + np.sin(self.theta) * np.sin(self.phi_r) * jmat(1, "y"))
            )
        elif self.N == 15:
            return -tensor(qeye(3),
                gamma_N15 * self.B0 * (np.cos(self.theta) * jmat(1/2, "z") + np.sin(self.theta) * np.cos(self.phi_r) * jmat(1/2, "x") + np.sin(self.theta) * np.sin(self.phi_r) * jmat(1/2, "y"))
            )
        elif self.N == 0 or self.N is None:
            return 0
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _HyperfineN(self):
        """
        Get the NV hamiltonian term accounting for the hyperfine coupling with Nitrogen.

        Returns
        -------
        Hyperfine Hamiltonian : Qobj
        """
        if self.N == 14:
            return -2.14 * tensor(jmat(1, "z"), jmat(1, "z")) - 2.7 * (tensor(jmat(1, "x"), jmat(1, "x")) + tensor(jmat(1, "y"), jmat(1, "y")))
        elif self.N == 15:
            return +3.03 * tensor(jmat(1, "z"), jmat(1 / 2, "z")) + 3.65 * (tensor(jmat(1, "x"), jmat(1 / 2, "x")) + tensor(jmat(1, "y"), jmat(1 / 2, "y")))
        if self.N == 0 or self.N is None:
            return 0
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")
        
    def _Quadrupole(self):
        """
        Get the quadrupole term

        Returns
        -------
        Quadrupole Hamiltonian : Qobj
        """
        if self.N==14:
            return - 5.01*tensor(qeye(3), jmat(1,'z')**2)
        elif self.N==15:
            return None
        elif self.N==0:
            return None
        else:
            raise ValueError(f"Invalid value for nitrogen isotope N. Expected either 14 or 15, got {self.N}.")
        
    def add_spin(self, H_spin):
        """
        Overwrites the parent class method by calling it and updating MW_H1 and RF_H1 attributes

        Parameters
        ----------
        H_spin : Qobj
            Hamiltonian of the extra spin       
        """
        super().add_spin(H_spin)

        self.MW_H1 = tensor(self.MW_H1, qeye(self.dim_add_spin))
        self.RF_H1 = tensor(self.RF_H1, qeye(self.dim_add_spin))