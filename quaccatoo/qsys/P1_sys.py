"""
This module contains P1 class for simulating P1 centers in diamond,
being a subclass of QSys.
"""

import warnings
from typing import Optional, Literal

import numpy as np
import scipy.constants as cte
from qutip import Qobj, jmat, qeye, tensor

from .qsys import QSys

gamma_e = cte.value("electron gyromag. ratio in MHz/T") * 1e-3  # MHz/mT
gamma_N14 = 3.077e-3
gamma_N15 = -4.316e-3

class P1(QSys):
    """
    P1 class contains attributes and methods to simulate the neutral substitutional nitrogen center in diamond.
    Due to Jahn-Teller effect this defect has a C_{3v} point symmetry, which means there are four isolated orientation families of P1 centers in diamond. In simulations the lab reference frame is chosen in such a way that it is a principal axis system (PAS) for one of the four families (rotation index 0). Hamiltonians of other orientation families are obtained by the transition from PAS to the corresponding orientation using rotation matrices. 

    Attributes
    ----------
    B0 : float
        Magnetic field intensity.
    rot_index : int
        Rotation index, integer between 0 and 3.
    R : list
        List of rotation matrices to go from the PAS (principal axis system) frame to the lab frame.
    N : int
        Nitrogen isotope, or 0 for no nuclear spin.
    units_B0 : str
        Units of the magnetic field (T, mT or G)
    theta : float | int
        Angle of the static magnetic field vector with respect to the z axis
    phi_r : float | int
        Azimutal angle of the static magnetic field vector within the xy plane
    theta_1 : float | int
        Angle of the MW magnetic field vector with respect to the z axis
    phi_r_1 : float | int
        Azimutal angle of the MW magnetic field vector within the xy plane
    units_angles : str
        Units of the angles (deg or rad)
    B0_vector : np.ndarray
        Unit vector of the static magnetic field in the lab frame
    B1_vector : np.ndarray
        Unit vector of the MW magnetic field in the lab frame 
    h1 : Qobj
        Standard microwave Hamiltonian for the P1 center corresponding to the electronic spin transitions.
    eigenstates : np.ndarray
        Array containing the eigenstates of the Hamiltonian.
    dim_add_spin : int
        Dimension of the added spin, if any.   

    Methods
    -------
    _rot_pas_to_lab
        Defines the rotation matrices to go from the PAS (principal axis system) frame to the lab frame.
    electron_zeeman
        Calculates the electron Zeeman Hamiltonian term.
    hyperfine
        Calculates the hyperfine Hamiltonian term.
    quadrupole
        Calculates the quadrupole Hamiltonian term.
    nuclear_zeeman
        Calculates the nuclear Zeeman Hamiltonian term.
    _set_h1
        Sets the standard microwave Hamiltonian for the P1 center corresponding to the electronic spin transitions.
    add_spin
        Adds an extra spin to the system and updates the microwave Hamiltonian accordingly.

    Notes
    -----
    The P1 class inherits the methods and attributes from the QSys class.
    """
    def __init__(
        self,
        B0 : float | int,
        rot_index : Literal[0, 1, 2, 3],
        N : Literal[15, 14, 0, None] = None,
        rho0 : Optional[Qobj | np.ndarray | int] = None,
        c_ops : Optional[Qobj | list[Qobj]] = None,
        units_B0 : Literal['T', 'mT', 'G'] = 'mT',
        theta : float | int = 0.0,
        phi_r : float | int = 0.0,
        theta_1 : float | int = 90.0,
        phi_r_1 : float | int = 0.0,
        units_angles : Literal['rad', 'deg'] = "deg",
        observable : Optional[Qobj | list[Qobj]] =None
        ) -> None:
        """
        Constructor for the P1 class.
        Takes the nitrogen isotope, the rotation index, the magnetic field intensity as inputs and calculates the Hamiltonian with all relevant attributes.

        Parameters
        ----------
        B0 : float
            Magnetic field intensity.
        rot_index : int
            Rotation index, integer between 0 and 3.
        N : 15 | 14 | 0 | None
            Nitrogen isotope, or 0 for no nuclear spin
        rho0 : Qobj | array | int
            Initial state of the system. Can be a Qobj, an array or an index number indicating the system eigenstates
        c_ops : Qobj | list(Qobj)
            List of collapse operators
        units_B0 : str
            Units of the magnetic field (T, mT or G)
        theta : float | int
            Angle of the static magnetic field vector with respect to the z axis
        phi_r : float | int
            Azimutal angle of the static magnetic field vector within the xy plane
        theta_1 : float | int
            Angle of the MW magnetic field vector with respect to the z axis
        phi_r_1 : float | int
            Azimutal angle of the MW magnetic field vector within the xy plane
        units_angles : str
            Units of the angles (deg or rad)
        B0_vector : np.ndarray
            Unit vector of the static magnetic field in the lab frame
        B1_vector : np.ndarray
            Unit vector of the MW magnetic field in the lab frame 
        observable : Qobj | list(Qobj)
            Observable to be measured                    
        """
        if not isinstance(B0, (int, float)):
            raise TypeError(f"B0 must be a real number, got {B0}: {type(B0)}.")
        else:
            self.B0 = B0

        if units_B0 is None:
            warnings.warn("No units for the magnetic field were given. The magnetic field will be considered in mT.")
        elif units_B0 == "T":
            B0 = B0 * 1e3
        elif units_B0 == "mT":
            pass
        elif units_B0 == "G":
            B0 = B0 * 1e-3
        else:
            raise ValueError(f"Invalid value for units_B0. Expected either 'G', 'mT' or 'T', got {units_B0}.")
        
        if not isinstance(theta, (int, float)) or not isinstance(phi_r, (int, float)) or not isinstance(theta_1, (int, float)) or not isinstance(phi_r_1, (int, float)):
            raise TypeError(
                f"Invalid type for theta or phi_r. Expected a float or int, got theta: {type(theta)}, phi_r: {type(phi_r)}, theta_1: {type(theta)}, phi_r_1: {type(phi_r)}."
            )
        elif units_angles == "deg":
            theta = np.deg2rad(theta)
            phi_r = np.deg2rad(phi_r)
            theta_1 = np.deg2rad(theta_1)
            phi_r_1 = np.deg2rad(phi_r_1)
        elif units_angles == "rad":
            pass
        else:
            raise ValueError(
                f"Invalid value for units_angles. Expected either 'deg' or 'rad', got {units_angles}."
            )

        self.B0_vector = np.array([[np.sin(theta)*np.cos(phi_r), np.sin(theta)*np.sin(phi_r), np.cos(theta)]])
        self.B1_vector = np.array([[np.sin(theta_1)*np.cos(phi_r_1), np.sin(theta_1)*np.sin(phi_r_1), np.cos(theta_1)]])

        if rot_index in range(4):
            self.rot_index = rot_index
        else:
            raise ValueError(f"Invalid value for rotation index r. Expected an integer between 0 and 3, got {rot_index}.")
        
        self.theta = theta
        self.phi_r = phi_r
        self.theta_1 = theta_1
        self.phi_r_1 = phi_r_1
        self._rot_pas_to_lab()
        self.N = N

        if N == 14:    
            H0 = self.electron_zeeman()+ self.hyperfine() + self.quadrupole() + self.nuclear_zeeman() 
        elif N == 15:
            H0 = self.electron_zeeman()+ self.hyperfine() + self.nuclear_zeeman()
        elif N == 0 or N is None:
            H0 = self.electron_zeeman()
        else:
            raise ValueError(f"Invalid value for nitrogen isotope N. Expected either 14 or 15, got {N}.")
        
        self.eigenstates = np.array([psi * psi.dag() for psi in H0.eigenstates()[1]])
        
        if observable is None:
            observable = None

        elif isinstance(observable, (list, np.ndarray)) and all(obs in range(len(self.eigenstates)+1) for obs in observable):
            observable = [self.eigenstates[obs] for obs in observable]

        elif observable in range(len(self.eigenstates)+1):
            observable = self.eigenstates[observable]
            if rho0 is None:
                rho0 = observable

        elif isinstance(observable, Qobj) and observable.shape == H0.shape:
            observable = observable

        elif isinstance(observable, (list, np.ndarray)) and all(isinstance(obs, Qobj) and obs.shape == H0.shape for obs in observable):
            observable = observable
        
        else:
            raise ValueError("Invalid value for observable. Expected a Qobj or a list of Qobj of the same dimensions as H0 and rho0.")

        super().__init__(H0, rho0, c_ops, observable, units_H0="MHz")

        self._set_h1()
    
    def _rot_pas_to_lab(
        self
        ) -> None:
        """
        Defines the rotation matrices to go from the PAS (principal axis system)
        frame to the lab frame.
        Start by defining the basis vectors of the PAS frame in the lab frame.
        Then generates the R list containing the rotation matrices.
        """
        basis = []
        basis.append(
            [1/np.sqrt(6)*np.array([[1, 1, -2]]).T,
             1/np.sqrt(2)*np.array([[1, -1, 0]]).T,
             1/np.sqrt(3)*np.array([[-1, -1, -1]]).T]
            )
        basis.append(
            [1/np.sqrt(6)*np.array([[2, 1, 1]]).T,
             1/np.sqrt(2)*np.array([[0, 1, -1]]).T,
             1/np.sqrt(3)*np.array([[-1, 1, 1]]).T]
             )
        basis.append(
            [1/np.sqrt(6)*np.array([[1, 2, 1]]).T,
             1/np.sqrt(2)*np.array([[-1, 0, 1]]).T,
             1/np.sqrt(3)*np.array([[1, -1, 1]]).T]
             )
        basis.append(
            [1/np.sqrt(6)*np.array([[1, 1, 2]]).T,
             1/np.sqrt(2)*np.array([[1, -1, 0]]).T,
             1/np.sqrt(3)*np.array([[1, 1, -1]]).T]
             )

        R_1 = np.c_[basis[0][0], basis[0][1], basis[0][2]]
        R_2 = np.c_[basis[1][0], basis[1][1], basis[1][2]]
        R_3 = np.c_[basis[2][0], basis[2][1], basis[2][2]]
        R_4 = np.c_[basis[3][0], basis[3][1], basis[3][2]]

        R_11 = R_1.T @ R_1
        R_12 = R_1.T @ R_2
        R_13 = R_1.T @ R_3
        R_14 = R_1.T @ R_4

        self.R = [R_11, R_12, R_13, R_14]

    def electron_zeeman(
        self
        ) -> Qobj:
        """
        Electron Zeeman Hamiltonian term,
        rotated according to the rotation index r.

        Returns
        -------
        Zeeman Hamiltonian : Qobj
        """
        H_ez = gamma_e*self.B0*(
            (self.R[self.rot_index] @ self.B0_vector.T)[0][0]*jmat(1/2, 'x') +
            (self.R[self.rot_index] @ self.B0_vector.T)[1][0]*jmat(1/2, 'y') +
            (self.R[self.rot_index] @ self.B0_vector.T)[2][0]*jmat(1/2, 'z')
        )

        if self.N == 14:
            return tensor(H_ez, qeye(3))
        
        elif self.N == 15:
            return tensor(H_ez, qeye(2))

        elif self.N == 0 or self.N is None:
            return H_ez
        
        else:
            raise ValueError(f"Invalid value for nitrogen isotope N. Expected either 14 or 15, got {self.N}.")

    def hyperfine(
        self
        ) -> Qobj:
        """
        Get the hyperfine term

        Returns
        -------
        Hyperfine Hamiltonian : Qobj
        """
        if self.N == 14:
            return (114.03*tensor(jmat(1/2,'z'), jmat(1,'z'))
                    + 81.32*(tensor(jmat(1/2,'x'), jmat(1,'x'))
                          + tensor(jmat(1/2,'y'), jmat(1,'y'))))
        elif self.N == 15:
            return (-159.73*tensor(jmat(1/2,'z'), jmat(1/2,'z'))
                    + -113.84*(tensor(jmat(1/2,'x'), jmat(1/2,'x'))
                          + tensor(jmat(1/2,'y'), jmat(1/2,'y'))))
        else:
            raise ValueError(f"Invalid value for nitrogen isotope N. Expected either 14 or 15, got {self.N}.")
    
    def quadrupole(
        self
        ) -> Qobj|int:
        """
        Get the quadrupole term

        Returns
        -------
        Quadrupole Hamiltonian : Qobj
        """
        if self.N == 14:
            return -3.97*tensor(qeye(2), jmat(1,'z')**2)
        elif self.N == 15:
            return 0
        else:
            raise ValueError(f"Invalid value for nitrogen isotope N. Expected either 14 or 15, got {self.N}.")
    
    def nuclear_zeeman(
        self
        ) -> Qobj | None:
        """
        Nuclear Zeeman Hamiltonian term

        Returns
        -------
        Zeeman Hamiltonian : Qobj
        """
        if self.N == 14:
            return -gamma_N14*self.B0*tensor(
                qeye(2),
                (self.R[self.rot_index] @ self.B0_vector.T)[0][0]*jmat(1, 'x') +
                (self.R[self.rot_index] @ self.B0_vector.T)[1][0]*jmat(1, 'y') +
                (self.R[self.rot_index] @ self.B0_vector.T)[2][0]*jmat(1, 'z')
                )
        
        elif self.N == 15:
            return -gamma_N15*self.B0*tensor(
                qeye(2),
                (self.R[self.rot_index] @ self.B0_vector.T)[0][0]*jmat(1/2, 'x') +
                (self.R[self.rot_index] @ self.B0_vector.T)[1][0]*jmat(1/2, 'y') +
                (self.R[self.rot_index] @ self.B0_vector.T)[2][0]*jmat(1/2, 'z')
                )
        
    def _set_h1(
        self
        ) -> None:
        """
        Sets the standard control Hamiltonian for the P1 center corresponding to the electronic spin transitions.
        """
        h1_e = 2*(
            (self.R[self.rot_index] @ self.B1_vector.T)[0][0]*jmat(1/2, 'x') +
            (self.R[self.rot_index] @ self.B1_vector.T)[1][0]*jmat(1/2, 'y') +
            (self.R[self.rot_index] @ self.B1_vector.T)[2][0]*jmat(1/2, 'z')
            )

        if self.N == 15:
            self.h1 = tensor(h1_e, qeye(2))

        elif self.N == 14:
             self.h1 = tensor(h1_e, qeye(3))

        elif self.N == 0 or self.N is None:
            self.h1 = h1_e

        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")
        
    def add_spin(
        self,
        H_spin : Qobj
        ) -> None:
        """
        Overwrites the parent class method by calling it and updating the h1 attribute

        Parameters
        ----------
        H_spin : Qobj
            Hamiltonian of the extra spin
        """
        super().add_spin(H_spin)

        self.h1 = tensor(self.h1, qeye(self.dim_add_spin))
