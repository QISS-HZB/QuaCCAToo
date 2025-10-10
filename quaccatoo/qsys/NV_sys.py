# TODO: add electric and crystal stress hamiltonians for NV

"""
This module contains NV class, which is a subclass of QSys.
"""

import warnings
from typing import Optional, Literal

import numpy as np
import scipy.constants as cte
from qutip import Qobj, basis, fock_dm, jmat, qeye, tensor

from .qsys import QSys

gamma_e = cte.value("electron gyromag. ratio in MHz/T") * 1e-3  # MHz/mT
gamma_N14 = 3.077e-3
gamma_N15 = -4.316e-3


class NV(QSys):
    """
    NV class contains attributes and methods to simulate the nitrogen vacancy center in diamond.

    Attributes
    ----------
    B0 : float
        Magnetic field
    N : 15, 14, 0 or None
        Nitrogen isotope, or 0 for no nuclear spin
    units_B0 : str
        Units of the magnetic field (T, mT or G)
    theta : float
        Angle of the magnetic field with respect to the NV axis
    phi_r : float
        Azimutal angle of the magnetic field with the NV axis
    units_angles : str
        Units of the angles (deg or rad)
    temp : float or None
        Temperature
    energy_levels : list
        List of energy levels of the Hamiltonian
    MW_freqs : numpy.ndarray
        Microwave frequencies
    RF_freqs : numpy.ndarray
        RF frequencies
    MW_h1 : Qobj
        Microwave Hamiltonian
    RF_h1 : Qobj
        RF Hamiltonian

    Methods
    -------
    _rho0_T
        Calculates the initial state of the system at low temperatures using the Boltzmann distribution
    _set_MW
        Sets the standard microwave Hamiltonian and pulse frequencies for the NV center corresponding to the electronic spin transitions
    _set_RF
        Sets the standard RF Hamiltonian and pulse frequencies for the NV center corresponding to the nuclear spin transitions
    zero_field
        Get the NV Hamiltonian term accounting for zero field splitting
    electron_zeeman
        Get the NV hamiltonian term accounting for the electron Zeeman effect
    nuclear_zeeman
        Get the NV hamiltonian term accounting for the nuclear (Nitrogen) Zeeman effect
    hyperfine_N
        Get the NV hamiltonian term accounting for the hyperfine coupling with Nitrogen
    quadrupole
        Get the quadrupole term
    add_spin
        Adds an extra spin to the NV system
    truncate
        Truncates the system to the given indexes

    Notes
    -----
    The NV class inherits the methods and attributes from the QSys class.
    """

    def __init__(
        self,
        B0: float | int,
        N: Literal[15, 14, 0, None],
        c_ops: Optional[Qobj | list[Qobj]] = None,
        units_B0: Literal["T", "mT", "G"] = "mT",
        theta: float | int = 0.0,
        phi_r: float | int = 0.0,
        units_angles: Literal["rad", "deg"] = "deg",
        temp: Optional[float | int] = None,
        units_temp: Literal["C", "K"] = "K",
        E: float | int = 0,
    ) -> None:
        """
        Constructor for the NV class.
        Takes the nitrogen isotope, the magnetic field intensity and angles with the
        quantization axis as inputs and calculates the Hamiltonian with all
        relevant attributes.

        Parameters
        ----------
        B0 : float
            Magnetic field
        N : 15 | 14 | 0 | None
            Nitrogen isotope, or 0 for no nuclear spin
        c_ops : list(Qobj)
            List of collapse operators
        units_B0 : str
            Units of the magnetic field (T, mT or G)
        theta : float | int
            Angle of the magnetic field with respect to the NV axis
        phi_r : float | int
            Angle of the magnetic field in the xy plane
        units_angles : str
            Units of the angles (deg or rad)
        temp : float | int
            Temperature
        units_temp : str
            Temperature units ('C'/'K')
        E : float | int
            Perpedicular component of the zero field splitting
        """
        if not isinstance(B0, (int, float)):
            raise TypeError(f"B0 must be a real number, got {B0}: {type(B0)}.")
        else:
            self.B0 = B0

        if units_B0 is None:
            warnings.warn(
                "No units for the magnetic field were given. The magnetic field will be considered in mT."
            )
        elif units_B0 == "T":
            self.B0 = B0 * 1e3
        elif units_B0 == "mT":
            pass
        elif units_B0 == "G":
            self.B0 = B0 * 1e-1
        else:
            raise ValueError(
                f"Invalid value for units_B0. Expected either 'G', 'mT' or 'T', got {units_B0}."
            )

        if not isinstance(theta, (int, float)) or not isinstance(phi_r, (int, float)):
            raise TypeError(
                f"Invalid type for theta or phi_r. Expected a float or int, got theta: {type(theta)}, phi_r: {type(phi_r)}."
            )
        elif units_angles == "deg":
            theta = np.deg2rad(theta)
            phi_r = np.deg2rad(phi_r)
        elif units_angles == "rad":
            pass
        else:
            raise ValueError(
                f"Invalid value for units_angles. Expected either 'deg' or 'rad', got {units_angles}."
            )

        if not isinstance(E, (int, float)):
            raise TypeError(f"E must be a real number, got {E}: {type(E)}.")
        else:
            self.E = E

        if temp is None:
            self.temp = temp
        elif isinstance(temp, (int, float)) and temp > 0:
            self.temp = temp
            if temp < 5.6 or temp > 700:
                warnings.warn(
                    "The operational temperature range for the Hamiltonian model is between 5.6 K to 700 K. Results might be inaccurate."
                )
        else:
            raise ValueError("T must be a positive real number.")

        # by default quaccatoo uses temperatures in Kelvin
        if units_temp == "K":
            pass
        elif units_temp == "C":
            self.temp += 273.15
        elif units_temp == "F":
            raise ValueError("'F' is not a valid unit for temperature, learn the metric system.")
        else:
            raise ValueError(f"Invalid value for units_temp. Expected either 'K' or 'C', got {units_temp}.")

        self.theta = theta
        self.phi_r = phi_r
        self.N = N

        # calculates the Hamiltonian for the given field and nitrogen isotope
        if N == 15:
            H0 = self.zero_field() + self.electron_zeeman() + self.hyperfine_N() + self.nuclear_zeeman()
            rho0 = tensor(fock_dm(3, 1), qeye(2) / 2)
            observable = tensor(fock_dm(3, 1), qeye(2))

        elif N == 14:
            H0 = (
                self.zero_field()
                + self.electron_zeeman()
                + self.hyperfine_N()
                + self.nuclear_zeeman()
                + self.quadrupole()
            )
            rho0 = tensor(fock_dm(3, 1), qeye(3) / 2)
            observable = tensor(fock_dm(3, 1), qeye(3))

        elif N == 0 or N is None:
            H0 = self.zero_field() + self.electron_zeeman()
            rho0 = basis(3, 1)
            observable = fock_dm(3, 1)

        else:
            raise ValueError(f"Invalid value for Nitrogen isotope. Expected either 14 or 15, got {N}.")

        super().__init__(H0, rho0, c_ops, observable, units_H0="MHz")

        if self.temp is not None:
            self._rho0_T()

        self._set_MW()
        self._set_RF()

    def _rho0_T(self) -> None:
        """
        Calculates the initial state of the system at low temperatures using the Boltzmann distribution.
        At room temperatures and moderate fields, the initial state of the nuclear spins is simply an identity matrix.

        Returns
        -------
        rho0 : Qobj
            Initial state of the system
        """
        # a loop to find the |0,1/2> and |0,-1/2> states
        max_1 = 0
        max_2 = 0
        max_3 = 0
        index_1 = None
        index_2 = None
        index_3 = None
        # iterates over all the eigenstates and find the one closest related to the |0,1/2> and |0,-1/2> states
        for idx_eig, val_eig in enumerate(self.eigenstates):
            if self.N == 15:
                proj_1 = np.abs(val_eig.overlap(basis(6, 2)))
                proj_2 = np.abs(val_eig.overlap(basis(6, 3)))

            elif self.N == 14:
                proj_1 = np.abs(val_eig.overlap(basis(9, 3)))
                proj_2 = np.abs(val_eig.overlap(basis(9, 4)))
                proj_3 = np.abs(val_eig.overlap(basis(9, 5)))
                if proj_3 > max_3:
                    # if the projection is higher than the previous maximum, update the maximum and the index
                    max_3 = proj_3
                    index_3 = idx_eig

            if proj_1 > max_1:
                max_1 = proj_1
                index_1 = idx_eig
            if proj_2 > max_2:
                max_2 = proj_2
                index_2 = idx_eig

        beta = -cte.h * 1e6 / (cte.Boltzmann * self.temp)

        if self.N == 15:
            # calculate the partition function based on the Hamiltonian eigenvalues
            Z = np.exp(beta * self.energy_levels[index_1]) + np.exp(beta * self.energy_levels[index_2])

            self.rho0 = tensor(
                fock_dm(3, 1),
                Qobj(
                    [
                        [np.exp(beta * self.energy_levels[index_1]), 0],
                        [0, np.exp(beta * self.energy_levels[index_2])],
                    ]
                )
                / Z,
            )

        elif self.N == 14:
            Z = (
                np.exp(beta * self.energy_levels[index_1])
                + np.exp(beta * self.energy_levels[index_2])
                + np.exp(beta * self.energy_levels[index_3])
            )

            self.rho0 = tensor(
                fock_dm(3, 1),
                Qobj(
                    [
                        [np.exp(beta * self.energy_levels[index_1]), 0, 0],
                        [0, np.exp(beta * self.energy_levels[index_2]), 0],
                        [0, 0, np.exp(beta * self.energy_levels[index_3])],
                    ]
                )
                / Z,
            )
        elif self.N == 0 or self.N is None:
            self.rho0 = basis(3, 1)
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _set_MW(self) -> None:
        """
        Sets the standard microwave Hamiltonian for the NV center corresponding to the electronic spin transitions.
        Sets the corresponding frequencies for microwave pulses with the transitions corresponding to the energy levels.
        """
        Rx_0 = Qobj([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        Rx_1 = Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        Ry_0 = Qobj([[1, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        Ry_1 = Qobj([[0, -1j, 0], [1j, 0, 0], [0, 0, 1]])

        if self.N == 15:
            self.MW_h1 = tensor(jmat(1, "x"), qeye(2)) * 2**0.5
            self.MW_Rx = [tensor(Rx_0, qeye(2)), tensor(Rx_1, qeye(2))]
            self.MW_Ry = [tensor(Ry_0, qeye(2)), tensor(Ry_1, qeye(2))]

            f1 = (np.sum(self.energy_levels[2:4]) - np.sum(self.energy_levels[1:2])) / 2
            f2 = (np.sum(self.energy_levels[4:6]) - np.sum(self.energy_levels[1:2])) / 2
            self.MW_freqs = np.array([f1, f2])

        elif self.N == 14:
            self.MW_h1 = tensor(jmat(1, "x"), qeye(3)) * 2**0.5
            self.MW_Rx = [tensor(Rx_0, qeye(3)), tensor(Rx_1, qeye(3))]
            self.MW_Ry = [tensor(Ry_0, qeye(3)), tensor(Ry_1, qeye(3))]

            f1 = (np.sum(self.energy_levels[3:6]) - np.sum(self.energy_levels[1:3])) / 3
            f2 = (np.sum(self.energy_levels[6:9]) - np.sum(self.energy_levels[1:3])) / 3
            self.MW_freqs = np.array([f1, f2])

        elif self.N == 0 or self.N is None:
            self.MW_h1 = tensor(jmat(1, "x")) * 2**0.5
            self.MW_Rx = [Rx_0, Rx_1]
            self.MW_Ry = [Ry_0, Ry_1]

            f1 = self.energy_levels[1]
            f2 = self.energy_levels[2]
            self.MW_freqs = np.array([f1, f2])

        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def _set_RF(self) -> None:
        """
        Sets the standard RF Hamiltonian for the NV center corresponding to the nuclear spin transitions.
        Sets the corresponding frequencies for RF pulses with the transitions corresponding to the energy levels.
        """
        Rx_0 = Qobj([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        Rx_1 = Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        Ry_0 = Qobj([[1, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        Ry_1 = Qobj([[0, -1j, 0], [1j, 0, 0], [0, 0, 1]])

        if self.N == 15:
            self.RF_h1 = tensor(qeye(3), jmat(1 / 2, "x")) * 2
            self.RF_Rx = self.RF_h1.copy()
            self.RF_Ry = tensor(qeye(3), jmat(1 / 2, "y")) * 2

            f1 = self.energy_levels[1]
            f2 = self.energy_levels[3] - self.energy_levels[2]
            f3 = self.energy_levels[5] - self.energy_levels[4]
            self.RF_freqs = np.array([f1, f2, f3])

        elif self.N == 14:
            self.RF_h1 = tensor(qeye(3), jmat(1, "x")) * 2**0.5

            Rx_0 = Qobj([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            Rx_1 = Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            Ry_0 = Qobj([[1, 0, 0], [0, 0, -1j], [0, 1j, 0]])
            Ry_1 = Qobj([[0, -1j, 0], [1j, 0, 0], [0, 0, 1]])
            self.RF_Rx = [tensor(qeye(3), Rx_0), tensor(qeye(3), Rx_1)]
            self.RF_Ry = [tensor(qeye(3), Ry_0), tensor(qeye(3), Ry_1)]

            # for the 14N isotope, the RF frequencies are more complicated as they need to respect the selection rule of Delta mI = +-1
            # the order of the ms states changes above the GSLAC
            if self.B0 <= 102.5:
                f1 = self.energy_levels[2] - self.energy_levels[1]  # 0 -> -1 at ms=0
                f2 = self.energy_levels[2]  # 0 -> +1 at ms=0
                f3 = self.energy_levels[5] - self.energy_levels[3]  # 0 -> -1 at ms=-1
                f4 = self.energy_levels[5] - self.energy_levels[4]  # 0 -> +1 at ms=-1
                f5 = self.energy_levels[8] - self.energy_levels[7]  # 0 -> -1 at ms=+1
                f6 = self.energy_levels[8] - self.energy_levels[6]  # 0 -> +1 at ms=-1
            else:
                f1 = self.energy_levels[2]  # 0 -> -1 at ms=-1
                f2 = self.energy_levels[2] - self.energy_levels[1]  # 0 -> +1 at ms=-1
                f3 = self.energy_levels[5] - self.energy_levels[4]  # 0 -> -1 at ms=0
                f4 = self.energy_levels[5] - self.energy_levels[3]  # 0 -> +1 at ms=0
                f5 = self.energy_levels[8] - self.energy_levels[7]  # 0 -> -1 at ms=+1
                f6 = self.energy_levels[8] - self.energy_levels[6]  # 0 -> +1 at ms=-1

            self.RF_freqs = np.array([f1, f2, f3, f4, f5, f6])

        elif self.N == 0 or self.N is None:
            self.RF_h1 = qeye(3)
            self.RF_Rx = qeye(3)
            self.RF_Ry = qeye(3)

        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def zero_field(self) -> Qobj:
        """Get the NV Hamiltonian term accounting for the zero field splitting.
        If the temperature attribute is set to a value below 295 K, the D parameter is calculated using a
        5th order polynomial function from X. D. Chen et al. Appl. Phys. Lett. 99, 161903 (2011).
        Otherwise, if the temperature is above 295 K, a 3rd order polynomial function from
        D. M. Toyli et al. Phys. Rev. X 2, 031001 (2012) is used.

        Returns
        -------
        Zero Field Hamiltonian : Qobj
        """
        if self.temp is not None:
            if self.temp <= 295:
                D = (
                    2.87771
                    - 4.625e-6 * self.temp
                    + 1.067e-7 * self.temp**2
                    - 9.325e-10 * self.temp**3
                    + 1.739e-12 * self.temp**4
                    - 1.838e-15 * self.temp**5
                ) * 1e3
            else:
                D = (2.8697 + 9.7e-5 * self.temp - 3.7e-7 * self.temp**2 + 1.7e-10 * self.temp**3) * 1e3
        else:
            D = 2.87e3

        H_zf = D * jmat(1, "z") ** 2 + self.E * (jmat(1, "x") ** 2 - jmat(1, "y") ** 2)

        if self.N == 14:
            return tensor(H_zf, qeye(3))

        elif self.N == 15:
            return tensor(H_zf, qeye(2))

        elif self.N == 0 or self.N is None:
            return H_zf
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def electron_zeeman(self) -> Qobj:
        """
        Get the NV hamiltonian term accounting for the electron Zeeman effect.

        Returns
        -------
        Electron Zeeman Hamiltonian : Qobj
        """
        H_ez = (
            gamma_e
            * self.B0
            * (
                np.cos(self.theta) * jmat(1, "z")
                + np.sin(self.theta) * np.cos(self.phi_r) * jmat(1, "x")
                + np.sin(self.theta) * np.sin(self.phi_r) * jmat(1, "y")
            )
        )

        if self.N == 14:
            return tensor(H_ez, qeye(3))

        elif self.N == 15:
            return tensor(H_ez, qeye(2))

        elif self.N == 0 or self.N is None:
            return H_ez
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def nuclear_zeeman(self) -> Qobj | int:
        """
        Get the NV hamiltonian term accounting for the nuclear (Nitrogen) Zeeman effect.

        Returns
        -------
        Nuclear Zeeman Hamiltonian : Qobj
        """

        if self.N == 14:
            return -tensor(
                qeye(3),
                gamma_N14
                * self.B0
                * (
                    np.cos(self.theta) * jmat(1, "z")
                    + np.sin(self.theta) * np.cos(self.phi_r) * jmat(1, "x")
                    + np.sin(self.theta) * np.sin(self.phi_r) * jmat(1, "y")
                ),
            )
        elif self.N == 15:
            return -tensor(
                qeye(3),
                gamma_N15
                * self.B0
                * (
                    np.cos(self.theta) * jmat(1 / 2, "z")
                    + np.sin(self.theta) * np.cos(self.phi_r) * jmat(1 / 2, "x")
                    + np.sin(self.theta) * np.sin(self.phi_r) * jmat(1 / 2, "y")
                ),
            )
        elif self.N == 0 or self.N is None:
            return 0
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def hyperfine_N(self) -> Qobj | int:
        """
        Get the NV hamiltonian term accounting for the hyperfine coupling with Nitrogen.

        Returns
        -------
        Hyperfine Hamiltonian : Qobj
        """
        if self.N == 14:
            return -2.14 * tensor(jmat(1, "z"), jmat(1, "z")) - 2.7 * (
                tensor(jmat(1, "x"), jmat(1, "x")) + tensor(jmat(1, "y"), jmat(1, "y"))
            )
        elif self.N == 15:
            return +3.03 * tensor(jmat(1, "z"), jmat(1 / 2, "z")) + 3.65 * (
                tensor(jmat(1, "x"), jmat(1 / 2, "x")) + tensor(jmat(1, "y"), jmat(1 / 2, "y"))
            )
        if self.N == 0 or self.N is None:
            return 0
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {self.N}.")

    def quadrupole(self) -> Qobj | int:
        """
        Get the quadrupole term

        Returns
        -------
        quadrupole Hamiltonian : Qobj
        """
        if self.N == 14:
            return -5.01 * tensor(qeye(3), jmat(1, "z") ** 2)
        elif self.N == 15 or self.N == 0:
            return 0
        else:
            raise ValueError(
                f"Invalid value for nitrogen isotope N. Expected either 14 or 15, got {self.N}."
            )

    def add_spin(self, H_spin: Qobj) -> None:
        """
        Overwrites the parent class method by calling it and updating MW_h1 and RF_h1 attributes

        Parameters
        ----------
        H_spin : Qobj
            Hamiltonian of the extra spin
        """
        super().add_spin(H_spin)

        self.MW_h1 = tensor(self.MW_h1, qeye(self.dim_add_spin))
        self.RF_h1 = tensor(self.RF_h1, qeye(self.dim_add_spin))

    def truncate(self, mS: Literal[1, 0, -1, None] = None, mI: Literal[1, 0, -1, None] = None) -> None:
        """
        Overwrites the parent class method by calling it and updating MW_h1 and RF_h1 attributes.
        The indexes to be removed are calculated according to the mS and mI parameters.

        Parameters
        ----------
        mS : '1', '0', '-1', None
            Electronic level to be excluded
        mI : '1', '0', '-1', None
            Electronic level to be excluded
        """
        if mS is None and mI is None:
            warnings.warn("No mS or mI parameters were given. The system will not be truncated.")
            return
        if mS not in {1, 0, -1, None}:
            raise ValueError(f"Invalid value for mS. Expected either 1, 0 or -1, got {mS}.")
        if mI == 1 / 2 or mI == -1 / 2:
            warnings.warn(
                "mI should be either 1, 0 or -1 for the NV system. The 15N isotope is already a two-level system and can't be truncated."
            )
        elif mI not in {1, 0, -1, None}:
            raise ValueError(f"Invalid value for mI. Expected either 1, 0 or -1, got {mI}.")

        # set the indexes to be removed and the dimensions of the new objects according to the mS and mI parameters
        if self.N == 0 or self.N is None:
            if mS == 1:
                indexes, dims = [0], [2]
            elif mS == 0:
                indexes, dims = [1], [2]
            elif mS == -1:
                indexes, dims = [2], [2]

        elif self.N == 15:
            if mS == 1:
                indexes, dims = [0, 1], [2, 2]
            elif mS == 0:
                indexes, dims = [2, 3], [2, 2]
            elif mS == -1:
                indexes, dims = [4, 5], [2, 2]

        elif self.N == 14:
            if mS == 1:
                indexes, dims = [0, 1, 2], [2]
            elif mS == 0:
                indexes, dims = [3, 4, 5], [2]
            elif mS == -1:
                indexes, dims = [6, 7, 8], [2]
            else:
                dims = [3]

            if mI == 1:
                indexes.extend([0, 3, 6])
                dims.append(2)
            elif mI == 0:
                indexes.extend([1, 4, 7])
                dims.append(2)
            elif mI == -1:
                indexes.extend([2, 5, 8])
                dims.append(2)
            else:
                dims.append(3)

        indexes = sorted(set(indexes))
        super().truncate(indexes)

        self.MW_h1 = Qobj(np.delete(np.delete(self.MW_h1.full(), indexes, axis=0), indexes, axis=1))
        self.RF_h1 = Qobj(np.delete(np.delete(self.RF_h1.full(), indexes, axis=0), indexes, axis=1))

        # corrrect the dimensions of the objects
        self.H0.dims = [dims, dims]
        self.MW_h1.dims = [dims, dims]
        self.RF_h1.dims = [dims, dims]

        if self.observable is not None:
            if isinstance(self.observable, Qobj):
                self.observable.dims = [dims, dims]
            elif isinstance(self.observable, list):
                for obs in self.observable:
                    obs.dims = [dims, dims]

        if self.rho0 is not None:
            if self.rho0.isket:
                if len(dims) == 1:
                    self.rho0.dims = [dims, [1]]
                elif len(dims) == 2:
                    self.rho0.dims = [dims, [1, 1]]
            else:
                self.rho0.dims = [dims, dims]

        if self.c_ops is not None:
            if isinstance(self.c_ops, Qobj):
                self.c_ops.dims = [dims, dims]
            elif isinstance(self.c_ops, list):
                for c_op in self.c_ops:
                    c_op.dims = [dims, dims]
