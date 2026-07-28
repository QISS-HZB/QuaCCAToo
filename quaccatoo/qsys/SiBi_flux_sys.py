"""
This module contains a class for modelling a Bismuth atom coupled to a flux superconducting qubit being a subclass of QSys.

This system is the same as used in Tikai Chang et al. 'Strong coupling of a superconducting flux qubit to single bismuth donors', Nature Communications 10.1038/s41467-025-64757-5.
"""

from typing import Literal

import numpy as np
from qutip import Qobj, jmat, qeye, tensor

from .constants import A_Bi, gamma_Bi, gamma_e
from .qsys import QSys

__all__ = ["SiBiFlux"]

####################################################################################################


class SiBiFlux(QSys):
    """
    Class for modelling a Bismuth atom coupled to a flux superconducting qubit being a subclass of QSys.

    This system is the same as used in Tikai Chang et al. 'Strong coupling of a superconducting flux qubit to single bismuth donors', Nature Communications 10.1038/s41467-025-64757-5.

    Attributes
    ----------
    B0 : float
        Intensity of the static magnetic field. Only set if spin is True.
    units_B0 : str
        Units of the static magnetic field (T, mT or G). Only set if spin is True.
    theta : float
        Polar angle of the static magnetic field vector with respect to the z axis.
        Only set if spin is True.
    phi_r : float
        Azimuthal angle of the static magnetic field vector within the xy plane.
        Only set if spin is True.
    units_angles : str
        Units of the angles (deg or rad). Only set if spin is True.
    N : bool
        Whether the bismuth nuclear spin (I=9/2) is included in the Hilbert space.
        Only set if spin is True.

    Methods
    -------
    _get_H_flux
        Calculates and returns the superconducting flux qubit Hamiltonian.
    electron_zeeman
        Calculates and returns the Hamiltonian of the electron Zeeman interaction.
    nuclear_zeeman
        Calculates and returns the Hamiltonian of the nuclear Zeeman interaction.
    hyperfine
        Calculates and returns the Hamiltonian of the hyperfine interaction between
        the donor electron and the bismuth nucleus.

    Notes
    -----
    The SiBiFlux class inherits the methods and attributes from the QSys class.

    The composite Hilbert space is ordered as flux qubit (dim 2), donor electron spin
    (dim 2) and, if N is True, bismuth nuclear spin (dim 10). Subsystems absent from
    the model are omitted from the tensor product instead of being traced over, such
    that the dimension of H0 depends on the flux, spin and N flags.
    """

    def __init__(
        self,
        *,
        rho0: Qobj | None,
        c_ops: Qobj | list[Qobj] | None = None,
        observable: Qobj | list[Qobj] | None,
        flux: bool = True,
        spin: bool = True,
        N: bool = False,
        Delta: float = 0.0,
        epsilon: float = 0.0,
        g: float = 0.0,
        B0: float = 0.0,
        units_B0: Literal["T", "mT", "G"] = "mT",
        theta: float = 0.0,
        phi_r: float = 0.0,
        units_angles: Literal["rad", "deg"] = "deg",
    ) -> None:
        """
        Constructor for the SiBiFlux class.
        Takes the flux qubit parameters and/or the spin.

        Parameters
        ----------
        rho0 : Qobj or None
            Initial state of the system, passed to the QSys constructor.
        c_ops : Qobj or list[Qobj] or None
            Collapse operators for the Lindblad master equation, passed to the QSys constructor.
        observable : Qobj or list[Qobj] or None
            Observable or list of observables to be measured, passed to the QSys constructor.
        flux : bool
            Whether the superconducting flux qubit is included in the system.
        spin : bool
            Whether the bismuth donor spin is included in the system.
        N : bool
            Whether the bismuth nuclear spin (I=9/2) is included, adding the nuclear
            Zeeman and hyperfine terms to the Hamiltonian. Only used if spin is True.
        Delta : float
            Coefficient of the sigma_z term of the flux qubit Hamiltonian, in MHz.
            Only used if flux is True.
        epsilon : float
            Coefficient of the sigma_x term of the flux qubit Hamiltonian, in MHz.
            Only used if flux is True.
        g : float
            Transverse coupling strength between the flux qubit and the donor electron
            spin, in MHz. Only used if both flux and spin are True.
        B0 : float
            Intensity of the static magnetic field. Only used if spin is True.
        units_B0 : str
            Units of the static magnetic field (T, mT or G). Only used if spin is True.
        theta : float
            Polar angle of the static magnetic field vector with respect to the z axis.
            Only used if spin is True.
        phi_r : float
            Azimuthal angle of the static magnetic field vector within the xy plane.
            Only used if spin is True.
        units_angles : str
            Units of the angles (deg or rad). Only used if spin is True.
        """
        for name, val in (("flux", flux), ("spin", spin), ("N", N)):
            if not isinstance(val, bool):
                raise TypeError(f"{name} must be a boolean, got {val}: {type(val)}.")

        H_flux = None
        if flux:
            if not isinstance(Delta, (int, float)):
                raise TypeError(f"Delta must be a real number, got {Delta}: {type(Delta)}.")

            if not isinstance(epsilon, (int, float)):
                raise TypeError(f"E must be a real number, got {epsilon}: {type(epsilon)}.")

            H_flux = self._get_H_flux(Delta, epsilon)

        H_SiBi = None
        if spin:
            self.B0, self.units_B0 = self._check_B0(B0, units_B0)
            self.theta, self.phi_r, self.units_angles = self._check_angles(
                theta, phi_r, units_angles
            )

            self.N = N
            if self.N:
                H_SiBi = self.electron_zeeman() + self.nuclear_zeeman() + self.hyperfine()

            elif not self.N:
                H_SiBi = self.electron_zeeman()

            else:
                raise TypeError(f"n must be a boolean, got {self.N}: {type(self.N)}.")

        # builds the Hamiltonian
        if flux and spin:
            if not isinstance(g, (int, float)):
                raise TypeError(f"g must be a real number, got {g}: {type(g)}.")

            if N:
                H_coupling = g * tensor(jmat(1 / 2, "x"), jmat(1 / 2, "x"), qeye(10))
                H0 = tensor(H_flux, qeye(2), qeye(10)) + tensor(qeye(2), H_SiBi) + H_coupling
            else:
                H_coupling = g * tensor(jmat(1 / 2, "x"), jmat(1 / 2, "x"))
                H0 = tensor(H_flux, qeye(2)) + tensor(qeye(2), H_SiBi) + H_coupling

        elif flux:
            H0 = H_flux
        elif spin:
            H0 = H_SiBi
        else:
            raise ValueError("Both flux and spin are False. At least one must be set to true")

        super().__init__(H0, rho0=rho0, c_ops=c_ops, observable=observable, units_H0="MHz")

    def _get_H_flux(self, Delta: float, epsilon: float) -> Qobj:
        """
        Calculates and returns the superconducting flux qubit hamiltonian

        Parameters
        ----------
        Delta : float
            Coefficient of the sigma_z term, in MHz.
        epsilon : float
            Coefficient of the sigma_x term, in MHz.

        Returns
        -------
        H_flux : Qobj
            Flux qubit Hamiltonian
        """
        return Delta * jmat(1 / 2, "z") + epsilon * jmat(1 / 2, "x")

    def electron_zeeman(
        self,
    ) -> Qobj:
        """
        Calculates and returns the hamiltonian for the electron Zeeman interaction.
        If the nuclear spin is present, it performs a tensor product with nuclear spin space.

        Returns
        -------
        H_ez : Qobj
            Electron Zeeman Hamiltonian
        """
        H_ez = (
            gamma_e
            * self.B0
            * (
                np.cos(self.theta) * jmat(1 / 2, "z")
                + np.sin(self.theta) * np.cos(self.phi_r) * jmat(1 / 2, "x")
                + np.sin(self.theta) * np.sin(self.phi_r) * jmat(1 / 2, "y")
            )
        )
        if self.N:
            return tensor(H_ez, qeye(10))
        else:
            return H_ez

    def nuclear_zeeman(
        self,
    ) -> Qobj:
        """
        Calculates the nuclear Zeeman Hamiltonian of the bismuth nuclear Zeeman interaction, with the field orientation set by theta and phi_r, embedded in the electron-nucleus product space through a tensor product with the identity on the electron.

        Returns
        -------
        H_nz : Qobj
            Nuclear Zeeman Hamiltonian
        """
        H_nz = (
            -gamma_Bi
            * self.B0
            * (
                np.cos(self.theta) * jmat(9 / 2, "z")
                + np.sin(self.theta) * np.cos(self.phi_r) * jmat(9 / 2, "x")
                + np.sin(self.theta) * np.sin(self.phi_r) * jmat(9 / 2, "y")
            )
        )
        return tensor(qeye(2), H_nz)

    def hyperfine(
        self,
    ) -> Qobj:
        """
        Calculates the hyperine Hamiltonian of the isotropic contact hyperfine interaction between the donor electron and the bismuth nucleus, acting on the electron-nucleus product space.
        The coupling constant A_Bi is 1475.4 MHz, giving the well known 7.377 GHz zero field splitting between the F=4 and F=5 manifolds.

        Returns
        -------
        H_hf : Qobj
            Hyperfine Hamiltonina
        """
        H_hf = A_Bi * (
            tensor(jmat(1 / 2, "x"), jmat(9 / 2, "x"))
            + tensor(jmat(1 / 2, "y"), jmat(9 / 2, "y"))
            + tensor(jmat(1 / 2, "z"), jmat(9 / 2, "z"))
        )
        return H_hf
