"""
This module contains dynamical decoupling pulse sequences, used in quantum sensing and for extending coherence of quantum systems.
"""

import warnings
from collections.abc import Callable

import numpy as np
from qutip import Qobj

from ..qsys.qsys import QSys
from .pulse_shapes import square_pulse
from .pulsed_sim import PulsedSim

__all__ = ["CPMG", "XY", "XY8"]

####################################################################################################


class CPMG(PulsedSim):
    """
    This class contains a Carr-Purcell-Meiboom-Gill sequence used in quantum sensing experiments.

    The CPMG sequence consists of a series of pi pulses and free evolution times,
    such that these periodicals inversions will cancel out oscillating noises except for frequencies corresponding to the pulse separation.

    Methods
    -------
    CPMG_sequence :
        defines the Carr-Purcell-Meiboom-Gill sequence for a given free evolution time tau and the set of attributes defined in the constructor,
        returning the final state. The sequence is to be called by the parallel_map method of QuTip.
    _get_pulse_profiles :
        Generates the pulse profiles for the CPMG sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
    plot_pulses :
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the CPMG sequence for a given tau and then plot them.

    Notes
    -----
    The CPMG sequence inherits the methods and attributes from the PulsedSim class.
    """

    # the sequence opens and closes with tau/2 around the pi pulses, therefore
    # tau must be at least twice the pi_pulse_duration to avoid negative free evolutions
    _min_tau_factor = 2

    def __init__(
        self,
        free_duration: np.ndarray | list[float],
        *,
        system: QSys,
        M: int,
        pi_pulse_duration: float,
        h1: Qobj | list[Qobj] | None = None,
        Rx: Qobj | None = None,
        Ry: Qobj | None = None,
        H2: tuple[Qobj, Callable] | None = None,
        projection_pulse: bool = True,
        pulse_shape: Callable = square_pulse,
        pulse_params: dict[str, float] | None = None,
        time_steps: int = 100,
        options: dict | None = None,
    ) -> None:
        """
        Class constructor for the Carr-Purcell-Meiboom-Gill sequence

        Parameters
        ----------
        free_duration : numpy array
            Time array for the simulation representing the free evolution time to be used as the variable attribute for the simulation
        system : QSys
            Quantum system object containing the initial state, internal time independent Hamiltonian and collapse operators
        M : int
            Order of the CPMG sequence
        pi_pulse_duration : float, int or 0
            Duration of the pi pulse. If set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        h1 : Qobj or list of Qobj
            Control Hamiltonian of the system.
        Rx : Qobj or None
            Rotation operator around the x-axis, used only if the pi_pulse_duration is set to 0.
        Ry : Qobj or None
            Rotation operator around the y-axis, used only if the pi_pulse_duration is set to 0.
        H2 : list(Qobj or function)
            Time dependent sensing Hamiltonian of the system
        projection_pulse : bool
            Boolean to determine if a final pi/2 pulse is to be included in order to project the measurement into the Sz basis
        pulse_shape : FunctionType or list(FunctionType)
            Pulse shape function or list of pulse shape functions representing the time modulation of h1
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions
        time_steps : int
            Number of time steps in the pulses for the simulation
        """
        super().__init__(system, H2=H2)
        self.sequence = self.CPMG_sequence
        self._check_attr_predef_seqs(
            h1,
            Rx,
            Ry,
            pulse_shape,
            pulse_params,
            options,
            time_steps,
            free_duration,
            pi_pulse_duration,
            M,
            projection_pulse,
        )

        base_pulse = self.pulse_params.copy()
        self.pulse_params = [{**base_pulse, "phi_t": 0}, {**base_pulse, "phi_t": -np.pi / 2}]

    def CPMG_sequence(self, tau: float) -> Qobj:
        """
        Defines the CPMG sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of an initial pi/2 pulse, and M pi-pulses separated by free evolution time tau.
        If projection_pulse is True, the sequence will include a final pi/2 pulse on Y axis to project the measurement into the Sz basis.
        If the pi_pulse_duration is set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time

        Returns
        -------
        rho : Qobj
            Final state
        """
        self._reset_sequence()

        if self.pi_pulse_duration == 0:
            # initial pi/2 pulse on Y
            self._delta_pulse(self.Ry_half)
            self._free_evolution(tau / 2, self.options)

            # repeat M times the pi pulse and free evolution of tau
            for idx_M in range(self.M):
                self._delta_pulse(self.Rx)
                if idx_M != self.M - 1:
                    self._free_evolution(tau, self.options)

            self._free_evolution(tau / 2, self.options)

            if self.projection_pulse:
                # final pi/2 pulse on Y
                self._delta_pulse(self.Ry_half)

        else:
            ps = tau - self.pi_pulse_duration
            # initial pi/2 pulse on Y
            self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[1])
            self._free_evolution(ps / 2 - self.pi_pulse_duration / 2, self.options)

            # repeat M times the pi pulse and free evolution of ps
            for idx_M in range(self.M):
                self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[0])
                if idx_M != self.M - 1:
                    self._free_evolution(ps, self.options)

            if self.projection_pulse:
                self._free_evolution(ps / 2 - self.pi_pulse_duration / 2, self.options)
                # final pi/2 pulse on Y
                self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[1])
            else:
                self._free_evolution(ps / 2, self.options)

        return self.rho

    def _get_pulse_profiles(self, tau: float | None = None) -> None:
        """
        Generates the pulse profiles for the CPMG sequence for a given tau.

        Parameters
        ----------
        tau : float
            Free evolution variable or pulse spacing for the  sequence.
        """
        tau = self._check_tau(tau)

        self.pulse_profiles = []
        ps = tau - self.pi_pulse_duration

        self._append_pulse_to_profile(0, self.pi_pulse_duration / 2, self.pulse_params[1])
        t0 = self.pi_pulse_duration / 2
        self.pulse_profiles.append(
            ["free_evo", [t0, t0 + ps / 2 - self.pi_pulse_duration / 2], None, None]
        )
        t0 += ps / 2 - self.pi_pulse_duration / 2

        for idx_M in range(self.M):
            self._append_pulse_to_profile(t0, self.pi_pulse_duration, self.pulse_params[0])
            t0 += self.pi_pulse_duration
            if idx_M != self.M - 1:
                self.pulse_profiles.append(["free_evo", [t0, t0 + ps], None, None])
                t0 += ps

        if self.projection_pulse:
            self.pulse_profiles.append(
                ["free_evo", [t0, t0 + ps / 2 - self.pi_pulse_duration / 2], None, None]
            )
            t0 += ps / 2 - self.pi_pulse_duration / 2
            self._append_pulse_to_profile(t0, self.pi_pulse_duration / 2, self.pulse_params[1])
            t0 += self.pi_pulse_duration / 2
        else:
            self.pulse_profiles.append(["free_evo", [t0, t0 + ps / 2], None, None])
            t0 += ps / 2

        self.total_time = t0

    def plot_pulses(
        self,
        figsize: tuple[int, int] = (6, 4),
        xlabel: str = "Time",
        ylabel: str = "Pulse Intensity",
        title: str = "Pulse Profiles of XY8 Sequence",
        tau: float | None = None,
    ) -> None:
        """
        Overwrites the plot_pulses method of the parent class in order to
        first generate the pulse profiles for the CPMG sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float
            Free evolution time for the  sequence. Contrary to the run method, tau must be a single number in order to plot the pulse profiles.
        """
        self._get_pulse_profiles(tau)
        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)


####################################################################################################


class XY(PulsedSim):
    """
    This class contains the XY-M pulse sequence.

    The sequence is composed of intercalated X and Y pi pulses and free evolutions repeated M times.
    It acts similar to the CPMG sequence, but the alternation of the pulse improves noise suppression on different axis.
    The initial and final pi/2 pulses are played on the Y axis, while the refocusing pi pulses alternate between X and Y.

    Notes
    -----
    The XY sequence inherits the methods and attributes from the PulsedSim class.

    On the phase convention of the pi/2 pulses: contrary to the XY8 sequence implemented in this
    module, which prepares and projects along X, the XY-M sequence here prepares and projects along Y.
    This is deliberate and must not be "harmonized" with XY8. Differently from XY8, whose eight pulse
    phase pattern is reproduced consistently across the literature, the plain XY-M sequence is not
    standardized in the field, with the axis of the initial pi/2 pulse varying between references.
    The convention used here is the one of the experimental setup the sequence was written for, and
    published results were obtained with it. The choice is not merely cosmetic, as the phase of the
    preparation pulse relative to the refocusing pulses determines whether the prepared state is
    locked along a refocusing axis, which changes how sensitive the sequence is to pulse errors.

    Methods
    -------
    XY_sequence :
        Defines the XY sequence for a given free evolution time tau and the set of attributes defined in the constructor,
        returning the final state. The sequence is to be called by the parallel_map method of QuTip.
    _get_pulse_profiles :
        Generates the pulse profiles for the XY-M sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
    plot_pulses :
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the XY-M sequence for a given tau and then plot them.
    """

    # the sequence opens and closes with tau/2 around the pi pulses, therefore
    # tau must be at least twice the pi_pulse_duration to avoid negative free evolutions
    _min_tau_factor = 2

    def __init__(
        self,
        free_duration: np.ndarray | list[float],
        *,
        system: QSys,
        M: int,
        pi_pulse_duration: float,
        h1: Qobj | list[Qobj] | None = None,
        Rx: Qobj | None = None,
        Ry: Qobj | None = None,
        H2: tuple[Qobj, Callable] | None = None,
        projection_pulse: bool = True,
        pulse_shape: Callable = square_pulse,
        pulse_params: dict[str, float] | None = None,
        time_steps: int = 100,
        options: dict | None = None,
    ) -> None:
        """
        Class constructor for the XY sequence

        Parameters
        ----------
        free_duration : numpy array
            Time array for the simulation representing the free evolution time to be used as the variable attribute for the simulation
        system : QSys
            Quantum system object containing the initial state, internal time independent Hamiltonian and collapse operators
        M : int
            Order of the XY sequence
        pi_pulse_duration : float, int or 0
            Duration of the pi pulse. If set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        h1 : Qobj or list of Qobj
            Control Hamiltonian of the system.
        Rx : Qobj or None
            Rotation operator around the x-axis, used only if the pi_pulse_duration is set to 0.
        Ry : Qobj or None
            Rotation operator around the y-axis, used only if the pi_pulse_duration is set to 0.
        H2 : Qobj, list(Qobj), optional
            Time dependent sensing Hamiltonian of the system
        pulse_shape : FunctionType, list(FunctionType), optional
            Pulse shape function or list of pulse shape functions representing the time modulation of h1
        pulse_params : dict, optional
            Dictionary of parameters for the pulse_shape functions
        time_steps : int, optional
            Number of time steps in the pulses for the simulation
        options : dict, optional
            Dictionary of solver options
        """
        super().__init__(system, H2=H2)
        self.sequence = self.XY_sequence
        self._check_attr_predef_seqs(
            h1,
            Rx,
            Ry,
            pulse_shape,
            pulse_params,
            options,
            time_steps,
            free_duration,
            pi_pulse_duration,
            M,
            projection_pulse,
        )

        base_pulse = self.pulse_params.copy()
        self.pulse_params = [{**base_pulse, "phi_t": 0}, {**base_pulse, "phi_t": -np.pi / 2}]

    def XY_sequence(self, tau: float) -> Qobj:
        """
        Defines the XY-M composed of intercalated pi pulses on X and Y axis with free evolutions of time tau repeated M times.
        The sequence starts with a pi/2 pulse on the Y axis, see the note on the phase convention in the class docstring.
        If projection_pulse is True, the sequence will include a final pi/2 pulse on Y axis to project the measurement into the Sz basis.
        If the pi_pulse_duration is set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time

        Returns
        -------
        rho : Qobj
            Final state
        """
        self._reset_sequence()

        if self.pi_pulse_duration == 0:
            # initial pi/2 pulse on Y, matching the finite pulse branch below
            self._delta_pulse(self.Ry_half)
            self._free_evolution(tau / 2, self.options)

            # repeat M times the pi X pulse, free evolution of ps, pi Y pulse and free evolution of ps
            for idx_M in range(self.M):
                self._delta_pulse(self.Rx)
                self._free_evolution(tau, self.options)
                self._delta_pulse(self.Ry)
                if idx_M != self.M - 1:
                    self._free_evolution(tau, self.options)

            # the sequence is symmetric, therefore it closes with tau/2 as it started,
            # giving a total sequence time of 2*M*tau
            self._free_evolution(tau / 2, self.options)

            if self.projection_pulse:
                self._delta_pulse(self.Ry_half)

        else:
            ps = tau - self.pi_pulse_duration
            # initial pi/2 pulse on Y, see the note on the phase convention in the class docstring
            self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[1])
            self._free_evolution(ps / 2 - self.pi_pulse_duration / 2, self.options)

            # repeat M times the pi X pulse, free evolution of ps, pi Y pulse and free evolution of ps
            for idx_M in range(self.M):
                self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[0])
                self._free_evolution(ps, self.options)
                self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[1])
                if idx_M != self.M - 1:
                    self._free_evolution(ps, self.options)

            if self.projection_pulse:
                self._free_evolution(ps / 2 - self.pi_pulse_duration / 2, self.options)
                # final pi/2 pulse on Y
                self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[1])
            else:
                self._free_evolution(ps / 2, self.options)

        return self.rho

    def _get_pulse_profiles(self, tau: float | None = None) -> None:
        """
        Generates the pulse profiles for the XY-M sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau : float
            free evolution variable or pulse spacing for the  sequence
        """
        tau = self._check_tau(tau)

        self.pulse_profiles = []
        ps = tau - self.pi_pulse_duration

        # the first and last pi/2 pulses are played on Y by XY_sequence, thus pulse_params[1]
        self._append_pulse_to_profile(0, self.pi_pulse_duration / 2, self.pulse_params[1])
        t0 = self.pi_pulse_duration / 2
        self.pulse_profiles.append(
            ["free_evo", [t0, t0 + ps / 2 - self.pi_pulse_duration / 2], None, None]
        )
        t0 += ps / 2 - self.pi_pulse_duration / 2

        for idx_M in range(2 * self.M):
            self._append_pulse_to_profile(t0, self.pi_pulse_duration, self.pulse_params[idx_M % 2])
            t0 += self.pi_pulse_duration
            if idx_M != 2 * self.M - 1:
                self.pulse_profiles.append(["free_evo", [t0, t0 + ps], None, None])
                t0 += ps

        if self.projection_pulse:
            self.pulse_profiles.append(
                ["free_evo", [t0, t0 + ps / 2 - self.pi_pulse_duration / 2], None, None]
            )
            t0 += ps / 2 - self.pi_pulse_duration / 2
            self._append_pulse_to_profile(t0, self.pi_pulse_duration / 2, self.pulse_params[1])
            t0 += self.pi_pulse_duration / 2
        else:
            self.pulse_profiles.append(["free_evo", [t0, t0 + ps / 2], None, None])
            t0 += ps / 2

        self.total_time = t0

    def plot_pulses(
        self,
        figsize: tuple[int, int] = (6, 6),
        xlabel: str = "Time",
        ylabel: str = "Pulse Intensity",
        title: str = "Pulse Profiles of XY8 Sequence",
        tau: float | None = None,
    ) -> None:
        """
        Overwrites the plot_pulses method of the parent class in order to
        first generate the pulse profiles for the XY-M sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float
            Free evolution time for the  sequence. Contrary to the run method, tau must be a single number in order to plot the pulse profiles.
        """
        # generate the pulse profiles for the given tau
        self._get_pulse_profiles(tau)

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)


####################################################################################################


class XY8(PulsedSim):
    """
    This contains the XY8-M sequence.

    The XY8-M is a further improvement from the XY-M sequence, where the X and Y pulses are group antisymmetrically in pairs of 4 as X-Y-X-Y-Y-X-Y-X,
    in order to improve noise suppression and pulse errors.

    Methods
    -------
    XY_sequence :
        Defines the XY8-M sequence for a given free evolution time tau and the set of attributes defined in the constructor,
        returning the final state. The sequence is to be called by the parallel_map method of QuTip.
    _get_pulse_profiles :
        Generates the pulse profiles for the XY8-M sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
    plot_pulses :
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the XY8-M sequence for a given tau and then plot them.

    Notes
    -----
    The XY8 sequence inherits the methods and attributes from the PulsedSim class.
    """

    # the sequence opens and closes with tau/2 around the pi pulses, therefore
    # tau must be at least twice the pi_pulse_duration to avoid negative free evolutions
    _min_tau_factor = 2

    def __init__(
        self,
        free_duration: np.ndarray | list[float],
        *,
        system: QSys,
        M: int,
        pi_pulse_duration: float,
        h1: Qobj | list[Qobj] | None = None,
        Rx: Qobj | None = None,
        Ry: Qobj | None = None,
        H2: tuple[Qobj, Callable] | None = None,
        projection_pulse: bool = True,
        pulse_shape: Callable = square_pulse,
        pulse_params: dict[str, float] | None = None,
        time_steps: int = 100,
        options: dict | None = None,
        RXY8: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Class constructor for the XY8 sequence

        Parameters
        ----------
        free_duration : numpy array
            Time array for the simulation representing the free evolution time to be used as the variable attribute for the simulation
        system : QSys
            Quantum system object containing the initial state, internal time independent Hamiltonian and collapse operators
        M : int
            Order of the XY sequence
        pi_pulse_duration : float, int or 0
            Duration of the pi pulse. If set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        h1 : Qobj or list of Qobj
            Control Hamiltonian of the system.
        Rx : Qobj or None
            Rotation operator around the x-axis, used only if the pi_pulse_duration is set to 0.
        Ry : Qobj or None
            Rotation operator around the y-axis, used only if the pi_pulse_duration is set to 0.
        H2 : Qobj, list(Qobj), optional
            Time dependent sensing Hamiltonian of the system
        pulse_shape : FunctionType, list(FunctionType), optional
            Pulse shape function or list of pulse shape functions representing the time modulation of h1
        pulse_params : dict, optional
            Dictionary of parameters for the pulse_shape functions
        time_steps : int, optional
            Number of time steps in the pulses for the simulation
        options : dict, optional
            Dictionary of solver options from Qutip
        projection_pulse : bool
            Boolean to determine if a final pi/2 pulse is to be included in order to project the measurement into the Sz basis
        RXY8 : bool
            Boolen to determine if a random phase is to be added to each XY8 block
        seed : int or None
            Seed for the random phases of the RXY8 sequence, so that the simulation is reproducible.
            Only used if RXY8 is True.
        """
        if not isinstance(RXY8, bool):
            raise ValueError(  # noqa: TRY004
                f"RXY8 must be a boolean value indicating whether to add a random phase to each XY8 block or not, got {RXY8}."
            )

        super().__init__(system, H2=H2)
        self.sequence = self.XY8_sequence
        self._check_attr_predef_seqs(
            h1,
            Rx,
            Ry,
            pulse_shape,
            pulse_params,
            options,
            time_steps,
            free_duration,
            pi_pulse_duration,
            M,
            projection_pulse,
        )

        if self.pi_pulse_duration == 0 and RXY8:
            warnings.warn(
                "RXY8 with delta pulses is not implemented, as it has not experimental relevance.",
                stacklevel=2,
            )
            random_phases = np.zeros(M)
        elif RXY8:
            # the phases are drawn once for the whole simulation, so that every point of the
            # variable is calculated with the same sequence
            random_phases = np.random.default_rng(seed).random(M) * 2 * np.pi
        else:
            random_phases = np.zeros(M)

        # generate the pulse parameters for the XY8 sequence with the correct order of the x and y pulses
        base_pulse = self.pulse_params.copy()

        self.pulse_params = []
        for idx_M in range(M):
            phi_x = 0 + random_phases[idx_M]
            phi_y = -np.pi / 2 + random_phases[idx_M]

            px = base_pulse.copy()
            py = base_pulse.copy()
            px["phi_t"] = phi_x
            py["phi_t"] = phi_y

            self.pulse_params.extend([px, py, px, py, py, px, py, px])

    def XY8_sequence(self, tau: float) -> Qobj:
        """
        Defines the XY8-M composed of 8 intercalated pi pulses on X and Y axis with free evolutions of time tau repeated M times.
        If random_phase is set to True, a random phase is added in each XY8 block.
        If projection_pulse is True, the sequence will include a final pi/2 pulse on X axis to project the measurement into the Sz basis.
        If the pi_pulse_duration is set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time

        Returns
        -------
        rho : Qobj
            Final state
        """
        self._reset_sequence()

        if self.pi_pulse_duration == 0:
            # initial pi/2 pulse on X
            self._delta_pulse(self.Rx_half)
            self._free_evolution(tau / 2, self.options)

            # repeat 8 times the block
            for idx_M in range(self.M):
                self._delta_pulse(self.Rx)
                self._free_evolution(tau, self.options)
                self._delta_pulse(self.Ry)
                self._free_evolution(tau, self.options)
                self._delta_pulse(self.Rx)
                self._free_evolution(tau, self.options)
                self._delta_pulse(self.Ry)
                self._free_evolution(tau, self.options)
                self._delta_pulse(self.Ry)
                self._free_evolution(tau, self.options)
                self._delta_pulse(self.Rx)
                self._free_evolution(tau, self.options)
                self._delta_pulse(self.Ry)
                self._free_evolution(tau, self.options)
                self._delta_pulse(self.Rx)
                if idx_M != self.M - 1:
                    self._free_evolution(tau, self.options)

            self._free_evolution(tau / 2, self.options)

            if self.projection_pulse:
                self._delta_pulse(self.Rx_half)

        else:
            ps = tau - self.pi_pulse_duration
            # initial pi/2 pulse on X
            self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[0])
            self._free_evolution(ps / 2 - self.pi_pulse_duration / 2, self.options)

            # repeat 8*M-1 times the pi pulse and free evolution of ps
            for idx_M in range(8 * self.M):
                self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[idx_M])
                if idx_M != 8 * self.M - 1:
                    self._free_evolution(ps, self.options)

            if self.projection_pulse:
                self._free_evolution(ps / 2 - self.pi_pulse_duration / 2, self.options)
                self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[0])
            else:
                self._free_evolution(ps / 2, self.options)

        return self.rho

    def _get_pulse_profiles(self, tau: float | None = None) -> None:
        """
        Generates the pulse profiles for the XY8-M sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau : float
            free evolution variable or pulse spacing for the  sequence
        """
        tau = self._check_tau(tau)

        self.pulse_profiles = []
        ps = tau - self.pi_pulse_duration

        self._append_pulse_to_profile(0, self.pi_pulse_duration / 2, self.pulse_params[0])
        t0 = self.pi_pulse_duration / 2
        self.pulse_profiles.append(
            ["free_evo", [t0, t0 + ps / 2 - self.pi_pulse_duration / 2], None, None]
        )
        t0 += ps / 2 - self.pi_pulse_duration / 2

        for idx_M in range(8 * self.M):
            self._append_pulse_to_profile(t0, self.pi_pulse_duration, self.pulse_params[idx_M])
            t0 += self.pi_pulse_duration
            if idx_M != 8 * self.M - 1:
                self.pulse_profiles.append(["free_evo", [t0, t0 + ps], None, None])
                t0 += ps

        if self.projection_pulse:
            self.pulse_profiles.append(
                ["free_evo", [t0, t0 + ps / 2 - self.pi_pulse_duration / 2], None, None]
            )
            t0 += ps / 2 - self.pi_pulse_duration / 2
            self._append_pulse_to_profile(t0, self.pi_pulse_duration / 2, self.pulse_params[0])
            t0 += self.pi_pulse_duration / 2
        else:
            self.pulse_profiles.append(["free_evo", [t0, t0 + ps / 2], None, None])
            t0 += ps / 2

        self.total_time = t0

    def plot_pulses(
        self,
        figsize: tuple[int, int] = (6, 4),
        xlabel: str = "Time",
        ylabel: str = "Pulse Intensity",
        title: str = "Pulse Profiles of XY8 Sequence",
        tau: float | None = None,
    ) -> None:
        """
        Overwrites the plot_pulses method of the parent class in order to first
        generate the pulse profiles for the XY8-M sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float
            Free evolution time for the  sequence. Contrary to the run method, tau must be a single number in order to plot the pulse profiles.
        """
        self._get_pulse_profiles(tau)
        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)
