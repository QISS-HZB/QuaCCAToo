"""
This module contains predefined basic pulsed experiments inheriting from the PulsedSim class as part of the QuaCCAToo package.

Classes
-------
- Rabi: resonant pulse of varying duration, such that the quantum system will undergo periodical transitions between the excited and ground states.
- PMR: Pulsed Magnetic Resonance (PMR) experiment, composed by a single pulse where the frequency is changed such that when it corresponds to a transition in the Hamiltonian of the system, the observable will be affected.
- Ramsey: Ramsey experiment, consisting of a free evolution that causes a phase accumulation between states in the system which can be used for interferometry.
- Hahn: Hahn echo experiment, consisting of two free evolutions with a pi pulse in the middle, in order to cancel out dephasings. The Hahn echo is usually used to measure the coherence time of a quantum system, however it can also be used to sense coupled spins.
"""

import numpy as np
from qutip import Qobj, mesolve, propagator
from typing import Callable, Optional

from .pulse_shapes import square_pulse
from .pulsed_sim import PulsedSim
from ..qsys.qsys import QSys

####################################################################################################


class Rabi(PulsedSim):
    """
    A class containing Rabi experiments.

    A Rabi sequence is composed of a resonant pulse of varying duration,
    such that the quantum system will undergo periodical transitions between the excited and ground states.

    Methods
    -------
    run :
        Runs the simulation and stores the results in the results attribute.

    Notes
    -----
    The Rabi sequence inherits the methods and attributes from the PulsedSim class.
    """

    def __init__(
        self,
        pulse_duration : np.ndarray | list[float | int],
        system: QSys,
        h1 : Qobj | list[Qobj],
        H2 : Optional[tuple[Qobj, Callable]] = None,
        pulse_shape : Callable = square_pulse,
        pulse_params : Optional[dict[str, float | int]] = None, 
        options : Optional[dict] = None
    ) -> None:
        """
        Constructor for the Rabi pulsed experiment class.

        Parameters
        ----------
        pulse_duration : numpy.ndarray
            Time array for the simulation representing the pulse duration to be used as the variable for the simulation.
        system : QSys
            Quantum system object containing the initial state, internal Hamiltonian and collapse operators.
        h1 : Qobj or list(Qobj)
            Control Hamiltonian of the system.
        H2 : Qobj or list(Qobj)
            Time-dependent sensing Hamiltonian of the system.
        pulse_shape : FunctionType or list(FunctionType)
            Pulse shape function or list of pulse shape functions representing the time modulation of h1.
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions.
        options : dict
            Dictionary of solver options from Qutip.
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(
            h1, None, None, pulse_shape, pulse_params, options, len(pulse_duration), None, None, None, None
        )

        # check whether pulse_duration is a numpy array and if it is, assign it to the object
        if (
            not isinstance(pulse_duration, (np.ndarray, list))
            or not np.all(np.isreal(pulse_duration))
            or not np.all(np.greater_equal(pulse_duration, 0))
        ):
            raise ValueError("pulse_duration must be a numpy array of real positive elements")
        else:
            self.total_time = pulse_duration[-1]
            self.variable = pulse_duration
            self.variable_name = f"Pulse Duration (1/{self.system.units_H0})"

        self._append_pulse_to_profile(0, self.total_time)

    def run(
        self
        ) -> None:
        """
        Overwrites the run method of the parent class. Runs the simulation and stores the results in the results attribute.
        If the system has no initial state, the propagator is calcualated.
        If an observable is given, the expectation values are stored in the results attribute.
        For the Rabi sequence, the calculation is optimally performed sequentially instead of in parallel over the pulse lengths,
        thus the run method from the parent class is overwritten.
        """
        if self.system.rho0 is None:
            self.U = propagator(
                self.Ht,
                2 * np.pi * self.variable,
                self.system.c_ops,
                options=self.options,
                args=self.pulse_params,
            )
        else:
            # calculates the density matrices in sequence using mesolve
            self.rho = mesolve(
                self.Ht,
                self.system.rho0,
                2 * np.pi * self.variable,
                self.system.c_ops,
                e_ops=[],
                options=self.options,
                args=self.pulse_params,
            ).states
            self._get_results()


####################################################################################################


class PMR(PulsedSim):
    """
    A class containing Pulsed Magnetic Resonance (PMR) experiments where the frequency is the variable being changed.

    The PMR consists of a single pulse of fixed length and changing frequency. If the frequency matches a resonance of the system,
    it will undergo some transition which will affect the observable.
    This way, the differences between energy levels can be determined with the linewidth usually limited by the pulse length.
    Here we make reference to optical detection as it is the most common detection scheme of pulsed magnetic resonance in color centers,
    however the method can be more general.

    Attributes
    ----------
    frequencies : numpy.ndarray
        Array of frequencies to run the simulation.

    Methods
    -------
    PMR_sequence :
        Defines the Pulsed Magnetic Resonance (PMR) sequence for a given frequency of the pulse.
        To be called by the parallel_map in run method.
    plot_pulses :
        Overwrites the plot_pulses method of the parent class in order to first define a pulse frequency to be plotted.

    Notes
    -----
    The PMR sequence inherits the methods and attributes from the PulsedSim class.
    """

    def __init__(
        self,
        frequencies : np.ndarray | list[float | int],
        system : QSys,
        pulse_duration : float | int,
        h1 : Qobj | list[Qobj],
        H2 : Optional[tuple[Qobj, Callable]] = None,
        pulse_shape : Callable = square_pulse,
        pulse_params : Optional[dict[str, float | int]] = None, 
        time_steps : int = 100,
        options : Optional[dict] = None
    ) -> None:
        """
        Constructor for the PMR pulsed experiment class.

        Parameters
        ----------
        frequencies : numpy.ndarray
            Array of frequencies to run the simulation.
        system : QSys
            Quantum system object containing the initial state, internal Hamiltonian and collapse operators.
        pulse_duration : float or int
            Duration of the pulse.
        h1 : Qobj or list(Qobj)
            Control Hamiltonian of the system.
        H2 : Qobj or list(Qobj)
            Time-dependent sensing Hamiltonian of the system.
        pulse_shape : FunctionType or list(FunctionType)
            Pulse shape function or list of pulse shape functions representing the time modulation of h1.
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions.
        time_steps : int
            Number of time steps in the pulses for the simulation.
        options : dict
            Dictionary of solver options from Qutip.
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(
            h1, None, None, pulse_shape, pulse_params, options, time_steps, None, None, None, None
        )

        # check whether frequencies is a numpy array or list and if it is, assign it to the object
        if (
            not isinstance(frequencies, (np.ndarray, list))
            or not np.all(np.isreal(frequencies))
            or not np.all(np.greater_equal(frequencies, 0))
        ):
            raise ValueError("frequencies must be a numpy array or list of real positive elements")
        else:
            self.variable = frequencies
            self.variable_name = f"Frequency ({self.system.units_H0})"

        # check whether pulse_duration is a numpy array and if it is, assign it to the object
        if not isinstance(pulse_duration, (float, int)) or pulse_duration <= 0:
            raise ValueError("pulse_duration must be a positive real number")
        else:
            self.total_time = pulse_duration

        # set the sequence attribute to the PMR_sequence method
        self.sequence = self.PMR_sequence
        self._append_pulse_to_profile(0, self.total_time)

    def PMR_sequence(
        self,
        f : float | int
        ) -> Qobj:
        """
        Defines the Pulsed Magnetic Resonance (PMR) sequence for a given frequency of the pulse.
        To be called by the parallel_map in run method.

        Parameters
        ----------
        f : float or int
            Frequency of the pulse.

        Returns
        -------
        rho : Qobj
            Final state.
        """
        self.pulse_params["f_pulse"] = f

        self._pulse(self.Ht, self.total_time, self.options, self.pulse_params)

        return self.rho

    def plot_pulses(
        self,
        figsize : tuple[int, int] = (6, 4),
        xlabel : str = "Time",
        ylabel : str = "Pulse Intensity",
        title : str = "Pulse Profiles",
        f_pulse : Optional[float | int] = None
    ):
        """
        Overwrites the plot_pulses method of the parent class in order to first define a pulse frequency to be plotted.

        Parameters
        ----------
        f_pulse : float or int
            Frequency of the pulse to be plotted.

        Notes
        -----
        The method uses the same parameters as plot_pulses
        """
        # if f_pulse is None, assign the first element of the variable attribute to the pulse_params dictionary
        if f_pulse is None:
            self.pulse_params["f_pulse"] = self.variable[0]
        # if f_pulse is a float or an integer, assign it to the pulse_params dictionary
        elif isinstance(f_pulse, (int, float)):
            self.pulse_params["f_pulse"] = f_pulse
        else:
            raise ValueError("f_pulse must be a float or an integer")

        super().plot_pulses(figsize, xlabel, ylabel, title)


####################################################################################################


class Ramsey(PulsedSim):
    """
    A class containing Ramsey experiments.

    The Ramsey sequence consists of a free evolution in the plane perpendicular to the quantization axis,
    which causes a phase accumulation between states in the system which can be used for sensing.

    Methods
    -------
    ramsey_sequence :
        Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of an initial pi/2 pulse and a single free evolution.
        The sequence is to be called by the parallel_map method of QuTip.
    _get_pulse_profiles :
        Generates the pulse profiles for the Ramsey sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.
    plot_pulses :
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the Ramsey sequence for a given tau and then plot them.

    Notes
    -----
    The Ramsey sequence inherits the methods and attributes from the PulsedSim class.
    """

    def __init__(
        self,
        free_duration : np.ndarray | list[float | int],
        system : QSys,
        pi_pulse_duration : float | int,
        h1 : Optional[Qobj | list[Qobj]] = None,
        Rx : Optional[Qobj] = None,
        H2 : Optional[tuple[Qobj, Callable]] = None,
        projection_pulse :  bool = True,
        pulse_shape : Callable = square_pulse,
        pulse_params : Optional[dict[str, float | int]] = None, 
        time_steps : int = 100,
        options : Optional[dict] = None
    ) -> None:
        """
        Class constructor for the Ramsey pulsed experiment class.

        Parameters
        ----------
        free_duration : numpy.ndarray
            Time array for the simulation representing the free evolution time to be used as the variable attribute for the simulation.
        system : QSys
            Quantum system object containing the initial state, internal Hamiltonian and collapse operators.
        pi_pulse_duration : float, int or 0
            Duration of the pi pulse. If set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        h1 : Qobj or list(Qobj)
            Control Hamiltonian of the system.
        Rx : Qobj or None
            Rotation operator around the x-axis, used only if the pi_pulse_duration is set to 0.
        pi_pulse_duration : float or int
            Duration of the pi pulse.
        H2 : Qobj or list(Qobj)
            Time-dependent sensing Hamiltonian of the system.
        projection_pulse : bool
            Boolean to determine if the measurement is to be performed in the Sz basis or not.
            If True, a final pi/2 pulse is included in order to project the result into the Sz basis, as for most color centers.
        pulse_shape : FunctionType or list(FunctionType)
            Pulse shape function or list of pulse shape functions representing the time modulation of h1.
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions.
        time_steps : int
            Number of time steps in the pulses for the simulation.
        options : dict
            Dictionary of solver options from Qutip.
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(
            h1,
            Rx,
            Rx,
            pulse_shape,
            pulse_params,
            options,
            time_steps,
            free_duration,
            pi_pulse_duration,
            None,
            projection_pulse,
        )
        self.sequence = self.ramsey_sequence

    def ramsey_sequence(
            self,
            tau : float | int
        ) -> Qobj:
        """
        Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of an initial pi/2 pulse and a single free evolution.
        If the projection_pulse is set to True, a final pi/2 pulse is included in order to project the result into the Sz basis.
        If the pi_pulse_duration is set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float or int
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final state.
        """
        if self.pi_pulse_duration == 0:
            self._delta_pulse(self.Rx_half)
            self._free_evolution(tau, self.options)

            if self.projection_pulse:
                self._delta_pulse(self.Rx_half)

        else:
            self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params)
            self._free_evolution(tau - self.pi_pulse_duration / 2, self.options)

            if self.projection_pulse:
                self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params)

        return self.rho

    def _get_pulse_profiles(
        self,
        tau : Optional[float | int] = None
        ) -> None:
        """
        Generates the pulse profiles for the Ramsey sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau : float | int
            Free evolution variable or pulse spacing for the Hahn echo sequence.
        """
        # check if tau is correctly defined
        if tau is None:
            tau = self.variable[-1]
        elif not isinstance(tau, (int, float)) or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration")

        self.pulse_profiles = []

        self._append_pulse_to_profile(0, self.pi_pulse_duration / 2)
        t0 = self.pi_pulse_duration / 2

        self.pulse_profiles.append(["free_evo", [t0, t0 + tau], None, None])
        t0 += tau

        if self.projection_pulse:
            self._append_pulse_to_profile(t0, self.pi_pulse_duration / 2)
            t0 += self.pi_pulse_duration / 2

        self.total_time = t0

    def plot_pulses(
        self,
        figsize: tuple[int, int] = (6, 4),
        xlabel : str = "Time",
        ylabel : str = "Pulse Intensity",
        title : str = "Pulse Profiles of Ramsey Sequence",
        tau : Optional[float | int] = None
        ) -> None:
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the Ramsey sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float | int
            Free evolution time for the Hahn echo sequence. Contrary to the run method, the free evolution must be a single number in order to plot the pulse profiles.

        Notes
        -----
        The method uses the same parameters as plot_pulses
        """
        # generate the pulse profiles for the Ramsey sequence for a given tau
        self._get_pulse_profiles(tau)

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)


####################################################################################################


class Hahn(PulsedSim):
    """
    A class containing Hahn echo experiments.

    The Hahn echo sequence consists of two free evolutions with a pi pulse in the middle, in order to cancel out dephasings.
    The Hahn echo is usually used to measure the coherence time of a quantum system, however it can also be used to sense coupled spins.

    Methods
    -------
    hahn_sequence :
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the constructor,
        returning the final state. The sequence is to be called by the parallel_map method of QuTip.
    _get_pulse_profiles :
        Generates the pulse profiles for the Hahn echo sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.
    plot_pulses :
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the Hahn echo sequence for a given tau and then plot them.

    Notes
    -----
    The Hahn echo sequence inherits the methods and attributes from the PulsedSim class.
    """
    def __init__(
        self,
        free_duration : np.ndarray | list[float | int],
        system : QSys,
        pi_pulse_duration : float | int,
        h1 : Optional[Qobj | list[Qobj]] = None,
        Rx : Optional[Qobj] = None,
        H2 : Optional[tuple[Qobj, Callable]] = None,
        projection_pulse :  bool = True,
        pulse_shape : Callable = square_pulse,
        pulse_params : Optional[dict[str, float | int]] = None, 
        time_steps : int = 100,
        options : Optional[dict] = None
    ) -> None:
        """
        Constructor for the Hahn echo pulsed experiment class, taking a specific free_duration to run the simulation.

        Parameters
        ----------
        free_duration : numpy.ndarray
            Time array for the simulation representing the free evolution time to be used as the variable attribute for the simulation.
        system : QSys
            Quantum system object containing the initial state, internal Hamiltonian and collapse operators.
        pi_pulse_duration : float, int or 0
            Duration of the pi pulse. If set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        h1 : Qobj or list of Qobj
            Control Hamiltonian of the system.
        Rx : Qobj or None
            Rotation operator around the x-axis, used only if the pi_pulse_duration is set to 0.
        H2 : Qobj or list of Qobj
            Time dependent sensing Hamiltonian of the system.
        projection_pulse : bool
            Boolean to determine if the measurement is to be performed in the Sz basis or not.
            If True, a final pi/2 pulse is included in order to project the result into the Sz basis, as done for the most color centers.
        pulse_shape : FunctionType or list of FunctionType
            Pulse shape function or list of pulse shape functions representing the time modulation of h1.
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions.
        time_steps : int
            Number of time steps in the pulses for the simulation.
        options : dict
            Dictionary of solver options from Qutip.
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(
            h1,
            Rx,
            Rx,
            pulse_shape,
            pulse_params,
            options,
            time_steps,
            free_duration,
            pi_pulse_duration,
            None,
            projection_pulse,
        )
        self.sequence = self.hahn_sequence

    def hahn_sequence(
        self,
        tau : float | int
        ) -> Qobj:
        """
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of a pi/2 pulse, a free evolution time tau, a pi pulse and another free evolution time tau followed by a pi/2 pulse, if the projection pulse is set to True.
        If the pi_pulse_duration is set to 0, the pulses are perfect delta pulses and the time-evolution is calculated with the rotation operator.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float or int
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final state.
        """
        # if pi_pulse_duration is 0, use the delta pulse method
        if self.pi_pulse_duration == 0:
            self._delta_pulse(self.Rx_half)
            self._free_evolution(tau, self.options)
            self._delta_pulse(self.Rx)
            self._free_evolution(tau, self.options)

            if self.projection_pulse:
                self._delta_pulse(self.Rx_half)

        else:
            # pulse separation time
            ps = tau - self.pi_pulse_duration

            self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params)
            self._free_evolution(ps, self.options)
            self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params)

            if self.projection_pulse:
                self._free_evolution(ps, self.options)
                self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params)
            else:
                self._free_evolution(ps + self.pi_pulse_duration / 2, self.options)

        return self.rho

    def _get_pulse_profiles(
        self,
        tau : Optional[float | int] = None
        ) -> None:
        """
        Generates the pulse profiles for the Hahn echo sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau : float | int
            Free evolution variable or pulse spacing for the Hahn echo sequence.
        """
        # check if tau is correctly defined
        if tau is None:
            tau = self.variable[-1]
        elif not isinstance(tau, (int, float)) or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration")

        self.pulse_profiles = []
        ps = tau - self.pi_pulse_duration

        # pi/2 pulse
        self._append_pulse_to_profile(0, self.pi_pulse_duration / 2)
        t0 = self.pi_pulse_duration / 2
        # free evolution
        self.pulse_profiles.append(["free_evo", [t0, t0 + ps], None, None])
        t0 += ps
        # pi pulse
        self._append_pulse_to_profile(t0, self.pi_pulse_duration)
        t0 += self.pi_pulse_duration

        if self.projection_pulse:
            # free evolution
            self.pulse_profiles.append(["free_evo", [t0, t0 + ps], None, None])
            t0 += ps
            # pi/2 pulse
            self._append_pulse_to_profile(t0, self.pi_pulse_duration / 2)
            t0 += self.pi_pulse_duration / 2
        else:
            # free evolution
            self.pulse_profiles.append(["free_evo", [t0, t0 + ps + self.pi_pulse_duration / 2], None, None])
            t0 += ps + self.pi_pulse_duration / 2

        self.total_time = t0

    def plot_pulses(
        self,
        figsize: tuple[int, int] = (6, 4),
        xlabel : str = "Time",
        ylabel : str = "Pulse Intensity",
        title : str = "Pulse Profiles of Hahn Echo Sequence",
        tau : Optional[float | int] = None
        ) -> None:
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the Hahn echo sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float | int
            Free evolution time for the Hahn echo sequence. Contrary to the run method, the free evolution must be a single number in order to plot the pulse profiles.

        Notes
        -----
        The method uses the same parameters as plot_pulses
        """
        # generate the pulse profiles for the Hahn echo sequence for a given tau
        self._get_pulse_profiles(tau)

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)
