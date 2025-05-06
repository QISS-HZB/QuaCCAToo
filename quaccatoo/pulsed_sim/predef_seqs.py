"""
This module contains predefined basic pulsed experiments inheriting from the PulsedSim class as part of the QuaCCAToo package.

Classes
-------
- Rabi: resonant pulse of varying duration, such that the quantum system will undergo periodical transitions between the excited and ground states.
- PMR: Pulsed Magnetic Resonance (PMR) experiment, composed by a single pulse where the frequency is changed such that when it corresponds to a transition in the Hamiltonian of the system, the observable will be affected.
- Ramsey: Ramsey experiment, consisting of a free evolution that causes a phase accumulation between states in the system which can be used for interferometry.
- Hahn: Hahn echo experiment, consisting of two free evolutions with a pi pulse in the middle, in order to cancel out dephasings. The Hahn echo is usually used to measure the coherence time of a quantum system, however it can also be used to sense coupled spins.
"""

import warnings

import numpy as np
from qutip import Qobj, mesolve, propagator

from .pulsed_sim import PulsedSim
from .pulse_shapes import square_pulse

####################################################################################################

class Rabi(PulsedSim):
    """
    A class containing Rabi experiments, inheriting from the PulsedSimulation class.

    A Rabi sequence is composed of a resonant pulse of varying duration,
    such that the quantum system will undergo periodical transitions between the excited and ground states.

    Attributes
    ----------
    pulse_duration : numpy.ndarray
        Time array for the simulation representing the pulse duration to be used as the variable for the simulation.

    Methods
    -------
    PulsedSimulation
    """

    def __init__(self, pulse_duration, system, H1, H2=None, pulse_shape=square_pulse, pulse_params=None, options=None):
        """
        Constructor for the Rabi pulsed experiment class.

        Parameters
        ----------
        pulse_duration : numpy.ndarray
            Time array for the simulation representing the pulse duration to be used as the variable for the simulation.
        system : QSys
            Quantum system object containing the initial density matrix, internal Hamiltonian and collapse operators.
        H1 : Qobj or list(Qobj)
            Control Hamiltonian of the system.
        H2 : Qobj or list(Qobj)
            Time-dependent sensing Hamiltonian of the system.
        pulse_shape : FunctionType or list(FunctionType)
            Pulse shape function or list of pulse shape functions representing the time modulation of H1.
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions.
        options : dict
            Dictionary of solver options from Qutip.
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(H1, pulse_shape, pulse_params, options, len(pulse_duration), None, None, None)

        # check whether pulse_duration is a numpy array and if it is, assign it to the object
        if not isinstance(pulse_duration, (np.ndarray, list)) or not np.all(np.isreal(pulse_duration)) or not np.all(np.greater_equal(pulse_duration, 0)):
            raise ValueError("pulse_duration must be a numpy array of real positive elements")
        else:
            self.total_time = pulse_duration[-1]
            self.variable = pulse_duration
            self.variable_name = f"Pulse Duration (1/{self.system.units_H0})"

        if isinstance(H1, Qobj):
            self.pulse_profiles = [[H1, pulse_duration, pulse_shape, self.pulse_params]]
        else:
            self.pulse_profiles = [[H1[i], pulse_duration, pulse_shape[i], self.pulse_params] for i in range(len(H1))]

    def run(self):
        """
        Overwrites the run method of the parent class. Runs the simulation and stores the results in the results attribute.
        If the system has no initial density matrix, the propagator is calcualated.
        If an observable is given, the expectation values are stored in the results attribute.
        For the Rabi sequence, the calculation is optimally performed sequentially instead of in parallel over the pulse lengths,
        thus the run method from the parent class is overwritten.
        """
        if self.system.rho0 is None:
            self.U = propagator(self.Ht, 2 * np.pi * self.variable, self.system.c_ops, options=self.options, args=self.pulse_params)
        else:   
        # calculates the density matrices in sequence using mesolve
            self.rho = mesolve(self.Ht, self.system.rho0, 2 * np.pi * self.variable, self.system.c_ops, e_ops=[], options=self.options, args=self.pulse_params).states
            self._get_results()

####################################################################################################

class PMR(PulsedSim):
    """
    A class containing Pulsed Magnetic Resonance (PMR) experiments where the frequency is the variable being changed,
    inheriting from the PulsedSim class.

    The PMR consists of a single pulse of fixed length and changing frequency. If the frequency matches a resonance of the system,
    it will undergo some transition which will affect the observable.
    This way, the differences between energy levels can be determined with the linewidth usually limited by the pulse length.
    Here we make reference to optical detection as it is the most common detection scheme of pulsed magnetic resonance in color centers,
    however the method can be more general.

    Attributes
    ----------
    frequencies : numpy.ndarray
        Array of frequencies to run the simulation.
    pulse_duration : float or int
        Duration of the pulse.

    Methods
    -------
    PMR_sequence(f)
        Defines the Pulsed Magnetic Resonance (PMR) sequence for a given frequency of the pulse.
        To be called by the parallel_map in run method.
    (Inherited from PulsedSimulation)
    """

    def __init__(self, frequencies, pulse_duration, system, H1, H2=None, pulse_shape=square_pulse, pulse_params=None, time_steps=100, options=None):
        """
        Constructor for the PMR pulsed experiment class.

        Parameters
        ----------
        frequencies : numpy.ndarray
            Array of frequencies to run the simulation.
        pulse_duration : float or int
            Duration of the pulse.
        system : QSys
            Quantum system object containing the initial density matrix, internal Hamiltonian and collapse operators.
        H1 : Qobj or list(Qobj)
            Control Hamiltonian of the system.
        H2 : Qobj or list(Qobj)
            Time-dependent sensing Hamiltonian of the system.
        pulse_shape : FunctionType or list(FunctionType)
            Pulse shape function or list of pulse shape functions representing the time modulation of H1.
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions.
        time_steps : int
            Number of time steps in the pulses for the simulation.
        options : dict
            Dictionary of solver options from Qutip.
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(H1, pulse_shape, pulse_params, options, time_steps, None, None, None)

        # check whether frequencies is a numpy array or list and if it is, assign it to the object
        if not isinstance(frequencies, (np.ndarray, list)) or not np.all(np.isreal(frequencies)) or not np.all(np.greater_equal(frequencies, 0)):
            raise ValueError("frequencies must be a numpy array or list of real positive elements")
        else:
            self.variable = frequencies
            self.variable_name = f"Frequency ({self.system.units_H0})"

        # check whether pulse_duration is a numpy array and if it is, assign it to the object
        if not isinstance(pulse_duration, (float, int)) or pulse_duration <= 0:
            raise ValueError("pulse_duration must be a positive real number")
        else:
            self.pulse_duration = pulse_duration

        # set the sequence attribute to the PMR_sequence method
        self.sequence = self.PMR_sequence

    def PMR_sequence(self, f):
        """
        Defines the Pulsed Magnetic Resonance (PMR) sequence for a given frequency of the pulse.
        To be called by the parallel_map in run method.

        Parameters
        ----------
        f : float
            Frequency of the pulse.

        Returns
        -------
        rho : Qobj
            Final density matrix.
        """
        self.pulse_params["f_pulse"] = f

        # run the simulation and return the final density matrix
        rho = mesolve(
            self.Ht,
            self.system.rho0,
            2 * np.pi * np.linspace(0, self.pulse_duration, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        return rho

    def plot_pulses(self, figsize=(6, 4), xlabel="Time", ylabel="Pulse Intensity", title="Pulse Profiles", f_pulse=None):
        """
        Overwrites the plot_pulses method of the parent class in order to first define a pulse frequency to be plotted.

        Parameters
        ----------
        f_pulse : float or int
            Frequency of the pulse to be plotted.

        (Inherited from PulsedSimulation.plot_pulses)
        """
        # if f_pulse is None, assign the first element of the variable attribute to the pulse_params dictionary
        if f_pulse is None:
            self.pulse_params["f_pulse"] = self.variable[0]
        # if f_pulse is a float or an integer, assign it to the pulse_params dictionary
        elif isinstance(f_pulse, (int, float)):
            self.pulse_params["f_pulse"] = f_pulse
        else:
            raise ValueError("f_pulse must be a float or an integer")

        self.total_time = self.pulse_duration

        super().plot_pulses(figsize, xlabel, ylabel, title)


####################################################################################################


class Ramsey(PulsedSim):
    """
    A class containing Ramsey experiments, inheriting from the PulsedSimulation class.

    Attributes
    ----------
    free_duration : numpy.ndarray
        Time array for the simulation representing the free evolution time to be used as the variable attribute for the simulation.
    pi_pulse_duration : float or int
        Duration of the pi pulse.
    projection_pulse : bool
        Boolean to determine if a final pi/2 pulse is to be included in order to project the measurement in the Sz basis.

    Methods
    -------
    ramsey_sequence(tau)
        Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of an initial pi/2 pulse and a single free evolution.
        The sequence is to be called by the parallel_map method of QuTip.
    ramsey_sequence_proj(tau)
        Defines the Ramsey sequence with final pi/2 pulse to project into the Sz basis.
    ramsey_sequence_H2(tau)
        Defines the Ramsey sequence considering time-dependent H2 or collapse operators.
    ramsey_sequence_proj_H2(tau)
        Defines the Ramsey sequence considering time-dependent H2 or collapse operators and a final pi/2 pulse.
    _get_pulse_profiles(tau)
        Generates the pulse profiles for the Ramsey sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.
    (Inherited from PulsedSimulation)
    """

    def __init__(
        self,
        free_duration,
        pi_pulse_duration,
        system,
        H1,
        H2=None,
        projection_pulse=True,
        pulse_shape=square_pulse,
        pulse_params=None,
        options=None,
        time_steps=100,
    ):
        """
        Class constructor for the Ramsey pulsed experiment class.

        Parameters
        ----------
        free_duration : numpy.ndarray
            Time array for the simulation representing the free evolution time to be used as the variable attribute for the simulation.
        system : QSys
            Quantum system object containing the initial density matrix, internal Hamiltonian and collapse operators.
        H1 : Qobj or list(Qobj)
            Control Hamiltonian of the system.
        pi_pulse_duration : float or int
            Duration of the pi pulse.
        H2 : Qobj or list(Qobj)
            Time-dependent sensing Hamiltonian of the system.
        projection_pulse : bool
            Boolean to determine if the measurement is to be performed in the Sz basis or not.
            If True, a final pi/2 pulse is included in order to project the result into the Sz basis, as for most color centers.
        pulse_shape : FunctionType or list(FunctionType)
            Pulse shape function or list of pulse shape functions representing the time modulation of H1.
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions.
        time_steps : int
            Number of time steps in the pulses for the simulation.
        options : dict
            Dictionary of solver options from Qutip.
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(H1, pulse_shape, pulse_params, options, time_steps, free_duration, pi_pulse_duration, None)

        # If projection_pulse is True, the sequence is set to the ramsey_sequence_proj method with the final projection pulse
        # otherwise it is set to the ramsey_sequence method without the projection pulse. If H2 or c_ops are given then uses the alternative methods _H2
        if projection_pulse:
            if H2 is not None or self.system.c_ops is not None:
                self.sequence = self.ramsey_sequence_proj_H2
            else:
                self.sequence = self.ramsey_sequence_proj
        elif not projection_pulse:
            if H2 is not None or self.system.c_ops is not None:
                self.sequence = self.ramsey_sequence_H2
            else:
                self.sequence = self.ramsey_sequence
        else:
            raise ValueError("projection_pulse must be a boolean")

        self.projection_pulse = projection_pulse

    def ramsey_sequence(self, tau):
        """
        Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of an initial pi/2 pulse and a single free evolution.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final density matrix.
        """
        # calculate the pulse separation time
        ps = tau - self.pi_pulse_duration / 2

        # perform initial pi/2 pulse
        rho = mesolve(
            self.Ht,
            self.system.rho0,
            2 * np.pi * np.linspace(0, self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        # perform the free evolution
        rho = (-1j * 2 * np.pi * self.system.H0 * ps).expm() * rho * ((-1j * 2 * np.pi * self.system.H0 * ps).expm()).dag()

        return rho

    def ramsey_sequence_proj(self, tau):
        """
        Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of an initial pi/2 pulse, a single free evolution, and a final pi/2 pulse to project the result into the Sz basis.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final density matrix.
        """
        # calculate the pulse separation time
        ps = tau - self.pi_pulse_duration

        # perform initial pi/2 pulse
        rho = mesolve(
            self.Ht,
            self.system.rho0,
            2 * np.pi * np.linspace(0, self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        # perform the free evolution
        rho = (-1j * 2 * np.pi * self.system.H0 * ps).expm() * rho * ((-1j * 2 * np.pi * self.system.H0 * ps).expm()).dag()

        # perform final pi/2 pulse
        t0 = self.pi_pulse_duration / 2 + ps
        rho = mesolve(
            self.Ht,
            rho,
            2 * np.pi * np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        return rho

    def ramsey_sequence_H2(self, tau):
        """
        Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of an initial pi/2 pulse and a single free evolution.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final density matrix.
        """
        ps = tau - self.pi_pulse_duration / 2

        # perform initial pi/2 pulse
        rho = mesolve(
            self.Ht,
            self.system.rho0,
            2 * np.pi * np.linspace(0, self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]
        t0 = self.pi_pulse_duration / 2

        # perform the free evolution
        rho = mesolve(self.H0_H2, rho, 2 * np.pi * np.linspace(t0, t0 + ps, self.time_steps), self.system.c_ops, e_ops=[], options=self.options, args=self.pulse_params).states[-1]

        return rho

    def ramsey_sequence_proj_H2(self, tau):
        """
        Defines the Ramsey sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of an initial pi/2 pulse, a single free evolution, and a final pi/2 pulse to project the result into the Sz basis.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final density matrix.
        """
        # calculate the pulse separation time
        ps = tau - self.pi_pulse_duration

        # perform initial pi/2 pulse
        rho = mesolve(
            self.Ht,
            self.system.rho0,
            2 * np.pi * np.linspace(0, self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]
        t0 = self.pi_pulse_duration / 2

        # perform the free evolution
        rho = mesolve(self.H0_H2, rho, 2 * np.pi * np.linspace(t0, t0 + ps, self.time_steps), self.system.c_ops, e_ops=[], options=self.options, args=self.pulse_params).states[-1]
        t0 += ps

        # perform final pi/2 pulse
        rho = mesolve(
            self.Ht,
            rho,
            2 * np.pi * np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        return rho

    def _get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the Ramsey sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau : float
            Free evolution variable or pulse spacing for the Hahn echo sequence.
        """
        # check if tau is None and if it is, assign the first element of the variable attribute to tau
        if tau is None:
            tau = self.variable[-1]
        # else if it is not a float or an integer, raise an error
        elif not isinstance(tau, (int, float)) or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration")

        # initialize the pulse_profiles attribute to an empty list
        self.pulse_profiles = []

        # if tau is None, assign the first element of the variable attribute to tau
        if tau is None:
            tau = self.variable[0]
        # if tau is not a float or an integer, raise an error
        elif not isinstance(tau, (int, float)):
            raise ValueError("tau must be a float or an integer")

        # if projection_pulse is True, include the final pi/2 pulse in the pulse_profiles
        if self.projection_pulse:
            # calculate the pulse separation time
            ps = tau - self.pi_pulse_duration

            # if only one control Hamiltonian is given, append the pulse_profiles with the Ramsey sequence
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append([self.H1, np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 = self.pi_pulse_duration / 2
                self.pulse_profiles.append([None, [t0, ps + t0], None, None])
                t0 += ps
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 += self.pi_pulse_duration / 2

            # otherwise if a list of control Hamiltonians is given, it sums over all H1 and appends to the pulse_profiles
            elif isinstance(self.H1, list):
                self.pulse_profiles.append([[self.H1[i], np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))])
                t0 = self.pi_pulse_duration / 2
                self.pulse_profiles.append([None, [t0, ps + t0], None, None])
                t0 += ps
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))])
                t0 += self.pi_pulse_duration / 2

        # if projection_pulse is false, do not include the final pi/2 pulse in the pulse_profiles
        else:
            # calculate the pulse separation time
            ps = tau - self.pi_pulse_duration / 2

            # if only one control Hamiltonian is given, append the pulse_profiles with the Ramsey sequence
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append([self.H1, np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 = self.pi_pulse_duration / 2
                self.pulse_profiles.append([None, [t0, ps + t0], None, None])
                t0 += ps

            # otherwise if a list of control Hamiltonians is given, it sums over all H1 and appends to the pulse_profiles
            elif isinstance(self.H1, list):
                self.pulse_profiles.append([[self.H1[i], np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))])
                t0 = self.pi_pulse_duration / 2
                self.pulse_profiles.append([None, [t0, ps + t0], None, None])
                t0 += ps

        # set the total time to t0
        self.total_time = t0

    def plot_pulses(self, figsize=(6, 4), xlabel="Time", ylabel="Pulse Intensity", title="Pulse Profiles of Ramsey Sequence", tau=None):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the Ramsey sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float
            Free evolution time for the Hahn echo sequence. Contrary to the run method, the free evolution must be a single number in order to plot the pulse profiles.
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot.
        xlabel : str
            Label of the x-axis.
        ylabel : str
            Label of the y-axis.
        title : str
            Title of the plot.
        """
        # generate the pulse profiles for the Ramsey sequence for a given tau
        self._get_pulse_profiles(tau)

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)


####################################################################################################


class Hahn(PulsedSim):
    """
    A class containing Hahn echo experiments, inheriting from the PulsedSimulation class.

    The Hahn echo sequence consists of two free evolutions with a pi pulse in the middle, in order to cancel out dephasings.
    The Hahn echo is usually used to measure the coherence time of a quantum system, however it can also be used to sense coupled spins.

    Attributes
    ----------
    free_duration : numpy.ndarray
        Time array of the free evolution times to run the simulation.
    pi_pulse_duration : float or int
        Duration of the pi pulse.
    projection_pulse : bool
        Boolean to determine if a final pi/2 pulse is to be included in order to project the measurement in the Sz basis.

    Methods
    -------
    hahn_sequence(tau)
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the constructor,
        returning the final density matrix. The sequence is to be called by the parallel_map method of QuTip.
    hahn_sequence_proj(tau)
        Defines the Hahn echo sequence with a final pi/2 pulse, in order to project the result into the Sz basis.
    hahn_sequence_H2(tau)
        Defines the Hahn echo sequence considering time-dependent H2 or collapse operators.
    hahn_sequence_proj_H2(tau)
        Defines the Hahn echo sequence considering time-dependent H2 or collapse operators and a final pi/2 pulse.
    (Inherited from PulsedSimulation)
    """

    def __init__(
        self,
        free_duration,
        pi_pulse_duration,
        system,
        H1,
        H2=None,
        projection_pulse=True,
        pulse_shape=square_pulse,
        pulse_params=None,
        options=None,
        time_steps=100,
    ):
        """
        Constructor for the Hahn echo pulsed experiment class, taking a specific free_duration to run the simulation and the pi_pulse_duration.

        Parameters
        ----------
        free_duration : numpy.ndarray
            Time array for the simulation representing the free evolution time to be used as the variable attribute for the simulation.
        system : QSys
            Quantum system object containing the initial density matrix, internal Hamiltonian and collapse operators.
        H1 : Qobj or list of Qobj
            Control Hamiltonian of the system.
        pi_pulse_duration : float or int
            Duration of the pi pulse.
        H2 : Qobj or list of Qobj
            Time dependent sensing Hamiltonian of the system.
        projection_pulse : bool
            Boolean to determine if the measurement is to be performed in the Sz basis or not.
            If True, a final pi/2 pulse is included in order to project the result into the Sz basis, as done for the most color centers.
        pulse_shape : FunctionType or list of FunctionType
            Pulse shape function or list of pulse shape functions representing the time modulation of H1.
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions.
        time_steps : int
            Number of time steps in the pulses for the simulation.
        options : dict
            Dictionary of solver options from Qutip.
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(H1, pulse_shape, pulse_params, options, time_steps, free_duration, pi_pulse_duration, None)

        # If projection_pulse is True, the sequence is set to the hahn_sequence_proj method with the final projection pulse to project the result into the Sz basis
        # otherwise it is set to the hahn_sequence method without the projection pulses
        if projection_pulse:
            if H2 is not None or self.system.c_ops is not None:
                self.sequence = self.hahn_sequence_proj_H2
            else:
                self.sequence = self.hahn_sequence_proj
        elif not projection_pulse:
            if H2 is not None or self.system.c_ops is not None:
                self.sequence = self.hahn_sequence_H2
            else:
                self.sequence = self.hahn_sequence
        else:
            raise ValueError("projection_pulse must be a boolean")

        self.projection_pulse = projection_pulse

    def hahn_sequence(self, tau):
        """
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the constructor.

        The sequence consists of an initial pi/2 pulse and two free evolutions with a pi pulse between them.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final density matrix.
        """
        # calculate pulse separation time
        ps = tau - self.pi_pulse_duration

        # perform the initial pi/2 pulse
        rho = mesolve(
            self.Ht,
            self.system.rho0,
            2 * np.pi * np.linspace(0, self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        # perform the first free evolution
        rho = (-1j * 2 * np.pi * self.system.H0 * ps).expm() * rho * ((-1j * 2 * np.pi * self.system.H0 * ps).expm()).dag()

        # changing pulse separation time for the second free evolution
        ps += self.pi_pulse_duration / 2

        # perform the pi pulse
        rho = mesolve(
            self.Ht,
            rho,
            2 * np.pi * np.linspace(ps, ps + self.pi_pulse_duration, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        # perform the second free evolution
        rho = (-1j * 2 * np.pi * self.system.H0 * ps).expm() * rho * ((-1j * 2 * np.pi * self.system.H0 * ps).expm()).dag()

        return rho

    def hahn_sequence_proj(self, tau):
        """
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the constructor.

        The sequence consists of a pi/2 pulse, a free evolution time tau, a pi pulse and another free evolution time tau followed by a pi/2 pulse.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final density matrix.
        """
        # calculate pulse separation time
        ps = tau - self.pi_pulse_duration

        # perform the initial pi/2 pulse
        rho = mesolve(
            self.Ht,
            self.system.rho0,
            2 * np.pi * np.linspace(0, self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        # perform the first free evolution
        rho = (-1j * 2 * np.pi * self.system.H0 * ps).expm() * rho * ((-1j * 2 * np.pi * self.system.H0 * ps).expm()).dag()

        # perform the pi pulse
        t0 = self.pi_pulse_duration / 2 + ps
        rho = mesolve(
            self.Ht,
            rho,
            2 * np.pi * np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        # perform the second free evolution
        rho = (-1j * 2 * np.pi * self.system.H0 * ps).expm() * rho * ((-1j * 2 * np.pi * self.system.H0 * ps).expm()).dag()

        # perform the final pi/2 pulse
        t0 += tau
        rho = mesolve(
            self.Ht,
            rho,
            2 * np.pi * np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        # if no observable is given, return the final density matrix
        return rho

    def hahn_sequence_H2(self, tau):
        """
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the constructor.

        The sequence consists of an initial pi/2 pulse and two free evolutions with a pi pulse between them.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final density matrix.
        """
        # calculate pulse separation time
        ps = tau - self.pi_pulse_duration

        # perform the initial pi/2 pulse
        rho = mesolve(
            self.Ht,
            self.system.rho0,
            2 * np.pi * np.linspace(0, self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]
        t0 = self.pi_pulse_duration / 2

        # perform the first free evolution
        rho = mesolve(self.H0_H2, rho, 2 * np.pi * np.linspace(t0, t0 + ps, self.time_steps), self.system.c_ops, e_ops=[], options=self.options, args=self.pulse_params).states[-1]
        t0 += ps

        # perform the pi pulse
        rho = mesolve(
            self.Ht,
            rho,
            2 * np.pi * np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]
        t0 += self.pi_pulse_duration

        # perform the second free evolution
        rho = mesolve(
            self.H0_H2,
            rho,
            2 * np.pi * np.linspace(t0, t0 + ps + self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        return rho

    def hahn_sequence_proj_H2(self, tau):
        """
        Defines the Hahn echo sequence for a given free evolution time tau and the set of attributes defined in the constructor.

        The sequence consists of a pi/2 pulse, a free evolution time tau, a pi pulse and another free evolution time tau followed by a pi/2 pulse.
        The sequence is to be called by the parallel_map method of QuTip.

        Parameters
        ----------
        tau : float
            Free evolution time.

        Returns
        -------
        rho : Qobj
            Final density matrix.
        """
        # calculate pulse separation time
        ps = tau - self.pi_pulse_duration

        # perform the initial pi/2 pulse
        rho = mesolve(
            self.Ht,
            self.system.rho0,
            2 * np.pi * np.linspace(0, self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]
        t0 = self.pi_pulse_duration / 2

        # perform the first free evolution
        rho = mesolve(self.H0_H2, rho, 2 * np.pi * np.linspace(t0, t0 + ps, self.time_steps), self.system.c_ops, e_ops=[], options=self.options, args=self.pulse_params).states[-1]
        t0 += ps

        # perform the pi pulse
        rho = mesolve(
            self.Ht,
            rho,
            2 * np.pi * np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]
        t0 += self.pi_pulse_duration

        # perform the second free evolution
        rho = mesolve(self.H0_H2, rho, 2 * np.pi * np.linspace(t0, t0 + ps, self.time_steps), self.system.c_ops, e_ops=[], options=self.options, args=self.pulse_params).states[-1]
        t0 += ps

        # perform the final pi/2 pulse
        rho = mesolve(
            self.Ht,
            rho,
            2 * np.pi * np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps),
            self.system.c_ops,
            e_ops=[],
            options=self.options,
            args=self.pulse_params,
        ).states[-1]

        return rho

    def _get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the Hahn echo sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau : float
            Free evolution variable or pulse spacing for the Hahn echo sequence.
        """
        # check if tau is None and if it is, assign the last element of the variable attribute to tau
        if tau is None:
            tau = self.variable[-1]
        # else if it is not a float or an integer, raise an error
        elif not isinstance(tau, (int, float)) or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration")

        # initialize the pulse_profiles attribute to an empty list and t0 to 0
        self.pulse_profiles = []
        t0 = 0

        # if projection_pulse is True, include the final pi/2 pulse in the pulse_profiles
        if self.projection_pulse:
            # if only one control Hamiltonian is given, append the pulse_profiles with the Hahn echo sequence as in the hahn_sequence method
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 += self.pi_pulse_duration / 2
                self.pulse_profiles.append([None, [t0, t0 + tau - self.pi_pulse_duration], None, None])
                t0 += tau - self.pi_pulse_duration
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 += self.pi_pulse_duration
                self.pulse_profiles.append([None, [t0, t0 + tau - self.pi_pulse_duration], None, None])
                t0 += tau - self.pi_pulse_duration
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 += self.pi_pulse_duration / 2

            # otherwise if a list of control Hamiltonians is given, it sums over all H1 and appends to the pulse_profiles the Hahn echo sequence as in the hahn_sequence method
            elif isinstance(self.H1, list):
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))])
                t0 += self.pi_pulse_duration / 2
                self.pulse_profiles.append([None, [t0, t0 + tau - self.pi_pulse_duration], None, None])
                t0 += tau - self.pi_pulse_duration
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))])
                t0 += self.pi_pulse_duration
                self.pulse_profiles.append([None, [t0, t0 + tau - self.pi_pulse_duration], None, None])
                t0 += tau - self.pi_pulse_duration
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))])
                t0 += self.pi_pulse_duration / 2

        # if projection_pulse is False, do not include the final pi/2 pulse in the pulse_profiles
        else:
            # if only one control Hamiltonian is given, append the pulse_profiles with the Hahn echo sequence as in the hahn_sequence method
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 += self.pi_pulse_duration / 2
                self.pulse_profiles.append([None, [t0, t0 + tau - self.pi_pulse_duration], None, None])
                t0 += tau - self.pi_pulse_duration
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params])
                t0 += self.pi_pulse_duration
                self.pulse_profiles.append([None, [t0, t0 + tau - self.pi_pulse_duration / 2], None, None])
                t0 += tau - self.pi_pulse_duration / 2

            # otherwise if a list of control Hamiltonians is given, it sums over all H1 and appends to the pulse_profiles the Hahn echo sequence as in the hahn_sequence method
            elif isinstance(self.H1, list):
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))])
                t0 += self.pi_pulse_duration / 2
                self.pulse_profiles.append([None, [t0, t0 + tau - self.pi_pulse_duration], None, None])
                t0 += tau - self.pi_pulse_duration
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params] for i in range(len(self.H1))])
                t0 += self.pi_pulse_duration
                self.pulse_profiles.append([None, [t0, t0 + tau - self.pi_pulse_duration / 2], None, None])
                t0 += tau - self.pi_pulse_duration / 2

        # set the total_time attribute to the total time of the pulse sequence
        self.total_time = t0

    def plot_pulses(self, figsize=(6, 6), xlabel="Time", ylabel="Pulse Intensity", title="Pulse Profiles", tau=None):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the Hahn echo sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float
            Free evolution time for the Hahn echo sequence. Contrary to the run method, the free evolution must be a single number in order to plot the pulse profiles.
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot.
        xlabel : str
            Label of the x-axis.
        ylabel : str
            Label of the y-axis.
        title : str
            Title of the plot.
        """
        # generate the pulse profiles for the Hahn echo sequence for a given tau
        self._get_pulse_profiles(tau)

        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)
