"""
This module contains dynamical decoupling pulse sequences, used in quantum sensing and for extending coherence of quantum systems.
"""

import numpy as np
from qutip import Qobj

from .pulse_shapes import square_pulse
from .pulsed_sim import PulsedSim

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

    def __init__(self, free_duration, system, M, pi_pulse_duration, H1, H2=None, projection_pulse=True, pulse_shape=square_pulse, pulse_params=None, options=None, time_steps=100):
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
        H1 : Qobj or list(Qobj)
            Control Hamiltonian of the system
        pi_pulse_duration : float or int
            Duration of the pi pulse
        H2 : list(Qobj or function)
            Time dependent sensing Hamiltonian of the system
        projection_pulse : bool
            Boolean to determine if a final pi/2 pulse is to be included in order to project the measurement into the Sz basis
        pulse_shape : FunctionType or list(FunctionType)
            Pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params : dict
            Dictionary of parameters for the pulse_shape functions
        time_steps : int
            Number of time steps in the pulses for the simulation
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(H1, pulse_shape, pulse_params, options, time_steps, free_duration, pi_pulse_duration, M, projection_pulse)
        self.sequence = self.CPMG_sequence

        base_pulse = pulse_params.copy()
        self.pulse_params = [
            {**base_pulse, "phi_t": 0},           
            {**base_pulse, "phi_t": -np.pi / 2}
        ]

    def CPMG_sequence(self, tau):
        """
        Defines the CPMG sequence for a given free evolution time tau and the set of attributes defined in the constructor.
        The sequence consists of an initial pi/2 pulse, and M pi-pulses separated by free evolution time tau.
        If projection_pulse is True, the sequence will include a final pi/2 pulse on Y axis to project the measurement into the Sz basis.
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
        ps = tau - self.pi_pulse_duration

        # initial pi/2 pulse on Y
        self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[1])
        self._free_evolution(ps/2 - self.pi_pulse_duration/2, self.options)

        # repeat M-1 times the pi pulse and free evolution of ps
        for itr_M in range(self.M - 1):
            self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[0])
            self._free_evolution(ps, self.options)

        # perform the last pi pulse on X
        self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[0])

        if self.projection_pulse:
            self._free_evolution(ps/2 - self.pi_pulse_duration/2, self.options)
            # final pi/2 pulse on Y
            self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[1])
        else:
            self._free_evolution(ps/2, self.options)

        return self.rho

    def _get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the CPMG sequence for a given tau.

        Parameters
        ----------
        tau : float
            Free evolution variable or pulse spacing for the  sequence.
        """
        # check if tau is correctly defined
        if tau is None:
            tau = self.variable[-1]
        elif not isinstance(tau, (int, float)) or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration")

        self.pulse_profiles = []

        # add the first pi/2 pulse on Y
        if isinstance(self.H1, Qobj):
            self.pulse_profiles.append([self.H1, np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params[1]])
        elif isinstance(self.H1, list):
            self.pulse_profiles.append([[self.H1[i], np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params[1]] for i in range(len(self.H1))])

        t0 = self.pi_pulse_duration / 2
        ps = tau - self.pi_pulse_duration
        
        # add the first free evolution
        self.pulse_profiles.append([None, [t0, t0 + ps/2 - self.pi_pulse_duration/2], None, None])
        t0 += ps/2 - self.pi_pulse_duration/2

        # add pulses and free evolution 8*M-1 times
        for itr_M in range(8*self.M - 1):
            # add a pi pulse on X
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[0]])
            elif isinstance(self.H1, list):
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))])
            t0 += self.pi_pulse_duration
            # add a free evolution
            self.pulse_profiles.append([None, [t0, t0 + ps], None, None])
            t0 += ps

        # add another pi pulse on X
        if isinstance(self.H1, Qobj):
            self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[0]])
        elif isinstance(self.H1, list):
            self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))])
        t0 += self.pi_pulse_duration

        if self.projection_pulse:
            # add the last free evolution
            self.pulse_profiles.append([None, [t0, t0 + ps/2 - self.pi_pulse_duration/2], None, None])
            t0 += ps/2 - self.pi_pulse_duration/2

            if isinstance(self.H1, Qobj):
                # add the last pi/2 pulse
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params[1]])
            elif isinstance(self.H1, list):
                # add the first pi/2 pulse
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params[1]] for i in range(len(self.H1))])

            t0 += self.pi_pulse_duration / 2
        else:
            # add the last free evolution
            self.pulse_profiles.append([None, [t0, t0 + ps/2], None, None])
            t0 += ps/2

        self.total_time = t0

    def plot_pulses(self, figsize=(6, 6), xlabel=None, ylabel="Expectation Value", title="Pulse Profiles", tau=None):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the CPMG sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float
            Free evolution time for the  sequence. Contrary to the run method, the free evolution must be a single number in order to plot the pulse profiles.
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot
        xlabel : str
            Label of the x-axis
        ylabel : str
            Label of the y-axis
        title : str
            Title of the plot
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

    Methods
    -------
    XY_sequence :
        Defines the XY sequence for a given free evolution time tau and the set of attributes defined in the constructor,
        returning the final state. The sequence is to be called by the parallel_map method of QuTip.
    _get_pulse_profiles :
        Generates the pulse profiles for the XY-M sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.
    plot_pulses :
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the XY-M sequence for a given tau and then plot them.

    Notes
    -----
    The XY sequence inherits the methods and attributes from the PulsedSim class.    
    """

    def __init__(self, free_duration, system, M, pi_pulse_duration, H1, H2=None, projection_pulse=True, pulse_shape=square_pulse, pulse_params=None, options=None, time_steps=100):
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
        H1 : Qobj, list(Qobj)
            Control Hamiltonian of the system
        pi_pulse_duration : float, int
            Duration of the pi pulse
        H2 : Qobj, list(Qobj), optional
            Time dependent sensing Hamiltonian of the system
        pulse_shape : FunctionType, list(FunctionType), optional
            Pulse shape function or list of pulse shape functions representing the time modulation of H1
        pulse_params : dict, optional
            Dictionary of parameters for the pulse_shape functions
        time_steps : int, optional
            Number of time steps in the pulses for the simulation
        options : dict, optional
            Dictionary of solver options
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(H1, pulse_shape, pulse_params, options, time_steps, free_duration, pi_pulse_duration, M, projection_pulse)
        self.sequence = self.XY_sequence

        base_pulse = pulse_params.copy()
        self.pulse_params = [
            {**base_pulse, "phi_t": 0},           
            {**base_pulse, "phi_t": -np.pi / 2}
        ]

    def XY_sequence(self, tau):
        """
        Defines the XY-M composed of intercalated pi pulses on X and Y axis with free evolutions of time tau repeated M times.
        If projection_pulse is True, the sequence will include a final pi/2 pulse on X axis to project the measurement into the Sz basis.
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
        ps = tau - self.pi_pulse_duration

        # initial pi/2 pulse on X
        self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[0])
        self._free_evolution(ps/2 - self.pi_pulse_duration/2, self.options)

        # repeat M-1 times the pi X pulse, free evolution of ps, pi Y pulse and free evolution of ps
        for itr_M in range(2 * self.M - 1):
            self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[itr_M % 2])
            self._free_evolution(ps, self.options)

        self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[1])

        if self.projection_pulse:
            self._free_evolution(ps/2 - self.pi_pulse_duration/2, self.options)
            # final pi/2 pulse on X
            self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[0])
        else:
            self._free_evolution(ps/2, self.options)

        return self.rho

    def _get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the XY-M sequence for a given tau.
        The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau : float
            free evolution variable or pulse spacing for the  sequence
        """
        # checl if tau is correctly define
        if tau is None:
            tau = self.variable[-1]
        elif not isinstance(tau, (int, float)) or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration")

        self.pulse_profiles = []

        # add the first pi/2 pulse on X axis
        if isinstance(self.H1, Qobj):
            self.pulse_profiles.append([self.H1, np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params[0]])
        elif isinstance(self.H1, list):
            self.pulse_profiles.append([[self.H1[i], np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))])

        t0 = self.pi_pulse_duration / 2
        ps = tau - self.pi_pulse_duration

        # add the first free evolution
        self.pulse_profiles.append([None, [t0, t0 + ps/2 - self.pi_pulse_duration/2], None, None])
        t0 += ps/2 - self.pi_pulse_duration/2

        # add pulses and free evolution M-1 times
        for itr_M in range(2 * self.M - 1):
            # add a pi pulse
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[itr_M % 2]])
            elif isinstance(self.H1, list):
                self.pulse_profiles.append(
                    [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[itr_M % 2]] for i in range(len(self.H1))]
                )
            t0 += self.pi_pulse_duration

            # add a free evolution
            self.pulse_profiles.append([None, [t0, t0 + ps], None, None])
            t0 += ps

        # add another pi pulse
        if isinstance(self.H1, Qobj):
            self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[1]])
        elif isinstance(self.H1, list):
            self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[1]] for i in range(len(self.H1))])
        t0 += self.pi_pulse_duration

        if self.projection_pulse:
            # add the last free evolution
            self.pulse_profiles.append([None, [t0, t0 + ps/2 - self.pi_pulse_duration/2], None, None])
            t0 += ps/2 - self.pi_pulse_duration/2

            if isinstance(self.H1, Qobj):
                # add the last pi/2 pulse
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params[0]])
            elif isinstance(self.H1, list):
                # add the first pi/2 pulse
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))])

            t0 += self.pi_pulse_duration / 2
        else:
            # add the last free evolution
            self.pulse_profiles.append([None, [t0, t0 + ps/2], None, None])
            t0 += ps/2

        self.total_time = t0

    def plot_pulses(self, figsize=(6, 6), xlabel=None, ylabel="Expectation Value", title="Pulse Profiles", tau=None):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the XY-M sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float
            Free evolution time for the  sequence. Contrary to the run method, the free evolution must be a single number in order to plot the pulse profiles.
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot
        xlabel : str
            Label of the x-axis
        ylabel : str
            Label of the y-axis
        title : str
            Title of the plot
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

    def __init__(self, free_duration, system, M, pi_pulse_duration, H1, H2=None, projection_pulse=True, pulse_shape=square_pulse, pulse_params=None, options=None, time_steps=100, RXY8=False):
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
        H1 : Qobj, list(Qobj)
            Control Hamiltonian of the system
        pi_pulse_duration : float, int
            Duration of the pi pulse
        H2 : Qobj, list(Qobj), optional
            Time dependent sensing Hamiltonian of the system
        pulse_shape : FunctionType, list(FunctionType), optional
            Pulse shape function or list of pulse shape functions representing the time modulation of H1
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
        """
        super().__init__(system, H2)
        self._check_attr_predef_seqs(H1, pulse_shape, pulse_params, options, time_steps, free_duration, pi_pulse_duration, M, projection_pulse)
        self.sequence = self.XY8_sequence

        if RXY8:
            random_phases = np.random.rand(M) * 2 * np.pi
        elif not RXY8:
            random_phases = np.zeros(M)
        else:
            raise ValueError("RXY8 must be a boolean value indicating weather to add a random phase to each XY8 block or not.")

        base_pulse = self.pulse_params.copy()

        self.pulse_params = []
        for itr_M in range(M):
            phi_x = 0 + random_phases[itr_M]
            phi_y = -np.pi / 2 + random_phases[itr_M]

            px = base_pulse.copy()
            py = base_pulse.copy()
            px["phi_t"] = phi_x
            py["phi_t"] = phi_y

            self.pulse_params.extend([px, py, px, py, py, px, py, px])
    
    def XY8_sequence(self, tau):
        """
        Defines the XY8-M composed of 8 intercalated pi pulses on X and Y axis with free evolutions of time tau repeated M times.
        If random_phase is set to True, a random phase is added in each XY8 block.
        If projection_pulse is True, the sequence will include a final pi/2 pulse on X axis to project the measurement into the Sz basis.
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
        ps = tau - self.pi_pulse_duration

        # initial pi/2 pulse on X
        self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[0])
        self._free_evolution(ps/2 - self.pi_pulse_duration/2, self.options)

        # repeat 8*M-1 times the pi pulse and free evolution of ps
        for itr_M in range(8*self.M - 1):
            self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[itr_M])
            self._free_evolution(ps, self.options)

        # perform the last pi pulse on X
        self._pulse(self.Ht, self.pi_pulse_duration, self.options, self.pulse_params[-1])

        if self.projection_pulse:
            self._free_evolution(ps/2 - self.pi_pulse_duration/2, self.options)
            self._pulse(self.Ht, self.pi_pulse_duration / 2, self.options, self.pulse_params[0])
        else:
            self._free_evolution(ps/2, self.options)

        return self.rho

    def _get_pulse_profiles(self, tau=None):
        """
        Generates the pulse profiles for the XY8-M sequence for a given tau. The pulse profiles are stored in the pulse_profiles attribute of the object.

        Parameters
        ----------
        tau : float
            free evolution variable or pulse spacing for the  sequence
        """
        # check if tau is correctly defined
        if tau is None:
            tau = self.variable[-1]
        elif not isinstance(tau, (int, float)) or tau < self.pi_pulse_duration:
            raise ValueError("tau must be a positive real number larger than pi_pulse_duration")

        self.pulse_profiles = []

        # add the first pi/2 pulse on X axis
        if isinstance(self.H1, Qobj):
            self.pulse_profiles.append([self.H1, np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params[0]])
        elif isinstance(self.H1, list):
            self.pulse_profiles.append([[self.H1[i], np.linspace(0, self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))])

        t0 = self.pi_pulse_duration / 2
        ps = tau  - self.pi_pulse_duration

        # add the first free evolution
        self.pulse_profiles.append([None, [t0, t0 + ps/2 - self.pi_pulse_duration/2], None, None])
        t0 += ps/2 - self.pi_pulse_duration/2

        # add pulses and free evolution M-1 times
        for itr_M in range(8 * self.M - 1):
            # add a pi pulse
            if isinstance(self.H1, Qobj):
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[itr_M % 8]])
            elif isinstance(self.H1, list):
                self.pulse_profiles.append(
                    [[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[itr_M % 8]] for i in range(len(self.H1))]
                )
            t0 += self.pi_pulse_duration

            # add a free evolution
            self.pulse_profiles.append([None, [t0, t0 + ps], None, None])
            t0 += ps

        # add another pi pulse
        if isinstance(self.H1, Qobj):
            self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape, self.pulse_params[0]])
        elif isinstance(self.H1, list):
            self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))])
        t0 += self.pi_pulse_duration

        if self.projection_pulse:
            # add the last free evolution
            self.pulse_profiles.append([None, [t0, t0 + ps/2 - self.pi_pulse_duration/2], None, None])
            t0 += ps/2 - self.pi_pulse_duration/2

            if isinstance(self.H1, Qobj):
                # add the last pi/2 pulse
                self.pulse_profiles.append([self.H1, np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape, self.pulse_params[0]])
            elif isinstance(self.H1, list):
                # add the first pi/2 pulse
                self.pulse_profiles.append([[self.H1[i], np.linspace(t0, t0 + self.pi_pulse_duration / 2, self.time_steps), self.pulse_shape[i], self.pulse_params[0]] for i in range(len(self.H1))])

            t0 += self.pi_pulse_duration / 2
        else:
            # add the last free evolution
            self.pulse_profiles.append([None, [t0, t0 + ps/2], None, None])
            t0 += ps/2

        # set the total_time attribute to the total time of the pulse sequence
        self.total_time = t0

    def plot_pulses(self, figsize=(6, 6), xlabel=None, ylabel="Expectation Value", title="Pulse Profiles", tau=None):
        """
        Overwrites the plot_pulses method of the parent class in order to first generate the pulse profiles for the XY8-M sequence for a given tau and then plot them.

        Parameters
        ----------
        tau : float
            Free evolution time for the  sequence. Contrary to the run method, the free evolution must be a single number in order to plot the pulse profiles.
        figsize : tuple
            Size of the figure to be passed to matplotlib.pyplot
        xlabel : str
            Label of the x-axis
        ylabel : str
            Label of the y-axis
        title : str
            Title of the plot
        """
        self._get_pulse_profiles(tau)
        # call the plot_pulses method of the parent class
        super().plot_pulses(figsize, xlabel, ylabel, title)