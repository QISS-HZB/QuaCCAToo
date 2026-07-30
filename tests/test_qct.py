"""
The testing framework for QuaCCAToo.

We use `pytest` for testing. See https://docs.pytest.org/en/stable/getting-started.html
for a quick introduction to pytest.

The gist is to use fixtures (`@pytest.fixture` decorator) for objects which are needed
repeatedly, and group related tests into an appropriately named test class.

Remember to mark long running tests with the `@pytest.mark.slow` decorator. These can then be run
with the `--runslow` CLI flag passed to `pytest`.
"""

import matplotlib
import numpy as np
import pytest
import scipy.constants as cte
from lmfit import Model
from qutip import basis, fock_dm, jmat, qeye, tensor

from quaccatoo import (
    CPMG,
    NV,
    P1,
    PMR,
    XY,
    XY8,
    Analysis,
    ExpData,
    Hahn,
    PulsedSim,
    QSys,
    Rabi,
    SiBiFlux,
    SpinLocking,
    compose_sys,
    load_quaccatoo,
    lorentzian_pulse,
    plot_histogram,
    save_quaccatoo,
    square_pulse,
)
from quaccatoo.analysis.fit_functions import (
    ExpDecayModel,
    GaussianModel,
    RabiModel,
    fit_sinc2,
    fit_two_lorentz_sym,
)

# the plotting methods are only called to check that they run, no window is needed
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# QSys fixture for reuse in multiple tests
@pytest.fixture
def qsys():
    delta = 1
    return QSys(
        H0=delta * jmat(1 / 2, "z"),
        rho0=fock_dm(2, 0),
        observable=jmat(1 / 2, "z") * 2,
        units_H0="MHz",
    )


# Test if the eigenstates of the qsys fixture are correct
class TestQSys:
    def test_states(self, qsys):
        assert (qsys.eigenstates[0].proj(), qsys.eigenstates[1].proj()) == (
            basis(2, 1).proj(),
            basis(2, 0).proj(),
        )

    def test_levels(self, qsys):
        assert np.array_equal(qsys.energy_levels, np.array([0, 1]))

    @pytest.mark.filterwarnings("ignore:Initial state not provided")
    def test_no_rho0(self):
        qsys_no_rho0 = QSys(H0=jmat(1 / 2, "z"), units_H0="MHz")
        assert qsys_no_rho0.rho0 is None
        assert PulsedSim(qsys_no_rho0).system.rho0 is None
        assert compose_sys(qsys_no_rho0, qsys_no_rho0).rho0 is None

    def test_truncate(self, qsys):
        trunc = QSys(H0=jmat(1, "z"), rho0=1, observable=fock_dm(3, 1), units_H0="MHz")
        trunc.truncate(0)
        assert trunc.H0.shape == (2, 2)

        with pytest.raises(ValueError):  # noqa: PT011
            qsys.truncate(5)
        with pytest.raises(ValueError):  # noqa: PT011
            qsys.truncate([0, 7])


# Tests for the NV class methods
class TestNV:
    def test_addspin(self):
        sys = NV(B0=200, units_B0="mT", N=0)
        GAMMA_C = 10.7084e-3
        azz = -130
        H2 = azz * tensor(jmat(1, "z"), jmat(1 / 2, "z")) - GAMMA_C * sys.B0 * tensor(
            qeye(3), jmat(1 / 2, "z")
        )
        sys.add_spin(H2)
        assert np.allclose(
            sys.energy_levels,
            np.array([0, 127.85832, 2797.84859722, 2799.99027722, 11207.83887444, 11339.98055444]),
        )

    def test_comp_trunc(self):
        NVb = NV(B0=18, units_B0="mT", N=0)
        NVa = NV(B0=25, units_B0="mT", N=14)
        NVb.truncate(mS=1)
        NVa.truncate(mS=1, mI=1)
        sys = compose_sys(NVb, NVa)
        assert np.allclose(
            sys.energy_levels,
            np.array(
                [
                    0,
                    4.93642778,
                    2167.23956813,
                    2174.31599591,
                    2365.55087505,
                    2370.48730283,
                    4532.79044318,
                    4539.86687097,
                ]
            ),
        )

    def test_lowT_rho0(self):
        rho0 = NV(N=15, B0=100, units_B0="T", temp=5.6, units_temp="K").rho0
        assert np.isclose(rho0[3, 3] - rho0[2, 2], 0.00185, atol=1e-5)

    # the nuclear spin can be truncated on its own, and the rotation operators of the
    # delta pulses have to follow the new dimensions
    def test_truncate_mI(self):
        sys = NV(B0=25, units_B0="mT", N=14)
        sys.truncate(mI=1)
        assert sys.H0.dims == [[3, 2], [3, 2]]
        assert all(R.shape == sys.H0.shape for R in sys.MW_Rx + sys.MW_Ry + sys.RF_Rx + sys.RF_Ry)


# Rabi object (fixture) used in the TestRabi class below
@pytest.fixture
def rabi_exp(qsys):
    w1 = 0.1
    delta = 1

    def custom_pulseX(t):
        return np.cos(delta * t)

    def custom_pulseY(t):
        return np.cos(delta * t - np.pi / 2)

    rabi_exp = Rabi(
        pulse_duration=np.linspace(0, 40, 100),
        system=qsys,
        h1=[w1 * jmat(1 / 2, "x"), w1 * jmat(1 / 2, "y")],
        pulse_shape=[custom_pulseX, custom_pulseY],
    )
    rabi_exp.run()
    return rabi_exp


class TestRabi:
    # Uses the rabi fixture defined above to check if the rabi frequency
    # is close to the expected value
    def test_tpi(self, rabi_exp):
        rabi_analysis = Analysis(rabi_exp)
        rabi_analysis.run_fit(fit_model=RabiModel())
        assert np.isclose(rabi_analysis.fit_params.best_values["Tpi"], 5, atol=1e-3)

    def test_fft(self, rabi_exp):
        rabi_analysis = Analysis(rabi_exp)
        rabi_analysis.run_FFT()
        assert np.isclose(rabi_analysis.get_peaks_FFT()[0], 1 / 2 / 5, atol=1e-3)

    # without an initial state the propagator is calculated instead of the density matrices
    @pytest.mark.filterwarnings("ignore:Initial state not provided")
    def test_propagator(self):
        rabi_U = Rabi(
            pulse_duration=np.linspace(0, 4, 10),
            system=QSys(H0=jmat(1 / 2, "z"), units_H0="MHz"),
            h1=0.1 * jmat(1 / 2, "x"),
            pulse_params={"f_pulse": 1},
        )
        rabi_U.run()
        assert len(rabi_U.U) == 10


class TestHahn:
    # @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")

    # check if the decay rate is close to the expected value
    def test_decay(self, qsys):
        w1 = 0.1
        delta = 1
        gamma = 0.1

        qsys.c_ops = gamma * jmat(1 / 2, "z") * 2
        hahn_exp = Hahn(
            free_duration=np.linspace(5, 25, 30),
            pi_pulse_duration=1 / 2 / w1,
            projection_pulse=True,
            system=qsys,
            h1=w1 * jmat(1 / 2, "x") * 2,
            pulse_shape=square_pulse,
            pulse_params={"f_pulse": delta},
        )
        hahn_exp.run()
        hahn_analysis = Analysis(hahn_exp)
        hahn_analysis.run_fit(fit_model=ExpDecayModel())
        assert np.allclose(
            [
                hahn_analysis.fit_params.best_values["Tc"],
                hahn_analysis.fit_params.best_values["amp"],
            ],
            [3.953, 1.905],
            atol=1e-3,
        )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_hahn_delta(self, qsys):
        gamma = 0.1
        qsys.c_ops = gamma * jmat(1 / 2, "z") * 2
        hahn_sim_delta = Hahn(
            free_duration=np.linspace(2.5, 25, 30),
            system=qsys,
            pi_pulse_duration=0,
            Rx=jmat(1 / 2, "x") * 2,
        )
        hahn_sim_delta.run()
        hahn_analysis = Analysis(hahn_sim_delta)
        hahn_analysis.run_fit(fit_model=ExpDecayModel())
        assert np.allclose(
            [
                hahn_analysis.fit_params.best_values["Tc"],
                hahn_analysis.fit_params.best_values["amp"],
            ],
            [3.9774, 1.0002],
            atol=1e-3,
        )


class TestXY:
    # Runs the XY sequence on an NV object
    # and checks if the center of the peak is in the expected position.
    # We don't use an outside fixture here since we need handcrafted values for this test
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_xy(self):
        qsys = NV(
            N=15,
            B0=39.4,
            units_B0="mT",
            theta=2.6,
            units_angles="deg",
        )
        w1 = 20
        XY_15N = XY(
            M=2,
            free_duration=np.linspace(0.25, 0.36, 30),
            pi_pulse_duration=1 / 2 / w1,
            system=qsys,
            h1=w1 * qsys.MW_h1,
            pulse_params={"f_pulse": qsys.MW_freqs[1]},
            time_steps=100,
        )
        XY_15N.run()
        XY_analysis = Analysis(XY_15N)
        XY_analysis.run_fit(fit_model=GaussianModel())
        assert np.allclose(
            [
                XY_analysis.fit_params.best_values["center"],
                XY_analysis.fit_params.best_values["amplitude"],
            ],
            [0.317, 0.010],
            atol=1e-3,
        )

    # the sequence is symmetric, therefore it lasts 2*M*tau with both realistic and delta pulses
    def test_duration(self, qsys):
        tau, M = 2.0, 2
        for pulses in (
            {"pi_pulse_duration": 0.2, "h1": jmat(1 / 2, "x") * 2, "pulse_params": {"f_pulse": 1}},
            {"pi_pulse_duration": 0, "Rx": jmat(1 / 2, "x") * 2, "Ry": jmat(1 / 2, "y") * 2},
        ):
            XY_sim = XY(free_duration=np.array([tau]), system=qsys, M=M, **pulses)
            XY_sim.sequence(tau)
            assert np.isclose(XY_sim.total_time, 2 * M * tau)


class TestXY8:
    @pytest.mark.slow

    # Runs the XY8 sequence on an NV object
    # and checks if the center of the peak is in the expected position.
    # We don't use an outside fixture here since we need handcrafted values for this test
    def test_xy8(self):
        qsys = NV(
            N=15,
            B0=39.4,
            units_B0="mT",
            theta=2.6,
            units_angles="deg",
        )
        w1 = 20
        XY8_15N = XY8(
            M=2,
            free_duration=np.linspace(0.25, 0.36, 30),
            pi_pulse_duration=1 / 2 / w1,
            system=qsys,
            h1=w1 * qsys.MW_h1,
            pulse_params={"f_pulse": qsys.MW_freqs[1]},
            time_steps=100,
        )
        XY8_15N.run()
        XY8_analysis = Analysis(XY8_15N)
        XY8_analysis.run_fit(fit_model=GaussianModel())
        assert np.allclose(
            [
                XY8_analysis.fit_params.best_values["center"],
                XY8_analysis.fit_params.best_values["amplitude"],
            ],
            [0.297, 0.024],
            atol=1e-3,
        )

    def test_duration(self, qsys):
        tau, M = 2.0, 1
        for pulses in (
            {"pi_pulse_duration": 0.2, "h1": jmat(1 / 2, "x") * 2, "pulse_params": {"f_pulse": 1}},
            {"pi_pulse_duration": 0, "Rx": jmat(1 / 2, "x") * 2, "Ry": jmat(1 / 2, "y") * 2},
        ):
            XY8_sim = XY8(free_duration=np.array([tau]), system=qsys, M=M, **pulses)
            XY8_sim.sequence(tau)
            assert np.isclose(XY8_sim.total_time, 8 * M * tau)

    # the random phases are drawn once, so that all the points share the same sequence
    def test_RXY8(self, qsys):
        phases = []
        for _ in range(2):
            RXY8_sim = XY8(
                free_duration=np.array([2.0]),
                system=qsys,
                M=2,
                pi_pulse_duration=0.2,
                h1=jmat(1 / 2, "x") * 2,
                pulse_params={"f_pulse": 1},
                RXY8=True,
                seed=7,
            )
            phases.append([pulse["phi_t"] for pulse in RXY8_sim.pulse_params])
        assert phases[0] == phases[1]


class TestCPMG:
    # Runs the CPMG sequence on an NV object
    # and checks if the center of the peak is in the expected position.
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_cpmg(self):
        qsys = NV(
            N=15,
            B0=40,
            units_B0="mT",
            theta=2,
            units_angles="deg",
        )
        w1 = 40
        cpmg = CPMG(
            free_duration=np.linspace(0.2, 0.5, 30),
            system=qsys,
            M=2,
            pi_pulse_duration=1 / 2 / w1,
            pulse_params={"f_pulse": qsys.MW_freqs[1]},
            h1=w1 * qsys.MW_h1,
        )
        cpmg.run()
        cpmg_analysis = Analysis(cpmg)
        cpmg_analysis.run_fit(fit_model=GaussianModel())
        assert np.allclose(
            [
                cpmg_analysis.fit_params.best_values["center"],
                cpmg_analysis.fit_params.best_values["amplitude"],
            ],
            [0.353, 0.003],
            atol=1e-3,
        )

    def test_duration(self, qsys):
        tau, M = 2.0, 2
        for pulses in (
            {"pi_pulse_duration": 0.2, "h1": jmat(1 / 2, "x") * 2, "pulse_params": {"f_pulse": 1}},
            {"pi_pulse_duration": 0, "Rx": jmat(1 / 2, "x") * 2, "Ry": jmat(1 / 2, "y") * 2},
        ):
            CPMG_sim = CPMG(free_duration=np.array([tau]), system=qsys, M=M, **pulses)
            CPMG_sim.sequence(tau)
            assert np.isclose(CPMG_sim.total_time, M * tau)

    # tau shorter than twice the pi pulse gives negative free evolutions
    def test_short_tau(self, qsys):
        with pytest.warns(UserWarning, match="free evolution time must be larger"):
            CPMG(
                free_duration=np.array([1.5]),
                system=qsys,
                M=1,
                pi_pulse_duration=1,
                h1=jmat(1 / 2, "x") * 2,
                pulse_params={"f_pulse": 1},
            )


class TestPODMR:
    @pytest.mark.slow

    # Runs the PODMR sequence on an NV object
    # and checks if the frequencies are close to the expected values
    # We don't use an outside fixture here since we need handcrafted values for this test
    def test_podmr(self):
        qsys = NV(N=15, B0=40, units_B0="mT")
        w1 = 0.3

        podmr_exp = PMR(
            frequencies=np.arange(1747, 1751.5, 0.5),
            pulse_duration=1 / 2 / w1,
            system=qsys,
            h1=w1 * qsys.MW_h1,
        )

        podmr_exp.run()
        podmr_analysis = Analysis(podmr_exp)

        podmr_analysis.run_fit(
            fit_model=Model(fit_two_lorentz_sym),
            guess={"A": 0.5, "gamma": 0.2, "f_mean": 1749, "f_delta": 3, "C": 1},
        )
        assert np.isclose(
            podmr_analysis.fit_params.best_values["f_mean"], 1.749e3, atol=1e-3
        )
        assert np.isclose(podmr_analysis.fit_params.best_values["f_delta"], 3., atol=1e-3)


class TestExpData:
    def test_expdata(self):
        qsys_exp = NV(
            N=15,
            units_B0="mT",
            B0=38.4,
        )
        exp_data = ExpData(file_path="./tests/data/Ex02_NV_rabi.dat")
        w1_exp = 16.72

        rabi_sim_exp = Rabi(
            pulse_duration=np.arange(0, 0.15, 3e-3),
            system=qsys_exp,
            h1=w1_exp * qsys_exp.MW_h1,
            pulse_params={"f_pulse": qsys_exp.MW_freqs[0]},
        )

        rabi_sim_exp.run()
        rabi_analysis_exp = Analysis(rabi_sim_exp)
        exp_data.variable *= 1e6
        rabi_analysis_exp.compare_with(exp_data)
        assert np.isclose(rabi_analysis_exp.pearson.slope, 2.5895, atol=1e-3)
        assert np.isclose(
            rabi_analysis_exp.pearson.intercept, -2.0453, atol=1e-3
        )

    def test_subtract(self):
        f = ExpData(file_path="./tests/data/xy82.dat", results_columns=[0, 1])
        sig = f.results[1] - f.results[0]
        f.subtract_results_columns(pos_col=1, neg_col=0)
        assert np.allclose(sig, f.results)

    def test_rescale(self):
        exp_data = ExpData(file_path="./tests/data/Ex02_NV_rabi.dat")
        data_scaled = 2 * exp_data.results
        exp_data.rescale_correction(2)
        assert np.allclose(data_scaled, exp_data.results)

    def test_offset(self):
        exp_data = ExpData(file_path="./tests/data/Ex02_NV_rabi.dat")
        data_corr = exp_data.results - 2
        exp_data.offset_correction(2)
        assert np.allclose(data_corr, exp_data.results)

    # a purely polynomial baseline has to be corrected to zero, on single and multiple columns
    def test_poly_base(self):
        exp_data = ExpData(file_path="./tests/data/xy82.dat")
        exp_data.results = 3 * exp_data.variable + 5
        exp_data.poly_base_correction(poly_order=1)
        assert np.allclose(exp_data.results, 0, atol=1e-6)

        columns = ExpData(file_path="./tests/data/xy82.dat", results_columns=[1, 2])
        columns.results = [3 * columns.variable + 5, columns.variable - 2]
        columns.poly_base_correction(poly_order=1)
        assert np.allclose(columns.results, 0, atol=1e-6)


class TestP1:
    def test_P1(self):
        B0 = (3911 - 1827) / 2 / 28.025
        w2 = 3
        freqs = np.linspace(900, 1200, 100)
        theta = 0
        phi = 0

        qsys = P1(
            B0=B0,
            rot_index=1,
            observable=1,
            N=14,
            theta=theta,
            phi_r=phi,
            theta_1=90 + theta,
            phi_r_1=phi,
        )

        sim = PMR(
            frequencies=freqs,
            pulse_duration=1 / 2 / w2,
            system=qsys,
            h1=w2 * qsys.h1,
        )

        sim.run()
        analysis = Analysis(sim)
        analysis.run_fit(
            fit_model=Model(fit_sinc2), guess={"A": 0.33, "gamma": 5, "f0": 1050, "C": 1}
        )
        # the fitted gamma recovers the Rabi frequency w2. It moved from 3.003 to 3.0046 when the
        # PMR pulse was corrected to start at t=0 instead of at pulse_duration, as the phase of the
        # drive at the beginning of the pulse enters through the counter rotating terms
        assert np.isclose(
            analysis.fit_params.best_values["gamma"], 3.0046, atol=1e-3
        )
        assert np.isclose(analysis.fit_params.best_values["f0"], 1050.85, atol=1e-3)


class TestSiBiFlux:
    def test_flux_only(self):
        Delta, epsilon = 7257.0, 400.0
        sys = SiBiFlux(
            rho0=basis(2, 0),
            observable=fock_dm(2, 0),
            Delta=Delta,
            epsilon=epsilon,
            spin=False,
        )
        assert sys.H0.shape == (2, 2)
        assert np.isclose(
            sys.energy_levels[1] - sys.energy_levels[0],
            np.sqrt(Delta**2 + epsilon**2),
            atol=1e-3,
        )

    def test_flux_spin_nuclear_dims(self):
        sys = SiBiFlux(
            rho0=tensor(basis(2, 0), basis(2, 0), basis(10, 0)),
            observable=tensor(jmat(1 / 2, "z"), qeye(2), qeye(10)),
            Delta=7257.0,
            B0=261.0,
            g=4 * 1.8,
            N=True,
        )
        assert sys.H0.dims == [[2, 2, 10], [2, 2, 10]]

    # the hyperfine coupling gives the known 7.377 GHz zero field splitting of the bismuth donor
    def test_hyperfine(self):
        sys = SiBiFlux(
            rho0=tensor(basis(2, 0), basis(10, 0)),
            observable=tensor(jmat(1 / 2, "z"), qeye(10)),
            flux=False,
            N=True,
            B0=0,
        )
        levels = np.unique(np.round(sys.energy_levels, 6))
        assert np.isclose(levels[-1] - levels[0], 7377, atol=1e-3)


class TestSpinLocking:
    @pytest.mark.slow
    def test_flipflop_freq(self):
        Delta, Omega, g_paper = 7257.0, 60.5, 1.8
        gamma_e = cte.value("electron gyromag. ratio in MHz/T") * 1e-3

        sys = SiBiFlux(
            rho0=tensor(basis(2, 0), basis(2, 0)),
            observable=tensor(jmat(1 / 2, "z"), qeye(2)),
            Delta=Delta,
            B0=(Delta + Omega) / gamma_e,
            g=4 * g_paper,
            N=False,
        )

        sim = SpinLocking(
            pulse_duration=np.linspace(0, 2, 20),
            system=sys,
            pi_pulse_duration=1 / 2 / Omega,
            h1=2 * Omega * tensor(jmat(1 / 2, "x"), qeye(2)),
            pulse_params={"f_pulse": Delta},
            options={"nsteps": 1e6},
        )
        sim.run()

        analysis = Analysis(sim)
        analysis.run_FFT()
        assert np.isclose(analysis.get_peaks_FFT()[0], g_paper, atol=0.1)

    # with delta pulses h1 is still needed, as it drives the system during the locking,
    # and it must not be added to the H0 of the system given by the user
    def test_delta_pulse(self, qsys):
        with pytest.raises(ValueError, match="h1 must still be given"):
            SpinLocking(
                pulse_duration=np.linspace(0, 2, 5),
                system=qsys,
                pi_pulse_duration=0,
                Ry=jmat(1 / 2, "y") * 2,
            )

        H0 = qsys.H0.copy()
        locking = SpinLocking(
            pulse_duration=np.linspace(0, 2, 5),
            system=qsys,
            pi_pulse_duration=0,
            Ry=jmat(1 / 2, "y") * 2,
            h1=0.1 * jmat(1 / 2, "x") * 2,
        )
        assert qsys.H0 == H0
        assert locking.system.H0 != H0


class TestPulseShapes:
    def test_lorentzian(self):
        value = lorentzian_pulse(np.array([0.0]), t_mid=0.5, gamma=1, f_pulse=1, phi_t=0)
        assert np.allclose(value, 1 / (1 + 0.5**2))


class TestPulsedSim:
    # the sensing Hamiltonian must stay a single term of Ht, also with several control Hamiltonians
    def test_H2(self, qsys):
        def sensing(t, **kwargs):  # noqa: ARG001
            return np.cos(t)

        rabi_H2 = Rabi(
            pulse_duration=np.linspace(0, 4, 10),
            system=qsys,
            h1=[0.1 * jmat(1 / 2, "x"), 0.1 * jmat(1 / 2, "y")],
            pulse_shape=[square_pulse, square_pulse],
            H2=[0.01 * jmat(1 / 2, "z"), sensing],
            pulse_params={"f_pulse": 1},
        )
        assert sensing not in rabi_H2.Ht
        rabi_H2.run()
        assert len(rabi_H2.results) == 10

    # a pulse must be appended to the profiles, without discarding the previous operations
    def test_pulse_profiles(self, qsys):
        seq = PulsedSim(qsys)
        seq.add_free_evolution(1)
        free_evo = seq.pulse_profiles[0]
        seq.add_pulse(
            1,
            [0.1 * jmat(1 / 2, "x"), 0.1 * jmat(1 / 2, "y")],
            pulse_shape=[square_pulse, square_pulse],
            pulse_params={"f_pulse": 1},
            time_steps=10,
        )
        assert len(seq.pulse_profiles) == 2
        assert seq.pulse_profiles[0] is free_evo

    # the sequences reset rho and the clock, so that they can also be called outside of run
    def test_reset(self, qsys):
        qsys.c_ops = 0.1 * jmat(1 / 2, "z") * 2
        free_duration = np.linspace(5, 25, 4)
        hahn_reset = Hahn(
            free_duration=free_duration,
            system=qsys,
            pi_pulse_duration=0,
            Rx=jmat(1 / 2, "x") * 2,
        )
        hahn_reset.run()
        direct = [
            np.real((hahn_reset.hahn_sequence(tau) * qsys.observable).tr()) for tau in free_duration
        ]
        assert np.allclose(hahn_reset.results, direct)


class TestAnalysis:
    # the comparison must not overwrite the results of the experiment given by the user
    def test_compare_with(self):
        experiment = ExpData(file_path="./tests/data/xy82.dat", results_columns=1)
        comparison = ExpData(file_path="./tests/data/xy82.dat", results_columns=2)
        results = comparison.results.copy()
        analysis = Analysis(experiment)
        analysis.compare_with(comparison)
        assert np.array_equal(results, comparison.results)
        assert analysis.comparison_results is not None

    # ExpData has no system attribute, the results are iterated instead of the observables
    def test_plot_columns(self):
        Analysis(ExpData(file_path="./tests/data/xy82.dat", results_columns=[1, 2])).plot_results()

    def test_figsize(self):
        plt.close("all")
        plot_histogram(fock_dm(2, 0), figsize=(9, 9))
        assert np.allclose(plt.gcf().get_size_inches(), (9, 9))
        with pytest.raises(ValueError, match="figsize"):
            plot_histogram(fock_dm(2, 0), figsize=(9, 9, 9))


########################################################################
# Standalone tests
########################################################################

# multiprocessing used by parallel_map doesn't seem to work without the following mess
# https://stackoverflow.com/questions/72766345/attributeerror-cant-pickle-local-object-in-multiprocessing
NV_pulse = NV(B0=25, units_B0="mT", N=14)
NV_pulse.truncate(mS=1, mI=1)


def hadamard_phi(phi_rf, **kwargs):
    sol_opt = {"nsteps": 1e6}
    seq_in = PulsedSim(NV_pulse)

    seq_in.add_pulse(
        kwargs["tpi_cnot"],
        kwargs["h1_cnot"],
        pulse_params={"f_pulse": kwargs["w0_cnot"], "phi_t": np.pi / 2},
        options=sol_opt,
    )

    seq_in.add_pulse(
        kwargs["tpi_rf"] / 2,
        kwargs["h1_rf"],
        pulse_params={"f_pulse": kwargs["w0_rf"], "phi_t": phi_rf},
        options=sol_opt,
    )
    seq_in.add_pulse(
        kwargs["tpi_mwa"],
        kwargs["h1_mwa"],
        pulse_params={"f_pulse": kwargs["w0_mwa"], "phi_t": np.pi / 2},
        options=sol_opt,
    )
    seq_in.add_pulse(
        kwargs["tpi_rf"] / 2,
        kwargs["h1_rf"],
        pulse_params={"f_pulse": kwargs["w0_rf"], "phi_t": phi_rf},
        options=sol_opt,
    )

    return seq_in.rho


def test_add_pulse():
    # Rabi frequencies of the spins in MHz
    w1_rf = 0.2
    w1_mwa = 16
    w1_cnot = 2.14 / 3**0.5

    # pi pulse times
    tpi_rf = 1 / (2 * w1_rf)
    tpi_mwa = 1 / (2 * w1_mwa)
    tpi_cnot = 1 / (2 * w1_cnot)

    # Larmor frequencies of the spins
    w0_rf = NV_pulse.RF_freqs[2]
    w0_mwa = NV_pulse.MW_freqs[0]
    w0_cnot = NV_pulse.energy_levels[2]

    # Hamiltonian terms for the RF and MW pulses

    NV_pulse.rho0 = tensor(basis(2, 0) - basis(2, 1), basis(2, 0) - basis(2, 1)).unit()

    NV_pulse.observable = [
        tensor(fock_dm(2, 0), fock_dm(2, 0)),
        tensor(fock_dm(2, 0), fock_dm(2, 1)),
        tensor(fock_dm(2, 1), fock_dm(2, 0)),
        tensor(fock_dm(2, 1), fock_dm(2, 1)),
    ]

    phi_array = np.arange(1.5, 3.3, 0.1)

    seq_phi = PulsedSim(NV_pulse)
    seq_args = {
        "tpi_rf": tpi_rf,
        "tpi_mwa": tpi_mwa,
        "tpi_cnot": tpi_cnot,
        "w0_rf": w0_rf,
        "w0_mwa": w0_mwa,
        "w0_cnot": w0_cnot,
        "h1_rf": w1_rf * NV_pulse.RF_h1,
        "h1_mwa": w1_mwa * NV_pulse.MW_h1,
        "h1_cnot": w1_cnot * NV_pulse.MW_h1,
    }

    seq_phi.run(phi_array, hadamard_phi, sequence_kwargs=seq_args)

    phi_rf = phi_array[np.argmax(seq_phi.results[0] ** 2 + seq_phi.results[3] ** 2)]
    assert np.isclose(phi_rf, 2.9)


def custom_Hahn(tau, **kwargs):
    ps = tau - kwargs["t_pi"]

    seq = PulsedSim(kwargs["qsys"])

    seq.add_pulse(
        duration=kwargs["t_pi"] / 2,
        h1=kwargs["h1"],
        pulse_shape=kwargs["pulse_shape"],
        pulse_params={"f_pulse": kwargs["delta"]},
    )
    seq.add_free_evolution(duration=ps)
    seq.add_pulse(
        duration=kwargs["t_pi"],
        h1=kwargs["h1"],
        pulse_shape=kwargs["pulse_shape"],
        pulse_params={"f_pulse": kwargs["delta"]},
    )
    seq.add_free_evolution(duration=ps)
    seq.add_pulse(
        duration=3 * kwargs["t_pi"] / 2,
        h1=kwargs["h1"],
        pulse_shape=kwargs["pulse_shape"],
        pulse_params={"f_pulse": kwargs["delta"]},
    )

    return seq.rho


def test_add_free_evolution():
    w0 = 1
    w1 = w0 / 10

    qsys = QSys(
        H0=w0 * jmat(1 / 2, "z"),
        rho0=basis(2, 0),
        observable=jmat(1 / 2, "z") * 2,
        units_H0="MHz",
    )
    sequence_kwargs = {
        "qsys": qsys,
        "h1": w1 * jmat(1 / 2, "x") * 2,
        "pulse_shape": square_pulse,
        "delta": w0,
        "t_pi": 1 / 2 / w1,
        "w1": w1,
    }

    custom_seq = PulsedSim(qsys)
    custom_seq.run(
        variable=np.linspace(5, 25, 30), sequence=custom_Hahn, sequence_kwargs=sequence_kwargs
    )
    analysis = Analysis(custom_seq)
    analysis.run_fit(fit_model=RabiModel())
    assert np.isclose(analysis.fit_params.best_values["amp"], 0.000623, rtol=1e-3)
    assert np.isclose(
        analysis.fit_params.best_values["Tpi"], 0.714, atol=1e-3
    )


def test_save_load(tmp_path):
    qsys_saved = NV(
        B0=4.2, units_B0="mT", theta=-45, units_angles="deg", N=14, temp=300, units_temp="K"
    )
    psim_saved = XY8(
        M=2,
        free_duration=np.linspace(0.25, 0.36, 30),
        pi_pulse_duration=1 / 2 / 10,
        system=qsys_saved,
        h1=10 * qsys_saved.MW_h1,
        pulse_params={"f_pulse": qsys_saved.MW_freqs[1]},
        time_steps=100,
    )

    qsys_path = str(tmp_path / "test_qsys")
    psim_path = str(tmp_path / "test_psim")

    save_quaccatoo(qsys_saved, qsys_path)
    save_quaccatoo(psim_saved, psim_path)

    loaded_qsys = load_quaccatoo(qsys_path)
    loaded_psim = load_quaccatoo(psim_path)

    assert set(loaded_qsys.__dict__.keys()) == set(qsys_saved.__dict__.keys())
    assert set(loaded_psim.__dict__.keys()) == set(psim_saved.__dict__.keys())
