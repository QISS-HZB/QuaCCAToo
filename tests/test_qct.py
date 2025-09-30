import numpy as np
import pytest
from lmfit import Model
from qutip import basis, fock_dm, jmat, qeye, sigmax, sigmay, sigmaz, tensor

from quaccatoo import (
    CPMG,
    NV,
    PMR,
    XY,
    XY8,
    Analysis,
    ExpData,
    Hahn,
    QSys,
    Rabi,
    compose_sys,
    square_pulse,
    P1,
    PMR,
    PulsedSim,
)
from quaccatoo.analysis.fit_functions import (
    ExpDecayModel,
    GaussianModel,
    RabiModel,
    fit_two_lorentz_sym,
    fit_sinc2,
)

"""
The testing framework for QuaCCAToo.

We use `pytest` for testing. See https://docs.pytest.org/en/stable/getting-started.html
for a quick introduction to pytest.

The gist is to use fixtures (`@pytest.fixture` decorator) for objects which are needed
repeatedly, and group related tests into an appropriately named test class.

Remember to mark long running tests with the `@pytest.mark.slow` decorator. These can then be run
with the `--runslow` CLI flag passed to `pytest`.
"""


# QSys fixture for reuse in multiple tests
@pytest.fixture
def qsys():
    delta = 1
    return QSys(H0=delta / 2 * sigmaz(), rho0=fock_dm(2, 0), observable=sigmaz(), units_H0="MHz")


# Test if the eigenstates of the qsys fixture are correct
class TestQSys:
    def test_states(self, qsys):
        assert (qsys.eigenstates[0], qsys.eigenstates[1]) == (
            -basis(2, 1),
            -basis(2, 0),
        )

    def test_levels(self, qsys):
        assert np.array_equal(qsys.energy_levels, np.array([0, 1]))


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
        pulse_duration=np.linspace(0, 40, 1000),
        system=qsys,
        h1=[w1 * sigmax() / 2, w1 * sigmay() / 2],
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


class TestHahn:
    # @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")

    # check if the decay rate is close to the expected value
    def test_decay(self, qsys):
        w1 = 0.1
        delta = 1
        gamma = 0.1

        qsys.c_ops = gamma * sigmaz()
        hahn_exp = Hahn(
            free_duration=np.linspace(5, 25, 30),
            pi_pulse_duration=1 / 2 / w1,
            projection_pulse=True,
            system=qsys,
            h1=w1 * sigmax(),
            pulse_shape=square_pulse,
            pulse_params={"f_pulse": delta},
        )
        hahn_exp.run()
        hahn_analysis = Analysis(hahn_exp)
        hahn_analysis.run_fit(fit_model=ExpDecayModel())
        assert np.allclose(
            [hahn_analysis.fit_params.best_values["Tc"], hahn_analysis.fit_params.best_values["amp"]],
            [3.953, 1.905],
            atol=1e-3,
        )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_hahn_delta(self, qsys):
        gamma = 0.1
        qsys.c_ops = gamma * sigmaz()
        hahn_sim_delta = Hahn(
            free_duration=np.linspace(2.5, 25, 30), system=qsys, pi_pulse_duration=0, Rx=sigmax()
        )
        hahn_sim_delta.run()
        hahn_analysis = Analysis(hahn_sim_delta)
        hahn_analysis.run_fit(fit_model=ExpDecayModel())
        assert np.allclose(
            [hahn_analysis.fit_params.best_values["Tc"], hahn_analysis.fit_params.best_values["amp"]],
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
            free_duration=np.linspace(0.25, 0.36, 100),
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
            [XY_analysis.fit_params.best_values["center"], XY_analysis.fit_params.best_values["amplitude"]],
            [0.317, 0.010],
            atol=1e-3,
        )


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
            free_duration=np.linspace(0.25, 0.36, 100),
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
            free_duration=np.linspace(0.2, 0.5, 100),
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


class TestPODMR:
    @pytest.mark.slow

    # Runs the PODMR sequence on an NV object
    # and checks if the frequencies are close to the expected values
    # We don't use an outside fixture here since we need handcrafted values for this test
    def test_podmr(self):
        qsys = NV(N=15, B0=40, units_B0="mT")
        w1 = 0.3

        podmr_exp = PMR(
            frequencies=np.arange(1745, 1753, 0.05),
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
        ) and np.isclose(podmr_analysis.fit_params.best_values["f_delta"], 3.029, atol=1e-3)


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
        rabi_analysis_exp.pearson
        assert np.isclose(rabi_analysis_exp.pearson.slope, 2.5895, atol=1e-3) and np.isclose(
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


class TestP1:
    def test_P1(self):
        B0 = (3911 - 1827) / 2 / 28.025
        w2 = 3
        freqs = np.linspace(900, 1200, 500)
        theta = 0
        phi = 0

        qsys = P1(
            B0=B0, rot_index=1, observable=1, N=14, theta=theta, phi_r=phi, theta_1=90 + theta, phi_r_1=phi
        )

        sim = PMR(
            frequencies=freqs,
            pulse_duration=1 / 2 / w2,
            system=qsys,
            h1=w2 * qsys.h1,
        )

        sim.run()
        analysis = Analysis(sim)
        analysis.run_fit(fit_model=Model(fit_sinc2), guess={"A": 0.33, "gamma": 5, "f0": 1050, "C": 1})
        assert np.isclose(analysis.fit_params.best_values["gamma"], 3.003, atol=1e-3) and np.isclose(
            analysis.fit_params.best_values["f0"], 1050.85, atol=1e-3
        )


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

    phi_array = np.arange(0, 2 * np.pi, 0.1)

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
        H0=w0 / 2 * sigmaz(),
        rho0=basis(2, 0),
        observable=sigmaz(),
        units_H0="MHz",
    )
    sequence_kwargs = {
        "qsys": qsys,
        "h1": w1 * sigmax(),
        "pulse_shape": square_pulse,
        "delta": w0,
        "t_pi": 1 / 2 / w1,
        "w1": w1,
    }

    custom_seq = PulsedSim(qsys)
    custom_seq.run(variable=np.linspace(5, 25, 30), sequence=custom_Hahn, sequence_kwargs=sequence_kwargs)
    analysis = Analysis(custom_seq)
    analysis.run_fit(fit_model=RabiModel())
    assert np.isclose(analysis.fit_params.best_values["amp"], 0.000623, rtol=1e-3) and np.isclose(
        analysis.fit_params.best_values["Tpi"], 0.714, atol=1e-3
    )
