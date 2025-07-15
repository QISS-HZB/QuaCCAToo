import numpy as np
import pytest
from lmfit import Model
from qutip import basis, fock_dm, sigmax, sigmay, sigmaz

from quaccatoo import NV, PMR, XY8, Analysis, ExpData, Hahn, QSys, Rabi, square_pulse
from quaccatoo.analysis.fit_functions import (
    ExpDecayModel,
    GaussianModel,
    RabiModel,
    fit_two_lorentz_sym,
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


# Rabi object (fixture) used in the TestRabi class below
@pytest.fixture
def rabi_exp(qsys):
    w1 = 0.1
    delta = 1

    def custom_pulseX(t):
        return np.cos(delta * t)

    def custom_pulseY(t):
        return np.cos(delta * t - np.pi / 2)

    return Rabi(
        pulse_duration=np.linspace(0, 40, 1000),
        system=qsys,
        H1=[w1 * sigmax() / 2, w1 * sigmay() / 2],
        pulse_shape=[custom_pulseX, custom_pulseY],
    )


class TestRabi:
    # Uses the rabi fixture defined above to check if the rabi frequency fit
    # is close to the expected value
    def test_tpi(self, rabi_exp):
        rabi_exp.run()
        rabi_analysis = Analysis(rabi_exp)
        rabi_analysis.run_fit(fit_model=RabiModel())
        assert np.isclose(rabi_analysis.fit_params.best_values["Tpi"], 5, atol=1e-3)


# Hahn object (fixture) used in the Test class below
@pytest.fixture
def hahn_exp(qsys):
    w1 = 0.1
    delta = 1
    gamma = 0.1

    qsys.c_ops = gamma * sigmaz()

    return Hahn(
        free_duration=np.linspace(5, 25, 30),
        pi_pulse_duration=1 / 2 / w1,
        projection_pulse=True,
        system=qsys,
        H1=w1 * sigmax(),
        pulse_shape=square_pulse,
        pulse_params={"f_pulse": delta},
    )


class TestHahn:
    # @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")

    # Uses the Hahn fixture defined above to check if the decay rate
    # is close to the expected value
    def test_decay(self, hahn_exp):
        hahn_exp.run()
        hahn_analysis = Analysis(hahn_exp)
        hahn_analysis.run_fit(fit_model=ExpDecayModel())
        assert np.isclose(hahn_analysis.fit_params.best_values["Tc"], 3.953, atol=1e-3)


class TestXY8:
    @pytest.mark.slow

    # Runs the XY8 sequence on an NV object
    # and checks if the center of the peak is in the expected position.
    # We don't use an outside fixture here since we need handcrafted values for this test
    def test_xy8_exp(self):
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
            H1=w1 * qsys.MW_H1,
            pulse_params={"f_pulse": qsys.MW_freqs[1]},
            time_steps=100,
        )
        XY8_15N.run()
        XY8_analysis = Analysis(XY8_15N)
        XY8_analysis.run_fit(fit_model=GaussianModel())
        assert 0.29 <= XY8_analysis.fit_params.best_values["center"] <= 0.31


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
            H1=w1 * qsys.MW_H1,
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
        exp_data = ExpData(file_path="./tests/data//Ex02_NV_rabi.dat")
        w1_exp = 16.72

        rabi_sim_exp = Rabi(
            pulse_duration=np.arange(0, 0.15, 3e-3),
            system=qsys_exp,
            H1=w1_exp * qsys_exp.MW_H1,
            pulse_params={"f_pulse": qsys_exp.MW_freqs[0]},
        )

        rabi_sim_exp.run()
        rabi_analysis_exp = Analysis(rabi_sim_exp)
        exp_data.variable *= 1e6
        rabi_analysis_exp.compare_with(exp_data)
        rabi_analysis_exp.pearson
        print(rabi_analysis_exp.pearson.intercept)
        assert np.isclose(rabi_analysis_exp.pearson.slope, 2.5895, atol=1e-3) and np.isclose(
            rabi_analysis_exp.pearson.intercept, -2.0453, atol=1e-3
        )
