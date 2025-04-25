import pytest

import numpy as np
from qutip import sigmax, sigmay, sigmaz, fock_dm, Qobj

from quaccatoo import QSys, Analysis, Rabi, Hahn, square_pulse, NV, XY8, PMR
from quaccatoo.analysis.fit_functions import (
    fit_exp_decay,
    fit_gaussian,
    fit_rabi,
    fit_two_lorentz_sym,
)


@pytest.fixture
def qsys():
    delta = 1
    return QSys(
        H0=delta / 2 * sigmaz(), rho0=fock_dm(2, 0), observable=sigmaz(), units_H0="MHz"
    )


class TestQSys:
    def test_states(self, qsys):
        assert (qsys.eigenstates[0], qsys.eigenstates[1]) == (
            Qobj([[0, 0], [0, 1]]),
            Qobj([[1, 0], [0, 0]]),
        )

    def test_levels(self, qsys):
        assert np.array_equal(qsys.energy_levels, np.array([0, 1]))


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
    def test_tpi(self, rabi_exp):
        rabi_exp.run()
        w1 = 0.1
        rabi_analysis = Analysis(rabi_exp)
        rabi_analysis.run_fit(fit_function=fit_rabi, guess=[1, 1 / 2 / w1, 0, 0])
        assert np.isclose(rabi_analysis.fit[1], 5, atol=1e-3)


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
    def test_decay(self, hahn_exp):
        hahn_exp.run()
        hahn_analysis = Analysis(hahn_exp)
        hahn_analysis.run_fit(fit_function=fit_exp_decay)
        assert np.isclose(hahn_analysis.fit[-1], 3.953, atol=1e-3)


class TestXY8:
    @pytest.mark.slow
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
        XY8_analysis.run_fit(fit_function=fit_gaussian, guess=[0.8, 0.02, 0.3, 0.01])
        assert 0.29 <= XY8_analysis.fit[-2] <= 0.31


class TestPODMR:
    @pytest.mark.slow
    def test_podmr(self):
        qsys = NV(N=15, B0=40, units_B0="mT")
        w1 = 0.3

        podmr_exp_2 = PMR(
            frequencies=np.arange(1745, 1753, 0.05),
            pulse_duration=1 / 2 / w1,
            system=qsys,
            H1=w1 * qsys.MW_H1,
        )

        podmr_exp_2.run()
        podmr_analysis = Analysis(podmr_exp_2)

        podmr_analysis.run_fit(
            fit_function=fit_two_lorentz_sym,
            guess=[0.5, 0.2, 1749, 3, 1],
        )
        assert np.isclose(podmr_analysis.fit[-3], 1.749e3, atol=1e-3) and np.isclose(
            podmr_analysis.fit[-2], 3.029, atol=1e-3
        )
