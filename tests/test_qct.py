import pytest

import numpy as np
from qutip import sigmax, sigmay, sigmaz, fock_dm, Qobj

from quaccatoo import QSys, Analysis, fit_rabi, Rabi

@pytest.fixture
def qsys():
    delta = 1
    return QSys(
        H0=delta/2 * sigmaz(),
        rho0=fock_dm(2, 0),
        observable=sigmaz(),
        units_H0='MHz'
    )

class TestQSys:
    def test_states(self, qsys):
        assert (qsys.eigenstates[0], qsys.eigenstates[1]) == (Qobj([[0,0],[0,1]]), Qobj([[1,0],[0,0]]))

    def test_levels(self, qsys):
        assert np.array_equal(qsys.energy_levels, np.array([0,1]))


@pytest.fixture
def rabi_exp(qsys):
    w1 = 0.1
    delta = 1

    def custom_pulseX(t):
        return np.cos(delta*t)

    def custom_pulseY(t):
        return np.cos(delta*t - np.pi/2)

    return Rabi(
        pulse_duration = np.linspace(0, 40, 1000),
        system = qsys,
        H1 = [w1*sigmax()/2, w1*sigmay()/2],
        pulse_shape = [custom_pulseX, custom_pulseY]
    )

class TestRabi:
    def test_tpi(self, rabi_exp):
        rabi_exp.run()
        rabi_analysis = Analysis(rabi_exp)
        rabi_analysis.run_fit(
            fit_function = fit_rabi,
            guess = [1, 1/2/0.1, 0, 0]
        )
        assert np.isclose(rabi_analysis.fit[1], 5, atol=1e-3)
