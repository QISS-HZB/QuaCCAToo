"""
This module collects the physical constants shared by the predefined quantum systems.

All gyromagnetic ratios are given in MHz/mT and all couplings in MHz, matching the default
units of the Hamiltonians in QuaCCAToo.
"""

import scipy.constants as cte

__all__ = []

####################################################################################################

# gyromagnetic ratios in MHz/mT
gamma_e = cte.value("electron gyromag. ratio in MHz/T") * 1e-3
gamma_N14 = 3.077e-3
gamma_N15 = -4.316e-3
gamma_Bi = 6.962e-3

# isotropic contact hyperfine coupling of the bismuth donor in silicon in MHz
A_Bi = 1475.4
