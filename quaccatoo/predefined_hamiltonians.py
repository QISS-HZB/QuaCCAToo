from qutip import tensor, jmat, qeye, fock_dm, Qobj
import numpy as np
import scipy.constants as cte

def NV_H0(B0, N, theta=0, phi_r=0):
    """
    Get the NV time independent component of the Hamiltonian

    Parameters
    ----------
    B0 : float
        Magnetic field in Gauss
    theta : float
        Polar angle of the magnetic field
    phi_r : float
        Azimuthal angle of the magnetic field
    N : int
        Nitrogen isotope

    Returns
    -------
    H0 : Qobj

    """
    theta = np.deg2rad(theta)
    phi_r = np.deg2rad(phi_r)

    #Zero-Field Splitting
    H0 = 2.87e3*tensor(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3, qeye(2))

    #Electron Zeeman
    H0 += 28025e-4*B0*tensor( np.sin(theta)*np.cos(phi_r)*jmat(1, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1, 'y') + np.cos(theta)*jmat(1,'z'), qeye(2))

    if N == 15:
        #Hyperfine Coupling
        H0 += 3.03*tensor(jmat(1,'z'), jmat(1/2,'z')) + 3.65*(tensor(jmat(1,'x'), jmat(1/2,'x')) + tensor(jmat(1,'y'), jmat(1/2,'y')))

        #15N Nuclear Zeeman Interaction
        H0 += 4.316e-4*B0*tensor(qeye(3),  np.sin(theta)*np.cos(phi_r)*jmat(1/2, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1/2, 'y') + np.cos(theta)*jmat(1/2,'z'))

    elif N == 14:
        #Hyperfine Coupling
        H0 += -2.7*tensor(jmat(1,'z'), jmat(1,'z')) -2.14*(tensor(jmat(1,'x'), jmat(1,'x')) + tensor(jmat(1,'y'), jmat(1,'y')))

        #Quadrupole Splitting
        H0 += -5.01*tensor(qeye(3), jmat(1,'z')**2)

        #14N Nuclear Zeeman Interaction
        H0 += -3.077e-4*B0*tensor(qeye(3), np.cos(theta)*jmat(1,'z') + np.sin(theta)*jmat(1,'x'))
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")
    
    return H0

def NV_H1(w1, N):
    """
    Get the NV time dependent component of the Hamiltonian

    Parameters
    ----------
    w1 : float
        Rabi frequency
    
    Returns
    -------
    H1 : Qobj

    """
    if N == 15:
        return w1/2**2*tensor(jmat(1,'x'), qeye(2))
    elif N == 14:
        return w1/2**2*tensor(jmat(1,'x'), qeye(3))
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")
    
def NV_rho0_lowT(B0, N, T, theta=0, phi_r=0):
    """
    Get the initial state of the NV center T low temperatures

    Parameters
    ----------
    B0 : float
        Magnetic field in Gauss
    theta : float
        Polar angle of the magnetic field
    phi_r : float
        Azimuthal angle of the magnetic field
    N : int
        Nitrogen isotope
    T : float
        Temperature in Kelvin

    Returns
    -------
    rho0 : Qobj

    """
    return None
    
def NV_rho0(N):
    """
    Get the initial state of the NV center at room temperature

    Parameters
    ----------
    N : int
        Nitrogen isotope
    Returns
    -------
    rho0 : Qobj

    """
    if N == 15:
        return tensor(fock_dm(3, 1), qeye(2)).unit()
    elif N == 14:
        return tensor(fock_dm(3, 1), qeye(3)).unit()
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")
    
def NV_observable(N):
    """
    Get the observable for the NV center, i.e. the fluorescence or population in ms=0 state

    Parameters
    ----------
    N : int
        Nitrogen isotope
    Returns
    -------
    observable : Qobj

    """
    if N == 15:
        return tensor(fock_dm(3, 1), qeye(2))
    elif N == 14:
        return tensor(fock_dm(3, 1), qeye(3))
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")

def NV_MW_freq(B0, N, theta=0, phi_r=0):
    """
    Get the eigenvalues of the NV center Hamiltonian

    Parameters
    ----------
    B0 : float
        Magnetic field in Gauss
    theta : float
        Polar angle of the magnetic field
    phi_r : float
        Azimuthal angle of the magnetic field
    N : int
        Nitrogen isotope
    Returns
    -------
    eigenvalues : array

    """
    H0 = NV_H0(B0, theta, phi_r, N)
    eigenvalues = H0.eigenenergies()

    if N == 15:
        return (np.sum(eigenvalues[2:4]) - np.sum(eigenvalues[0:2]))/2, (np.sum(eigenvalues[4:6]) - np.sum(eigenvalues[0:2]))/2
    elif N == 14:
        return (np.sum(eigenvalues[3:6]) - np.sum(eigenvalues[0:3]))/3, (np.sum(eigenvalues[6:9]) - np.sum(eigenvalues[0:3]))/3
