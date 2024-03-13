import numpy as np
from qutip import jmat, tensor, qeye, fock_dm
import scipy as sp
import os

gamma_e = sp.constants.value('electron gyromag. ratio in MHz/T')
gamma_e = gamma_e/1e4       # MHz/gauss

gamma_N14 = -3.077/1e4      # MHz/gauss
gamma_N15 = 4.316/1e4       # MHz/gauss

# TODO constants for hyperfine params


def ZeroField(N, D=2.87e3, E=0):
    """Get the NV Hamiltonian term accounting for zero field splitting.

    Parameters
    ----------
    N : int
        Mass number of Nitrogen (either 14 (spin 1) or 15 (spin 1/2))
    D : float, optional
        Axial component of magnetic dipole-dipole interaction, by default 2.87e3 MHz (NV)
    E : float, optional
        Non axial compononet, by default 0. Usually much (1000x) smaller than `D`

    Returns
    -------
    qobj / ndarray
        The zero field hamiltonian term for the NV as a `qutip.qobj`

    Raises
    ------
    ValueError
        If the value of `N` isn't `14` or `15`
    """
    if N == 14:
        return tensor(D*(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3) + E*(jmat(1,'x')**2  - jmat(1,'y')**2), qeye(3))
    elif N == 15:
        return tensor(D*(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3) + E*(jmat(1,'x')**2  - jmat(1,'y')**2), qeye(2))
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")


def ElectronZeeman(N,B0,theta_deg=0,phi_deg=0):
    """Get the NV hamiltonian term accounting for the electron Zeeman effect.

    Parameters
    ----------
    N : int
        Mass number of Nitrogen (either 14 (spin 1) or 15 (spin 1/2))
    B0 : float
        Magnitude of the magnetic field (Gauss)
    theta_deg : float, optional
        Misalignment of the magnetic field in **degrees** with the z axis. The default is 0, which implies no misalignment.
    phi_deg : float, optional
        Misalignment of the magnetic field in **degrees** with the x axis. The default is 0, which implies no field along y axis.

    Returns
    -------
    qobj / ndarray
        The electron zeeman hamiltonian term for the NV as a `qutip.qobj`

    Raises
    ------
    ValueError
        If the value of `N` isn't `14` or `15`
    """
    theta = theta_deg*np.pi/180
    phi = phi_deg*np.pi/180
    if N == 14:
        return tensor(gamma_e*B0*(np.cos(theta)*jmat(1,'z') + np.sin(theta)*np.cos(phi)*jmat(1,'x') + np.sin(theta)*np.sin(phi)*jmat(1,'y')), qeye(3))
    if N == 15:
        return tensor(gamma_e*B0*(np.cos(theta)*jmat(1,'z') + np.sin(theta)*np.cos(phi)*jmat(1,'x') + np.sin(theta)*np.sin(phi)*jmat(1,'y')), qeye(2))
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")


def NuclearZeeman(N,B0,theta_deg=0,phi_deg=0):
    """Get the NV hamiltonian term accounting for the nuclear (Nitrogen) Zeeman effect.

    Parameters
    ----------
    N : int
        Mass number of Nitrogen (either 14 (spin 1) or 15 (spin 1/2))
    B0 : float
        Magnitude of the magnetic field (Gauss)
    theta_deg : float, optional
        Misalignment of the magnetic field in **degrees** with the z axis. The default is 0, which implies no misalignment.
    phi_deg : float, optional
        Misalignment of the magnetic field in **degrees** with the x axis. The default is 0, which implies no field along y axis.

    Returns
    -------
    qobj / ndarray
        The nuclear (Nitrogen) zeeman hamiltonian term for the NV as a `qutip.qobj`

    Raises
    ------
    ValueError
        If the value of `N` isn't `14` or `15`
    """
    theta = theta_deg*np.pi/180
    phi = phi_deg*np.pi/180
    if N == 14:
        return tensor(gamma_N14*B0*(np.cos(theta)*jmat(1,'z') + np.sin(theta)*np.cos(phi)*jmat(1,'x') + np.sin(theta)*np.sin(phi)*jmat(1,'y')), qeye(3))
    elif N == 15:
        return tensor(gamma_N15*B0*(np.cos(theta)*jmat(1,'z') + np.sin(theta)*np.cos(phi)*jmat(1,'x') + np.sin(theta)*np.sin(phi)*jmat(1,'y')), qeye(2))
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")


# TODO reference for correct values of the hyperfine tensor
def HyperFineN(N):
    """Get the NV hamiltonian term accounting for the hyperfine coupling with Nitrogen.

    Parameters
    ----------
    N : int
        Mass number of Nitrogen (either 14 (spin 1) or 15 (spin 1/2))

    Returns
    -------
    qobj / ndarray
        The Nitrogen hyperfine coupling hamiltonian term for the NV as a `qutip.qobj`

    Raises
    ------
    ValueError
        If the value of `N` isn't `14` or `15`
    """
    if N == 14:
        return -2.14*tensor(jmat(1,'z'), jmat(1,'z')) - 2.7*(tensor(jmat(1,'x'), jmat(1,'x')) + tensor(jmat(1,'y'), jmat(1,'y')))
    elif N == 15:
        return -3.03*tensor(jmat(1,'z'), jmat(1/2,'z')) - 3.65*(tensor(jmat(1,'x'), jmat(1/2,'x')) + tensor(jmat(1,'y'), jmat(1/2,'y')))
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")



def InitState(N):
    """Get the NV hamiltonian term accounting for the hyperfine coupling with Nitrogen.

    Parameters
    ----------
    N : int
        Mass number of Nitrogen (either 14 (spin 1) or 15 (spin 1/2))

    Returns
    -------
    qobj / ndarray
        The intial state (ms=0) of the NV as a `qutip.qobj`

    Raises
    ------
    ValueError
        If the value of `N` isn't `14` or `15`
    """
    if N == 14:
        return tensor(fock_dm(3,1), qeye(3)).unit()
    elif N == 15:
        return tensor(fock_dm(3,1), qeye(2)).unit()
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")


def GetPulsef1(N,H0):
    """_summary_

    Parameters
    ----------
    N : _type_
        _description_
    H0 : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
       If the value of `N` isn't `14` or `15`
    """
    H0_eig = H0.eigenenergies()
    if N == 14:
        return (np.sum(H0_eig[3:6]) - np.sum(H0_eig[0:3]))/3
    elif N == 15:
        return (np.sum(H0_eig[2:4]) - np.sum(H0_eig[0:2]))/2
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")


def GetPulsef2(N,H0):
    """_summary_

    Parameters
    ----------
    N : _type_
        _description_
    H0 : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        If the value of `N` isn't `14` or `15`
    """
    H0_eig = H0.eigenenergies()
    if N == 14:
        return (np.sum(H0_eig[6:9]) - np.sum(H0_eig[0:3]))/3
    elif N == 15:
        return (np.sum(H0_eig[4:6]) - np.sum(H0_eig[0:2]))/2
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")


def create_npz(name, npy_loc='.', npz_loc='.', clean=False):
    """Create a single npz archive from all the npy files in a directory.

    Parameters
    ----------
    name : string
        name of the `npz` file
    npy_loc : str, optional
        Path to the `npy` files, by default '.' (current dir)
    npz_loc : str, optional
        Path where the `npz` file is to be saved, by default '.' (current dir)
    clean : bool, optional
        Pass it as `True` to delete the `npy` files, by default False
    """
    files = sorted([f for f in os.listdir(npy_loc) if f.endswith('.npy')])
    data_dict = {}
    for file in files:
        data_dict[os.path.splitext(file)[0]] = np.load(f'{npy_loc}/{file}')
    np.savez(npz_loc+'/'+name, **data_dict)

    if clean:
        for file in files:
            os.remove(f'{npy_loc}/{file}')

def H1x(t, args) :
    return np.cos(t)


def GetTimeDepHamiltonian(H0,args):
    if string:
        return [H0, [tensor(),str()]]
    else:
        return [H0,[tensor(),H1x]]