from qutip import tensor, jmat, qeye, fock_dm, Qobj, basis
import numpy as np
import scipy.constants as cte

def NV_H0(B0, N, units_B0='G', theta=0, phi_r=0, units_H0='MHz', units_angles='deg'):
    """
    Gets the NV time independent component of the Hamiltonian.

    Parameters
    ----------
    B0 (float, int): Magnetic field in Gauss
    theta (float, int): Polar angle of the magnetic field
    units_B0 (str): Units of the magnetic field, either 'G', 'mT' or 'T'
    units_H0 (str): Units of the Hamiltonian, either 'MHz', 'GHz', 'kHz', 'J' or 'rad/s'
    phi_r (float, int): Azimuthal angle of the magnetic field
    N (14 or 15): Nitrogen isotope

    Returns
    -------
    H0 (Qobj): Time independent component of the Hamiltonian
    """
    # converts the magnetic field to Gauss
    if not isinstance(B0, (int, float)):
        raise ValueError(f"B0 must be a real number, got {B0}.")
    else:
        if units_B0 == 'T':
            B0 = B0*1e4
        elif units_B0 == 'mT':
            B0 = B0*10
        elif units_B0 == 'G':
            pass
        else:
            raise ValueError(f"Invalid value for units_B0. Expected either 'G', 'mT' or 'T', got {units_B0}.")
    if not isinstance(theta, (int, float)) or not isinstance(phi_r, (int, float)):
        raise ValueError(f"Invalid value for theta or phi_r. Expected a number, got {theta} or {phi_r}.")
    else:
        # converts the angles to radians
        if units_angles == 'deg':
            theta = np.deg2rad(theta)
            phi_r = np.deg2rad(phi_r)
        elif units_angles == 'rad':
            pass
        else:
            raise ValueError(f"Invalid value for units_angles. Expected either 'deg' or 'rad', got {units_angles}.")

    # calculates the Hamiltonian for the given field and nitrogen isotope
    if N == 15:
        H0 = (  2.87e3*tensor(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3, qeye(2)) # Zero Field Splitting
              
                + 28025e-4*B0*tensor( np.sin(theta)*np.cos(phi_r)*jmat(1, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1, 'y') + np.cos(theta)*jmat(1,'z'), qeye(2)) # Electron Zeeman

                + 3.03*tensor(jmat(1,'z'), jmat(1/2,'z')) + 3.65*(tensor(jmat(1,'x'), jmat(1/2,'x')) + tensor(jmat(1,'y'), jmat(1/2,'y'))) # Hyperfine Coupling

                + 4.316e-4*B0*tensor(qeye(3),  np.sin(theta)*np.cos(phi_r)*jmat(1/2, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1/2, 'y') + np.cos(theta)*jmat(1/2,'z')) ) # 15N Nuclear Zeeman

    elif N == 14:
        H0 = (  2.87e3*tensor(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3, qeye(3)) # Zero Field Splitting
              
                + 28025e-4*B0*tensor( np.sin(theta)*np.cos(phi_r)*jmat(1, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1, 'y') + np.cos(theta)*jmat(1,'z'), qeye(3)) # Electron Zeeman
                
                - 2.7*tensor(jmat(1,'z'), jmat(1,'z')) -2.14*(tensor(jmat(1,'x'), jmat(1,'x')) + tensor(jmat(1,'y'), jmat(1,'y'))) # Hyperfine Coupling
                                
                - 5.01*tensor(qeye(3), jmat(1,'z')**2) # Quadrupole Interaction

                - 3.077e-4*B0*tensor(qeye(3), np.cos(theta)*jmat(1,'z') + np.sin(theta)*jmat(1,'x'))) # 14N Nuclear Zeeman
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")
    
    # converts the Hamiltonian to the desired units
    if units_H0 == 'MHz':
        return H0
    if units_H0 == 'GHz':
        return H0/1e3
    if units_H0 == 'kHz':
        return H0*1e3
    if units_H0 == 'J':
        return H0*cte.h*1e6
    if units_H0 == 'rad/s':
        return 2*np.pi*H0*1e6
    else:
        raise ValueError(f"Invalid value for units_H0. Expected either 'MHz', 'GHz', 'kHz', 'J' or 'rad/s', got {units_H0}.")   

def NV_H1(w1, N):
    """
    Get the NV time dependent component of the Hamiltonian. H1 will have the same units as w1. In this case we assume that MW field has no effect on the nuclear spin.

    Parameters
    ----------
    w1 (float, int): Rabi frequency
    
    Returns
    -------
    H1 (Qobj): Time dependent component of the Hamiltonian
    """
    # check weather w1 is a number
    if not isinstance(w1, (int, float)):
        raise ValueError(f"w1 must be a real number.")
    else:
        # check the nitrogen isotope
        if N == 15:
            return w1/2**.5*tensor(jmat(1,'x'), qeye(2))
        elif N == 14:
            return w1/2*.5*tensor(jmat(1,'x'), qeye(3))
        else:
            raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")
    
def NV_rho0_lowT(B0, N, T, units_T='K', units_B0='G', theta=0, phi_r=0):
    """
    Get the initial state of the NV center T low temperatures and high fields by the Boltzmann distribution.

    Parameters
    ----------
    B0 (float, int): Magnetic field in Gauss
    theta (float, int): Polar angle of the magnetic field
    units_B0 (str): Units of the magnetic field, either 'G', 'mT' or 'T'
    phi_r (float, int): Azimuthal angle of the magnetic field
    N (14 or 15): Nitrogen isotope
    T (float, int): Temperature in Kelvin

    Returns
    -------
    rho0 (Qobj): Initial density matrix
    """
    # calculate the internal= Hamiltonian
    H0 = NV_H0(B0, N, units_B0, theta, phi_r, units_H0='J')

    # check if the temperature is a positive real number
    if not isinstance(T, (int, float)) and T > 0:
        raise ValueError(f"T must be a positive real number.")
    
    # check the units and convert the temperature to Kelvin
    if units_T == 'K':
        pass
    elif units_T == 'C':
        T += 273.15
    else:
        raise ValueError(f"Invalid value for units_T. Expected either 'K' or 'C', got {units_T}.")

    # get the eigenstates of the Hamiltonian
    H0_eig = H0.eigenstates()

    # a loop to find the |0,1/2> and |0,-1/2> states
    max_1= 0
    max_2= 0
    max_3 = 0
    index_1 = None
    index_2 = None
    index_3 = None
    # iterates over all the eigenstates and find the one closest related to the |0,1/2> and |0,-1/2> states
    for itr in range(len(H0_eig[0])):
        if N == 15:
            proj_1 = np.abs(H0_eig[1][itr].overlap(basis(6, 2)))
            proj_2 = np.abs(H0_eig[1][itr].overlap(basis(6, 3)))

        elif N == 14:
            proj_1 = np.abs(H0_eig[1][itr].overlap(basis(9, 3)))
            proj_2 = np.abs(H0_eig[1][itr].overlap(basis(9, 4)))
            proj_3 = np.abs(H0_eig[1][itr].overlap(basis(9, 5)))
            if proj_3 > max_3:
                # if the projection is higher than the previous maximum, update the maximum and the index
                max_3 = proj_3
                index_3 = itr

        if proj_1 > max_1:
            max_1 = proj_1
            index_1 = itr
        if proj_2 > max_2:
            max_2 = proj_2
            index_2 = itr

    beta = -1/cte.Boltzmann*T

    if N == 15:
        # calculate the partition function based on the Hamiltonian eigenvalues
        Z = np.exp(beta*H0_eig[0][index_1]) + np.exp(beta*H0_eig[0][index_2])
        
        return tensor(fock_dm(3,1), Qobj([ [np.exp(beta*H0_eig[0][index_1]), 0] , [0, np.exp(beta*H0_eig[0][index_2])] ])/ Z )
    
    elif N == 14:
        Z = np.exp(beta*H0_eig[0][index_1]) + np.exp(beta*H0_eig[0][index_2]) + np.exp(beta*H0_eig[0][index_3])

        return tensor(fock_dm(3,1), Qobj([ [np.exp(beta*H0_eig[0][index_1]), 0, 0] , [0, np.exp(beta*H0_eig[0][index_2]), 0], [0, 0, np.exp(beta*H0_eig[0][index_3])] ])/ Z ) 
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")
    
    
def NV_rho0(N):
    """
    Get the initial state of the NV center at room temperature and not so high fields. In this case the initial state of the nuclear spin is very well approximated by an identity matrix.

    Parameters
    ----------
    N (14 or 15): Nitrogen isotope

    Returns
    -------
    rho0 (Qobj): Initial density matrix
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
    Get the initial state of the NV center at room temperature

    Parameters
    ----------
    N (14 or 15): Nitrogen isotope

    Returns
    -------
    observable (Qobj): Initial density matrix
    """
    if N == 15:
        return tensor(fock_dm(3, 1), qeye(2))
    elif N == 14:
        return tensor(fock_dm(3, 1), qeye(3))
    else:
        raise ValueError(f"Invalid value for Nitrogen. Expected either 14 or 15, got {N}.")

def NV_MW_freqs(B0, N, units_B0='G', theta=0, phi_r=0, units_freq='MHz', units_angles='deg'):
    """
    Get the resonant frequencies of the NV center by calculating the energy difference between the different electronic spin states.

    Parameters
    ----------
    B0 (float, int): Magnetic field in Gauss
    theta (float, int): Polar angle of the magnetic field
    units_B0 (str): Units of the magnetic field, either 'G', 'mT' or 'T'
    phi_r (float, int): Azimuthal angle of the magnetic field
    N (14 or 15): Nitrogen isotope
    units_freq (str): Units of the frequencies, either 'MHz', 'GHz', 'kHz', 'J' or 'rad/s'

    Returns
    -------
    MW_freqs (float): Microwave frequencies of the given Hamiltonian
    """
    H0 = NV_H0(B0, N, units_B0, theta, phi_r, units_freq, units_angles)
    eigenvalues = H0.eigenenergies()

    if N == 15:
        return (np.sum(eigenvalues[2:4]) - np.sum(eigenvalues[0:2]))/2, (np.sum(eigenvalues[4:6]) - np.sum(eigenvalues[0:2]))/2
    elif N == 14:
        return (np.sum(eigenvalues[3:6]) - np.sum(eigenvalues[0:3]))/3, (np.sum(eigenvalues[6:9]) - np.sum(eigenvalues[0:3]))/3
