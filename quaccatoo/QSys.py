from qutip import tensor, jmat, qeye, fock_dm, Qobj, basis
import numpy as np
import scipy.constants as cte
import matplotlib.pyplot as plt
import warnings

def plot_energy_B0(B0, H0, figsize=(6, 4), energy_lim = None, xlabel='Magnetic Field', ylabel='Energy (MHz)'):
    """
    
    """
    # check if figsize is a tuple of two positive floats
    if not (isinstance(figsize, tuple) or len(figsize) == 2):
        raise ValueError("figsize must be a tuple of two positive floats")
    
    # check if B0 is a numpy array or list and if all elements are real numbers
    if not isinstance(B0 , (np.ndarray, list)) and all(isinstance(b, (int, float)) for b in B0):
        raise ValueError("B0 must be a list or a numpy array of real numbers")
    
    # check if H0 is a list of Qobj of the same size as B0
    if not isinstance(H0, list) or not all(isinstance(h, Qobj) for h in H0) or len(H0) != len(B0):
        raise ValueError("H0 must be a list of Qobj of the same size as B0")
    
    energy_levels = []
    
    for itr_B0 in range(len(B0)):
        H0_eig = H0[itr_B0].eigenenergies()
        energy_levels.append(H0_eig - H0_eig[0])

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for itr_e in range(len(energy_levels[0])):
        ax.plot(B0, [energy_levels[itr_B0][itr_e] for itr_B0 in range(len(B0))])

    if isinstance(energy_lim, tuple) and len(energy_lim) == 2:
        ax.set_ylim(energy_lim)
    
    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)
    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel)

    fig.suptitle('Energy Levels')


class QSys:
    def __init__(self, H0, rho0, c_ops=None, observable=None, units_H0='MHz'):

        # check if rho0 and H0 are Qobj and if they have the same dimensions
        if not isinstance(rho0, Qobj) or not isinstance(H0, Qobj):
            raise ValueError("H0 and rho0 must be a Qobj")
        else:
            if H0.shape != rho0.shape:
                raise ValueError("H0 and rho0 must have the same dimensions")
                # if they are correct, assign them to the objects
            else:
                self.rho0 = rho0
        
        # check if observable is not None, or if it is a Qobj of the same dimension as H0 and rho0, or a list of Qobj
        if observable is None:
            self.observable = None
        elif (isinstance(observable, Qobj) and observable.shape == H0.shape) or (isinstance(observable, list) and all(isinstance(obs, Qobj) for obs in observable) and all(obs.shape == H0.shape for obs in observable)):
            self.observable = observable
        else:
            raise ValueError("Invalid value for observable. Expected a Qobj or a list of Qobj of the same dimensions as H0 and rho0.")
        
        # check if c_ops is a list of Qobj with the same dimensions as H0
        if c_ops == None:
            self.c_ops = c_ops
        elif isinstance(c_ops, list):
            if not all(isinstance(op, Qobj) and op.shape == self.H0.shape for op in c_ops):
                raise ValueError("All items in c_ops must be Qobj with the same dimensions as H0")
            else:
                self.c_ops = c_ops
        else:
            raise ValueError("c_ops must be a list of Qobj or None")
        
        # if the units are in frequency, assign the Hamiltonian as it is
        if not isinstance(units_H0, str):
            raise ValueError("units_H0 must be a string")
        elif units_H0 == 'MHz' or units_H0 == 'GHz' or units_H0 == 'kHz' or units_H0 == 'Hz' or units_H0 == 'eV':
            pass
        else:
            warnings.warn(f"Warning: Invalid value for units_H0. Expected either units of frequencies or 'eV', got {units_H0}. The Hamiltonian will be considered in MHz.")

        self.H0 = H0
        self.units_H0 = units_H0
        
        # calculate the eigenstates of the Hamiltonian
        H_eig = H0.eigenenergies()
        # substracts the ground state energy from all the eigenstates to get the lowest level at 0
        self.energy_levels = H_eig - H_eig[0]
        
    def plot_energy(self, figsize=(2, 6), energy_lim = None):
        """
        """
        # check if figsize is a tuple of two positive floats
        if not (isinstance(figsize, tuple) or len(figsize) == 2):
            raise ValueError("figsize must be a tuple of two positive floats")

        fig, ax = plt.subplots(figsize)

        for itr in range(self.energy_levels.size):
            ax.axhline(y=self.energy_levels[itr])

        ax.set_ylabel(f'Energy ({self.units_H0})')
        ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if energy_lim == None:
            pass
        elif len(energy_lim) == 2:
            ax.set_xlim(energy_lim[0], energy_lim[1])
        else:
            raise ValueError("freq_lim must be a tuple of two floats")

class NV(QSys):
    def __init__(self, B0, N, units_B0='mT', theta=0, phi_r=0, units_angles='deg'):
        """
        """
        # converts the magnetic field to Gauss
        if not isinstance(B0, (int, float)):
            raise ValueError(f"B0 must be a real number, got {B0}.")
        else:
            if units_B0 == 'T':
                B0 = B0*1e3
            elif units_B0 == 'mT':
                B0 = B0
            elif units_B0 == 'G':
                B0 = B0*1e-3
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
            
        # 
        self.units_H0 = 'MHz'
            
        # calculates the Hamiltonian for the given field and nitrogen isotope
        if N == 15:
            self.H0 = (  2.87e3*tensor(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3, qeye(2)) # Zero Field Splitting                
                    + 28.025*B0*tensor( np.sin(theta)*np.cos(phi_r)*jmat(1, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1, 'y') + np.cos(theta)*jmat(1,'z'), qeye(2)) # Electron Zeeman
                    + 3.03*tensor(jmat(1,'z'), jmat(1/2,'z')) + 3.65*(tensor(jmat(1,'x'), jmat(1/2,'x')) + tensor(jmat(1,'y'), jmat(1/2,'y'))) # Hyperfine Coupling
                    + 4.316e-3*B0*tensor(qeye(3),  np.sin(theta)*np.cos(phi_r)*jmat(1/2, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1/2, 'y') + np.cos(theta)*jmat(1/2,'z')) ) # 15N Nuclear Zeeman
            
            self.rho0 = tensor(fock_dm(3,1), qeye(2)).unit()
            self.observable = tensor(fock_dm(3,1), qeye(2))

        elif N == 14:
            self.H0 = (  2.87e3*tensor(jmat(1,'z')**2 - (jmat(1,'x')**2 + jmat(1,'y')**2 + jmat(1,'z')**2)/3, qeye(3)) # Zero Field Splitting                
                    + 28.025*B0*tensor( np.sin(theta)*np.cos(phi_r)*jmat(1, 'x') + np.sin(theta)*np.sin(phi_r)*jmat(1, 'y') + np.cos(theta)*jmat(1,'z'), qeye(3)) # Electron Zeeman                   
                    - 2.7*tensor(jmat(1,'z'), jmat(1,'z')) -2.14*(tensor(jmat(1,'x'), jmat(1,'x')) + tensor(jmat(1,'y'), jmat(1,'y'))) # Hyperfine Coupling                                    
                    - 5.01*tensor(qeye(3), jmat(1,'z')**2) # Quadrupole Interaction
                    - 3.077e-3*B0*tensor(qeye(3), np.cos(theta)*jmat(1,'z') + np.sin(theta)*jmat(1,'x'))) # 14N Nuclear Zeeman
            
            self.rho0 = tensor(fock_dm(3,1), qeye(3)).unit()
            self.observable = tensor(fock_dm(3,1), qeye(3))
        else:
            raise ValueError(f"Invalid value for Nitrogen isotope. Expected either 14 or 15, got {N}.")
        
        self.N = N
        # get the eigenstates of the Hamiltonian
        H0_eig = self.H0.eigenstates()
        self.energy_levels = H0_eig[0] - H0_eig[0][0]
    
    def rho0_lowT(self, T, units_T='K'):

        
        # check the units and convert the temperature to Kelvin
        if units_T == 'K':
            pass
        elif units_T == 'C':
            T += 273.15
        else:
            raise ValueError(f"Invalid value for units_T. Expected either 'K' or 'C', got {units_T}.")

        # check if the temperature is a positive real number
        if not isinstance(T, (int, float)) and T > 0:
            raise ValueError(f"T must be a positive real number.")
        
        # a loop to find the |0,1/2> and |0,-1/2> states
        max_1= 0
        max_2= 0
        max_3 = 0
        index_1 = None
        index_2 = None
        index_3 = None
        # iterates over all the eigenstates and find the one closest related to the |0,1/2> and |0,-1/2> states
        for itr in range(len(self.energy_levels)):
            if self.N == 15:
                proj_1 = np.abs(self.energy_levels[itr].overlap(basis(6, 2)))
                proj_2 = np.abs(self.energy_levels[itr].overlap(basis(6, 3)))

            elif self.N == 14:
                proj_1 = np.abs(self.energy_levels[itr].overlap(basis(9, 3)))
                proj_2 = np.abs(self.energy_levels[itr].overlap(basis(9, 4)))
                proj_3 = np.abs(self.energy_levels[itr].overlap(basis(9, 5)))
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

        if self.N == 15:
            # calculate the partition function based on the Hamiltonian eigenvalues
            Z = np.exp(beta*self.energy_levels[index_1]) + np.exp(beta*self.energy_levels[index_2])
            
            self.rho0 = tensor(fock_dm(3,1), Qobj([ [np.exp(beta*self.energy_levels[index_1]), 0] , [0, np.exp(beta*self.energy_levels[index_2])] ])/ Z )
        
        elif self.N == 14:
            Z = np.exp(beta*self.energy_levels[index_1]) + np.exp(beta*self.energy_levels[index_2]) + np.exp(beta*self.energy_levels[index_3])

            self.rho0 = tensor(fock_dm(3,1), Qobj([ [np.exp(beta*self.energy_levels[index_1]), 0, 0] , [0, np.exp(beta*self.energy_levels[index_2]), 0], [0, 0, np.exp(beta*self.energy_levels[index_3])] ])/ Z ) 

        return self.rho0

    def MW_freqs(self):
        """
        
        """

        if self.N == 15:
            f1 = (np.sum(self.energy_levels[2:4]) - np.sum(self.energy_levels[0:2]))/2
            f2 = (np.sum(self.energy_levels[4:6]) - np.sum(self.energy_levels[0:2]))/2
        elif self.N == 14:
            f1 = (np.sum(self.energy_levels[3:6]) - np.sum(self.energy_levels[0:3]))/3
            f2 = (np.sum(self.energy_levels[6:9]) - np.sum(self.energy_levels[0:3]))/3
        
        return f1, f2
        
    def RF_freqs(self):
        """
        
        """

        if self.N == 15:
            f1 = self.energy_levels[1] - self.energy_levels[0]
            f2 = self.energy_levels[3] - self.energy_levels[2]
            f3 = self.energy_levels[5] - self.energy_levels[4]
            return f1, f2, f3
        elif self.N == 14:
            f11 = self.energy_levels[1] - self.energy_levels[0]
            f12= self.energy_levels[2] - self.energy_levels[1]
            f21 = self.energy_levels[4] - self.energy_levels[3]
            f22 = self.energy_levels[5] - self.energy_levels[4]
            f31 = self.energy_levels[7] - self.energy_levels[6]
            f32 = self.energy_levels[8] - self.energy_levels[7]
            return f11, f12, f21, f22, f31, f32
        
    def MW_H1(self):
        """
        """
        if self.N == 15:
            return tensor(jmat(1, 'x'), qeye(2))*2**.5
        elif self.N == 14:
            return tensor(jmat(1, 'x'), qeye(3))*2**.5
        
    def RF_H1(self):
        """
        """
        if self.N == 15:
            return tensor(qeye(3), jmat(1/2, 'x'))*2
        elif self.N == 14:
            return tensor(qeye(3), jmat(1/2, 'x'))*2
