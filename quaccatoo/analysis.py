import numpy as np

def fit_rabi(t, A, T, C):
    return A*np.cos(2*np.pi*t/T) + C

def fit_rabi_decay(t, A, T, phi, C, Tc, n):
    return A*np.cos(2*np.pi*t/T + phi)*np.exp(-(x/Tc)**n) + C

def fit_hahn_mod(t, A, B, C, f1, f2):
    return ( A - B*np.sin(2*np.pi*f1*t/2)**2*np.sin(2*np.pi*f2*t/2)**2 ) + C

def fit_hahn_mod_decay(t, A, B, C, f1, f2, Tc, n):
    return np.exp(- (t/Tc)**n)*( A - B*np.sin(2*np.pi*f1*t/2)**2*np.sin(2*np.pi*f2*t/2)**2 ) + C