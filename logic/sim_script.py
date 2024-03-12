import numpy as np
from qutip import jmat, tensor, qeye, fock_dm, mesolve, Options, parallel_map, ptrace
from scipy.optimize import curve_fit
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt

import quaccatoo.nv as nv

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-N", type=int)
parser.add_argument("-B", type=float)
parser.add_argument("-m", type=int)
parser.add_argument("--sim", type=str)
parser.add_argument("--theta", type=float)
parser.add_argument("--w1", type=float)
parser.add_argument("--trabimax", type=float)


args = parser.parse_args()

N_value = args.N
B_value = args.B
m_value = args.m
theta_value = args.theta
task_value = args.sim
w1_value = args.w1

print(args)


def H1x(t, args):
    return 2*np.pi*args['w1']*np.cos(args['w_pulse']*(t + args['phi']))


def HAHN(x, args):

    tau = x - args['t_pi']/2
    t_pi = np.linspace(0, args['t_pi'], 100)

    r = mesolve(Hx, args['rho0'], t_pi/2, [], [], options= Options(nsteps=1e5), args={'phi': 0, 'w1':args['w1'], 'w_pulse':args['w_pulse']}) #You need to increase number of steps for ODE to converge
    r = (-1j*H0*tau).expm()*r.states[t_pi.size - 1]*((-1j*H0*tau).expm()).dag()
    phi = x
    r = mesolve(Hx, r, t_pi, [], [], options= Options(nsteps=1e5), args={'phi': phi, 'w1':args['w1'], 'w_pulse':args['w_pulse']})
    r = (-1j*H0*tau).expm()*r.states[t_pi.size - 1]*((-1j*H0*tau).expm()).dag()
    phi = phi + tau + args['t_pi']
    r = mesolve(Hx, r, t_pi/2, [], [], options= Options(nsteps=1e5), args={'phi': phi, 'w1':args['w1'], 'w_pulse':args['w_pulse']}) 

    p = np.abs(fock_dm(3,1).overlap(ptrace(r.states[t_pi.size - 1], 0)))**2

    return p


def fit_rabi(x, A, B, C):
    return A*np.cos(2*np.pi*x/B)**2 + C

Hzf = nv.ZeroField(N_value)
Hez = nv.ElectronZeeman(N_value, B_value, theta_value)
Hhf = nv.HyperFineN(N_value)
Hnz = nv.NuclearZeeman(N_value, B_value,theta_value)

H0 = 2*np.pi*(Hzf + Hez + Hhf + Hnz)

if N_value == 14:
    Hx = [H0, [tensor(jmat(1,'x'), qeye(3)), H1x]]
    e_ops = [tensor(fock_dm(3,1), qeye(3))]

elif N_value == 15:
    Hx = [H0, [tensor(jmat(1,'x'), qeye(2)), H1x]]
    e_ops = [tensor(fock_dm(3,1), qeye(2))]


if m_value == -1:
    w_pulse = nv.GetPulsef1(N_value,H0)
elif m_value == 1:
    w_pulse = nv.GetPulsef2(N_value,H0)
    


rho0 = nv.InitState(N_value)
t_rabi = np.linspace(0, args.trabimax, 200)

w1 = w1_value

tpi = 0.70694882/(w1+0.11130492)
guess_B = 4*tpi

options = Options(nsteps=1e5)

# print(H0)
# print(Hx)
# print(e_ops)
# print(w_pulse)

r_rabi = mesolve(Hx, rho0, t_rabi, [], e_ops, args={'phi': 0, 'w1':w1, 'w_pulse':w_pulse}, options=options)
# plt.plot(t_rabi,r_rabi.expect[0])
# plt.show()

# print(r_rabi.expect[0])

guess_A = max(r_rabi.expect[0]) - min(r_rabi.expect[0])
guess_C = min(r_rabi.expect[0])

params, cov = curve_fit(fit_rabi, t_rabi, r_rabi.expect[0], p0=[guess_A, guess_B, guess_C])

np.savez('./data/rabi',t=t_rabi,r=r_rabi.expect[0],fit=params)

if task_value == 'Hahn':
    hahn_args ={
    'H0': H0,
    't_pi': params[1]/4,
    'rho0': rho0,
    'w1': w1,
    'w_pulse': w_pulse
    }
    tau =  np.linspace(0.1, 50, 500)
    r_hahn = parallel_map(HAHN, tau, task_args=(hahn_args,), num_cpus=6)
    np.savez('./data/hahn',t=tau,r=r_hahn)