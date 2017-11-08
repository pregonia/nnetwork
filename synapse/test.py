import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Constants
g_syn = 0.1  # mS/cm^2, maximal synaptic conductance
E_syn = -75  # mV, reversal potential
th_syn = 0   # mV, synaptic constant

al = 12      # 1/msec, channel opening rate
bt = 0.1     # 1/msec, channel closing reate

dt = 0.1     # msec, time step for RK4

# Sample input signal
t = np.arange(0, 1, 0.001)
N = len(t)
# print N

stim = np.arange(-4,1,0.5)
stim_2 = np.arange(0,20,0.1)
V_pre_samp = np.zeros(N)
V_pre_samp[:] = -64
V_pre_samp[100:110] = 163*np.exp(stim-1) - 64
V_pre_samp[110:310] = -0.5*np.exp(1-stim_2) - 64

# sinv = np.arange(0, 2*np.pi, 2*np.pi/(N))
# v1 = np.sin(10*sinv)
# v2 = np.sin(7*sinv)**2
# V_pre_samp = -40 + 30*(v1 + v2)

# plt.plot(V_pre_samp)
# plt.show()
# exit()

# Sampling method
def m_V_pre(V_pre, t_i):
    i = np.where(t == t_i)
    return V_pre[i]

# Numerical method
def rk4(f, t, X, V):
    ''' Solves the ODE using 4th order Runge-Kutta '''
    k1 = f(t, X, V)
    k2 = f(t+(dt/2), X+(dt*k1/2), V)
    k3 = f(t+(dt/2), X+(dt*k2/2), V)
    k4 = f(t+dt, X+(dt*k3), V)
    y = X + dt*(k1 + (2*k2) + (2*k3) + k4)/6

    return y


# Equations
def f_I_syn(s, V):
    ''' Returns the synaptic current '''
    return g_syn * s * (V-E_syn)

def f_V_pre(V_pre):
    ''' Returns the normalized concentration of
        postsynaptic transmitter-receptor complex '''
    return 1. / (1 + np.exp((th_syn-V_pre)/2))

def f_dsdt(t, s, V_pre):
    ''' Returns the gating variable '''
    exc = al * f_V_pre(V_pre) * (1-s)
    inh = bt * s
    dsdt =  exc - inh 
    return dsdt


# Run
si = 0

s_samp, I_syn_samp = [], []
for ti, V_pre in zip(t, V_pre_samp):
    sj = rk4(f_dsdt, ti, si, V_pre)
    Ij = f_I_syn(sj, V_pre)

    # print sj, Ij
    s_samp.append(sj)
    I_syn_samp.append(Ij)

    print "%4e, %4e, %4e, %4e" % (si, sj, ti, V_pre)
    si = sj


s_samp = np.array(s_samp)
I_syn_samp = np.array(I_syn_samp)

# plt.plot(s_samp)

plt.figure(1)
ax1 = plt.subplot(311)
plt.plot(V_pre_samp, label='V_pre')
plt.legend()

plt.subplot(312, sharex=ax1)
plt.plot(s_samp, label='S')
plt.legend()

plt.subplot(313, sharex=ax1)
plt.plot(-I_syn_samp, label='I_syn')
plt.legend()
# plt.xlim([0,2000])
# plt.ylim([-10,10])


plt.show()

