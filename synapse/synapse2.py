import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Constants
g_syn = 0.1    # mS/cm^2, maximal synaptic conductance
E_syn = -75.0  # mV, reversal potential
th_syn = 0.0   # mV, 

al = 12.0      # 1/msec, channel opening rate
bt = 0.1       # 1/msec, channel closing reate


# Sample input signal
t = np.arange(0, 8, 0.01)
stim = np.arange(-4,1,0.5)
stim_2 = np.arange(0,20,0.1)

V_pre_samp = np.zeros(len(t))
V_pre_samp[:] = -64
V_pre_samp[100:110] = 163*np.exp(stim-1) - 64
# V_pre_samp[110:310] = -0.5*np.exp(1-stim_2) - 64



# plt.plot(V_pre_samp)
# plt.show()

# exit()

# Sampling method
def m_V_pre(V_pre, t_i):
    i = np.where(t == t_i)
    return V_pre[i]


# Equations
def f_I_syn(s, V):
    ''' Returns the synaptic current '''
    return g_syn*s*(V - E_syn)

def f_V_pre(V_pre):
    ''' Returns the normalized concentration of
        postsynaptic transmitter-receptor complex '''
    return 1.0/(1.0+np.exp(-(V_pre - th_syn)/2.0))

def f_dsdt(s, t, V_pre):
    ''' Returns the gating variable '''
    # print 't', t
    dsdt = al*f_V_pre(V_pre)*(1-s) - bt*s
    return dsdt


# Run
s_0 = [0]*len(V_pre_samp)
# s_0 = [0]

s = odeint(f_dsdt, s_0, t, (V_pre_samp, ))
print s.shape

# plt.plot(s)
# plt.plot(s[:,-1])
# plt.plot(f_I_syn(s[:,-1], V_pre_samp))
plt.plot(V_pre_samp)
plt.show()

