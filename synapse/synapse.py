import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Constants
g_syn = 0.1  # mS/cm^2, maximal synaptic conductance
E_syn = -75  # mV, reversal potential
th_syn = 0   # mV, 

al = 12      # 1/msec, channel opening rate
bt = 0.1     # 1/msec, channel closing reate


# Sample input signal
t = np.arange(0, 8, 0.01)
stim = np.arange(-4,1,0.5)
stim_2 = np.arange(0,20,0.1)

V_pre_samp = np.zeros(len(t))
V_pre_samp[:] = -64
V_pre_samp[100:110] = 163*np.exp(stim-1) - 64
V_pre_samp[110:310] = -0.5*np.exp(1-stim_2) - 64



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
    return 1./(1+np.exp(-(V_pre - th_syn)/2))

def f_dsdt(s, t, V_pre):
    ''' Returns the gating variable '''
    print 't', t
    dsdt = al*f_V_pre(V_pre)*(1-s) - bt*s
    return dsdt


# Run
s_0 = 0

s_samp, I_syn_samp = [], []
t_j = 0
for V_pre, t_i in zip(V_pre_samp, t):
    print V_pre, "--------------------------------------"
    s = odeint(f_dsdt, s_0, [t_j, t_i], (V_pre, ))
    # s = odeint(f_dsdt, s_0, np.array([0,1]), (V_pre, ))

    t_j = t_i
    print 's', s    
#     s = s[1,0]
#     I_syn = f_I_syn(s, V_pre)
    
#     s_samp.append(s)
#     I_syn_samp.append(I_syn)


# s_samp = np.array(s_samp)
# I_syn_samp = np.array(I_syn_samp)
# print I_syn_samp.shape

exit()

plt.plot(s_samp)
# plt.plot(I_syn_samp)
# plt.xlim([90,120])
plt.show()

