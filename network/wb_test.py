import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns

cp = sns.color_palette("hls", 7)
sns.set_palette(cp)
sns.set_color_codes()

plt.figure(figsize=(18, 10))
for bt in [1./10, 1./50, 1./100]:
	# Constants
	g_syn = 0.1  # mS/cm^2, maximal synaptic conductance
	E_syn = -75  # mV, reversal potential
	th_syn = 0   # mV, synaptic constant

	al = 12      # 1/msec, channel opening rate
	# bt = 1./10     # 1/msec, channel closing reate

	dt = 0.0001     # msec, time step for RK4

	# Sample input signal
	t = np.arange(0, 2, 0.0001)
	N = len(t)

	stim = np.arange(-4,1,0.0005)
	stim_2 = np.arange(0,20,0.01)
	print stim_2.shape
	V_pre_samp = np.zeros(N)
	V_pre_samp[:] = -64
	V_pre_samp[100:10100] = 163*np.exp(stim-1) - 64
	V_pre_samp[10100:12100] = -0.5*np.exp(1-stim_2) - 64

	# plt.figure(figsize=(18, 4))
	# plt.plot(V_pre_samp, 'r', label='V_pre')
	# plt.ylabel('mV')
	# plt.xlabel('ms')
	# plt.legend()
	# plt.show()

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


	# Run the simulation
	si = 0 # initial value for s

	s_samp, I_syn_samp = [], []
	for ti, V_pre in zip(t, V_pre_samp):
	    sj = rk4(f_dsdt, ti, si, V_pre)
	    Ij = f_I_syn(sj, V_pre)

	    s_samp.append(sj)
	    I_syn_samp.append(Ij)

	    si = sj
	    
	s_samp = np.array(s_samp)
	I_syn_samp = np.array(I_syn_samp)


	# Plot the results
	# plt.figure(figsize=(18, 10))
	ax1 = plt.subplot(311)
	plt.plot(V_pre_samp, label='V_pre')
	plt.ylabel('$V_{pre}$ ($mV$)')

	plt.subplot(312, sharex=ax1)
	plt.plot(s_samp, label='S %.2f'%bt)
	plt.ylabel('$s$')

	# plt.subplot(313, sharex=ax1)
	# plt.plot(I_syn_samp, label='I_syn')
	# plt.ylabel('$I_{syn}$ ($mA$)')
	# plt.xlabel('t (ms)')
	# plt.xlim([100,120])
plt.legend()
plt.show()