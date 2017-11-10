
"""
This code compiles the following synapse utility functions and methods:
- Type-specific functions
"""
import numpy as np
import scipy as sp
import pandas as pd

#______________________________________________________________
# Synapse configurations

# g_syn = 0.1     # mS/cm^2, maximal synaptic conductance
th_syn = 0      # mV, synaptic constant

al = 12         # 1/msec, channel opening rate

# Integration
dt = 0.01       # msec, time step for RK4

# Temperature
q10_syn = 1.8   # Q_10 constant for temperature (interneurons)

#______________________________________________________________
# Synapse functions

#===============================================
# Type-specific functions
#===============================================

def E_syn(typ):
    """ Returns reverse potential (mV) based on type """
    if typ == 'pyr': return 0.
    elif typ == 'exc': return 0.
    elif typ == 'sin': return -75.
    elif typ == 'fin': return -75.
    else: print "Error: unknown type %s." % typ
  
def g_syn(g=None):
    """ Returns maximal synaptic conductance (mS/cm^2) based on type """
    return g if g else 0.1


def bt(typ):
    """ Returns channel closing rate (1/msec) based on type """
    if typ == 'pyr': return 1./50
    elif typ == 'exc': return 1./50
    elif typ == 'sin': return 1./100
    elif typ == 'fin': return 1./10
    else: print "Error: unknown type %s." % typ
    

def sgn(typ):
    """ Returns current sign (+/-) based on type """
    if typ == 'pyr': return 1
    elif typ == 'exc': return 1
    elif typ == 'sin': return -1
    elif typ == 'fin': return -1
    else: print "Error: unknown type %s." % typ
    

#===============================================
# Temperature function
#===============================================

def phi(T):
    """ Temperature factor Phi for interneurons """
    return q10_syn**((T-31)/10.)


#===============================================
# ODE solver for neuron
#===============================================

def syn_rk4(f, t, X, V, typ, dt=0.01):
    """ 
    Solves the ODE using 4th order Runge-Kutta 
    
    Parameters:
    f: differential equation
    t: time point
    X: ODE variables
    V: pre-synaptic current
    dt: time step
    """
    k1 = f(t, X, V, typ)
    k2 = f(t+(dt/.2), X+(np.multiply(dt,k1/.2)), V, typ)
    k3 = f(t+(dt/.2), X+(np.multiply(dt,k2/.2)), V, typ)
    k4 = f(t+dt, X+(np.multiply(dt,k3)), V, typ)
    y = X + np.multiply(dt/.6, (k1 + 
                (np.multiply(2,k2)) + 
                (np.multiply(2,k3)) + k4))  
    return y


#===============================================
# Synaptic functions
#===============================================

def f_I_syn(s, V, typ, g=None):
    """ Returns the synaptic current """
    return g_syn(g) * s * (V-E_syn(typ))

def f_V_pre(V_pre):
    """
    Returns the normalized concentration of
    postsynaptic transmitter-receptor complex 
    """
    return 1. / (1 + np.exp((th_syn-V_pre)/2))

def f_dsdt(t, s, V_pre, typ):
    """ Returns the gating variable """
    exc = al * f_V_pre(V_pre) * (1-s)
    inh = bt(typ) * s
    dsdt =  exc - inh 
    return dsdt


#===============================================
# Synapse simulation
#===============================================

def run(t, V_pre=0, s=0, typ='pyr', g=None, T=15):
    """ 
    Computes the current state of the synapse given pre-synaptic voltage
    """

    # Update gating variable
    s_ = syn_rk4(f_dsdt, t, s, V_pre, typ, dt=dt)

    # Compute synaptic current
    ph = phi(T)
    print ph
    I_syn_ = sgn(typ) * ph * f_I_syn(s_, V_pre, typ, g)

    return s_, I_syn_
