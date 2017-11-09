
"""
This code compiles the following Hodgkin-Huxley neuron utility functions and methods:
- Channel gating functions
- Temperature function
- Ionic current function
- RK4 ODE solver
- Neuron simulator
"""
import numpy as np
import scipy as sp
import pandas as pd


#______________________________________________________________
# Neuron configurations

C_m = 1.0       # membrane capacitance, uF/cm^2

g_Na = 120.0    # Na maximum conductances, mS/cm^2
g_K = 36.0      # K maximum conductances, mS/cm^2
g_L = 0.3       # Leak maximum conductances, mS/cm^2

E_Na = 50.0     # Na Nernst reversal potentials, mV
E_K = -77.0     # K Nernst reversal potentials, mV
E_L = -54.387   # Leak Nernst reversal potentials, mV

phi = lambda T: 1 	# temperature function; returns 1 always

dt = 0.01       # msec, time step for RK4


#______________________________________________________________
# Neuron functions

#===============================================
# Channel gating functions
#===============================================

def alpha_m(V):
    return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

def beta_m(V):
    return 4.0*np.exp(-(V+65.0) / 18.0)

def alpha_h(V):
    return 0.07*np.exp(-(V+65.0) / 20.0)

def beta_h( V):
    return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

def alpha_n(V):
    return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

def beta_n(V):
    return 0.125*np.exp(-(V+65) / 80.0)


#===============================================
# Temperature function
#===============================================

def T_phi(T):
    """ Temperature factor Q10 """
    return 3**((T-6.3)/10.)


#===============================================
# Ionic current functions
#===============================================

def I_Na(V, m, h, T=0):
    """ Sodium current """
    ph = phi(T)
    return g_Na * (ph*m)**3 * (ph*h) * (V - E_Na)

def I_K(V, n, T=0):
    """ Potassium current """
    ph = phi(T)
    return g_K  * (ph*n)**4 * (V - E_K)

def I_L(V):
    """ Leak current """        
    return g_L * (V - E_L)

def dALLdt(t, X, I_ext):
    """ Calculates membrane potential & activation variables """
    V, m, h, n = X

    dVdt = (I_ext - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
    dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
    dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
    return np.array([dVdt, dmdt, dhdt, dndt])


#===============================================
# ODE solver for neuron
#===============================================

def neu_rk4(f, t, X, I_ext, dt=0.01):
    """ 
    Solves the ODE using 4th order Runge-Kutta 
    
    Parameters:
    f: differential equation
    t: time point
    X: ODE variables
    I_ext: external current
    dt: time step
    """
    k1 = f(t, X, I_ext)
    k2 = f(t+(dt/.2), X+(np.multiply(dt,k1/.2)), I_ext)
    k3 = f(t+(dt/.2), X+(np.multiply(dt,k2/.2)), I_ext)
    k4 = f(t+dt, X+(np.multiply(dt,k3)), I_ext)
    y = X + np.multiply(dt/.6, (k1 + 
                (np.multiply(2,k2)) + 
                (np.multiply(2,k3)) + k4))  
    return y


#===============================================
# Neuron simulation
#===============================================

def run(t, V, m, h, n, I_ext=0):
    """ 
    Computes the current state of the neuron given the external current 
    """

    X = np.array([V, m, h, n])
    V_, m_, h_, n_ = neu_rk4(dALLdt, t, X, I_ext, dt=dt)
    
    return V_, m_, h_, n_