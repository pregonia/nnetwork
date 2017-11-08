"""
This code implements a Hodgkin-Huxley neuron model
"""
import numpy as np
import scipy as sp
import pandas as pd


class HH():
    """ 
    Hodgkin-Huxley neuron model
    """

    #_____________________________________________________
    # Constants

    C_m = 1.0       # membrane capacitance, uF/cm^2

    g_Na = 120.0    # Na maximum conductances, mS/cm^2
    g_K = 36.0      # K maximum conductances, mS/cm^2
    g_L = 0.3       # Leak maximum conductances, mS/cm^2

    E_Na = 50.0     # Na Nernst reversal potentials, mV
    E_K = -77.0     # K Nernst reversal potentials, mV
    E_L = -54.387   # Leak Nernst reversal potentials, mV

    dt = 0.01       # msec, time step for RK4

    #_____________________________________________________
    # Neuron object instantiation

    def __init__(self, temp_on=False, T=6.3, V=-65, typ='fin'):
        """ Initializes neuron """
        
        # Time-dependent states
        self.prop = ['V', 'I_ext', 'm', 'h', 'n']           
        self.state_df = pd.DataFrame(index=[0], columns=self.prop)

        self.V = V          # membrane potential, mV
        self.m = 0.       # conductance variable
        self.h = 0.        # conductance variable
        self.n = 0.       # conductance variable
        # self.m = 0.05       # conductance variable
        # self.h = 0.6        # conductance variable
        # self.n = 0.32       # conductance variable

        self.update_state(0,
                          V = self.V,
                          I_ext = 0, 
                          m = self.m,
                          h = self.h,
                          n = self.n)


        # Neuron type (pyr, exc, fin, sin)
        self.typ = typ


        # Temperature settings
        self.T = T                          # temperature, C
        self.phi = lambda T: 1              # returns 1 always
        if temp_on: self.phi = self.T_phi   # returns phi factor
        

    def __repr__(self):
        """ Returns neuron identity """
        return self.typ.upper() + " neuron: V = %.2f"%self.V


    #_____________________________________________________
    # Channel gating functions

    def alpha_m(self, V):
        return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        return 4.0*np.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        return 0.07*np.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        return 0.125*np.exp(-(V+65) / 80.0)


    #_____________________________________________________
    # Temperature function

    def T_phi(self, T):
        """ Temperature factor Q10 """
        return 3**((T-6.3)/10.)


    #_____________________________________________________
    # Ionic current functions

    def I_Na(self, V, m, h):
        """ Sodium current """
        phi = self.phi(self.T)
        return self.g_Na * (phi*m)**3 * (phi*h) * (V - self.E_Na)

    def I_K(self, V, n):
        """ Potassium current """
        phi = self.phi(self.T)
        return self.g_K  * (phi*n)**4 * (V - self.E_K)

    def I_L(self, V):
        """ Leak current """        
        return self.g_L * (V - self.E_L)

    def dALLdt(self, t, X, I_ext):
        """ Calculates membrane potential & activation variables """
        V, m, h, n = X

        dVdt = (I_ext - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return np.array([dVdt, dmdt, dhdt, dndt])


    #_____________________________________________________
    # ODE solver
    def rk4(self, f, t, X, I_ext):
        """ 
        Solves the ODE using 4th order Runge-Kutta 
        """
        dt = self.dt
        k1 = f(t, X, I_ext)
        k2 = f(t+(dt/.2), X+(np.multiply(dt,k1/.2)), I_ext)
        k3 = f(t+(dt/.2), X+(np.multiply(dt,k2/.2)), I_ext)
        k4 = f(t+dt, X+(np.multiply(dt,k3)), I_ext)
        y = X + np.multiply(dt/.6, (k1 + 
                    (np.multiply(2,k2)) + 
                    (np.multiply(2,k3)) + k4))  

        return y


    #_____________________________________________________
    # Simulate neuron
    def run(self, t, I_ext=0):
        """ 
        Computes the current state of the neuron given the external current 
        """

        X = np.array([self.V, self.m, self.h, self.n])
        self.V, self.m, self.h, self.n = self.rk4(self.dALLdt, t, X, I_ext)
        
        self.update_state(t,
                          V = self.V,
                          I_ext = I_ext, 
                          m = self.m,
                          h = self.h,
                          n = self.n)


    def get_current_state(self, col=None):
        """ 
        Returns the current state of the neuron 
        Returns the current value of a variable if given
        """
        if col:
            return float(self.state_df.tail()[col])
        else:
            return self.state_df.tail()


    def update_state(self, t, V=0., I_ext=0., m=0., h=0., n=0.):
        """ 
        Updates the current state of the neuron 
        """
        self.state_df.ix[t, 'V'] = V
        self.state_df.ix[t, 'I_ext'] = I_ext
        self.state_df.ix[t, 'm'] = m
        self.state_df.ix[t, 'h'] = h
        self.state_df.ix[t, 'n'] = n


