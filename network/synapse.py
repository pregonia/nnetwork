"""
This code implements a synapse model
"""
import numpy as np
import scipy as sp
import pandas as pd


class WB():
    """ 
    Wang-Buzsaki synapse model
    """


    #_____________________________________________________
    # Constants

    g_syn = 0.1     # mS/cm^2, maximal synaptic conductance
    th_syn = 0      # mV, synaptic constant
    
    al = 12         # 1/msec, channel opening rate

    dt = 0.01       # msec, time step for RK4

    #_____________________________________________________
    # Synapse object instantiation

    def __init__(self, typ='fin'):
        """ Initializes synapse """
        if typ == 'pyr':        # Pyramidal, excitatory
            self.E_syn = 0.     # mV, reversal potential
            self.bt = 1./50     # 1/msec, channel closing rate
            self.sgn = 1        # negative current

        elif typ == 'exc':      # Non-pyramidal, excitatory
            self.E_syn = 0.     # mV, reversal potential
            self.bt = 1./50     # 1/msec, channel closing rate
            self.sgn = 1        # negative current

        elif typ == 'sin':      # Slow, inhibitory
            self.E_syn = -75    # mV, reversal potential
            self.bt = 1./100    # 1/msec, channel closing rate
            self.sgn = -1       # negative current

        elif typ == 'fin':      # Fast, inhibitory
            self.E_syn = -75    # mV, reversal potential
            self.bt = 1./10     # 1/msec, channel closing rate
            self.sgn = -1       # positive current
        
        self.typ = typ

        # Initialize time-dependent states
        self.prop = ['I_syn', 's']           
        self.state_df = pd.DataFrame(index=[0], columns=self.prop)

        self.s = 0
        self.I_syn = 0

        self.update_state(0, I_syn = self.I_syn, s = self.s)


    def __repr__(self):
        return self.typ + " synapse"


    #_____________________________________________________
    # ODE solver

    def rk4(self, f, t, X, V):
        """
        Solves the ODE using 4th order Runge-Kutta
        """
        dt = self.dt

        k1 = f(t, X, V)
        k2 = f(t+(dt/.2), X+(np.multiply(dt,k1/.2)), V)
        k3 = f(t+(dt/.2), X+(np.multiply(dt,k2/.2)), V)
        k4 = f(t+dt, X+(np.multiply(dt,k3)), V)
        y = X + np.multiply(dt/.6, (k1 + 
                    (np.multiply(2,k2)) + 
                    (np.multiply(2,k3)) + k4))  

        return y


    #_____________________________________________________
    # Equations

    def f_I_syn(self, s, V):
        """ Returns the synaptic current """ 
        return self.g_syn * s * (V-self.E_syn)

    def f_V_pre(self, V_pre):
        """
        Returns the normalized concentration of
        postsynaptic transmitter-receptor complex 
        """
        return 1. / (1 + np.exp((self.th_syn-V_pre)/2))

    def f_dsdt(self, t, s, V_pre):
        """ Returns the gating variable """
        exc = self.al * self.f_V_pre(V_pre) * (1-s)
        inh = self.bt * s
        dsdt =  exc - inh 
        return dsdt


    #_____________________________________________________
    # Simulate synapse

    def run(self, t, V_pre=0):
        """ 
        Computes the current state of the synapse given pre-synaptic voltage
        """

        # Update gating variable
        self.s = self.rk4(self.f_dsdt, t, self.s, V_pre)
        
        # Compute synaptic current
        self.I_syn = self.sgn * self.f_I_syn(self.s, V_pre)

        self.update_state(t, I_syn = self.I_syn, s = self.s)



    def get_current_state(self, col=None):
        """ 
        Returns the current state of the synapse 
        Returns the current value of a variable if given
        """
        if col:
            return float(self.state_df.tail()[col])
        else:
            return self.state_df.tail()


    def update_state(self, t, I_syn=0., s=0.):
        """ 
        Updates the current state of the neuron 
        """
        self.state_df.ix[t, 'I_syn'] = I_syn
        self.state_df.ix[t, 's'] = s



