"""
This code implements a neuronal network.
- Uses Wendling's model of 4 types of neurons
- Optimized code (non-OOP)
- Tests different g_syn (inh and exc)
"""
import numpy as np
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import utils_neuron as uneu
import utils_synapse as usyn

import time

import argparse

import os 
net_path = os.path.dirname(os.path.realpath(__file__))
rt_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
plt_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'plots'))

# from bigfloat import *

#______________________________________________________________
# Plot settings

cp = sns.color_palette("hls", 7)
sns.set_palette(cp)
sns.set_color_codes()
sns.set() 


#______________________________________________________________
# Simulation function

def run(g_syn_inh, g_syn_exc):
    #______________________________________________________________
    # Simulation settings

    # Randomization settings
    np.random.seed(200716341)

    # Clock settings
    start_time = 0                  # msec
    end_time = 5.                 # msec
    inter_time = 0.01               # msec
    print "Simulating %.2f ms (%d time points)" %\
          (end_time, 
           np.arange(start_time, end_time, inter_time).shape[0])

    # Network components
    n_pyr = 4                       # number of pyramidal neurons
    n_exc = 4                       # number of slow neurons
    n_sin = 4                       # number of slow neurons
    n_fin = 4                       # number of fast neurons
    C = 4                           # neuron connection constant

    # n_pyr = 250                     # number of pyramidal neurons
    # n_exc = 135                     # number of slow neurons
    # n_sin = 48                      # number of slow neurons
    # n_fin = 108                     # number of fast neurons
    # C = 135
    n_neu = n_pyr + n_exc + n_sin + n_fin   # total number of neurons in network

    neu_typ_db = []                 # list of neurons type
    neu_V_db = []                   # 2D list of neurons potential (thru time)
    neu_h_db = []                   # 2D list of neurons h factor (thru time)
    neu_m_db = []                   # 2D list of neurons m factor (thru time)
    neu_n_db = []                   # 2D list of neurons n factor (thru time)

    syn_link_db = []                # list of pre- and post-synaptic neurons (per synapse)
    syn_link_dict = {}              # dictionary version of syn_link_db
    syn_pre_dict = {}               # dictionary with pre-synaptic neurons as keys
    syn_post_dict = {}              # dictionary with post-synaptic neurons as keys
    syn_typ_db = []                 # list of synapse types
    syn_I_db = []                   # 2D list of post-synaptic current (thru time)
    syn_s_db = []                   # 2D list of s factor (thru time)

    neu_I_syn_db = [[0]*n_neu]      # 2D list of neuronal external current


    # Network constants
    I_mu = 10                       # mean, bias current 
    I_sigma = 0.02                  # standard deviation, bias current 
    p_syn = 1                       # percent of synaptic connections (per neuron)


    #______________________________________________________________
    # Initialize network
    time_init_neu = time.time()
    print "Initializing network..." 

    #===============================================
    # Initialize neurons
    #===============================================
    print "\nCreating neurons..."
    print "Neurons:", n_neu
    print "Pyramidal:", n_pyr
    print "Excitatory:", n_exc
    print "Slow inhibitory:", n_sin
    print "Fast inhibitory:", n_fin


    for i in range(n_neu):
        # generate random initial voltage
        v = np.random.uniform(-60,30) 

        # assign neuron type
        typ = 'pyr'
        if i >= (n_pyr+n_exc+n_sin):
            typ = 'fin'
        elif i >= (n_pyr+n_exc):
            typ = 'sin'
        elif i >= n_pyr:
            typ = 'exc'

        # store to database
        neu_typ_db.append(typ)
        neu_V_db.append(v)
        neu_m_db.append(0)
        neu_h_db.append(0)
        neu_n_db.append(0)
                
    neu_V_db = [neu_V_db]
    neu_m_db = [neu_m_db]
    neu_h_db = [neu_h_db]
    neu_n_db = [neu_n_db]


    # for db in [neu_typ_db, neu_V_db, neu_m_db, neu_h_db, neu_n_db]:
    #     db = np.array(db)

    print("(Running time: %0.4f seconds)" % (time.time() - time_init_neu))
    time_init_syn = time.time()

    #===============================================
    # Randomize links
    #===============================================
    print "\nGenerating random synaptic links..."

    # Pre-synaptic neuron types by index
    pre_pyr = range(n_pyr)
    pre_exc = range(n_pyr, n_pyr+n_exc)
    pre_sin = range(n_pyr+n_exc, n_pyr+n_exc+n_sin)
    pre_fin = range(n_pyr+n_exc+n_sin, n_neu)

    # Post-synaptic links
    def post_pyr(C):
        pre_exc_choice = [] if not pre_exc else np.random.choice(pre_exc, int(np.ceil(1.00*C)), replace=False)
        pre_sin_choice = [] if not pre_sin else np.random.choice(pre_sin, int(np.ceil(0.25*C)), replace=False)
        pre_fin_choice = [] if not pre_fin else np.random.choice(pre_fin, int(np.ceil(0.30*C)), replace=False)

        return np.unique(np.hstack((
               pre_exc_choice,
               pre_sin_choice,
               pre_fin_choice))).ravel()
        # return np.unique(np.hstack((
        #        np.random.choice(pre_exc, int(np.ceil(1.00*C)), replace=False),
        #        np.random.choice(pre_sin, int(np.ceil(0.25*C)), replace=False),
        #        np.random.choice(pre_fin, int(np.ceil(0.30*C)), replace=False)))).ravel()

    def post_exc(C): 
        pre_pyr_choice = [] if not pre_pyr else np.unique(np.random.choice(pre_pyr, int(np.ceil(0.80*C)), replace=False))
        return pre_pyr_choice
        # return np.unique(np.random.choice(pre_pyr, int(np.ceil(0.80*C)), replace=False))

    def post_sin(C):
        pre_pyr_choice = [] if not pre_pyr else np.random.choice(pre_pyr, int(np.ceil(0.25*C)), replace=False)
        pre_fin_choice = [] if not pre_fin else np.random.choice(pre_fin, int(np.ceil(0.10*C)), replace=False)

        return np.unique(np.hstack((
               pre_pyr_choice,
               pre_fin_choice))).ravel()
        # return np.unique(np.hstack((
        #        np.random.choice(pre_pyr, int(np.ceil(0.25*C)), replace=False),
        #        np.random.choice(pre_fin, int(np.ceil(0.10*C)), replace=False)))).ravel()

    def post_fin(C):
        pre_pyr_choice = [] if not pre_pyr else np.unique(np.random.choice(pre_pyr, int(np.ceil(0.80*C)), replace=False))

        return pre_pyr_choice
        # return np.unique(np.random.choice(pre_pyr, int(np.ceil(0.80*C)), replace=False))


    # Connect pre-synaptic neurons according to type
    n_syn = post_pyr(C).shape[0]*n_pyr +\
            post_exc(C).shape[0]*n_exc +\
            post_sin(C).shape[0]*n_sin +\
            post_fin(C).shape[0]*n_fin

    start_idx, end_idx = 0, 0
    for pre in range(n_neu):
        typ = neu_typ_db[pre]

        # select synaptic link generator function
        # based on neuron type
        post_fun = post_pyr
        if typ == 'exc':
            post_fun = post_exc
        elif typ == 'sin':
            post_fun = post_sin
        elif typ == 'fin':
            post_fun = post_fin

        post_ls = post_fun(C)
        for post in post_ls:
            syn_link_db.append([pre, post])
            syn_typ_db.append(typ)

        syn_link_dict[pre] = post_ls

        end_idx += post_ls.shape[0]
        syn_pre_dict[pre] = range(start_idx, end_idx)
        start_idx = end_idx

    # Find all post-synaptic neurons
    syn_post = [db[1] for db in syn_link_db]
    for post in range(n_neu):
        syn_post_dict[post] = [i for i, x in enumerate(syn_post) if x == post]

    # print np.reshape(syn_link_db, newshape=(n_syn, 2)).astype(int).T

    print "Outbound connections:"
    print "Total:", n_syn
    print "Pyramidal: %d x %d neurons" % (post_pyr(C).shape[0], n_pyr)
    print "Excitatory: %d x %d neurons" % (post_exc(C).shape[0], n_exc)
    print "Slow inhibitory: %d x %d neurons" % (post_sin(C).shape[0], n_sin)
    print "Fast inhibitory: %d x %d neurons" % (post_fin(C).shape[0], n_fin)

    #===============================================
    # Initialize synapse
    #===============================================
    syn_s_db.append([0]*n_syn)
    syn_I_db.append([0]*n_syn)

    print("(Running time: %0.4f seconds)" % (time.time() - time_init_syn))


    #______________________________________________________________
    # # Run simulation
    time_run = time.time()
    print "\nRunning simulation..."

    ti = 0
    for t in np.arange(start_time, end_time, inter_time):
        # if not (t*100)%100: print "Time:", t, "ms"
        time_iter = time.time()
        print "Time:", t, "ms",

        # Run neurons
        neu_V, neu_m, neu_h, neu_n = [], [], [], []
        for ni, I_syn in zip(range(n_neu), neu_I_syn_db[-1]):
            
            # Add stimulus
            I_rand = np.random.normal(loc=I_mu, scale=I_sigma)
            I_app = I_rand

            # Get latest values of neuron ni
            V = neu_V_db[-1][ni]    
            m = neu_m_db[-1][ni]
            h = neu_h_db[-1][ni]
            n = neu_n_db[-1][ni]

            V_, m_, h_, n_ = uneu.run(t, V, m, h, n, I_ext=I_syn+I_app)

            neu_V.append(V_)
            neu_m.append(m_)
            neu_h.append(h_)
            neu_n.append(n_)

        # Update neuron values
        neu_V_db.append(neu_V)
        neu_m_db.append(neu_m)
        neu_h_db.append(neu_h)
        neu_n_db.append(neu_n)


        # Forward pass voltage through synapses
        # pre_ni: pre-synaptic neuron id
        syn_I, syn_s = [], []
        for pre_ni in range(n_neu):

            # Get pre-synaptic voltage
            V_pre = neu_V_db[-1][pre_ni]

            # Run synapse
            for si in syn_pre_dict[pre_ni]:
                s = syn_s_db[-1][si]
                typ = syn_typ_db[si]

                g = g_syn_inh
                if typ == 'pyr' or typ == 'exc':
                    g = g_syn_exc

                s_, I_syn_ = usyn.run(t, V_pre=V_pre, s=s, typ=typ, g=g)

                syn_I.append(I_syn_)
                syn_s.append(s_)

        # Update synapse values
        syn_I_db.append(syn_I)
        syn_s_db.append(syn_s)


        # Collect all post-synaptic currents
        # post_n: post-synaptic neuron (id)
        neu_I_syn = []
        for post_si in range(n_neu):
            # Get all pre-synaptic links to post_n
            pre_s = syn_post_dict[post_si]

            # Get all post-synaptic current
            psc = [syn_I_db[-1][pre_si] for pre_si in pre_s]
            
            # Get the mean post-synaptic voltage
            syn_mean = 0 if not psc else np.mean(psc)
            neu_I_syn.append(syn_mean)


        # Update post-synaptic current values
        neu_I_syn_db.append(neu_I_syn)


        print "(Running time: %0.4f seconds)" % (time.time() - time_iter)

    print("\n(Running time: %0.4f seconds)" % (time.time() - time_run))

    # ______________________________________________________________
    # Plot results

    neu_V_db = np.array(neu_V_db)
    lfp = np.mean([neu_V_db[:,pre] for pre in pre_pyr], axis=0)[1:]

    x = np.arange(start_time, end_time, inter_time)
    y = lfp

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(18, 4))
    plt.title(r'\textbf{LFP, $g_{inh}=$%.4f, $g_{exc}=$%.4f}' % (g_syn_inh, g_syn_exc))
    plt.plot(x, y)
    plt.ylabel(r'\textbf{Membrane potential} (mV)')
    plt.xlabel(r'\textbf{Time} (ms)')
    # plt.show()

    f_name = 'LFP_p%d_e%d_s%d_f%d_%.2fms_gin%.4f_gex%.4f.png' %\
            (n_pyr, n_exc, n_sin, n_fin, 
             end_time, g_syn_inh, g_syn_exc)

    f_path = os.path.abspath(os.path.join(plt_path, f_name))
    fig.savefig(f_path)
    exit()



#______________________________________________________________
# Main function

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-gin', type=float, default=0.1)
    ap.add_argument('-gex', type=float, default=0.1)

    ops = ap.parse_args()

    print "Running network simulation..."
    print "-g_syn_inh:", ops.gin
    print "-g_syn_exc:", ops.gex

    run(g_syn_inh=ops.gin, g_syn_exc=ops.gex)

