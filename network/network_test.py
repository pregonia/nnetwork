"""
This code implements a neuronal network.
"""
import numpy as np
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import neuron
import synapse


#______________________________________________________________
# Plot settings

cp = sns.color_palette("hls", 7)
sns.set_palette(cp)
sns.set_color_codes()


#______________________________________________________________
# Simulation settings

# Flags
ex_fl = True

# Clock settings
start_time = 0                  # msec
end_time = 10                   # msec
inter_time = 0.01               # msec

# Network components
n_neurons = 2                   # number of neurons in network
neurons_db = []                 # list of neurons
synapses_db = []                # list of out-going synapses (one per neuron)
links_db = {}                   # dict of synaptic connections
I_syn_db = np.zeros(n_neurons)  # array of neuronal external current


# Network constants
I_mu = 10                        # mean, bias current 
I_sigma = 0.02                  # standard deviation, bias current 
p_syn = 0.5                     # percent of synaptic connections (per neuron)
M_syn = int(n_neurons * p_syn)  # number of synaptic connections (per neuron)


# Initialize neurons
for i in range(n_neurons):
    nn = neuron.HH(V=np.random.uniform(-60,30))
    neurons_db.append(nn)

    sy = synapse.WB(ex_fl)
    synapses_db.append(sy)


# Initialize synapse
# for i in range(n_neurons):
#     nn = range(n_neurons)
#     nn.remove(i)

#     links_db[i] = np.random.choice(nn, M_syn, replace=False) 
links_db[0] = [1]
links_db[1] = []


print "Initializing network..." 
print "Interneurons:", n_neurons
print "Synaptic connections (per neuron):", M_syn
print pd.DataFrame(links_db).T


#______________________________________________________________________________
# Run simulation
print "\nRunning simulation..."

for t in np.arange(start_time, end_time, inter_time):
    if not (t % 0.1): print "Time:", t, "ms"

    # Run neurons
    for nn, I_syn in zip(neurons_db, I_syn_db):
        # Add stimulus
        I_rand = np.random.normal(loc=I_mu, scale=I_sigma)
        I_app = I_rand
        # I_app = I_rand if np.abs(t - 0.1) > 0.1 else 0 
        # I_app = 10*(t>.100) - 10*(t>.200) + 35*(t>.300) - 35*(t>.400)

        if ex_fl: nn.run(t, +I_syn+I_app)
        else: nn.run(t, -I_syn+I_app)

    # Forward pass voltage through synapses
    # post_n: post-synaptic neuron 
    # pre_n: pre-synaptic neuron 
    for post_n, pre_n in links_db.iteritems():

        # Get all pre-synaptic voltage linked to post_n
        lnk = [neurons_db[i].V for i in pre_n]
        
        # Get the mean pre-synaptic voltage
        V_pre = np.mean(lnk)

        # Run synapse
        sy = synapses_db[post_n]
        sy.run(t, V_pre)

    # Collect all post-synaptic currents
    for post_n, pre_n in links_db.iteritems():
        # Get all post-synaptic voltage linked to post_n
        lnk = [synapses_db[i].I_syn for i in pre_n]
        
        # Get the mean post-synaptic voltage
        I_syn_db[post_n] = np.mean(lnk)


#______________________________________________________________________________
# Plot results

plt.figure(figsize=(18, 10))
plt.title("Interneuron Membrane Potential")
ax = None
for nn, i in zip(neurons_db, range(n_neurons)):
    idx = (n_neurons * 100) + (10) + (i+1)
    ax = plt.subplot(idx)

    ax.plot(nn.state_df['V'])
    plt.ylabel('$V (mV)$')
    plt.xlabel('s')


plt.show()
