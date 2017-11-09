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

import time

#______________________________________________________________
# Plot settings

cp = sns.color_palette("hls", 7)
sns.set_palette(cp)
sns.set_color_codes()


#______________________________________________________________
# Simulation settings

# Randomization settings
np.random.seed(200716341)

# Clock settings
start_time = 0                  # msec
end_time = 10.                  # msec
inter_time = 0.01               # msec
print "Simulating %.2f ms (%d time points)" %\
      (end_time, 
       np.arange(start_time, end_time, inter_time).shape[0])

# Network components
n_pyr = 4                     # number of pyramidal neurons
n_exc = 4                     # number of slow neurons
n_sin = 4                     # number of slow neurons
n_fin = 4                     # number of fast neurons
C = 4
# n_pyr = 250                     # number of pyramidal neurons
# n_exc = 135                     # number of slow neurons
# n_sin = 48                      # number of slow neurons
# n_fin = 108                     # number of fast neurons
# C = 135
n_neurons = n_pyr + n_exc + n_sin + n_fin   # total number of neurons in network

neurons_db = []                 # list of neurons
synapses_db = []                # dataframe of synapses
links_db = {}                   # dataframe of synaptic connections
I_syn_db = np.zeros(n_neurons)  # array of neuronal external current


# Network constants
I_mu = 10                       # mean, bias current 
I_sigma = 0.02                  # standard deviation, bias current 
p_syn = 1                       # percent of synaptic connections (per neuron)


#______________________________________________________________________________
# Initialize network
time_init_neu = time.time()
print "Initializing network..." 

# Initialize neurons
print "\nCreating neurons..."
print "Neurons:", n_neurons
print "Pyramidal:", n_pyr
print "Excitatory:", n_exc
print "Slow inhibitory:", n_sin
print "Fast inhibitory:", n_fin


for i in range(n_neurons):
    v = np.random.uniform(-60,30) # generate random initial voltage

    # assign neuron type
    typ = 'pyr'
    if i >= (n_pyr+n_exc+n_sin):
        typ = 'fin'
    elif i >= (n_pyr+n_exc):
        typ = 'sin'
    elif i >= n_pyr:
        typ = 'exc'

    nn = neuron.HH(V=v, typ=typ)
    neurons_db.append(nn)

    # synapses_db.append(sy)

print("(Running time: %0.4f seconds)" % (time.time() - time_init_neu))
time_init_syn = time.time()


# Generate links
print "\nConfiguring synaptic links..."

# Pre-synaptic neuron types by index
pre_pyr = range(n_pyr)
pre_exc = range(n_pyr, n_pyr+n_exc)
pre_sin = range(n_pyr+n_exc, n_pyr+n_exc+n_sin)
pre_fin = range(n_pyr+n_exc+n_sin, n_neurons)

# Post-synaptic links
def post_pyr(C):
    return np.unique(np.hstack((
           np.random.choice(pre_exc, int(np.ceil(1.00*C)), replace=False),
           np.random.choice(pre_sin, int(np.ceil(0.25*C)), replace=False),
           np.random.choice(pre_fin, int(np.ceil(0.30*C)), replace=False)))).ravel()

def post_exc(C): 
    return np.unique(np.random.choice(pre_pyr, int(np.ceil(0.80*C)), replace=False))

def post_sin(C):
    return np.unique(np.hstack((
           np.random.choice(pre_pyr, int(np.ceil(0.25*C)), replace=False),
           np.random.choice(pre_fin, int(np.ceil(0.10*C)), replace=False)))).ravel()

def post_fin(C):
    return np.unique(np.random.choice(pre_pyr, int(np.ceil(0.80*C)), replace=False))

# Connect pre-synaptic neurons according to type
links_db = {}
for pre in range(n_neurons):
    nn = neurons_db[pre]
    if nn.typ == 'pyr':
        links_db.update({pre: post_pyr(C)})
    elif nn.typ == 'exc':
        links_db.update({pre: post_exc(C)})
    elif nn.typ == 'sin':
        links_db.update({pre: post_sin(C)})
    elif nn.typ == 'fin':
        links_db.update({pre: post_fin(C)})

print "Outbound connections:"
print "Pyramidal:", post_pyr(C).shape[0]
print "Excitatory:", post_exc(C).shape[0]
print "Slow inhibitory:", post_sin(C).shape[0]
print "Fast inhibitory:", post_fin(C).shape[0]


# Collect synaptic links into a dataframe
df = pd.DataFrame([links_db]).T
df.columns = ['links']
df['neuron'] = df.index

df['type'] = (['pyr'] * n_pyr) + (['exc'] * n_exc) +\
             (['sin'] * n_sin) + (['fin'] * n_fin) 

links_db = df[['neuron', 'type', 'links']]

# print links_db.to_string(index=False)


# Initialize synapse
print "\nCreating synapses..."
df = pd.DataFrame()
for pre in range(n_neurons):
    print "... neuron", pre
    links = links_db.iloc[pre]['links']

    for post in links:
        typ = neurons_db[pre].typ
        syn = synapse.WB(typ=typ)

        df = df.append({'pre':  pre, 
                        'post': post, 
                        'syn': syn},
                        ignore_index=True)


df['pre'] = df['pre'].astype(int)    
df['post'] = df['post'].astype(int) 
synapses_db = df[['pre', 'post', 'syn']]   
# print synapses_db.to_string(index=False)

print("(Running time: %0.4f seconds)" % (time.time() - time_init_syn))


#______________________________________________________________________________
# Run simulation
time_run = time.time()
print "\nRunning simulation..."

ti = 0
for t in np.arange(start_time, end_time, inter_time):
    if not (t*100)%100: print "Time:", t, "ms"

    for nn, I_syn in zip(neurons_db, I_syn_db):
        # Add stimulus
        I_rand = np.random.normal(loc=I_mu, scale=I_sigma)
        I_app = I_rand*0
        # I_app = I_rand if np.abs(t - 0.1) > 0.1 else 0 
        # I_app = 10*(t>.100) - 10*(t>.200) + 35*(t>.300) - 35*(t>.400)

        nn.run(t, I_syn+I_app)


    # Forward pass voltage through synapses
    # print "forward"
    # pre_n: pre-synaptic neuron (id)
    for pre_n in range(n_neurons):
        pre_df = synapses_db[synapses_db['pre'] == pre_n]
        pre_syn = pre_df['syn'].tolist()

        # Get pre-synaptic voltage
        V_pre = neurons_db[pre_n].V

        # Run synapse
        for syn in pre_syn:
            syn.run(t, V_pre)


    # Collect all post-synaptic currents
    # post_n: post-synaptic neuron (id)
    # print "postsyn"
    for post_n in range(n_neurons):
        # Get all pre-synaptic links to post_n
        pre_links = synapses_db[synapses_db['post'] == post_n]

        pre_n = pre_links['pre'].tolist()
        post_syn = pre_links['syn'].tolist()

        # Get all post-synaptic current
        psc = [syn.I_syn for syn in post_syn]
        
        # Get the mean post-synaptic voltage
        syn_mean = 0 if not psc else np.mean(psc)
        I_syn_db[post_n] = syn_mean

print("(Running time: %0.4f seconds)" % (time.time() - time_run))

#______________________________________________________________________________
# Plot results

lfp = np.mean([neurons_db[pre].state_df['V'] for pre in pre_pyr], axis=0)

plt.figure(figsize=(18, 4))
plt.title("Local Field Potential Over Time")
plt.plot(lfp)
plt.ylabel('$mV$')
plt.xlabel('ms')
plt.show()

exit()



plt.figure(figsize=(18, 10))
plt.title("Interneuron Membrane Potential")
ax = None
for nn, i in zip(neurons_db, range(n_neurons)):
    idx = (n_neurons * 100) + (10) + (i+1)
    ax = plt.subplot(idx)

    ax.plot(nn.state_df['V'], label=nn.typ)
    ax.legend()
    plt.ylabel('$V (mV)$')

plt.xlabel('ms')
plt.show()
