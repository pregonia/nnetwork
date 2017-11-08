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

# Clock settings
start_time = 0                  # msec
end_time = 100                  # msec
inter_time = 0.01               # msec

# Network components
n_neurons = 4                   # number of neurons in network
neurons_db = []                 # list of neurons
synapses_db = []                # dataframe of synapses
links_db = {}                   # dataframe of synaptic connections
I_syn_db = np.zeros(n_neurons)  # array of neuronal external current


# Network constants
I_mu = 10                       # mean, bias current 
I_sigma = 0.02                  # standard deviation, bias current 
p_syn = 1                       # percent of synaptic connections (per neuron)
M_syn = int(n_neurons * p_syn)  # number of synaptic connections (per neuron)

p_pyr = 0.25                    # percent of pyramidal synapses (per neuron)
p_exc = 0.25                    # percent of excitatory synapses (per neuron)
p_sin = 0.25                    # percent of slow synapses (per neuron)
p_fin = 0.25                    # percent of fast synapses (per neuron)

n_pyr = int(n_neurons * p_pyr)              # number of pyramidal neurons
n_exc = int(n_neurons * p_exc)              # number of slow neurons
n_sin = int(np.ceil(n_neurons * p_sin))     # number of slow neurons
n_fin = n_neurons - n_pyr - n_exc - n_sin   # number of fast neurons

M_pyr = int(M_syn * p_pyr)              # number of pyramidal synapses (per neuron)
M_exc = int(M_syn * p_exc)              # number of slow synapses (per neuron)
M_sin = int(np.ceil(M_syn * p_sin))     # number of slow synapses (per neuron)
M_fin = M_syn - M_pyr - M_exc - M_sin   # number of fast synapses (per neuron)


#______________________________________________________________________________
# Initialize network
print "Initializing network..." 

# Initialize neurons
print "\nCreating neurons..."
print "Neurons:", n_neurons
print "Pyramidal:", n_pyr
print "Excitatory:", n_exc
print "Slow inhibitory:", n_sin
print "Fast inhibitory:", n_fin


syn_typ = ['pyr', 'exc', 'sin', 'fin']  # types of synapses
syn_p = [p_pyr, p_exc, p_sin, p_fin]    # probability / weights

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


# Generate links
print "\nConfiguring synaptic links..."

# Pre-synaptic neuron types by index
pre_pyr = range(n_pyr)
pre_exc = range(n_pyr, n_pyr+n_exc)
pre_sin = range(n_pyr+n_exc, n_pyr+n_exc+n_sin)
pre_fin = range(n_pyr+n_exc+n_sin, n_neurons)

# Post-synaptic links
post_pyr = pre_exc + pre_sin + pre_fin
post_exc = pre_pyr
post_sin = pre_pyr + pre_fin
post_fin = pre_pyr

nn_pre = [pre_pyr, pre_exc, pre_sin, pre_fin]
nn_post= [post_pyr, post_exc, post_sin, post_fin]

# Connect pre-synaptic neurons according to type
links_db = {}
for pre, post in zip(nn_pre, nn_post):
    links_db.update({pr: post for pr in pre})

# Collect synaptic links into a dataframe
df = pd.DataFrame([links_db]).T
df.columns = ['links']
df['neuron'] = df.index

df['type'] = (['pyr'] * n_pyr) + (['exc'] * n_exc) +\
             (['sin'] * n_sin) + (['fin'] * n_fin) 

links_db = df[['neuron', 'type', 'links']]

print links_db.to_string(index=False)


# Initialize synapse
print "\nCreating synapses..."
df = pd.DataFrame()
for pre in range(n_neurons):
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
print synapses_db.to_string(index=False)



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

        nn.run(t, I_syn+I_app)
        

    # Forward pass voltage through synapses
    # pre_n: pre-synaptic neuron (id)
    for pre_n in range(n_neurons):
        pre_df = synapses_db[synapses_db['pre'] == pre_n]
        pre_syn = pre_df['syn'].tolist()

        # Get pre-synaptic voltage
        V_pre = neurons_db[pre_n].V

        # Run synapse
        for syn in pre_syn:
            syn.run(t, V_pre)

    # print I_syn_db

    # Collect all post-synaptic currents
    # post_n: post-synaptic neuron (id)
    for post_n in range(n_neurons):
        # Get all pre-synaptic links to post_n
        pre_links = synapses_db[synapses_db['post'] == post_n]
        pre_n = pre_links['pre'].tolist()
        post_syn = pre_links['syn'].tolist()

        # Get all post-synaptic current
        psc = [syn.I_syn for syn in post_syn]
        
        # Get the mean post-synaptic voltage
        I_syn_db[post_n] = np.mean(psc)



# exit()

#______________________________________________________________________________
# Plot results

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
