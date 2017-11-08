import nest
import nest.voltage_trace
import pylab


neuron = nest.Create('iaf_psc_alpha')
sine = nest.Create('ac_generator', 1,
                    {'amplitude': 100.0,
                    'frequency': 2.0})
noise = nest.Create('poisson_generator', 2,
                    [{'rate': 70000.0},
                     {'rate': 20000.0}])
voltmeter = nest.Create('voltmeter', 1, 
                       {'withgid': True})


nest.Connect(sine, neuron)
nest.Connect(voltmeter, neuron)
nest.Connect(noise[:1], neuron, syn_spec={'weight': 1.0, 'delay': 1.0})
nest.Connect(noise[1:], neuron, syn_spec={'weight': -1.0, 'delay': 1.0})
nest.Simulate(1000.0)
nest.voltage_trace.from_device(voltmeter)
pylab.show()






# import nest
# import nest.raster-plot
# import pylab

# g       = 5.0
# eta     = 2.0
# delay   = 1.5
# tau_m   = 20.0

# V_th    = 20.0
# N_E     = 8000
# N_I     = 2000
# N_neurons = N_E + N_I

# C_E     = N_E/10
# C_I     = N_I/10
