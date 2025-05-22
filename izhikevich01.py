# This script simulates a network of 10 Izhikevich neurons using the Brian2 simulator.
# It models the membrane potential dynamics and spike generation of each neuron,
# including synaptic interactions between them.
# The script records spike times and membrane potentials during a 1-second simulation.
# Finally, it visualizes three plots aligned by simulation time:
# 1) Raster plot showing spike times of all neurons,
# 2) Membrane potential trace of the first neuron,
# 3) Sum of membrane potentials of all 10 neurons over time.
# This helps to analyze both individual neuron activity and collective network dynamics.


from brian2 import *

duration = 1 * second

a = 0.02
b = 0.2
c = -65
d = 8

izh_eqs = '''
dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn) / ms : 1
du/dt = a*(b*v - u) / ms : 1
I_syn : 1
'''

N = 10
neurons = NeuronGroup(N, model=izh_eqs,
                      threshold='v >= 30',
                      reset='v = c; u += d',
                      method='euler')

neurons.v = c
neurons.u = b * neurons.v
neurons.I_syn = 0

input_current = TimedArray([5, 0, 0, 0], dt=250*ms)
neurons.run_regularly('I_syn = input_current(t)')

syn = Synapses(neurons, neurons, on_pre='I_syn_post += 2')
syn.connect(condition='i != j')
syn.delay = 5*ms

spike_mon = SpikeMonitor(neurons)
voltage_mon = StateMonitor(neurons, 'v', record=True)

run(duration)

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 8))

# Растер спайків
ax1 = plt.subplot(311)
ax1.plot(spike_mon.t/ms, spike_mon.i, '.k')
ax1.set_xlabel('Час (мс)')
ax1.set_ylabel('Нейрон')
ax1.set_title('Растер спайків')
ax1.set_xlim(0, duration/ms)  # Вирівнюємо час

# Потенціал 0-го нейрона
ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(voltage_mon.t/ms, voltage_mon.v[0])
ax2.set_ylabel('v (без одиниць)')
ax2.set_title('Потенціал нейрона 0')

# Сумарний потенціал усіх 10 нейронів
ax3 = plt.subplot(313, sharex=ax1)
sum_v = np.sum(voltage_mon.v, axis=0)
ax3.plot(voltage_mon.t/ms, sum_v)
ax3.set_xlabel('Час (мс)')
ax3.set_ylabel('Сумарний потенціал')
ax3.set_title('Сумарний потенціал 10 нейронів')

plt.tight_layout()
plt.show()
