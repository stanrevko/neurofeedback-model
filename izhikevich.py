#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation as animation

# --- Network & simulation parameters ---
Ne, Ni = 800, 200             # excitatory and inhibitory neuron counts
N = Ne + Ni
T = 2000                      # total simulation time (ms)
dt = 1.0                      # time step (ms)
steps = int(T / dt)

# --- Izhikevich neuron parameters ---
re = np.random.rand(Ne)
ri = np.random.rand(Ni)

a = np.concatenate((0.02 * np.ones(Ne), 0.02 + 0.08 * ri))
b = np.concatenate((0.20 * np.ones(Ne), 0.25 - 0.05 * ri))
c = np.concatenate((-65 + 15 * re**2,    -65 * np.ones(Ni)))
d = np.concatenate((  8 - 6 * re**2,      2 * np.ones(Ni)))

# --- Synaptic connectivity matrix ---
S = np.hstack((
    0.5 * np.random.rand(N, Ne),     # excitatory→all
    -1.0 * np.random.rand(N, Ni)     # inhibitory→all
))

# --- State variables ---
v = -65.0 * np.ones(N)       # membrane potentials
u = b * v                     # recovery variables

# --- Buffers for real-time plotting ---
spike_buffer = deque(maxlen=5000)          # store (t, neuron_index)
eeg_buffer   = deque([0.0] * 1000,  maxlen=1000)

# --- EEG low-pass filter state ---
alpha = 0.9               # filter coefficient (cutoff ~15 Hz)
E_prev = 0.0              # previous EEG value

# --- Set up Matplotlib figure ---
fig, (ax_raster, ax_eeg) = plt.subplots(2, 1, figsize=(8, 6))

# Raster plot setup
raster_scatter = ax_raster.scatter([], [], s=2, c='k')
ax_raster.set_xlim(0, T/2)
ax_raster.set_ylim(0, N)
ax_raster.set_ylabel('Neuron index')

# EEG plot setup
ax_eeg_line, = ax_eeg.plot([], [], c='b')
ax_eeg.set_xlim(-1000, 0)
ax_eeg.set_ylim(-100, 100)
ax_eeg.set_xlabel('Time (ms)')
ax_eeg.set_ylabel('EEG (a.u.)')

def init():
    """
    Initialize the raster scatter and EEG line.
    Use an empty (0×2) array to clear the scatter properly.
    """
    raster_scatter.set_offsets(np.empty((0, 2)))
    ax_eeg_line.set_data([], [])
    return raster_scatter, ax_eeg_line

def update(frame):
    """
    Perform one simulation time‐step:
    1. Integrate all neurons via two‐step Euler for v and Euler for u.
    2. Detect spikes, reset v & u, record spike times, and deliver synaptic inputs.
    3. Compute and low‐pass filter the mean excitatory potential for EEG.
    4. Update the raster and EEG plots.
    """
    global v, u, E_prev

    t = frame * dt

    # 1. Generate thalamic (noisy) input
    I = np.concatenate((5.0 * np.random.randn(Ne),
                        2.0 * np.random.randn(Ni)))

    # 2. Find spiking neurons
    fired = np.where(v >= 30.0)[0]
    if fired.size > 0:
        for neuron in fired:
            spike_buffer.append((t, neuron))
        # Reset after-spike variables
        v[fired] = c[fired]
        u[fired] += d[fired]
        # Deliver synaptic pulses
        I += np.sum(S[:, fired], axis=1)

    # 3. Two half‐steps Euler integration for v
    dv = 0.04 * v**2 + 5 * v + 140 - u + I
    v += 0.5 * dt * dv
    dv = 0.04 * v**2 + 5 * v + 140 - u + I
    v += 0.5 * dt * dv

    # 4. Euler integration for u
    u += dt * a * (b * v - u)

    # 5. Compute EEG: low‐pass filter of mean excitatory v
    v_exc = v[:Ne].mean()
    E_curr = alpha * E_prev + (1 - alpha) * v_exc
    E_prev = E_curr
    eeg_buffer.append(E_curr)

    # 6. Update raster plot
    coords = np.array(spike_buffer)
    if coords.size > 0:
        raster_scatter.set_offsets(coords)

    # 7. Update EEG plot (last 1000 ms)
    eeg_data = np.array(eeg_buffer)
    ax_eeg_line.set_data(np.arange(-len(eeg_data), 0), eeg_data)

    return raster_scatter, ax_eeg_line

# Create the animation (disable blitting to avoid _resize_id bug)
ani = animation.FuncAnimation(
    fig, update, frames=steps,
    init_func=init, interval=1, blit=False
)

plt.tight_layout()
plt.show()
