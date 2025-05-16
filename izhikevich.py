#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation as animation
from scipy.signal import welch

# --- Speed-up and plot settings ---
speedup = 10           # simulation steps per frame (e.g., 10× faster)
raster_window = 1000   # ms window width for raster; set >= T for full simulation view

# --- Network & simulation parameters ---
Ne, Ni = 800, 200             # excitatory and inhibitory counts
N = Ne + Ni
T = 2000                      # total simulation time (ms)
base_dt = 1.0                 # base time step (ms)
dt = base_dt
total_steps = int(T / dt)

# Compute sampling frequency for EEG buffer (in Hz)
fs = 1000.0 / dt

# Determine frames and steps per update
skip = speedup                # simulation steps per animation frame
frame_count = int(np.ceil(total_steps / skip))

# --- Izhikevich neuron parameters ---
re = np.random.rand(Ne)
ri = np.random.rand(Ni)
a = np.concatenate((0.02 * np.ones(Ne), 0.02 + 0.08 * ri))
b = np.concatenate((0.20 * np.ones(Ne), 0.25 - 0.05 * ri))
c = np.concatenate((-65 + 15 * re**2,    -65 * np.ones(Ni)))
d = np.concatenate((  8 - 6 * re**2,      2 * np.ones(Ni)))

# --- Synaptic connectivity ---
S = np.hstack((
    0.5 * np.random.rand(N, Ne),     # excitatory→all
    -1.0 * np.random.rand(N, Ni)      # inhibitory→all
))

# --- State variables ---
v = -65.0 * np.ones(N)       # membrane potentials
u = b * v                     # recovery variables

# --- Buffers for plotting ---
spike_buffer = deque(maxlen=5000)
eeg_buffer   = deque([0.0] * 1000, maxlen=1000)

# --- EEG filter state ---
alpha = 0.9               # filter coefficient (~15 Hz cutoff)
E_prev = 0.0              # previous EEG value

# Global step counter
current_step = 0

# --- Matplotlib setup ---
fig, (ax_raster, ax_eeg, ax_spec) = plt.subplots(3, 1, figsize=(8, 10))

# Raster plot
raster_scatter = ax_raster.scatter([], [], s=2, c='k')
ax_raster.set_ylabel('Neuron index')
ax_raster.set_ylim(0, N)
ax_raster.set_xlim(0, raster_window)

# EEG time-series plot
ax_eeg_line, = ax_eeg.plot([], [])
ax_eeg.set_xlabel('Time (ms)')
ax_eeg.set_ylabel('EEG (a.u.)')
ax_eeg.set_xlim(-len(eeg_buffer), 0)
ax_eeg.set_ylim(-80, -20)

# Frequency-Amplitude plot (power spectrum)
spec_line, = ax_spec.plot([], [])
ax_spec.set_xlabel('Frequency (Hz)')
ax_spec.set_ylabel('Amplitude')
ax_spec.set_xlim(0, 50)  # limit to 0–50 Hz
ax_spec.set_ylim(0, None)


def init():
    raster_scatter.set_offsets(np.empty((0, 2)))
    ax_eeg_line.set_data([], [])
    spec_line.set_data([], [])
    return raster_scatter, ax_eeg_line, spec_line


def update(_frame):
    global current_step, v, u, E_prev
    # Run multiple simulation steps
    for _ in range(skip):
        t = current_step * dt

        # 1. Noisy input
        I = np.concatenate((5.0 * np.random.randn(Ne),
                            2.0 * np.random.randn(Ni)))

        # 2. Spike detection & reset
        fired = np.where(v >= 30.0)[0]
        if fired.size > 0:
            for neuron in fired:
                spike_buffer.append((t, neuron))
            v[fired] = c[fired]
            u[fired] += d[fired]
            I += np.sum(S[:, fired], axis=1)

        # 3. Two-step Euler for v
        dv = 0.04 * v**2 + 5 * v + 140 - u + I
        v += 0.5 * dt * dv
        dv = 0.04 * v**2 + 5 * v + 140 - u + I
        v += 0.5 * dt * dv

        # 4. Euler for u
        u += dt * a * (b * v - u)

        # 5. EEG proxy
        v_exc = v[:Ne].mean()
        E_curr = alpha * E_prev + (1 - alpha) * v_exc
        E_prev = E_curr
        eeg_buffer.append(E_curr)

        current_step += 1

    # Update raster scatter
    coords = np.array(spike_buffer)
    if coords.size > 0:
        x_min = max(0, t - raster_window)
        raster_scatter.set_offsets(coords)
        ax_raster.set_xlim(x_min, x_min + raster_window)

    # Update EEG time-series
    eeg_data = np.array(eeg_buffer)
    ax_eeg_line.set_data(np.arange(-len(eeg_data), 0), eeg_data)

    # Compute and update power spectrum via Welch
    f, Pxx = welch(eeg_data, fs=fs, nperseg=min(len(eeg_data), 256))
    spec_line.set_data(f, Pxx)
    ax_spec.set_ylim(0, np.max(Pxx) * 1.1)

    return raster_scatter, ax_eeg_line, spec_line

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=frame_count,
    init_func=init, interval=1, blit=False
)

plt.tight_layout()
plt.show()
