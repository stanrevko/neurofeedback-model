#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation as animation
from scipy.signal import welch
import matplotlib.gridspec as gridspec

# --- Speed-up and plot settings ---
speedup = 10           # simulation steps per frame
raster_window = 1000   # ms window width for raster (1 second)

# --- Network & simulation parameters ---
Ne, Ni = 800, 200
N = Ne + Ni
T = 2000                      # total time (ms)
dt = 1.0                     # time step (ms)
total_steps = int(T / dt)

# EEG sampling frequency (Hz)
fs = 1000.0 / dt

# Simulation stepping
skip = speedup
frame_count = int(np.ceil(total_steps / skip))

# --- Izhikevich parameters ---
re = np.random.rand(Ne)
ri = np.random.rand(Ni)
a = np.concatenate((0.02 * np.ones(Ne), 0.02 + 0.08 * ri))
b = np.concatenate((0.2 * np.ones(Ne), 0.25 - 0.05 * ri))
c = np.concatenate((-65 + 15 * re**2, -65 * np.ones(Ni)))
d = np.concatenate((8 - 6 * re**2, 2 * np.ones(Ni)))

# Connectivity
S = np.hstack((0.5 * np.random.rand(N, Ne), -1.0 * np.random.rand(N, Ni)))

# State variables
v = -65.0 * np.ones(N)       # membrane potentials
u = b * v                     # recovery variables

# Buffers
spike_buffer = deque()       # holds (time, neuron_index) for current window
eeg_buffer = deque([0.0] * 1000, maxlen=1000)

# EEG filter
alpha = 0.9
E_prev = 0.0

# Time tracking
t = 0
t_sec = -1

# --- Figure & axes via GridSpec ---
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1], wspace=0.4)
ax_raster = fig.add_subplot(gs[0, 0])
ax_eeg    = fig.add_subplot(gs[1, 0])
ax_spec   = fig.add_subplot(gs[2, 0])
ax_fb     = fig.add_subplot(gs[:, 1])  # feedback spans all rows

# Raster axis
raster_scatter = ax_raster.scatter([], [], s=2, c='k')
ax_raster.set_xlim(0, raster_window)
ax_raster.set_ylim(0, N)
ax_raster.set_xlabel('Time within second (ms)')
ax_raster.set_ylabel('Neuron index')

# EEG time-series axis
ax_eeg_line, = ax_eeg.plot([], [])
ax_eeg.set_xlim(-len(eeg_buffer), 0)
ax_eeg.set_ylim(-80, -20)
ax_eeg.set_xlabel('Time (ms)')
ax_eeg.set_ylabel('EEG (a.u.)')

# Spectrum axis
spec_line, = ax_spec.plot([], [])
ax_spec.set_xlim(0, 50)
ax_spec.set_ylim(0, 0.4)
ax_spec.set_xlabel('Frequency (Hz)')
ax_spec.set_ylabel('Amplitude')

# Neurofeedback bar (persistent) and text label
barh = ax_fb.barh([0], [0], height=0.5, color='g')[0]
fb_text = ax_fb.text(0, 0, '', va='center', ha='left', fontsize=12)
ax_fb.set_xlim(0, 1)
ax_fb.set_ylim(-0.5, 0.5)
ax_fb.set_yticks([])
ax_fb.set_xlabel('Peak / Integral')
ax_fb.set_title('Neurofeedback')

# Initialization function
def init():
    raster_scatter.set_offsets(np.empty((0, 2)))
    ax_eeg_line.set_data([], [])
    spec_line.set_data([], [])
    barh.set_width(0)
    fb_text.set_text('')
    return raster_scatter, ax_eeg_line, spec_line, barh, fb_text

# Update function
def update(_frame):
    global v, u, E_prev, t, t_sec
    # Run multiple simulation steps per frame
    for _ in range(skip):
        t += dt
        current_sec = int(t // raster_window)
        if current_sec != t_sec:
            spike_buffer.clear()
            t_sec = current_sec

        # 1. Noisy input from thalamus
        I = np.concatenate((5.0 * np.random.randn(Ne), 2.0 * np.random.randn(Ni)))
        # 2. Detect spikes
        fired = np.where(v >= 30.0)[0]
        if fired.size > 0:
            for neuron in fired:
                spike_buffer.append((t, neuron))
            v[fired] = c[fired]
            u[fired] += d[fired]
            I += np.sum(S[:, fired], axis=1)

        # 3. Two-step Euler integration for v
        dv = 0.04 * v**2 + 5 * v + 140 - u + I
        v += 0.5 * dt * dv
        dv = 0.04 * v**2 + 5 * v + 140 - u + I
        v += 0.5 * dt * dv

        # 4. Euler integration for u
        u += dt * a * (b * v - u)

        # 5. EEG proxy: mean excitatory potential + low-pass
        v_exc = v[:Ne].mean()
        E_curr = alpha * E_prev + (1.0 - alpha) * v_exc
        E_prev = E_curr
        eeg_buffer.append(E_curr)

    # Update raster plot
    if spike_buffer:
        times, neurons = zip(*spike_buffer)
        rel_times = np.array(times) - t_sec * raster_window
        offsets = np.column_stack((rel_times, neurons))
        raster_scatter.set_offsets(offsets)
    else:
        raster_scatter.set_offsets(np.empty((0, 2)))

    # Update EEG trace
    eeg_data = np.array(eeg_buffer)
    ax_eeg_line.set_data(np.arange(-len(eeg_data), 0), eeg_data)

    # Update power spectrum
    f, Pxx = welch(eeg_data, fs=fs, nperseg=min(len(eeg_data), 256))
    spec_line.set_data(f, Pxx)

    # Compute alpha-band integral and normalize to maximum amplitude
    alpha_mask = (f >= 8) & (f <= 12)
    P_alpha = Pxx[alpha_mask]
    integral_alpha = np.sum(P_alpha) *10
    max_amplitude = (12-8) * 0.4
    percent = integral_alpha / max_amplitude if max_amplitude > 0 else 0.0

    # Update neurofeedback bar and label with percent
    barh.set_width(percent)
    fb_text.set_x(percent + 0.02)
    fb_text.set_text(f"{percent*100:.1f}%")

    return raster_scatter, ax_eeg_line, spec_line, barh, fb_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=frame_count, init_func=init, interval=1, blit=False)

plt.tight_layout()
plt.show()
