#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Signal Simulation and Real-time Alpha Rhythm Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import time

class EEGSimulator:
    """
    Class for EEG signal simulation and alpha rhythm visualization
    """
    def __init__(self,
                sampling_rate=250,     # Sampling rate (Hz)
                buffer_size=1000,      # Buffer size for display (samples)
                alpha_band=(8, 12),    # Alpha rhythm frequency range (Hz)
                window_size=500,       # Window size for spectrum analysis (samples)
                update_interval=100):  # Plot update interval (ms)

        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.alpha_band = alpha_band
        self.window_size = window_size
        self.update_interval = update_interval

        # Buffer for storing EEG data
        self.eeg_buffer = np.zeros(buffer_size)

        # Time axis
        self.time_axis = np.linspace(0, buffer_size / sampling_rate, buffer_size)

        # Parameters for signal generation
        self.alpha_amplitude = 1.0     # initial alpha rhythm amplitude
        self.alpha_freq = 10.0         # alpha frequency (Hz)
        self.alpha_direction = 1       # direction of amplitude change
        self.alpha_phase = 0           # current phase
        self.noise_level = 0.5         # noise level

        # Alpha activity over time
        self.alpha_trend = []
        self.alpha_time = []
        self.start_time = time.time()

        # Plot setup
        self.setup_plot()

    def setup_plot(self):
        """Plot configuration"""
        self.fig = plt.figure(figsize=(12, 8))

        # EEG signal plot
        self.ax_eeg = self.fig.add_subplot(3, 1, 1)
        self.ax_eeg.set_title('EEG Signal')
        self.ax_eeg.set_xlabel('Time (s)')
        self.ax_eeg.set_ylabel('Amplitude (μV)')
        self.eeg_line, = self.ax_eeg.plot(self.time_axis, self.eeg_buffer)

        # Spectrum plot
        self.ax_spectrum = self.fig.add_subplot(3, 1, 2)
        self.ax_spectrum.set_title('Signal Spectrum')
        self.ax_spectrum.set_xlabel('Frequency (Hz)')
        self.ax_spectrum.set_ylabel('Power')
        self.ax_spectrum.set_xlim(0, 30)
        self.freqs = np.fft.rfftfreq(self.window_size, 1.0/self.sampling_rate)
        self.spectrum_line, = self.ax_spectrum.plot(self.freqs, np.zeros_like(self.freqs))

        # Highlighting alpha range
        alpha_min, alpha_max = self.alpha_band
        self.ax_spectrum.axvspan(alpha_min, alpha_max, alpha=0.3, color='red')
        self.ax_spectrum.text(alpha_min + (alpha_max - alpha_min) / 2, 0.1, 'Alpha',
                             horizontalalignment='center', color='white', fontweight='bold')

        # Alpha activity trend plot
        self.ax_trend = self.fig.add_subplot(3, 1, 3)
        self.ax_trend.set_title('Alpha Activity Dynamics')
        self.ax_trend.set_xlabel('Time (s)')
        self.ax_trend.set_ylabel('Alpha Power')
        self.trend_line, = self.ax_trend.plot([], [])

        self.fig.tight_layout()

    def generate_sample(self):
        """Generate a new EEG signal sample"""
        # Generate alpha wave with variable amplitude
        self.alpha_phase += 2 * np.pi * self.alpha_freq / self.sampling_rate
        self.alpha_amplitude += 0.01 * self.alpha_direction

        # Change direction if amplitude reaches limits
        if self.alpha_amplitude > 2.0:
            self.alpha_direction = -1
        elif self.alpha_amplitude < 0.2:
            self.alpha_direction = 1

        # Main signal - alpha rhythm
        alpha_signal = self.alpha_amplitude * np.sin(self.alpha_phase)

        # Add other rhythms and noise
        theta_signal = 0.3 * np.sin(2 * np.pi * 6 * time.time())
        beta_signal = 0.2 * np.sin(2 * np.pi * 20 * time.time())
        noise = self.noise_level * np.random.normal()

        return alpha_signal + theta_signal + beta_signal + noise

    def update_plot(self, frame):
        """Update the plot"""
        # Shift buffer to the left
        self.eeg_buffer[:-1] = self.eeg_buffer[1:]

        # Generate a new sample
        self.eeg_buffer[-1] = self.generate_sample()

        # Update EEG plot
        self.eeg_line.set_ydata(self.eeg_buffer)
        self.ax_eeg.set_ylim(min(self.eeg_buffer) - 0.5, max(self.eeg_buffer) + 0.5)

        # Calculate spectrum using Fast Fourier Transform
        if frame % 5 == 0:  # Update spectrum less frequently to reduce computational load
            # Use Hann window to reduce spectral leakage
            windowed_data = self.eeg_buffer[-self.window_size:] * signal.windows.hann(self.window_size)
            spectrum = np.abs(np.fft.rfft(windowed_data))**2
            self.spectrum_line.set_ydata(spectrum)
            self.ax_spectrum.set_ylim(0, max(spectrum) * 1.2)

            # Calculate alpha rhythm power
            alpha_mask = (self.freqs >= self.alpha_band[0]) & (self.freqs <= self.alpha_band[1])
            alpha_power = np.mean(spectrum[alpha_mask]) if np.any(alpha_mask) else 0

            # Add value to trend
            current_time = time.time() - self.start_time
            self.alpha_time.append(current_time)
            self.alpha_trend.append(alpha_power)

            # Limit trend length
            if len(self.alpha_trend) > 100:
                self.alpha_trend = self.alpha_trend[-100:]
                self.alpha_time = self.alpha_time[-100:]

            # Update trend plot
            self.trend_line.set_data(self.alpha_time, self.alpha_trend)
            self.ax_trend.set_xlim(min(self.alpha_time), max(self.alpha_time))
            if self.alpha_trend:
                self.ax_trend.set_ylim(0, max(self.alpha_trend) * 1.2)

        return self.eeg_line, self.spectrum_line, self.trend_line

    def run(self):
        """Start the animation"""
        self.animation = FuncAnimation(
            self.fig, self.update_plot,
            interval=self.update_interval, blit=True)
        plt.show()


if __name__ == "__main__":
    # Create and run EEG simulator
    simulator = EEGSimulator(
        sampling_rate=250,         # Hz
        buffer_size=1000,          # samples
        alpha_band=(8, 12),        # Hz
        window_size=500,           # samples
        update_interval=40)        # ms

    print("Starting EEG simulation and alpha rhythm visualization...")
    print("Press Ctrl+C in the terminal or close the plot window to exit.")

    simulator.run()