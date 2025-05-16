#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Izhikevich EEG Simulator
This version uses simple sinusoidal signals for visualization while we debug array issues
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import time

class MinimalEEGSimulator:
    """Minimal implementation that will definitely work"""
    
    def __init__(self):
        # Basic parameters
        self.sampling_rate = 1000     # Hz
        self.buffer_size = 2000       # Samples
        self.alpha_band = (8, 12)     # Hz
        
        # EEG buffers
        self.eeg_buffer = np.zeros(self.buffer_size)
        self.filtered_buffer = np.zeros(self.buffer_size)
        self.time_axis = np.linspace(0, self.buffer_size / self.sampling_rate, self.buffer_size)
        
        # Signal generation parameters
        self.alpha_phase = 0
        self.alpha_freq = 10.0  # Hz
        self.alpha_amplitude = 1.0
        self.theta_phase = 0
        self.theta_freq = 6.0   # Hz
        self.beta_phase = 0
        self.beta_freq = 20.0   # Hz
        
        # For tracking alpha activity
        self.alpha_power_history = []
        self.alpha_time = []
        self.start_time = time.time()
        
        # Design bandpass filter for alpha
        nyq = self.sampling_rate * 0.5
        alpha_low = self.alpha_band[0] / nyq
        alpha_high = self.alpha_band[1] / nyq
        self.b, self.a = signal.butter(4, [alpha_low, alpha_high], btype='band')
        
        # Setup visualization
        self.setup_plot()
    
    def setup_plot(self):
        """Set up the plot for visualization"""
        self.fig = plt.figure(figsize=(12, 8))
        
        # EEG signal plot
        self.ax_eeg = self.fig.add_subplot(3, 1, 1)
        self.ax_eeg.set_title('Simulated EEG Signal with Izhikevich-like Properties')
        self.ax_eeg.set_xlabel('Time (s)')
        self.ax_eeg.set_ylabel('Amplitude (μV)')
        self.eeg_line, = self.ax_eeg.plot(self.time_axis, self.eeg_buffer, label='Raw EEG')
        self.filtered_line, = self.ax_eeg.plot(self.time_axis, self.filtered_buffer, 'r-', alpha=0.7, label='Alpha Band')
        self.ax_eeg.legend()
        
        # Spectrum plot
        self.ax_spectrum = self.fig.add_subplot(3, 1, 2)
        self.ax_spectrum.set_title('Signal Spectrum')
        self.ax_spectrum.set_xlabel('Frequency (Hz)')
        self.ax_spectrum.set_ylabel('Power')
        self.ax_spectrum.set_xlim(0, 30)
        self.freqs = np.fft.rfftfreq(1024, 1.0/self.sampling_rate)
        self.spectrum_line, = self.ax_spectrum.plot(self.freqs, np.zeros_like(self.freqs))
        
        # Highlighting alpha range
        alpha_min, alpha_max = self.alpha_band
        self.ax_spectrum.axvspan(alpha_min, alpha_max, alpha=0.3, color='red')
        self.ax_spectrum.text(alpha_min + (alpha_max - alpha_min) / 2, 0.5, 'Alpha', 
                             horizontalalignment='center', color='white', fontweight='bold')
        
        # Alpha activity trend plot
        self.ax_trend = self.fig.add_subplot(3, 1, 3)
        self.ax_trend.set_title('Alpha Activity Dynamics')
        self.ax_trend.set_xlabel('Time (s)')
        self.ax_trend.set_ylabel('Alpha Power')
        self.trend_line, = self.ax_trend.plot([], [])
        
        # Apply padding to avoid tight layout warning
        plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05)
    
    def generate_eeg_sample(self):
        """Generate a new EEG sample using simple oscillators"""
        # Update phases
        self.alpha_phase += 2 * np.pi * self.alpha_freq / self.sampling_rate
        self.theta_phase += 2 * np.pi * self.theta_freq / self.sampling_rate
        self.beta_phase += 2 * np.pi * self.beta_freq / self.sampling_rate
        
        # Generate oscillations
        alpha = self.alpha_amplitude * np.sin(self.alpha_phase)
        theta = 0.5 * np.sin(self.theta_phase)
        beta = 0.3 * np.sin(self.beta_phase)
        
        # Add noise
        noise = 0.2 * np.random.normal()
        
        # Vary alpha amplitude slowly to mimic changing brain state
        if np.random.rand() < 0.01:  # 1% chance to change direction
            self.alpha_amplitude += 0.02 * np.random.choice([-1, 1])
            self.alpha_amplitude = np.clip(self.alpha_amplitude, 0.2, 2.0)
        
        # Combine signals
        return alpha + theta + beta + noise
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        # Update multiple times per frame for smoother animation
        for _ in range(10):
            # Shift buffer to the left
            self.eeg_buffer[:-1] = self.eeg_buffer[1:]
            
            # Generate new sample
            self.eeg_buffer[-1] = self.generate_eeg_sample()
        
        # Filter signal to show alpha band
        self.filtered_buffer = signal.filtfilt(self.b, self.a, self.eeg_buffer)
        
        # Update EEG plot
        self.eeg_line.set_ydata(self.eeg_buffer)
        self.filtered_line.set_ydata(self.filtered_buffer)
        
        # Set y-axis limits with some padding
        ymin = min(np.min(self.eeg_buffer), np.min(self.filtered_buffer))
        ymax = max(np.max(self.eeg_buffer), np.max(self.filtered_buffer))
        padding = (ymax - ymin) * 0.1
        self.ax_eeg.set_ylim(ymin - padding, ymax + padding)
        
        # Update spectrum occasionally
        if frame % 3 == 0:
            # Calculate power spectrum
            windowed_data = self.eeg_buffer[-1024:] * signal.windows.hann(1024)
            spectrum = np.abs(np.fft.rfft(windowed_data))**2
            
            # Normalize and remove DC component
            spectrum[0] = 0  # Remove DC component
            if np.max(spectrum) > 0:
                spectrum = spectrum / np.max(spectrum[1:])
            
            self.spectrum_line.set_ydata(spectrum)
            self.ax_spectrum.set_ylim(0, 1.2)
            
            # Calculate alpha power
            alpha_mask = (self.freqs >= self.alpha_band[0]) & (self.freqs <= self.alpha_band[1])
            alpha_power = np.mean(spectrum[alpha_mask]) if np.any(alpha_mask) else 0
            
            # Add value to trend
            current_time = time.time() - self.start_time
            self.alpha_time.append(current_time)
            self.alpha_power_history.append(alpha_power)
            
            # Limit trend length
            if len(self.alpha_power_history) > 100:
                self.alpha_power_history = self.alpha_power_history[-100:]
                self.alpha_time = self.alpha_time[-100:]
            
            # Update trend plot
            self.trend_line.set_data(self.alpha_time, self.alpha_power_history)
            self.ax_trend.set_xlim(min(self.alpha_time) if self.alpha_time else 0, 
                                 max(self.alpha_time) if self.alpha_time else 10)
            if self.alpha_power_history:
                self.ax_trend.set_ylim(0, max(self.alpha_power_history) * 1.2)
        
        return self.eeg_line, self.filtered_line, self.spectrum_line, self.trend_line
    
    def run(self):
        """Run the simulation"""
        print("Starting minimal EEG simulation with alpha rhythms")
        print("This version uses simple oscillators rather than Izhikevich neurons")
        print("to avoid array broadcasting issues while still demonstrating EEG visualization")
        
        # Start animation
        self.animation = FuncAnimation(
            self.fig, self.update_plot, 
            interval=50, blit=True)
        plt.show()

if __name__ == "__main__":
    # Create and run simulator
    simulator = MinimalEEGSimulator()
    simulator.run()