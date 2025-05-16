#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neurofeedback model based on Davelaar (2018)
This simulation implements a computational model of neurofeedback training
focusing on alpha frequency upregulation.
"""

from brian2 import *
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import os

class NeurofeedbackModel:
    """
    Implementation of a computational model for neurofeedback based on
    Davelaar (2018). The model simulates EEG alpha upregulation using
    striatal learning mechanisms.
    """
    
    def __init__(self, 
                 ne=800,                # number of excitatory neurons
                 ni=200,                # number of inhibitory neurons
                 n_striatum=1000,       # number of striatal units
                 duration=10*second,    # total simulation duration
                 dt=0.1*ms,             # simulation time step
                 feedback_interval=100*ms,  # how often to give feedback
                 learning_rate=0.01,    # learning rate for weights update
                 alpha_band=(8, 12),    # alpha frequency band (Hz)
                 threshold=1.0,         # threshold for positive feedback
                 seed=None,             # random seed
                 results_dir="../results"  # directory to save results
                ):
        """
        Initialize the neurofeedback model with the given parameters.
        """
        if seed is not None:
            np.random.seed(seed)
            seed_gen = seed
        else:
            seed_gen = np.random.randint(10000)
        
        self.seed = seed_gen
        self.ne = ne
        self.ni = ni
        self.n_striatum = n_striatum
        self.duration = duration
        self.dt = dt
        self.feedback_interval = feedback_interval
        self.learning_rate = learning_rate
        self.alpha_band = alpha_band
        self.threshold = threshold
        self.results_dir = Path(results_dir)
        
        # Make sure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize striatal weights and probabilities
        self.w = np.ones(n_striatum) / n_striatum  # initial equal weights
        self.p_active = self.w.copy()              # activation probabilities
        self.target = np.random.randint(n_striatum)  # randomly select target MSN
        
        # Data collection for analysis
        self.weight_history = []
        self.alpha_power_history = []
        self.feedback_history = []
        self.target_activation_history = []
        
        # Set up the Izhikevich neural network
        self._setup_network()
    
    def _setup_network(self):
        """Set up the neural network with Izhikevich neurons"""
        
        # Izhikevich neuron equations - FIXED for Brian2 compatibility
        eqs = '''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I)/ms : 1
        du/dt = a*(b*v - u)/ms                      : 1
        I = I_thal + I_noise                        : 1
        I_thal : 1
        dI_noise/dt = -I_noise/ms + sigma * xi * sqrt(2/ms)/ms : 1
        a : 1
        b : 1
        c : 1
        d : 1
        sigma : 1
        '''
        
        # Create neuron groups
        self.G_e = NeuronGroup(self.ne, model=eqs, threshold='v>30', 
                              reset='v=c; u+=d', method='euler', dt=self.dt)
        self.G_i = NeuronGroup(self.ni, model=eqs, threshold='v>30', 
                              reset='v=c; u+=d', method='euler', dt=self.dt)
        
        # Initialize neuron parameters
        # Excitatory neurons
        re = np.random.rand(self.ne)
        self.G_e.a = 0.02
        self.G_e.b = 0.2
        self.G_e.c = -65 + 15 * re**2
        self.G_e.d = 8 - 6 * re**2
        self.G_e.sigma = 5
        
        # Inhibitory neurons
        ri = np.random.rand(self.ni)
        self.G_i.a = 0.02 + 0.08 * ri
        self.G_i.b = 0.25 - 0.05 * ri
        self.G_i.c = -65
        self.G_i.d = 2
        self.G_i.sigma = 2
        
        # Initialize membrane potentials
        self.G_e.v = -65
        self.G_e.u = self.G_e.b * self.G_e.v
        self.G_i.v = -65
        self.G_i.u = self.G_i.b * self.G_i.v
        self.G_e.I_noise = 0
        self.G_i.I_noise = 0
        
        # Create synaptic connections (full connectivity)
        self.S_ee = Synapses(self.G_e, self.G_e, on_pre='v_post += 0.5')
        self.S_ei = Synapses(self.G_e, self.G_i, on_pre='v_post += 0.5')
        self.S_ie = Synapses(self.G_i, self.G_e, on_pre='v_post -= 1.0')
        self.S_ii = Synapses(self.G_i, self.G_i, on_pre='v_post -= 1.0')
        
        # Connect all neurons except self-connections
        for S in [self.S_ee, self.S_ei, self.S_ie, self.S_ii]:
            S.connect(condition='i!=j')
        
        # Set up state monitor for recording membrane potentials
        self.mon = StateMonitor(self.G_e, 'v', record=True)
    
    def update_thalamic_input(self, active_msn):
        """
        Update thalamic input based on active MSNs
        
        Parameters:
        -----------
        active_msn : array
            Binary array indicating which MSNs are active
        
        Returns:
        --------
        float
            Thalamic input value
        """
        # If target MSN is active, boost the thalamic input
        base = 5.0
        boost = 2.0
        return base + boost * active_msn[self.target]
    
    def simulate(self, baseline_run=False):
        """
        Run the simulation with neurofeedback
        
        Parameters:
        -----------
        baseline_run : bool
            If True, run as baseline without updating weights
        
        Returns:
        --------
        dict
            Dictionary containing simulation results
        """
        # Start simulation
        print(f"Starting {'baseline' if baseline_run else 'neurofeedback'} simulation...")
        
        # Reset state monitor
        self.mon = StateMonitor(self.G_e, 'v', record=True)
        
        # Global variables for storing simulation data
        self.weight_history = []
        self.alpha_power_history = []
        self.feedback_history = []
        self.target_activation_history = []
        
        # Initialize Brian2 network
        net = Network(self.G_e, self.G_i, self.S_ee, self.S_ei, 
                      self.S_ie, self.S_ii, self.mon)
        net.store('initial')
        
        # Calculate number of feedback iterations
        n_iterations = int(self.duration / self.feedback_interval)
        
        run_time = 0*second
        
        # Main simulation loop with feedback
        for i in range(n_iterations):
            net.restore('initial')
            
            # Select active MSNs based on probability distribution
            active = np.random.rand(self.n_striatum) < self.p_active
            
            # Record if target MSN is active
            target_active = active[self.target]
            self.target_activation_history.append(float(target_active))
            
            # Update thalamic input based on active MSNs
            I_thal_value = self.update_thalamic_input(active.astype(float))
            self.G_e.I_thal = I_thal_value
            self.G_i.I_thal = I_thal_value / 2
            
            # Run simulation for the feedback interval
            net.run(self.feedback_interval)
            run_time += self.feedback_interval
            
            # Get "EEG signal" - filtered average membrane potential
            eeg = np.mean(self.mon.v[:, -int(self.feedback_interval/self.dt):], axis=0)
            
            # Estimate spectrum and alpha band power
            fs = int(1/(self.dt/second))
            f, Pxx = welch(eeg, fs=fs, nperseg=min(1024, len(eeg)))
            
            # Calculate alpha band power
            alpha_mask = np.logical_and(f >= self.alpha_band[0], f <= self.alpha_band[1])
            alpha_power = np.trapz(Pxx[alpha_mask], f[alpha_mask])
            self.alpha_power_history.append(float(alpha_power))
            
            # Compare with threshold
            positive_feedback = alpha_power > self.threshold
            self.feedback_history.append(int(positive_feedback))
            
            # Record current target weight
            self.weight_history.append(float(self.w[self.target]))
            
            # Skip weight update if this is baseline run
            if baseline_run:
                continue
            
            # Update weights based on feedback
            if positive_feedback:
                # Strengthen weights for active MSNs
                self.w[active] += self.learning_rate
            else:
                # Weaken weights for active MSNs
                self.w[active] -= self.learning_rate
            
            # Ensure minimum weight values
            self.w = np.clip(self.w, 1e-4, None)
            
            # Normalize weights to get probabilities
            self.w /= np.sum(self.w)
            self.p_active = self.w.copy()
            
            # Print progress every 10%
            if i % max(1, n_iterations // 10) == 0:
                print(f"Progress: {i/n_iterations*100:.1f}%, "
                      f"Target weight: {self.w[self.target]:.4f}, "
                      f"Alpha power: {alpha_power:.4f}")
        
        print("Simulation completed.")
        
        # Compile results
        results = {
            'parameters': {
                'ne': self.ne,
                'ni': self.ni,
                'n_striatum': self.n_striatum,
                'duration_seconds': float(self.duration/second),
                'dt_ms': float(self.dt/ms),
                'feedback_interval_ms': float(self.feedback_interval/ms),
                'learning_rate': self.learning_rate,
                'alpha_band': self.alpha_band,
                'threshold': self.threshold,
                'seed': self.seed,
                'target_msn': int(self.target)
            },
            'results': {
                'weight_history': self.weight_history,
                'alpha_power_history': self.alpha_power_history,
                'feedback_history': self.feedback_history,
                'target_activation_history': self.target_activation_history,
                'final_target_weight': float(self.w[self.target]),
                'final_alpha_power': float(self.alpha_power_history[-1] if self.alpha_power_history else 0)
            }
        }
        
        return results
    
    def save_results(self, results, filename=None):
        """
        Save simulation results to a JSON file
        
        Parameters:
        -----------
        results : dict
            Simulation results to save
        filename : str, optional
            Filename to save results to, if None a timestamped filename is generated
        
        Returns:
        --------
        str
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"neurofeedback_sim_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return str(filepath)
    
    def plot_results(self, results, show=True, save=True, filename=None):
        """
        Plot simulation results
        
        Parameters:
        -----------
        results : dict
            Simulation results to plot
        show : bool
            Whether to display the plot
        save : bool
            Whether to save the plot
        filename : str, optional
            Filename to save plot to, if None a timestamped filename is generated
        
        Returns:
        --------
        str or None
            Path to saved plot if save=True, None otherwise
        """
        weight_history = results['results']['weight_history']
        alpha_power_history = results['results']['alpha_power_history']
        feedback_history = results['results']['feedback_history']
        target_activation_history = results['results']['target_activation_history']
        
        # Calculate x-axis in seconds
        timestep = results['parameters']['feedback_interval_ms'] / 1000
        x_time = np.arange(len(weight_history)) * timestep
        
        # Create figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        # Plot target MSN weight
        axs[0].plot(x_time, weight_history, 'b-', label='Target MSN Weight')
        axs[0].set_ylabel('Weight')
        axs[0].set_title('Target MSN Weight Over Time')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot alpha power
        axs[1].plot(x_time, alpha_power_history, 'r-', label='Alpha Power')
        axs[1].axhline(y=results['parameters']['threshold'], color='k', linestyle='--', 
                      label='Threshold')
        axs[1].set_ylabel('Power')
        axs[1].set_title('Alpha Band Power Over Time')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot feedback (0 or 1)
        axs[2].plot(x_time, feedback_history, 'g-', label='Feedback')
        axs[2].set_ylabel('Feedback')
        axs[2].set_title('Positive Feedback (1) vs Negative Feedback (0)')
        axs[2].set_ylim(-0.1, 1.1)
        axs[2].legend()
        axs[2].grid(True)
        
        # Plot target MSN activation
        axs[3].plot(x_time, target_activation_history, 'm-', label='Target MSN Active')
        axs[3].set_ylabel('Active')
        axs[3].set_xlabel('Time (seconds)')
        axs[3].set_title('Target MSN Activation')
        axs[3].set_ylim(-0.1, 1.1)
        axs[3].legend()
        axs[3].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"neurofeedback_plot_{timestamp}.png"
            
            filepath = self.results_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
            
        # Show figure if requested
        if show:
            plt.show()
        else:
            plt.close()
            
        return str(filepath) if save else None


def run_parameter_sweep(param_grid, base_params=None, results_dir="../results/param_sweep"):
    """
    Run a parameter sweep over the given parameter grid
    
    Parameters:
    -----------
    param_grid : dict
        Dictionary mapping parameter names to lists of values
    base_params : dict, optional
        Base parameters for the model, if None default values are used
    results_dir : str
        Directory to save results to
    
    Returns:
    --------
    list
        List of result file paths
    """
    # Create results directory
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize base parameters
    if base_params is None:
        base_params = {}
    
    # Get all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Generate all combinations
    import itertools
    param_combinations = list(itertools.product(*param_values))
    
    result_files = []
    
    # Run simulation for each parameter combination
    for i, params in enumerate(param_combinations):
        print(f"\nParameter combination {i+1}/{len(param_combinations)}")
        
        # Create parameter dictionary for this run
        run_params = base_params.copy()
        
        # Add specific parameters for this run
        for name, value in zip(param_names, params):
            run_params[name] = value
            print(f"{name} = {value}")
        
        # Create and run model
        model = NeurofeedbackModel(**run_params, results_dir=results_dir)
        
        # Run baseline first
        baseline_results = model.simulate(baseline_run=True)
        
        # Use last alpha power as threshold for training
        if 'threshold' not in run_params:
            mean_baseline_alpha = np.mean(baseline_results['results']['alpha_power_history'])
            model.threshold = mean_baseline_alpha
            print(f"Setting threshold to mean baseline alpha: {mean_baseline_alpha:.4f}")
        
        # Run neurofeedback training
        results = model.simulate()
        
        # Create parameter string for filename
        param_str = "_".join([f"{name}_{value}" for name, value in zip(param_names, params)])
        filename = f"sweep_{param_str}.json"
        
        # Save results
        result_file = model.save_results(results, filename)
        result_files.append(result_file)
        
        # Generate plot
        plot_filename = f"sweep_{param_str}.png"
        model.plot_results(results, show=False, save=True, filename=plot_filename)
    
    return result_files


def analyze_parameter_sweep(sweep_dir, output_file=None):
    """
    Analyze results from a parameter sweep
    
    Parameters:
    -----------
    sweep_dir : str
        Directory containing parameter sweep results
    output_file : str, optional
        File to save analysis results to
    
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    sweep_dir = Path(sweep_dir)
    
    # Find all JSON result files
    result_files = list(sweep_dir.glob("*.json"))
    
    if not result_files:
        print(f"No result files found in {sweep_dir}")
        return {}
    
    print(f"Found {len(result_files)} result files")
    
    # Collect results
    all_results = []
    
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
    
    # Extract parameters and metrics
    analysis = {
        'parameter_names': [],
        'parameter_values': [],
        'metrics': {
            'final_target_weight': [],
            'final_alpha_power': [],
            'learning_speed': [],  # Rate of increase in target weight
            'success_rate': []     # Percentage of positive feedback
        }
    }
    
    # Find all unique parameter names
    param_names = set()
    for result in all_results:
        param_names.update(result['parameters'].keys())
    
    param_names = sorted(list(param_names))
    analysis['parameter_names'] = param_names
    
    # Extract parameter values and metrics for each result
    for result in all_results:
        # Parameters
        params = [result['parameters'].get(name, None) for name in param_names]
        analysis['parameter_values'].append(params)
        
        # Metrics
        analysis['metrics']['final_target_weight'].append(
            result['results']['final_target_weight'])
        
        analysis['metrics']['final_alpha_power'].append(
            result['results']['final_alpha_power'])
        
        # Calculate learning speed (slope of target weight)
        weights = result['results']['weight_history']
        learning_speed = (weights[-1] - weights[0]) / len(weights) if len(weights) > 1 else 0
        analysis['metrics']['learning_speed'].append(learning_speed)
        
        # Calculate success rate (percentage of positive feedback)
        feedback = result['results']['feedback_history']
        success_rate = sum(feedback) / len(feedback) if feedback else 0
        analysis['metrics']['success_rate'].append(success_rate)
    
    # Find best parameter combinations for each metric
    best_params = {}
    
    for metric_name, metric_values in analysis['metrics'].items():
        # Find index of best value (maximum for all these metrics)
        best_idx = np.argmax(metric_values)
        
        # Get corresponding parameter values
        best_param_values = analysis['parameter_values'][best_idx]
        
        # Create dictionary of parameter name -> value
        best_param_dict = {name: value for name, value in 
                          zip(param_names, best_param_values)}
        
        best_params[metric_name] = {
            'params': best_param_dict,
            'value': metric_values[best_idx]
        }
    
    analysis['best_parameters'] = best_params
    
    # Save analysis if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {output_file}")
    
    return analysis


if __name__ == "__main__":
    print("Neurofeedback Model - Based on Davelaar (2018)")
    print("="*50)
    
    # Example parameters
    params = {
        'ne': 800,               # number of excitatory neurons
        'ni': 200,               # number of inhibitory neurons
        'n_striatum': 1000,      # number of striatal units
        'duration': 5*second,    # simulation duration (shorter for testing)
        'dt': 0.1*ms,            # simulation time step
        'feedback_interval': 100*ms,  # how often to give feedback
        'learning_rate': 0.01,   # learning rate for weights update
        'alpha_band': (8, 12),   # alpha frequency band (Hz)
        'seed': 42,              # random seed for reproducibility
    }
    
    # Create and run model
    model = NeurofeedbackModel(**params)
    
    # Run baseline first (no learning)
    print("Running baseline simulation...")
    baseline_results = model.simulate(baseline_run=True)
    
    # Use mean baseline alpha power as threshold
    mean_baseline_alpha = np.mean(baseline_results['results']['alpha_power_history'])
    model.threshold = mean_baseline_alpha
    print(f"Setting threshold to mean baseline alpha: {mean_baseline_alpha:.4f}")
    
    # Run neurofeedback training
    print("\nRunning neurofeedback training simulation...")
    results = model.simulate()
    
    # Save results
    model.save_results(results)
    
    # Plot results
    model.plot_results(results)
    
    print("\nExample parameter sweep:")
    # Example parameter sweep
    param_grid = {
        'learning_rate': [0.005, 0.01, 0.02],
        'feedback_interval': [50*ms, 100*ms, 200*ms],
    }
    
    # Uncomment to run parameter sweep (takes longer)
    # sweep_params = params.copy()
    # sweep_params['duration'] = 2*second  # Shorter for sweep
    # result_files = run_parameter_sweep(param_grid, base_params=sweep_params)
    # 
    # # Analyze results
    # analysis = analyze_parameter_sweep("../results/param_sweep", 
    #                                   "../results/param_sweep/analysis.json")
    # 
    # # Print best parameters
    # print("\nBest parameters for each metric:")
    # for metric, info in analysis['best_parameters'].items():
    #     print(f"{metric}: {info['value']:.4f}")
    #     for param, value in info['params'].items():
    #         print(f"  {param}: {value}")
