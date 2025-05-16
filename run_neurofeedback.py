#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neurofeedback Model - Based on Davelaar (2018)

This script serves as a launcher for the neurofeedback model and parameter optimization.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Add the src directory to the path
src_dir = Path(__file__).parent / "src"
sys.path.append(str(src_dir))

# Import our modules
from neurofeedback_model import NeurofeedbackModel, run_parameter_sweep, analyze_parameter_sweep
from parameter_optimization import ParameterOptimizer

def run_single_simulation(args):
    """Run a single neurofeedback simulation"""
    from brian2 import second, ms
    
    print("Running single neurofeedback simulation")
    
    # Setup parameters
    params = {
        'ne': args.ne,
        'ni': args.ni,
        'n_striatum': args.n_striatum,
        'duration': args.duration * second,
        'dt': args.dt * ms,
        'feedback_interval': args.feedback_interval * ms,
        'learning_rate': args.learning_rate,
        'alpha_band': (args.alpha_band_min, args.alpha_band_max),
        'threshold': args.threshold,
        'seed': args.seed,
        'results_dir': args.results_dir
    }
    
    # Create and run model
    model = NeurofeedbackModel(**params)
    
    # Run baseline first if requested
    if args.run_baseline:
        print("Running baseline simulation...")
        baseline_results = model.simulate(baseline_run=True)
        
        if args.auto_threshold:
            # Use mean baseline alpha power as threshold
            mean_baseline_alpha = sum(baseline_results['results']['alpha_power_history']) / len(baseline_results['results']['alpha_power_history'])
            model.threshold = mean_baseline_alpha
            print(f"Setting threshold to mean baseline alpha: {mean_baseline_alpha:.4f}")
    
    # Run neurofeedback training
    print("\nRunning neurofeedback training simulation...")
    results = model.simulate()
    
    # Save results
    model.save_results(results)
    
    # Plot results
    model.plot_results(results, show=not args.no_plot)
    
    return results

def run_parameter_sweep_with_args(args):
    """Run a parameter sweep with the given arguments"""
    from brian2 import second, ms
    
    print("Running parameter sweep")
    
    # Setup base parameters
    base_params = {
        'ne': args.ne,
        'ni': args.ni,
        'n_striatum': args.n_striatum,
        'duration': args.duration * second,
        'dt': args.dt * ms,
        'results_dir': args.results_dir
    }
    
    # Setup parameter grid
    param_grid = {}
    
    if args.sweep_learning_rate:
        min_val, max_val, steps = args.sweep_learning_rate
        param_grid['learning_rate'] = np.linspace(min_val, max_val, steps)
    
    if args.sweep_feedback_interval:
        min_val, max_val, steps = args.sweep_feedback_interval
        # Convert to seconds for the model
        param_grid['feedback_interval'] = [val * ms for val in np.linspace(min_val, max_val, steps)]
    
    if args.sweep_threshold:
        min_val, max_val, steps = args.sweep_threshold
        param_grid['threshold'] = np.linspace(min_val, max_val, steps)
    
    # Run parameter sweep
    result_files = run_parameter_sweep(param_grid, base_params=base_params, 
                                     results_dir=args.results_dir)
    
    # Analyze results
    analysis_file = os.path.join(args.results_dir, "sweep_analysis.json")
    analysis = analyze_parameter_sweep(args.results_dir, analysis_file)
    
    return analysis

def run_optimization(args):
    """Run parameter optimization with the given arguments"""
    from brian2 import second, ms
    
    print("Running parameter optimization")
    
    # Setup base parameters
    base_params = {
        'ne': args.ne,
        'ni': args.ni,
        'n_striatum': args.n_striatum,
        'duration': args.duration * second,
        'dt': args.dt * ms,
        'results_dir': args.results_dir
    }
    
    # Setup parameter ranges
    param_ranges = {}
    
    if args.opt_learning_rate:
        min_val, max_val = args.opt_learning_rate
        param_ranges['learning_rate'] = (min_val, max_val)
    
    if args.opt_feedback_interval:
        min_val, max_val = args.opt_feedback_interval
        # Convert to seconds for the optimizer
        param_ranges['feedback_interval'] = (min_val / 1000, max_val / 1000)
    
    if args.opt_threshold:
        min_val, max_val = args.opt_threshold
        param_ranges['threshold'] = (min_val, max_val)
    
    # Create optimizer
    optimizer = ParameterOptimizer(
        base_params=base_params,
        param_ranges=param_ranges,
        results_dir=args.results_dir,
        n_jobs=args.n_jobs
    )
    
    # Run grid search if requested
    if args.grid_search:
        optimizer.run_grid_search(
            grid_steps=args.grid_steps,
            random_subset=args.random_subset,
            seed=args.seed
        )
    
    # Train model if requested
    if args.train_model:
        optimizer.train_surrogate_model()
    
    # Visualize if requested
    if args.visualize:
        optimizer.visualize_surrogate_model()
    
    # Optimize if requested
    best_params = None
    if args.optimize:
        best_trial = optimizer.optimize_parameters(n_trials=args.n_trials)
        best_params = best_trial['params']
    
    # Validate if requested
    if args.validate:
        if best_params is None:
            print("No optimal parameters found. Run optimization first.")
        else:
            optimizer.validate_optimal_parameters(best_params, n_runs=args.n_runs, seed=args.seed)
    
    return optimizer

def main():
    """Main function parsing command line arguments"""
    parser = argparse.ArgumentParser(description='Neurofeedback Model - Based on Davelaar (2018)')
    
    # Common parameters
    parser.add_argument('--ne', type=int, default=800,
                        help='Number of excitatory neurons')
    parser.add_argument('--ni', type=int, default=200,
                        help='Number of inhibitory neurons')
    parser.add_argument('--n-striatum', type=int, default=1000,
                        help='Number of striatal units')
    parser.add_argument('--duration', type=float, default=10,
                        help='Simulation duration in seconds')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Simulation time step in milliseconds')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save results to')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Single simulation mode
    sim_parser = subparsers.add_parser('simulate', help='Run a single neurofeedback simulation')
    sim_parser.add_argument('--feedback-interval', type=float, default=100,
                          help='Feedback interval in milliseconds')
    sim_parser.add_argument('--learning-rate', type=float, default=0.01,
                          help='Learning rate for weight updates')
    sim_parser.add_argument('--alpha-band-min', type=float, default=8,
                          help='Minimum frequency of alpha band in Hz')
    sim_parser.add_argument('--alpha-band-max', type=float, default=12,
                          help='Maximum frequency of alpha band in Hz')
    sim_parser.add_argument('--threshold', type=float, default=1.0,
                          help='Threshold for positive feedback')
    sim_parser.add_argument('--run-baseline', action='store_true',
                          help='Run baseline simulation first')
    sim_parser.add_argument('--auto-threshold', action='store_true',
                          help='Automatically set threshold based on baseline')
    sim_parser.add_argument('--no-plot', action='store_true',
                          help='Do not show plots')
    
    # Parameter sweep mode
    sweep_parser = subparsers.add_parser('sweep', help='Run parameter sweep')
    sweep_parser.add_argument('--sweep-learning-rate', type=float, nargs=3, metavar=('MIN', 'MAX', 'STEPS'),
                            help='Sweep learning rate from MIN to MAX with STEPS steps')
    sweep_parser.add_argument('--sweep-feedback-interval', type=float, nargs=3, metavar=('MIN', 'MAX', 'STEPS'),
                            help='Sweep feedback interval from MIN to MAX with STEPS steps (in ms)')
    sweep_parser.add_argument('--sweep-threshold', type=float, nargs=3, metavar=('MIN', 'MAX', 'STEPS'),
                            help='Sweep threshold from MIN to MAX with STEPS steps')
    
    # Optimization mode
    opt_parser = subparsers.add_parser('optimize', help='Run parameter optimization')
    opt_parser.add_argument('--opt-learning-rate', type=float, nargs=2, metavar=('MIN', 'MAX'),
                          help='Optimize learning rate between MIN and MAX')
    opt_parser.add_argument('--opt-feedback-interval', type=float, nargs=2, metavar=('MIN', 'MAX'),
                          help='Optimize feedback interval between MIN and MAX (in ms)')
    opt_parser.add_argument('--opt-threshold', type=float, nargs=2, metavar=('MIN', 'MAX'),
                          help='Optimize threshold between MIN and MAX')
    opt_parser.add_argument('--grid-search', action='store_true',
                          help='Run grid search')
    opt_parser.add_argument('--grid-steps', type=int, default=3,
                          help='Number of steps for each parameter in grid search')
    opt_parser.add_argument('--random-subset', type=int, default=None,
                          help='Randomly select this many parameter combinations')
    opt_parser.add_argument('--train-model', action='store_true',
                          help='Train surrogate model')
    opt_parser.add_argument('--visualize', action='store_true',
                          help='Visualize surrogate model')
    opt_parser.add_argument('--optimize', action='store_true',
                          help='Run evolutionary optimization')
    opt_parser.add_argument('--n-trials', type=int, default=5,
                          help='Number of optimization trials')
    opt_parser.add_argument('--validate', action='store_true',
                          help='Validate optimal parameters')
    opt_parser.add_argument('--n-runs', type=int, default=3,
                          help='Number of validation runs')
    opt_parser.add_argument('--n-jobs', type=int, default=-1,
                          help='Number of parallel jobs (-1 for all cores)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Run appropriate mode
    if args.mode == 'simulate':
        run_single_simulation(args)
    elif args.mode == 'sweep':
        run_parameter_sweep_with_args(args)
    elif args.mode == 'optimize':
        run_optimization(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
