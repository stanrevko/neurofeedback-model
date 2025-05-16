#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter optimization for neurofeedback model

This script implements a parameter optimization framework to find the most
effective neurofeedback training parameters based on the model from Davelaar (2018).
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution
import itertools
import multiprocessing
import time
import sys
import argparse
from datetime import datetime

# Import our neurofeedback model
from neurofeedback_model import NeurofeedbackModel, run_parameter_sweep, analyze_parameter_sweep

class ParameterOptimizer:
    """
    Parameter optimizer for neurofeedback model using machine learning
    and evolutionary algorithms to find optimal training parameters.
    """
    
    def __init__(self, 
                 base_params=None,
                 param_ranges=None,
                 results_dir="../results/optimization",
                 n_jobs=-1):
        """
        Initialize the parameter optimizer
        
        Parameters:
        -----------
        base_params : dict, optional
            Base parameters for the model, if None default values are used
        param_ranges : dict, optional
            Dictionary mapping parameter names to (min, max) tuples for optimization
        results_dir : str
            Directory to save results to
        n_jobs : int
            Number of parallel jobs to run, -1 uses all available cores
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base parameters
        if base_params is None:
            # Extract constants from neurofeedback_model
            from brian2 import second, ms
            
            self.base_params = {
                'ne': 800,               # number of excitatory neurons
                'ni': 200,               # number of inhibitory neurons
                'n_striatum': 1000,      # number of striatal units
                'duration': 5*second,    # simulation duration 
                'dt': 0.1*ms,            # simulation time step
                'feedback_interval': 100*ms,  # how often to give feedback
                'learning_rate': 0.01,   # learning rate for weights update
                'alpha_band': (8, 12),   # alpha frequency band (Hz)
            }
        else:
            self.base_params = base_params
        
        # Parameter ranges for optimization
        if param_ranges is None:
            self.param_ranges = {
                'learning_rate': (0.001, 0.05),
                'feedback_interval': (50e-3, 500e-3),  # in seconds
                'threshold': (0.8, 1.2),  # relative to baseline
            }
        else:
            self.param_ranges = param_ranges
            
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
        # Storage for collected data
        self.collected_data = []
        self.ml_model = None
    
    def run_grid_search(self, grid_steps=3, random_subset=None, seed=None):
        """
        Run a grid search over parameter space
        
        Parameters:
        -----------
        grid_steps : int
            Number of steps for each parameter in the grid
        random_subset : int, optional
            If provided, randomly select this many parameter combinations
        seed : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        list
            List of result dictionaries
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create grid for each parameter
        param_grid = {}
        for param_name, (param_min, param_max) in self.param_ranges.items():
            param_grid[param_name] = np.linspace(param_min, param_max, grid_steps)
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Take random subset if requested
        if random_subset is not None and random_subset < len(all_combinations):
            indices = np.random.choice(len(all_combinations), random_subset, replace=False)
            selected_combinations = [all_combinations[i] for i in indices]
        else:
            selected_combinations = all_combinations
        
        print(f"Running grid search with {len(selected_combinations)} parameter combinations")
        
        # Create parameter grid for sweep
        sweep_grid = {name: [comb[i] for comb in selected_combinations] 
                     for i, name in enumerate(param_names)}
        
        # Run parameter sweep
        result_files = run_parameter_sweep(
            sweep_grid, 
            base_params=self.base_params,
            results_dir=str(self.results_dir / "grid_search")
        )
        
        # Load and return results
        results = []
        for file in result_files:
            with open(file, 'r') as f:
                results.append(json.load(f))
        
        # Add to collected data
        self.collected_data.extend(results)
        
        return results
    
    def train_surrogate_model(self, test_size=0.2, random_state=42):
        """
        Train a machine learning model to predict performance from parameters
        
        Parameters:
        -----------
        test_size : float
            Fraction of data to use for testing
        random_state : int
            Random seed for reproducibility
        
        Returns:
        --------
        tuple
            (model, X_train, X_test, y_train, y_test)
        """
        if not self.collected_data:
            print("No data collected yet. Run grid_search first.")
            return None
        
        # Extract features (parameters) and targets (metrics)
        X = []
        y = []
        
        param_names = sorted(self.param_ranges.keys())
        
        for result in self.collected_data:
            # Extract parameters
            params = [result['parameters'].get(name, None) for name in param_names]
            
            # Some parameter values might be in seconds, convert to proper value
            params = [float(p) if isinstance(p, (int, float)) else p for p in params]
            
            # Extract target metric (we'll use final_target_weight as our objective)
            target = result['results']['final_target_weight']
            
            X.append(params)
            y.append(target)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train a Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=self.n_jobs
        )
        model.fit(X_train, y_train)
        
        # Save the model
        self.ml_model = model
        self.param_names = param_names
        
        # Calculate and print model performance
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Random Forest R² score on training data: {train_score:.4f}")
        print(f"Random Forest R² score on test data: {test_score:.4f}")
        
        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Parameter importance:")
        for i in range(len(param_names)):
            print(f"{param_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return model, X_train, X_test, y_train, y_test
    
    def visualize_surrogate_model(self, resolution=20):
        """
        Visualize the surrogate model predictions across the parameter space
        
        Parameters:
        -----------
        resolution : int
            Resolution of the visualization grid
        
        Returns:
        --------
        dict
            Dictionary with visualization data
        """
        if self.ml_model is None:
            print("No surrogate model trained yet. Run train_surrogate_model first.")
            return None
        
        # Create parameter grid for visualization
        param_names = self.param_names
        param_grid = {}
        
        for param_name in param_names:
            param_min, param_max = self.param_ranges[param_name]
            param_grid[param_name] = np.linspace(param_min, param_max, resolution)
        
        # For 2D visualization, use the two most important parameters
        importances = self.ml_model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:2]
        
        top_params = [param_names[i] for i in top_indices]
        
        print(f"Visualizing the most important parameters: {top_params}")
        
        # Create 2D mesh grid for visualization
        xx, yy = np.meshgrid(
            param_grid[top_params[0]], 
            param_grid[top_params[1]]
        )
        
        # Create prediction inputs
        X_viz = []
        
        for i in range(resolution):
            for j in range(resolution):
                params = []
                for name in param_names:
                    if name == top_params[0]:
                        params.append(xx[i, j])
                    elif name == top_params[1]:
                        params.append(yy[i, j])
                    else:
                        # Use the middle value for other parameters
                        param_min, param_max = self.param_ranges[name]
                        params.append((param_min + param_max) / 2)
                X_viz.append(params)
        
        X_viz = np.array(X_viz)
        
        # Predict performance
        y_pred = self.ml_model.predict(X_viz)
        
        # Reshape for visualization
        zz = y_pred.reshape(resolution, resolution)
        
        # Create visualization plot
        plt.figure(figsize=(10, 8))
        
        # Contour plot
        contour = plt.contourf(xx, yy, zz, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Predicted Target Weight')
        
        # Mark the best predicted point
        best_idx = np.argmax(y_pred)
        best_x = X_viz[best_idx, top_indices[0]]
        best_y = X_viz[best_idx, top_indices[1]]
        plt.scatter(best_x, best_y, color='red', marker='x', s=100, label='Best predicted')
        
        # Mark the actual best point from data
        actual_best_idx = np.argmax([r['results']['final_target_weight'] for r in self.collected_data])
        actual_best = self.collected_data[actual_best_idx]
        actual_best_x = actual_best['parameters'].get(top_params[0], None)
        actual_best_y = actual_best['parameters'].get(top_params[1], None)
        
        plt.scatter(actual_best_x, actual_best_y, color='white', marker='o', s=100, 
                   edgecolor='black', label='Best observed')
        
        plt.xlabel(top_params[0])
        plt.ylabel(top_params[1])
        plt.title('Surrogate Model Predictions')
        plt.legend()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"surrogate_model_viz_{timestamp}.png"
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        print(f"Visualization saved to {filepath}")
        
        # Also create 1D plots showing the effect of each parameter
        fig, axs = plt.subplots(len(param_names), 1, figsize=(10, 3*len(param_names)))
        
        for i, param_name in enumerate(param_names):
            param_values = param_grid[param_name]
            predictions = []
            
            for val in param_values:
                X_pred = []
                for name in param_names:
                    if name == param_name:
                        X_pred.append(val)
                    else:
                        # Use the middle value for other parameters
                        param_min, param_max = self.param_ranges[name]
                        X_pred.append((param_min + param_max) / 2)
                
                # Predict and store
                pred = self.ml_model.predict([X_pred])[0]
                predictions.append(pred)
            
            # Plot
            axs[i].plot(param_values, predictions)
            axs[i].set_xlabel(param_name)
            axs[i].set_ylabel('Predicted Target Weight')
            axs[i].set_title(f'Effect of {param_name}')
            axs[i].grid(True)
        
        plt.tight_layout()
        
        # Save 1D plots
        filename = f"param_effects_{timestamp}.png"
        filepath = self.results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        print(f"Parameter effects plot saved to {filepath}")
        
        return {
            'best_predicted': {
                'params': {param_names[i]: X_viz[best_idx, i] for i in range(len(param_names))},
                'value': y_pred[best_idx]
            },
            'best_observed': {
                'params': {k: v for k, v in actual_best['parameters'].items() if k in param_names},
                'value': actual_best['results']['final_target_weight']
            }
        }
    
    def optimize_parameters(self, n_trials=10, max_iter=100, popsize=15):
        """
        Use evolutionary optimization to find optimal parameters
        
        Parameters:
        -----------
        n_trials : int
            Number of trials to run
        max_iter : int
            Maximum number of iterations per trial
        popsize : int
            Population size for differential evolution
        
        Returns:
        --------
        dict
            Dictionary with optimization results
        """
        if self.ml_model is None:
            print("No surrogate model trained yet. Run train_surrogate_model first.")
            return None
        
        # Define the bounds for differential evolution
        bounds = [self.param_ranges[name] for name in self.param_names]
        
        # Define objective function (negative because we maximize)
        def objective(x):
            return -self.ml_model.predict([x])[0]
        
        # Run multiple trials
        results = []
        
        for trial in range(n_trials):
            print(f"Starting optimization trial {trial+1}/{n_trials}")
            
            # Run differential evolution
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=max_iter,
                popsize=popsize,
                tol=1e-4,
                disp=True
            )
            
            # Store result
            params = {name: result.x[i] for i, name in enumerate(self.param_names)}
            predicted_value = -result.fun
            
            print(f"Trial {trial+1} completed:")
            print(f"Predicted target weight: {predicted_value:.4f}")
            for name, value in params.items():
                print(f"{name}: {value:.6f}")
            
            results.append({
                'trial': trial,
                'success': result.success,
                'params': params,
                'predicted_value': predicted_value
            })
        
        # Find best result
        best_trial = max(results, key=lambda x: x['predicted_value'])
        
        print("\nBest optimization result:")
        print(f"Predicted target weight: {best_trial['predicted_value']:.4f}")
        for name, value in best_trial['params'].items():
            print(f"{name}: {value:.6f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                'trials': results,
                'best_trial': best_trial
            }, f, indent=2)
        
        print(f"Optimization results saved to {filepath}")
        
        return best_trial
    
    def validate_optimal_parameters(self, params, n_runs=5, seed=None):
        """
        Validate the predicted optimal parameters with actual simulations
        
        Parameters:
        -----------
        params : dict
            Dictionary with parameter values to validate
        n_runs : int
            Number of validation runs to perform
        seed : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Dictionary with validation results
        """
        # Convert to proper parameters for the model
        from brian2 import second, ms
        model_params = self.base_params.copy()
        
        for name, value in params.items():
            if name == 'feedback_interval':
                model_params[name] = value * second
            else:
                model_params[name] = value
        
        print(f"Validating optimal parameters with {n_runs} runs")
        
        # Run multiple validation simulations
        results = []
        
        for i in range(n_runs):
            # Set different seed for each run
            run_seed = None if seed is None else seed + i
            model_params['seed'] = run_seed
            
            print(f"Validation run {i+1}/{n_runs}")
            
            # Create and run model
            model = NeurofeedbackModel(
                **model_params, 
                results_dir=str(self.results_dir / "validation")
            )
            
            # Run simulation
            result = model.simulate()
            
            # Calculate some metrics
            final_target_weight = result['results']['final_target_weight']
            final_alpha_power = result['results']['final_alpha_power']
            positive_feedback = sum(result['results']['feedback_history']) / len(result['results']['feedback_history'])
            
            print(f"  Final target weight: {final_target_weight:.4f}")
            print(f"  Final alpha power: {final_alpha_power:.4f}")
            print(f"  Positive feedback rate: {positive_feedback:.2f}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_run_{i+1}_{timestamp}.json"
            model.save_results(result, filename)
            
            # Plot results
            plot_filename = f"validation_run_{i+1}_{timestamp}.png"
            model.plot_results(result, show=False, save=True, filename=plot_filename)
            
            # Store metrics
            results.append({
                'run': i+1,
                'final_target_weight': final_target_weight,
                'final_alpha_power': final_alpha_power,
                'positive_feedback_rate': positive_feedback
            })
        
        # Calculate statistics
        metrics = {
            'final_target_weight': [r['final_target_weight'] for r in results],
            'final_alpha_power': [r['final_alpha_power'] for r in results],
            'positive_feedback_rate': [r['positive_feedback_rate'] for r in results]
        }
        
        statistics = {}
        for metric_name, values in metrics.items():
            statistics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Print statistics
        print("\nValidation statistics:")
        for metric_name, stats in statistics.items():
            print(f"{metric_name}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
        
        # Save validation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                'params': params,
                'runs': results,
                'statistics': statistics
            }, f, indent=2)
        
        print(f"Validation results saved to {filepath}")
        
        return {
            'params': params,
            'results': results,
            'statistics': statistics
        }


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Neurofeedback Model Parameter Optimization')
    
    parser.add_argument('--grid-search', action='store_true',
                        help='Run grid search')
    parser.add_argument('--grid-steps', type=int, default=3,
                        help='Number of steps for each parameter in grid search')
    parser.add_argument('--random-subset', type=int, default=None,
                        help='Randomly select this many parameter combinations')
    
    parser.add_argument('--train-model', action='store_true',
                        help='Train surrogate model')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize surrogate model')
    
    parser.add_argument('--optimize', action='store_true',
                        help='Run optimization')
    parser.add_argument('--n-trials', type=int, default=5,
                        help='Number of optimization trials')
    
    parser.add_argument('--validate', action='store_true',
                        help='Validate optimal parameters')
    parser.add_argument('--n-runs', type=int, default=3,
                        help='Number of validation runs')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ParameterOptimizer(
        param_ranges={
            'learning_rate': (0.001, 0.05),
            'feedback_interval': (50e-3, 300e-3),  # in seconds
            'threshold': (0.8, 1.2),  # relative to baseline
        }
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


if __name__ == "__main__":
    main()
