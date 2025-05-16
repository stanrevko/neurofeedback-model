#!/bin/bash

# Neurofeedback Model Example Run Script

# Create directories
mkdir -p results/example

# Example 1: Run a single simulation with baseline and auto-threshold
echo "Example 1: Running single simulation with baseline"
python run_neurofeedback.py simulate --duration 5 --learning-rate 0.01 --run-baseline --auto-threshold --results-dir results/example/single_run

# Example 2: Run a parameter sweep over learning rates and feedback intervals
echo "Example 2: Running parameter sweep"
python run_neurofeedback.py sweep --sweep-learning-rate 0.001 0.05 3 --sweep-feedback-interval 50 300 3 --results-dir results/example/sweep

# Example 3: Run parameter optimization
echo "Example 3: Running parameter optimization"
python run_neurofeedback.py optimize --opt-learning-rate 0.001 0.05 --opt-feedback-interval 50 300 --opt-threshold 0.8 1.2 --grid-search --grid-steps 3 --train-model --visualize --duration 3 --results-dir results/example/optimize

echo "Examples completed. Results are stored in results/example directory."

# Make the script executable
chmod +x run_example.sh
