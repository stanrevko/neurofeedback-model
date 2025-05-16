# Neurofeedback Computational Model

This project implements a computational model of neurofeedback based on the work of Davelaar (2018). The model simulates the process of EEG neurofeedback training, specifically focusing on alpha frequency upregulation, and provides tools for parameter optimization to improve training efficiency.

## Overview

Neurofeedback is a form of biofeedback that uses real-time monitoring of brain activity to teach self-regulation of brain function. This model implements the computational theory proposed by Davelaar (2018) that advocates a critical role of the striatum in modulating EEG frequencies during neurofeedback training.

The model has two main components:
1. A neural network simulation using spiking neurons (Izhikevich model) that generates EEG-like signals
2. A reinforcement learning mechanism that modifies striatal weights based on feedback

## Installation

### Requirements

- Python 3.8 or higher
- The following packages:
  - Brian2 (for neural simulation)
  - NumPy
  - SciPy
  - Matplotlib
  - Pandas
  - Seaborn
  - Scikit-learn

### Setup

1. Clone this repository:
```
git clone https://github.com/your-username/neurofeedback-model.git
cd neurofeedback-model
```

2. Install the package and dependencies:
```
pip install -e .
```

## Usage

The model can be used in three main modes:

### 1. Single Simulation

Run a single neurofeedback simulation with specified parameters:

```
python run_neurofeedback.py simulate --duration 10 --learning-rate 0.01 --run-baseline --auto-threshold
```

This will run a simulation for 10 seconds, using a learning rate of 0.01, with baseline measurement and automatic threshold setting.

### 2. Parameter Sweep

Run a parameter sweep to compare different parameter combinations:

```
python run_neurofeedback.py sweep --sweep-learning-rate 0.001 0.05 5 --sweep-feedback-interval 50 300 5
```

This will run simulations with learning rates from 0.001 to 0.05 (5 values) and feedback intervals from 50ms to 300ms (5 values).

### 3. Parameter Optimization

Optimize parameters using surrogate modeling and evolutionary algorithms:

```
python run_neurofeedback.py optimize --opt-learning-rate 0.001 0.05 --opt-feedback-interval 50 300 --grid-search --train-model --visualize --optimize --validate
```

This will:
1. Define parameter ranges for learning rate and feedback interval
2. Run grid search over these ranges
3. Train a surrogate model on the results
4. Visualize the surrogate model's predictions
5. Run optimization to find the best parameters
6. Validate the optimal parameters with additional simulations

## Model Parameters

- `ne`: Number of excitatory neurons (default: 800)
- `ni`: Number of inhibitory neurons (default: 200)
- `n_striatum`: Number of striatal units (default: 1000)
- `duration`: Simulation duration in seconds (default: 10)
- `dt`: Simulation time step in milliseconds (default: 0.1)
- `feedback_interval`: How often to give feedback in milliseconds (default: 100)
- `learning_rate`: Learning rate for weight updates (default: 0.01)
- `alpha_band`: Alpha frequency band in Hz (default: (8, 12))
- `threshold`: Threshold for positive feedback (default: 1.0)
- `seed`: Random seed for reproducibility (default: None)

## Project Structure

```
neurofeedback-model/
│
├── src/
│   ├── neurofeedback_model.py   # Main model implementation
│   └── parameter_optimization.py # Parameter optimization tools
│
├── results/                    # Directory for results
│   ├── grid_search/            # Results from parameter sweeps
│   ├── optimization/           # Results from parameter optimization
│   └── validation/             # Results from validation runs
│
├── notebooks/                  # Jupyter notebooks for analysis
│
├── run_neurofeedback.py        # Command-line interface
├── setup.py                    # Setup script for installation
└── README.md                   # This readme file
```

## Theoretical Background

The model is based on Davelaar's multi-stage theory of neurofeedback learning:

1. **Striatal-exploration stage**: The frontal or executive system generates representations that activate the striatum, which in turn modulates the thalamus and thereby the EEG signal. Learning occurs by modifying the synaptic connections between frontal and striatal representations.

2. **Thalamic-consolidation stage**: The synaptic modifications occur between the striatum and the thalamic nuclei, leading to changes that can be observed in baseline recordings.

3. **Interoceptive-homeostasis stage**: A subjective internal representation of the target brain state becomes associated with the reward and serves as a secondary reinforcer.

This implementation focuses primarily on the first stage, the striatal-exploration phase, demonstrating how the brain finds the right striatal representation to produce the desired EEG pattern.

## Reference

Davelaar, E. J. (2018). Mechanisms of neurofeedback: A computation-theoretic approach. Neuroscience, 378, 175-188.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
