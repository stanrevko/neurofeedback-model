# Makefile for Neurofeedback Model

.PHONY: install test run clean example

# Default Python interpreter
PYTHON = python3

# Install package and dependencies
install:
	$(PYTHON) -m pip install -e .

# Run test
test:
	$(PYTHON) -c "import sys; from src.neurofeedback_model import NeurofeedbackModel; print('Neurofeedback model imported successfully!')"

# Run example simulation
run:
	$(PYTHON) run_neurofeedback.py simulate --duration 3 --run-baseline --auto-threshold

# Run full example set
example:
	$(PYTHON) run_neurofeedback.py simulate --duration 3 --learning-rate 0.01 --run-baseline --auto-threshold --results-dir results/example/single_run
	$(PYTHON) run_neurofeedback.py sweep --sweep-learning-rate 0.001 0.05 2 --sweep-feedback-interval 50 200 2 --duration 2 --results-dir results/example/sweep

# Clean up generated files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Clean all results
clean-results:
	rm -rf results/
