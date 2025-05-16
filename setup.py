#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="neurofeedback_model",
    version="0.1.0",
    description="Computational model of neurofeedback based on Davelaar (2018)",
    author="Neurofeedback Researcher",
    author_email="researcher@example.com",
    url="https://github.com/your-username/neurofeedback-model",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={},
    scripts=["run_neurofeedback.py"],
    install_requires=[
        "brian2>=2.5.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Neuroscience",
    ],
)
