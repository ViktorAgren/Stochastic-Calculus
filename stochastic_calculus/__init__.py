"""Stochastic Calculus Library

A comprehensive library for stochastic process simulation, option pricing,
and financial mathematics.

Main modules:
- processes: Stochastic process implementations (Brownian, OU, CIR, Heston, etc.)
- pricing: Option pricing models (Black-Scholes, Monte Carlo, PDE solvers)
- visualization: Plotting and analysis tools
- calibration: Parameter estimation methods
"""

# Core exports
from .core.protocols import Drift, Sigma, InitialValue

# Process exports
from .processes.brownian import BrownianMotion, GeometricBrownianMotion
from .processes.mean_reverting import OrnsteinUhlenbeckProcess, CIRProcess
from .processes.stochastic_vol import HestonProcess

# Visualization exports
from .visualization import ProcessPlotter, StatisticalPlotter

__version__ = "0.1.0"
__author__ = "Stochastic Calculus Project"

__all__ = [
    # Core
    "Drift",
    "Sigma",
    "InitialValue",
    # Processes
    "BrownianMotion",
    "GeometricBrownianMotion",
    "OrnsteinUhlenbeckProcess",
    "CIRProcess",
    "HestonProcess",
    # Visualization
    "ProcessPlotter",
    "StatisticalPlotter",
]
