"""Core utility functions for stochastic processes."""

import numpy as np
from numpy.typing import NDArray


def validate_positive(value: float, name: str) -> None:
    """Validate parameter is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_probability(p: float, name: str) -> None:
    """Validate parameter is a valid probability in [0, 1]."""
    if not 0 <= p <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {p}")


def calculate_realized_volatility(
    prices: NDArray[np.float64], annualization_factor: float = 252
) -> NDArray[np.float64]:
    """
    Calculate realized volatility from price series.

    Args:
        prices: Price series
        annualization_factor: Factor to annualize volatility (252 for daily data)

    Returns:
        Realized volatility
    """
    log_returns = np.diff(np.log(prices), axis=0)
    return np.sqrt(annualization_factor) * np.std(log_returns, axis=0)


def apply_absorption_boundary(
    values: NDArray[np.float64], lower_bound: float = 0.0
) -> NDArray[np.float64]:
    """Apply absorption boundary condition (e.g., for CIR/Heston volatility)."""
    return np.maximum(values, lower_bound)
