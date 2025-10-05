"""Brownian motion specific utility functions."""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


def validate_correlation(rho: float) -> None:
    """Validate correlation parameter is in [-1, 1]."""
    if not -1 <= rho <= 1:
        raise ValueError(f"Correlation must be in [-1, 1], got {rho}")


def generate_correlated_brownian(
    n_steps: int,
    n_processes: int,
    correlation: Optional[float] = None,
    dt: float = 1.0,
    random_state: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Generate correlated Brownian motion increments.

    Args:
        n_steps: Number of time steps
        n_processes: Number of parallel processes
        correlation: Correlation coefficient between processes
        dt: Time increment
        random_state: Random seed

    Returns:
        Array of shape (n_steps, n_processes) with Brownian increments
    """
    rng = np.random.default_rng(random_state)

    if correlation is None or n_processes == 1:
        return rng.normal(0, np.sqrt(dt), (n_steps, n_processes))

    validate_correlation(correlation)

    # Generate independent Brownian motions
    dW = rng.normal(0, np.sqrt(dt), (n_steps, n_processes))

    if n_processes == 2:
        # Simple case for two processes
        dW[:, 1] = correlation * dW[:, 0] + np.sqrt(1 - correlation**2) * dW[:, 1]
    else:
        # For multiple processes, use Cholesky decomposition approach
        corr_matrix = np.full((n_processes, n_processes), correlation)
        np.fill_diagonal(corr_matrix, 1.0)

        try:
            L = np.linalg.cholesky(corr_matrix)
            dW = dW @ L.T
        except np.linalg.LinAlgError:
            # Fallback: make each process correlated with the first one
            for i in range(1, n_processes):
                independent = rng.normal(0, np.sqrt(dt), n_steps)
                dW[:, i] = (
                    correlation * dW[:, 0] + np.sqrt(1 - correlation**2) * independent
                )

    return dW


def estimate_correlation_from_increments(increments: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Estimate correlation matrix from process increments.

    Args:
        increments: Array of shape (n_steps, n_processes)

    Returns:
        Correlation matrix
    """
    return np.corrcoef(increments.T)