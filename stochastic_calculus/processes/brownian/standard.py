"""Standard Brownian motion implementation."""

from typing import Optional, NamedTuple, Any
import numpy as np

from .utils import generate_correlated_brownian, validate_correlation
from ...core.protocols import StochasticProcess


class BrownianPaths(NamedTuple):
    """Container for Brownian motion simulation results."""

    increments: np.ndarray  # dW increments
    paths: np.ndarray  # Cumulative paths W(t)


class BrownianMotion(StochasticProcess):
    """
    Standard Brownian motion simulator.

    Simulates one or more correlated Brownian motion processes.
    """

    def __init__(
        self, 
        n_processes: int = 1, 
        correlation: Optional[float] = None,
        W_0: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize Brownian motion simulator.

        Args:
            n_processes: Number of parallel Brownian motions
            correlation: Correlation between processes (if n_processes > 1)
            W_0: Initial positions for Brownian motions (default: 0.0 for all)
        """
        self.n_processes = n_processes
        self.correlation = correlation
        
        # Set initial values
        if W_0 is None:
            self.W_0 = np.zeros(n_processes)
        else:
            self.W_0 = np.asarray(W_0)
            if self.W_0.shape != (n_processes,):
                raise ValueError(f"W_0 must have shape ({n_processes},), got {self.W_0.shape}")

        if correlation is not None:
            validate_correlation(correlation)

    def simulate(
        self, n_steps: int, dt: float = 1.0, random_state: Optional[int] = None
    ) -> BrownianPaths:
        """
        Simulate Brownian motion paths.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed for reproducibility

        Returns:
            BrownianPaths with increments and cumulative paths
        """
        # Generate correlated increments
        increments = generate_correlated_brownian(
            n_steps, self.n_processes, self.correlation, dt, random_state
        )

        # Calculate cumulative paths starting from initial values
        paths = np.zeros((n_steps + 1, self.n_processes))
        paths[0] = self.W_0  # Start from initial positions
        paths[1:] = self.W_0 + np.cumsum(increments, axis=0)

        return BrownianPaths(increments, paths)

    def get_initial_value_names(self) -> list[str]:
        """Return names of initial value parameters for Brownian motion."""
        return ["W_0"]

    def set_initial_values(self, **kwargs) -> None:
        """Set initial values on the Brownian motion process."""
        if "W_0" in kwargs:
            W_0_value = kwargs["W_0"]
            if np.isscalar(W_0_value):
                self.W_0 = np.full(self.n_processes, W_0_value)
            else:
                self.W_0 = np.asarray(W_0_value)
                if self.W_0.shape != (self.n_processes,):
                    raise ValueError(f"W_0 must have shape ({self.n_processes},), got {self.W_0.shape}")

    def get_initial_values(self) -> dict[str, Any]:
        """Get current initial values."""
        return {"W_0": self.W_0.copy()}

    def get_parameters(self) -> dict[str, Any]:
        """Get process parameters."""
        return {
            "process_type": "BrownianMotion",
            "n_processes": self.n_processes,
            "correlation": self.correlation,
        }


def simulate_single_brownian(
    n_steps: int, dt: float = 1.0, random_state: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for single Brownian motion.

    Returns:
        Tuple of (increments, cumulative_path)
    """
    bm = BrownianMotion(n_processes=1)
    result = bm.simulate(n_steps, dt, random_state)
    return result.increments.flatten(), result.paths.flatten()


def estimate_correlation_matrix(brownian_paths: np.ndarray) -> np.ndarray:
    """
    Estimate correlation matrix from Brownian motion increments.

    Args:
        brownian_paths: Array of shape (n_steps+1, n_processes)

    Returns:
        Correlation matrix
    """
    increments = np.diff(brownian_paths, axis=0)
    return np.corrcoef(increments.T)
