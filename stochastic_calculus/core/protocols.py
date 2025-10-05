"""Core protocol definitions for stochastic processes and components."""

from typing import Protocol, Optional, Any
import numpy as np
from numpy.typing import NDArray


class Drift(Protocol):
    """Protocol for drift processes in stochastic differential equations."""

    @property
    def sample_size(self) -> int:
        """The number of time steps in the drift process."""
        ...

    @property
    def n_processes(self) -> int:
        """The number of parallel drift processes."""
        ...

    def get_drift(self, random_state: Optional[int] = None) -> NDArray[np.float64]:
        """
        Generate drift matrix.

        Returns:
            2D array of shape (sample_size, n_processes)
        """
        ...


class Sigma(Protocol):
    """Protocol for volatility/diffusion processes."""

    @property
    def sample_size(self) -> int:
        """The number of time steps in the volatility process."""
        ...

    @property
    def n_processes(self) -> int:
        """The number of parallel volatility processes."""
        ...

    def get_volatility(self, random_state: Optional[int] = None) -> NDArray[np.float64]:
        """
        Generate volatility matrix.

        Returns:
            2D array of shape (sample_size, n_processes)
        """
        ...


class InitialValue(Protocol):
    """Protocol for initial values of stochastic processes."""

    @property
    def n_processes(self) -> int:
        """The number of initial values."""
        ...

    def get_initial_values(self, random_state: Optional[int] = None) -> NDArray[np.float64]:
        """
        Generate initial values.

        Returns:
            1D array of length n_processes
        """
        ...


class StochasticProcess(Protocol):
    """Base protocol for all stochastic processes."""

    def simulate(
        self, n_steps: int, dt: float = 1.0, random_state: Optional[int] = None
    ) -> Any:
        """
        Simulate the stochastic process.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed for reproducibility

        Returns:
            Simulation results (typically array or custom result container)
        """
        ...

    def simulate_with_initial_values(
        self,
        n_steps: int,
        initial_values: dict[str, Any],
        dt: float = 1.0,
        random_state: Optional[int] = None,
    ) -> Any:
        """
        Default implementation: Store current values, set new ones, simulate, restore.
        
        Processes can override this if they need custom behavior.
        """
        # Store current values
        current_values = {}
        for key, value in initial_values.items():
            if hasattr(self, key):
                current_values[key] = getattr(self, key)
                setattr(self, key, value)
        
        try:
            # Run simulation
            return self.simulate(n_steps, dt, random_state)
        finally:
            # Restore original values
            for key, value in current_values.items():
                setattr(self, key, value)

    def get_initial_value_names(self) -> list[str]:
        """
        Return names of initial value parameters for this process.

        Returns:
            List of parameter names that can be used as initial values
        """
        ...

    def set_initial_values(self, **kwargs) -> None:
        """
        Set initial values on the process.

        Args:
            **kwargs: Initial value parameters to set
        """
        ...

    def get_initial_values(self) -> dict[str, Any]:
        """
        Get current initial values.

        Returns:
            Dictionary of current initial value parameters
        """
        ...

    def get_parameters(self) -> dict[str, Any]:
        """Get process parameters as dictionary."""
        ...


class OptionPricer(Protocol):
    """Protocol for option pricing models."""

    def price(self, strike: float, expiry: float, option_type: str = "call") -> float:
        """
        Calculate option price.

        Args:
            strike: Strike price
            expiry: Time to expiry
            option_type: "call" or "put"

        Returns:
            Option price
        """
        ...

    def greeks(self, strike: float, expiry: float, option_type: str = "call") -> dict[str, float]:
        """
        Calculate option Greeks.

        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        ...
