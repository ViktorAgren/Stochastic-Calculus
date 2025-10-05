"""Geometric Brownian motion implementation."""

from typing import Optional, Union, NamedTuple, Any
import numpy as np

from ...core.protocols import Drift, Sigma, InitialValue, StochasticProcess
from .utils import generate_correlated_brownian
from ...core.utils import validate_positive




class GBMResult(NamedTuple):
    """Container for GBM simulation results."""

    prices: np.ndarray
    log_prices: np.ndarray

class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion simulator.

    Models asset prices using the SDE:
    dS = μ S dt + σ S dW
    """

    def __init__(
        self,
        drift: Drift,
        volatility: Sigma,
        initial_prices: InitialValue,
        correlation: Optional[float] = None,
        S_0: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        """
        Initialize GBM with dependency injection.

        Args:
            drift: Drift process implementation
            volatility: Volatility process implementation
            initial_prices: Initial price implementation
            correlation: Correlation between processes
            S_0: Override initial prices (if provided, overrides initial_prices component)
        """
        self.drift = drift
        self.volatility = volatility
        self.initial_prices = initial_prices
        self.correlation = correlation

        # Validate compatible dimensions
        if (
            drift.n_processes != volatility.n_processes
            or drift.n_processes != initial_prices.n_processes
        ):
            raise ValueError(
                "Drift, volatility, and initial prices must have same n_processes"
            )

        if drift.sample_size != volatility.sample_size:
            raise ValueError("Drift and volatility must have same sample_size")

        self.n_processes = drift.n_processes
        self.n_steps = drift.sample_size
        
        # Handle initial price override for standardized interface
        if S_0 is not None:
            if np.isscalar(S_0):
                self.S_0 = np.full(self.n_processes, S_0)
            else:
                self.S_0 = np.asarray(S_0)
                if self.S_0.shape != (self.n_processes,):
                    raise ValueError(f"S_0 must have shape ({self.n_processes},), got {self.S_0.shape}")
        else:
            # Use initial prices from component
            self.S_0 = self.initial_prices.get_initial_values()

    def simulate(
        self, n_steps: int, dt: float = 1.0, random_state: Optional[int] = None
    ) -> GBMResult:
        """
        Simulate GBM paths.

        Args:
            n_steps: Number of time steps (overrides component n_steps)
            dt: Time increment
            random_state: Random seed

        Returns:
            GBMResult with price and log-price paths
        """
        # Use provided n_steps or fall back to component n_steps
        actual_n_steps = n_steps if n_steps != self.n_steps else self.n_steps
        
        drift_matrix = self.drift.get_drift(random_state)
        vol_matrix = self.volatility.get_volatility(random_state)
        
        # Adjust matrices if n_steps differs from component size
        if actual_n_steps != self.n_steps:
            # Repeat or truncate to match requested n_steps
            if actual_n_steps > self.n_steps:
                # Repeat the last values
                repeats = actual_n_steps - self.n_steps
                drift_matrix = np.vstack([drift_matrix, np.tile(drift_matrix[-1:], (repeats, 1))])
                vol_matrix = np.vstack([vol_matrix, np.tile(vol_matrix[-1:], (repeats, 1))])
            else:
                # Truncate
                drift_matrix = drift_matrix[:actual_n_steps]
                vol_matrix = vol_matrix[:actual_n_steps]

        dW = generate_correlated_brownian(
            actual_n_steps, self.n_processes, self.correlation, dt, random_state
        )

        time_integrals = np.cumsum((drift_matrix - 0.5 * vol_matrix**2) * dt, axis=0)
        stochastic_integrals = np.cumsum(vol_matrix * dW, axis=0)

        log_prices = np.zeros((actual_n_steps + 1, self.n_processes))
        log_prices[0] = np.log(self.S_0)  # Use standardized initial values
        log_prices[1:] = log_prices[0] + time_integrals + stochastic_integrals

        prices = np.exp(log_prices)

        return GBMResult(prices, log_prices)

    def get_initial_value_names(self) -> list[str]:
        """Return names of initial value parameters for GBM."""
        return ["S_0"]

    def set_initial_values(self, **kwargs) -> None:
        """Set initial values on the GBM process."""
        if "S_0" in kwargs:
            S_0_value = kwargs["S_0"]
            if np.isscalar(S_0_value):
                self.S_0 = np.full(self.n_processes, S_0_value)
            else:
                self.S_0 = np.asarray(S_0_value)
                if self.S_0.shape != (self.n_processes,):
                    raise ValueError(f"S_0 must have shape ({self.n_processes},), got {self.S_0.shape}")

    def get_initial_values(self) -> dict[str, Any]:
        """Get current initial values."""
        return {"S_0": self.S_0.copy()}


    def get_parameters(self) -> dict[str, Any]:
        """Get process parameters."""
        return {
            "process_type": "GeometricBrownianMotion",
            "n_processes": self.n_processes,
            "n_steps": self.n_steps,
            "correlation": self.correlation,
        }




def estimate_gbm_parameters(price_data: np.ndarray, dt: float = 1.0) -> dict:
    """
    Estimate GBM parameters from price data.

    For GBM: dS = μS dt + σS dW
    Log returns: d(log S) = (μ - σ²/2) dt + σ dW

    So: μ = mean(log_returns)/dt + σ²/2
        σ = std(log_returns)/√dt

    Args:
        price_data: Price series (1D or 2D array)
        dt: Time increment

    Returns:
        Dictionary with estimated mu and sigma
    """
    log_returns = np.diff(np.log(price_data), axis=0)

    if price_data.ndim == 1:
        # Estimate sigma first (volatility per unit time)
        sigma_est = np.std(log_returns, ddof=1) / np.sqrt(dt)
        # Then estimate mu (drift per unit time)
        # From d(ln S) = (μ - σ²/2)dt + σ dW, we have:
        # μ = E[d(ln S)]/dt + σ²/2
        mu_est = np.mean(log_returns) / dt + 0.5 * sigma_est**2
        return {"mu": float(mu_est), "sigma": float(sigma_est)}
    else:
        # Multi-asset case
        sigma_est = np.std(log_returns, axis=0, ddof=1) / np.sqrt(dt)
        mu_est = np.mean(log_returns, axis=0) / dt + 0.5 * sigma_est**2
        return {"mu": tuple(mu_est), "sigma": tuple(sigma_est)}
