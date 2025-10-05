"""Ornstein-Uhlenbeck process implementation."""

from typing import Optional, Union, NamedTuple, Any
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression

from ...core.utils import validate_positive
from ...core.protocols import StochasticProcess
from ..brownian.standard import BrownianMotion


@dataclass
class OUParameters:
    """Parameters for Ornstein-Uhlenbeck process."""

    alpha: float  # Mean reversion speed
    gamma: float  # Long-term mean
    beta: float  # Volatility scale

    def __post_init__(self) -> None:
        """Validate parameters."""
        validate_positive(self.alpha, "mean reversion speed alpha")
        validate_positive(self.beta, "volatility beta")


class OUResult(NamedTuple):
    """Container for OU process simulation results."""

    paths: np.ndarray


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """
    Ornstein-Uhlenbeck mean-reverting process.

    Solves the SDE: dX = α(γ - X)dt + β dW
    """

    def __init__(
        self,
        parameters: Union[OUParameters, tuple[OUParameters, ...]],
        n_processes: Optional[int] = None,
        correlation: Optional[float] = None,
        X_0: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        """
        Initialize OU process.

        Args:
            parameters: Single OUParameters or tuple for multiple processes
            n_processes: Number of processes (ignored if parameters is tuple)
            correlation: Correlation between processes
            X_0: Initial values (default: gamma values)
        """
        self.correlation = correlation
        self.X_0 = X_0

        if isinstance(parameters, tuple):
            self.parameters = parameters
            self.n_processes = len(parameters)
        else:
            if n_processes is None:
                raise ValueError("n_processes required when parameters is not tuple")
            self.parameters = (parameters,) if n_processes == 1 else parameters
            self.n_processes = n_processes

    def simulate(
        self, n_steps: int, dt: float = 1.0, random_state: Optional[int] = None
    ) -> OUResult:
        """
        Simulate OU process using exact solution.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed

        Returns:
            OUResult with simulated paths
        """
        # Generate correlated Brownian motions using composition
        brownian = BrownianMotion(
            n_processes=self.n_processes, correlation=self.correlation
        )
        brownian_result = brownian.simulate(n_steps, dt, random_state)
        dW = brownian_result.increments

        # Initialize paths
        paths = np.zeros((n_steps + 1, self.n_processes))

        # Set initial conditions
        X_0 = self.X_0
        if X_0 is None:
            if isinstance(self.parameters, tuple):
                paths[0] = [p.gamma for p in self.parameters]
            else:
                paths[0] = self.parameters.gamma
        else:
            paths[0] = X_0

        # Simulate each process
        for i in range(self.n_processes):
            params = self._get_params_for_process(i)
            paths[:, i] = self._simulate_single_ou(
                n_steps, dt, params, paths[0, i], dW[:, i]
            )

        return OUResult(paths)

    def get_initial_value_names(self) -> list[str]:
        """Return names of initial value parameters for OU process."""
        return [StandardInitialValueNames.PROCESS_VALUE]

    def set_initial_values(self, **kwargs) -> None:
        """Set initial values on the OU process."""
        if StandardInitialValueNames.PROCESS_VALUE in kwargs:
            self.X_0 = kwargs[StandardInitialValueNames.PROCESS_VALUE]

    def get_initial_values(self) -> dict[str, Any]:
        """Get current initial values."""
        return {StandardInitialValueNames.PROCESS_VALUE: self.X_0}

    def _get_params_for_process(self, i: int) -> OUParameters:
        """Get parameters for process i."""
        if isinstance(self.parameters, tuple):
            return self.parameters[i]
        return self.parameters

    def _simulate_single_ou(
        self, n_steps: int, dt: float, params: OUParameters, X_0: float, dW: np.ndarray
    ) -> np.ndarray:
        """Simulate single OU process using exact solution."""
        time_grid = np.arange(n_steps + 1) * dt
        exp_neg_alpha_t = np.exp(-params.alpha * time_grid)

        # Calculate stochastic integral
        stochastic_integral = self._calculate_ou_integral(time_grid, dW, params)

        # Exact solution
        paths = (
            X_0 * exp_neg_alpha_t
            + params.gamma * (1 - exp_neg_alpha_t)
            + params.beta * exp_neg_alpha_t * stochastic_integral
        )

        return paths

    def _calculate_ou_integral(
        self, time_grid: np.ndarray, dW: np.ndarray, params: OUParameters
    ) -> np.ndarray:
        """Calculate ∫₀ᵗ exp(αs) dW_s."""
        n_steps = len(dW)

        integral = np.zeros(n_steps + 1)
        for i in range(n_steps):
            t_i = time_grid[i]
            exp_alpha_t = np.exp(params.alpha * t_i)
            integral[i + 1] = integral[i] + exp_alpha_t * dW[i]

        return integral

    def get_parameters(self) -> dict:
        """Get process parameters."""
        return {
            "process_type": "OrnsteinUhlenbeckProcess",
            "parameters": self.parameters,
            "n_processes": self.n_processes,
            "correlation": self.correlation,
        }


def estimate_ou_parameters(data: np.ndarray, dt: float = 1.0) -> OUParameters:
    """
    Estimate OU parameters using OLS regression.

    Args:
        data: Time series data (1D array)
        dt: Time increment

    Returns:
        Estimated OUParameters
    """
    if data.ndim != 1:
        raise ValueError("Data must be 1D array for single process estimation")

    # Prepare regression: dX = a + b*X + noise
    y = np.diff(data) / dt
    X = data[:-1].reshape(-1, 1)

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)

    # Extract parameters
    b_coef = reg.coef_[0]
    a_coef = reg.intercept_

    alpha = -b_coef
    gamma = a_coef / alpha if alpha != 0 else 0.0

    # Estimate volatility from residuals
    y_pred = reg.predict(X)
    residuals = y - y_pred
    beta = np.sqrt(dt) * np.std(residuals)

    return OUParameters(
        alpha=max(alpha, 0.001),  # Ensure positive
        gamma=gamma,
        beta=max(beta, 0.001),  # Ensure positive
    )


def simulate_correlated_ou_processes(
    n_steps: int,
    parameters: Union[OUParameters, tuple[OUParameters, ...]],
    n_processes: Optional[int] = None,
    correlation: Optional[float] = None,
    dt: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function for correlated OU processes.

    Returns:
        Array of shape (n_steps+1, n_processes)
    """
    ou = OrnsteinUhlenbeckProcess(parameters, n_processes, correlation)
    result = ou.simulate(n_steps, dt, random_state=random_state)
    return result.paths
