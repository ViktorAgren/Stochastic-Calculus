"""Cox-Ingersoll-Ross process implementation."""

from typing import Optional, Union, NamedTuple
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression

from ...core.utils import validate_positive
from ..brownian.utils import generate_correlated_brownian


@dataclass
class CIRParameters:
    """Parameters for Cox-Ingersoll-Ross process."""

    a: float  # Mean reversion speed
    b: float  # Long-term mean level
    c: float  # Volatility scale

    def __post_init__(self) -> None:
        """Validate parameters and Feller condition."""
        validate_positive(self.a, "mean reversion speed a")
        validate_positive(self.b, "long-term mean b")
        validate_positive(self.c, "volatility c")

        if 2 * self.a * self.b < self.c**2:
            raise ValueError(
                f"Feller condition violated: 2ab ({2*self.a*self.b:.4f}) < c² ({self.c**2:.4f})"
            )


class CIRResult(NamedTuple):
    """Container for CIR process simulation results."""

    paths: np.ndarray


class CIRProcess:
    """
    Cox-Ingersoll-Ross square-root process.

    Solves the SDE: dσ² = a(b - σ²)dt + c√σ² dW
    """

    def __init__(
        self,
        parameters: Union[CIRParameters, tuple[CIRParameters, ...]],
        n_processes: Optional[int] = None,
        correlation: Optional[float] = None,
    ) -> None:
        """
        Initialize CIR process.

        Args:
            parameters: Single CIRParameters or tuple for multiple processes
            n_processes: Number of processes (ignored if parameters is tuple)
            correlation: Correlation between processes
        """
        self.correlation = correlation

        if isinstance(parameters, tuple):
            self.parameters = parameters
            self.n_processes = len(parameters)
        else:
            if n_processes is None:
                raise ValueError("n_processes required when parameters is not tuple")
            self.parameters = parameters
            self.n_processes = n_processes

    def simulate(
        self,
        n_steps: int,
        dt: float = 1.0,
        sigma_0: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ) -> CIRResult:
        """
        Simulate CIR process using Euler-Maruyama with absorption.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            sigma_0: Initial values (uses b if None)
            random_state: Random seed

        Returns:
            CIRResult with simulated paths
        """
        # Generate correlated Brownian motions
        dW = generate_correlated_brownian(
            n_steps, self.n_processes, self.correlation, dt, random_state
        )

        paths = np.zeros((n_steps + 1, self.n_processes))

        if sigma_0 is None:
            if isinstance(self.parameters, tuple):
                paths[0] = [p.b for p in self.parameters]
            else:
                paths[0] = self.parameters.b
        else:
            paths[0] = sigma_0
        for i in range(self.n_processes):
            params = self._get_params_for_process(i)
            paths[:, i] = self._simulate_single_cir(
                n_steps, dt, params, paths[0, i], dW[:, i]
            )

        return CIRResult(paths)

    def _get_params_for_process(self, i: int) -> CIRParameters:
        """Get parameters for process i."""
        if isinstance(self.parameters, tuple):
            return self.parameters[i]
        return self.parameters

    def _simulate_single_cir(
        self,
        n_steps: int,
        dt: float,
        params: CIRParameters,
        sigma_0: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """Simulate single CIR process using Euler-Maruyama scheme."""
        paths = np.zeros(n_steps + 1)
        paths[0] = sigma_0

        for t in range(n_steps):
            sigma_t = max(paths[t], 0.0)  # Apply absorption boundary
            sqrt_sigma_t = np.sqrt(sigma_t)

            # Euler-Maruyama step
            d_sigma = (
                params.a * (params.b - sigma_t) * dt + params.c * sqrt_sigma_t * dW[t]
            )

            paths[t + 1] = paths[t] + d_sigma

            # Apply absorption boundary again
            paths[t + 1] = max(paths[t + 1], 0.0)

            # Check for numerical issues
            if np.isnan(paths[t + 1]):
                raise ValueError(
                    f"CIR simulation failed at step {t}. "
                    f"Try smaller time step or different parameters."
                )

        return paths

    def get_parameters(self) -> dict:
        """Get process parameters."""
        return {
            "process_type": "CIRProcess",
            "parameters": self.parameters,
            "n_processes": self.n_processes,
            "correlation": self.correlation,
        }

    def simulate_with_initial_value(
        self, n_steps: int, dt: float = 1.0, sigma_0: Optional[Union[float, np.ndarray]] = None, random_state: Optional[int] = None
    ) -> CIRResult:
        """
        Simulate CIR process with specific initial value.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            sigma_0: Initial value(s)
            random_state: Random seed

        Returns:
            CIRResult with simulated paths
        """
        return self.simulate(n_steps=n_steps, dt=dt, sigma_0=sigma_0, random_state=random_state)


def estimate_cir_parameters(data: np.ndarray, dt: float = 1.0) -> CIRParameters:
    """
    Estimate CIR parameters using OLS on transformed equation.

    The CIR SDE can be written as:
    dσ²/√σ² = (a/√σ²)dt + (-a)√σ² dt + c dW

    This gives: d(σ²)/√σ² = ab(1/√σ²) - a√σ² + c dW

    Args:
        data: Time series of CIR process values (1D array)
        dt: Time increment

    Returns:
        Estimated CIRParameters
    """
    if data.ndim != 1:
        raise ValueError("Data must be 1D array for single process estimation")

    # Ensure non-negative values
    data = np.maximum(data, 1e-8)

    # Calculate increments and transformed variables
    d_sigma = np.diff(data) / dt
    sqrt_sigma = np.sqrt(data[:-1])

    # Regression equation: d_sigma/sqrt_sigma = a*b*(1/sqrt_sigma) - a*sqrt_sigma + noise
    y = d_sigma / sqrt_sigma
    X = np.column_stack([1.0 / sqrt_sigma, sqrt_sigma])

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)

    # Extract parameters
    ab_coef = reg.coef_[0]  # Coefficient of 1/√σ
    neg_a_coef = reg.coef_[1]  # Coefficient of √σ

    a = -neg_a_coef
    b = ab_coef / a if a > 0 else 1.0

    # Estimate c from residuals
    y_pred = reg.predict(X)
    residuals = y - y_pred
    c = np.sqrt(dt) * np.std(residuals)

    # Ensure parameters are valid
    a = max(a, 0.001)
    b = max(b, 0.001)
    c = max(c, 0.001)

    # Adjust to satisfy Feller condition if necessary
    if 2 * a * b < c**2:
        # Increase a or b to satisfy condition
        a = (c**2) / (2 * b) + 0.001

    return CIRParameters(a=a, b=b, c=c)


def simulate_correlated_cir_processes(
    n_steps: int,
    parameters: Union[CIRParameters, tuple[CIRParameters, ...]],
    n_processes: Optional[int] = None,
    correlation: Optional[float] = None,
    dt: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function for correlated CIR processes.

    Returns:
        Array of shape (n_steps+1, n_processes)
    """
    cir = CIRProcess(parameters, n_processes, correlation)
    result = cir.simulate(n_steps, dt, random_state=random_state)
    return result.paths
