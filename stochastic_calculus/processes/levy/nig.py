"""Normal Inverse Gaussian Lévy process implementation."""

from typing import Optional, Union, NamedTuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.special import kv

from ...core.utils import validate_positive
from .base import LevyProcess


@dataclass
class NIGParameters:
    """Parameters for Normal Inverse Gaussian process."""

    alpha: float  # Tail heaviness parameter (α > 0)
    beta: float  # Asymmetry parameter (|β| < α)
    delta: float  # Scale parameter (δ > 0)
    mu: float  # Location parameter

    def __post_init__(self) -> None:
        """Validate parameters."""
        validate_positive(self.alpha, "tail parameter alpha")
        validate_positive(self.delta, "scale parameter delta")

        if abs(self.beta) >= self.alpha:
            raise ValueError(
                f"Asymmetry constraint violated: |β| ({abs(self.beta):.4f}) >= α ({self.alpha:.4f})"
            )


class NIGResult(NamedTuple):
    """Container for NIG simulation results."""

    paths: np.ndarray  # Process paths X(t)
    inverse_gaussian: np.ndarray  # Inverse Gaussian subordinator increments
    brownian: np.ndarray  # Underlying Brownian motion increments


class NIGProcess(LevyProcess):
    """
    Normal Inverse Gaussian Lévy process.

    The NIG process is defined as:
    X(t) = μt + βδ²(IG(t) - t) + σ√IG(t) * Z

    Where:
    - IG(t) is an Inverse Gaussian subordinator
    - Z is standard normal
    - Parameters satisfy α > |β| ≥ 0, δ > 0
    """

    def __init__(
        self,
        parameters: Union[NIGParameters, tuple[NIGParameters, ...]],
        n_processes: Optional[int] = None,
        correlation: Optional[float] = None,
    ) -> None:
        """
        Initialize NIG process.

        Args:
            parameters: Single NIGParameters or tuple for multiple processes
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
        self, n_steps: int, dt: float = 1.0, random_state: Optional[int] = None
    ) -> NIGResult:
        """
        Simulate NIG process paths.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed

        Returns:
            NIGResult with paths, inverse Gaussian, and Brownian components
        """
        rng = np.random.default_rng(random_state)

        paths = np.zeros((n_steps + 1, self.n_processes))
        ig_increments = np.zeros((n_steps, self.n_processes))
        brownian_increments = np.zeros((n_steps, self.n_processes))

        for i in range(self.n_processes):
            params = self._get_params_for_process(i)
            gamma = np.sqrt(params.alpha**2 - params.beta**2)

            # Generate Inverse Gaussian subordinator increments
            # IG(dt) with shape parameter dt and rate parameter γ²
            ig_shape = dt
            ig_rate = gamma**2

            # Use inverse Gaussian distribution from scipy
            ig_increments_i = (
                stats.invgauss.rvs(
                    mu=ig_shape / ig_rate, scale=1.0, size=n_steps, random_state=rng
                )
                * ig_rate
            )

            ig_increments[:, i] = ig_increments_i

            # Generate correlated Brownian motion
            brownian_i = rng.normal(0, 1, n_steps)
            brownian_increments[:, i] = brownian_i

            # Construct NIG increments
            nig_increments = (
                params.mu * dt
                + params.beta * params.delta**2 * (ig_increments_i - dt)
                + params.delta * np.sqrt(ig_increments_i) * brownian_i
            )

            # Construct cumulative path
            paths[1:, i] = np.cumsum(nig_increments)

        # Apply correlation if specified and multiple processes
        if self.correlation is not None and self.n_processes > 1:
            paths = self._apply_correlation(paths, n_steps, dt)

        return NIGResult(paths, ig_increments, brownian_increments)

    def _get_params_for_process(self, i: int) -> NIGParameters:
        """Get parameters for process i."""
        if isinstance(self.parameters, tuple):
            return self.parameters[i]
        return self.parameters

    def _apply_correlation(
        self, paths: np.ndarray, n_steps: int, dt: float
    ) -> np.ndarray:
        """Apply correlation structure to multiple processes."""
        if self.correlation == 0.0:
            return paths

        # Extract increments
        increments = np.diff(paths, axis=0)

        # Apply correlation using Cholesky decomposition
        correlation_matrix = np.full(
            (self.n_processes, self.n_processes), self.correlation
        )
        np.fill_diagonal(correlation_matrix, 1.0)

        try:
            chol = np.linalg.cholesky(correlation_matrix)
            correlated_increments = increments @ chol.T

            # Reconstruct paths
            correlated_paths = np.zeros_like(paths)
            correlated_paths[1:, :] = np.cumsum(correlated_increments, axis=0)
            return correlated_paths
        except np.linalg.LinAlgError:
            # Fallback if correlation matrix is not positive definite
            return paths

    def characteristic_function(
        self, u: Union[float, np.ndarray], t: float = 1.0
    ) -> Union[complex, np.ndarray]:
        """
        Characteristic function of NIG process.

        φ(u,t) = exp(t * δ * (√(α² - β²) - √(α² - (β + iu)²)) + iuμt)
        """
        params = self._get_params_for_process(0)

        # Convert to complex for proper computation
        u_complex = np.asarray(u, dtype=complex)

        # Compute characteristic function
        gamma = np.sqrt(params.alpha**2 - params.beta**2)
        sqrt_term = np.sqrt(params.alpha**2 - (params.beta + 1j * u_complex) ** 2)

        exponent = (
            t * params.delta * (gamma - sqrt_term) + 1j * u_complex * params.mu * t
        )

        result = np.exp(exponent)

        # Return scalar if input was scalar
        if np.isscalar(u):
            return complex(result)
        return result

    def levy_measure_density(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Lévy measure density for NIG process.

        ν(dx) = (αδ/π) * exp(βx) * K₁(α|x|) / |x| dx

        where K₁ is the modified Bessel function of the second kind.
        """
        params = self._get_params_for_process(0)

        x = np.asarray(x)
        abs_x = np.abs(x)

        # Avoid division by zero
        mask = abs_x > 1e-10
        result = np.zeros_like(x, dtype=float)

        if np.any(mask):
            x_nonzero = x[mask]
            abs_x_nonzero = abs_x[mask]

            # Constants
            c1 = params.alpha * params.delta / np.pi

            # Exponential term
            exp_term = np.exp(params.beta * x_nonzero)

            # Bessel function K₁
            bessel_k1 = kv(1, params.alpha * abs_x_nonzero)

            result[mask] = c1 * exp_term * bessel_k1 / abs_x_nonzero

        return result if not np.isscalar(x) else float(result)

    def pdf(
        self, x: Union[float, np.ndarray], t: float = 1.0
    ) -> Union[float, np.ndarray]:
        """
        Probability density function of NIG distribution.

        Args:
            x: Point(s) to evaluate
            t: Time parameter

        Returns:
            Density value(s)
        """
        params = self._get_params_for_process(0)

        x = np.asarray(x)
        centered_x = x - params.mu * t

        # NIG density formula
        alpha_t = params.alpha * np.sqrt(params.delta**2 * t + centered_x**2)

        # Constants
        c1 = params.alpha * params.delta * t / np.pi
        c2 = np.exp(
            params.delta * t * np.sqrt(params.alpha**2 - params.beta**2)
            + params.beta * centered_x
        )

        # Bessel function K₁
        bessel_k1 = kv(1, alpha_t)

        result = c1 * c2 * bessel_k1 / np.sqrt(params.delta**2 * t + centered_x**2)

        return result if not np.isscalar(x) else float(result)

    def get_parameters(self) -> dict:
        """Get process parameters."""
        return {
            "process_type": "NIGProcess",
            "parameters": self.parameters,
            "n_processes": self.n_processes,
            "correlation": self.correlation,
        }


def estimate_nig_parameters(
    data: np.ndarray, dt: float = 1.0, method: str = "method_of_moments"
) -> NIGParameters:
    """
    Estimate NIG parameters from data.

    Args:
        data: Time series data (1D array)
        dt: Time increment
        method: Estimation method

    Returns:
        Estimated NIGParameters
    """
    if data.ndim != 1:
        raise ValueError("Data must be 1D array for single process estimation")

    increments = np.diff(data) / dt

    if method == "method_of_moments":
        # Sample moments
        mean_inc = np.mean(increments)
        var_inc = np.var(increments, ddof=1)
        skew_inc = stats.skew(increments)
        kurt_inc = stats.kurtosis(increments)

        # Method of moments for NIG is complex
        # Use simplified approach based on moment relationships

        # Rough initial estimates
        mu_est = mean_inc

        # Use kurtosis to estimate tail heaviness
        alpha_est = max(1.0, np.sqrt(kurt_inc + 3))

        # Use skewness for asymmetry
        beta_est = np.clip(skew_inc * 0.5, -alpha_est * 0.9, alpha_est * 0.9)

        # Use variance for scale
        delta_est = max(0.1, np.sqrt(var_inc * 0.5))

        return NIGParameters(
            alpha=float(alpha_est),
            beta=float(beta_est),
            delta=float(delta_est),
            mu=float(mu_est),
        )
    else:
        raise ValueError(f"Estimation method '{method}' not implemented")


def create_simple_nig(
    alpha: float, beta: float, delta: float, mu: float = 0.0
) -> NIGProcess:
    """
    Create a simple NIG process.

    Args:
        alpha: Tail heaviness parameter
        beta: Asymmetry parameter
        delta: Scale parameter
        mu: Location parameter

    Returns:
        NIGProcess instance
    """
    params = NIGParameters(alpha=alpha, beta=beta, delta=delta, mu=mu)
    return NIGProcess(params, n_processes=1)
