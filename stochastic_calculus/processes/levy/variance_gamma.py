"""Variance Gamma Lévy process implementation."""

from typing import Optional, Union, NamedTuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.special import kv

from ...core.utils import validate_positive
from ...core.protocols import StochasticProcess
from ..brownian.standard import BrownianMotion
from .base import LevyProcess, TimeChangedBrownianMotion


@dataclass
class VarianceGammaParameters:
    """Parameters for Variance Gamma process."""

    theta: float  # Drift parameter (asymmetry)
    sigma: float  # Volatility parameter (scale)
    nu: float  # Time change rate parameter (shape)

    def __post_init__(self) -> None:
        """Validate parameters."""
        validate_positive(self.sigma, "volatility sigma")
        validate_positive(self.nu, "time change rate nu")


class VarianceGammaResult(NamedTuple):
    """Container for Variance Gamma simulation results."""

    paths: np.ndarray  # Process paths X(t)
    subordinator: np.ndarray  # Gamma subordinator increments
    brownian: np.ndarray  # Underlying Brownian motion increments


class VarianceGammaProcess(LevyProcess, StochasticProcess):
    """
    Variance Gamma Lévy process.

    The Variance Gamma process is defined as:
    X(t) = θ * G(t) + σ * W(G(t))

    Where:
    - G(t) is a Gamma subordinator with rate 1/ν and shape t/ν
    - W(s) is standard Brownian motion
    - θ controls asymmetry/drift
    - σ controls volatility scaling
    - ν controls jump activity (smaller ν = more jumps)
    """

    def __init__(
        self,
        parameters: Union[VarianceGammaParameters, tuple[VarianceGammaParameters, ...]],
        n_processes: Optional[int] = None,
        correlation: Optional[float] = None,
    ) -> None:
        """
        Initialize Variance Gamma process.

        Args:
            parameters: Single VarianceGammaParameters or tuple for multiple processes
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
            self.parameters = (parameters,) if n_processes == 1 else parameters
            self.n_processes = n_processes

    def simulate(
        self, n_steps: int, dt: float = 1.0, random_state: Optional[int] = None
    ) -> VarianceGammaResult:
        """
        Simulate Variance Gamma process paths.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed

        Returns:
            VarianceGammaResult with paths, subordinator, and Brownian components
        """
        rng = np.random.default_rng(random_state)

        paths = np.zeros((n_steps + 1, self.n_processes))
        subordinator_increments = np.zeros((n_steps, self.n_processes))
        brownian_increments = np.zeros((n_steps, self.n_processes))

        for i in range(self.n_processes):
            params = self._get_params_for_process(i)

            # Generate Gamma subordinator increments
            # G(dt) ~ Gamma(shape=dt/ν, scale=ν)
            gamma_increments = rng.gamma(
                shape=dt / params.nu, scale=params.nu, size=n_steps
            )
            subordinator_increments[:, i] = gamma_increments

            # Generate time-changed Brownian motion
            bm_increments = TimeChangedBrownianMotion.simulate_time_changed_bm(
                n_steps=n_steps,
                subordinator=gamma_increments,
                drift=params.theta,
                volatility=params.sigma,
                random_state=(
                    rng.integers(0, 2**31) if random_state is not None else None
                ),
            )
            brownian_increments[:, i] = bm_increments

            # Construct cumulative path
            paths[1:, i] = np.cumsum(bm_increments)

        # Apply correlation if specified and multiple processes using BrownianMotion
        if self.correlation is not None and self.n_processes > 1:
            paths = self._apply_correlation_using_brownian(
                paths, n_steps, dt, random_state
            )

        return VarianceGammaResult(paths, subordinator_increments, brownian_increments)

    def _get_params_for_process(self, i: int) -> VarianceGammaParameters:
        """Get parameters for process i."""
        if isinstance(self.parameters, tuple):
            return self.parameters[i]
        return self.parameters

    def _apply_correlation_using_brownian(
        self, paths: np.ndarray, n_steps: int, dt: float, random_state: Optional[int]
    ) -> np.ndarray:
        """Apply correlation structure using BrownianMotion composition."""
        if self.correlation == 0.0:
            return paths

        # Generate correlated Brownian motions using BrownianMotion
        correlated_brownian = BrownianMotion(
            n_processes=self.n_processes, correlation=self.correlation
        )
        brownian_result = correlated_brownian.simulate(n_steps, dt, random_state)
        correlated_increments = brownian_result.increments

        # Scale by the magnitude of original increments
        original_increments = np.diff(paths, axis=0)
        magnitude_scaling = np.std(original_increments, axis=0) / np.std(
            correlated_increments, axis=0
        )

        # Apply scaling and offset to match original characteristics
        scaled_increments = correlated_increments * magnitude_scaling

        # Reconstruct paths
        correlated_paths = np.zeros_like(paths)
        correlated_paths[1:, :] = np.cumsum(scaled_increments, axis=0)
        return correlated_paths

    def _apply_correlation(
        self, paths: np.ndarray, n_steps: int, dt: float
    ) -> np.ndarray:
        """Apply correlation structure to multiple processes (legacy method)."""
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
        Characteristic function of Variance Gamma process.

        φ(u,t) = (1 - iuθν + σ²u²ν/2)^(-t/ν)
        """
        params = self._get_params_for_process(0)

        # Convert to complex for proper computation
        u_complex = np.asarray(u, dtype=complex)

        # Characteristic function formula
        exponent = -t / params.nu
        base = (
            1
            - 1j * u_complex * params.theta * params.nu
            + (params.sigma**2 * u_complex**2 * params.nu) / 2
        )

        result = base**exponent

        # Return scalar if input was scalar
        if np.isscalar(u):
            return complex(result)
        return result

    def levy_measure_density(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Lévy measure density for Variance Gamma process.

        ν(dx) = (1/|x|) * exp(θx/σ²) / (ν * σ² * sqrt(2π)) * K₁(|x|*sqrt(2/ν)/σ) dx

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
            c1 = 1 / (params.nu * params.sigma**2 * np.sqrt(2 * np.pi))

            # Exponential term
            exp_term = np.exp(params.theta * x_nonzero / (params.sigma**2))

            # Bessel function argument
            bessel_arg = abs_x_nonzero * np.sqrt(2 / params.nu) / params.sigma

            # Modified Bessel function K₁
            bessel_k1 = kv(1, bessel_arg)

            result[mask] = c1 * exp_term * bessel_k1 / abs_x_nonzero

        return result if not np.isscalar(x) else float(result)

    def get_parameters(self) -> dict:
        """Get process parameters."""
        return {
            "process_type": "VarianceGammaProcess",
            "parameters": self.parameters,
            "n_processes": self.n_processes,
            "correlation": self.correlation,
        }


def estimate_vg_parameters(
    data: np.ndarray, dt: float = 1.0, method: str = "method_of_moments"
) -> VarianceGammaParameters:
    """
    Estimate Variance Gamma parameters from data.

    Args:
        data: Time series data (1D array)
        dt: Time increment
        method: Estimation method

    Returns:
        Estimated VarianceGammaParameters
    """
    if data.ndim != 1:
        raise ValueError("Data must be 1D array for single process estimation")

    increments = np.diff(data) / dt

    if method == "method_of_moments":
        # Sample moments
        mean_inc = np.mean(increments)
        var_inc = np.var(increments, ddof=1)
        kurt_inc = stats.kurtosis(increments)

        # Method of moments equations for VG
        # E[X] = θ, Var[X] = σ² + θ²ν, Skew[X] = 2θ³ν²(σ²+θ²ν)^(-3/2)
        # Kurt[X] = 3(1 + 2ν(σ²+θ²ν))

        # Estimate θ from mean
        theta_est = mean_inc

        # Estimate ν from kurtosis
        excess_kurt = kurt_inc
        nu_est = max(0.01, excess_kurt / 6.0)  # Rough approximation

        # Estimate σ from variance
        sigma_sq_est = var_inc - theta_est**2 * nu_est
        sigma_est = max(0.01, np.sqrt(abs(sigma_sq_est)))

        return VarianceGammaParameters(
            theta=float(theta_est), sigma=float(sigma_est), nu=float(nu_est)
        )
    else:
        raise ValueError(f"Estimation method '{method}' not implemented")


def create_simple_vg(theta: float, sigma: float, nu: float) -> VarianceGammaProcess:
    """
    Create a simple Variance Gamma process.

    Args:
        theta: Drift/asymmetry parameter
        sigma: Volatility parameter
        nu: Time change rate parameter

    Returns:
        VarianceGammaProcess instance
    """
    params = VarianceGammaParameters(theta=theta, sigma=sigma, nu=nu)
    return VarianceGammaProcess(params, n_processes=1)
