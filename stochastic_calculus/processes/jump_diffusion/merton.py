"""Merton Jump Diffusion model implementation."""

from typing import Optional, Union, NamedTuple, Any
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression

from ...core.utils import validate_positive
from ...core.protocols import StochasticProcess
from ..brownian.standard import BrownianMotion


@dataclass
class MertonParameters:
    """Parameters for Merton Jump Diffusion model."""

    mu: float  # Drift coefficient
    sigma: float  # Diffusion volatility
    jump_intensity: float  # λ - average number of jumps per unit time
    jump_mean: float  # μ_J - mean of log jump sizes
    jump_volatility: float  # σ_J - volatility of log jump sizes

    def __post_init__(self) -> None:
        """Validate parameters."""
        validate_positive(self.sigma, "diffusion volatility sigma")
        validate_positive(self.jump_intensity, "jump intensity lambda")
        validate_positive(self.jump_volatility, "jump volatility sigma_J")


class MertonResult(NamedTuple):
    """Container for Merton jump diffusion simulation results."""

    prices: np.ndarray  # Asset prices S(t)
    log_prices: np.ndarray  # Log prices ln(S(t))
    jump_times: list  # List of jump times for each path
    jump_sizes: list  # List of jump sizes for each path


class MertonJumpDiffusion(StochasticProcess):
    """
    Merton Jump Diffusion model.

    Solves the SDE:
    dS = μS dt + σS dW + S dJ

    Where J is a compound Poisson process with:
    - Jump intensity λ
    - Log-normal jump sizes: ln(1 + k) ~ N(μ_J, σ_J²)
    """

    def __init__(
        self,
        parameters: Union[MertonParameters, tuple[MertonParameters, ...]],
        n_processes: Optional[int] = None,
        correlation: Optional[float] = None,
        S_0: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        """
        Initialize Merton Jump Diffusion model.

        Args:
            parameters: Single MertonParameters or tuple for multiple assets
            n_processes: Number of processes (ignored if parameters is tuple)
            correlation: Correlation between diffusion processes
            S_0: Initial asset prices (default: 100.0)
        """
        self.correlation = correlation
        self.S_0 = S_0

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
    ) -> MertonResult:
        """
        Simulate Merton Jump Diffusion paths.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed

        Returns:
            MertonResult with prices, log prices, and jump information
        """
        rng = np.random.default_rng(random_state)

        S_0 = self.S_0
        if S_0 is None:
            S_0 = 100.0

        if np.isscalar(S_0):
            S_0 = np.full(self.n_processes, S_0)
        else:
            S_0 = np.asarray(S_0)

        prices = np.zeros((n_steps + 1, self.n_processes))
        log_prices = np.zeros((n_steps + 1, self.n_processes))
        jump_times: list[list[float]] = [[] for _ in range(self.n_processes)]
        jump_sizes: list[list[float]] = [[] for _ in range(self.n_processes)]

        prices[0] = S_0
        log_prices[0] = np.log(S_0)

        # Generate correlated Brownian motions for diffusion part using composition
        brownian = BrownianMotion(
            n_processes=self.n_processes, correlation=self.correlation
        )
        brownian_result = brownian.simulate(n_steps, dt, random_state)
        dW = brownian_result.increments

        # Simulate each asset
        for i in range(self.n_processes):
            params = self._get_params_for_process(i)

            # Adjust drift for jump compensation
            # E[e^X - 1] where X ~ N(μ_J, σ_J²)
            jump_compensation = (
                np.exp(params.jump_mean + 0.5 * params.jump_volatility**2) - 1
            )
            adjusted_drift = params.mu - params.jump_intensity * jump_compensation

            # Simulate path with jumps
            log_S = log_prices[0, i]

            for t in range(n_steps):
                # Diffusion part
                log_S += adjusted_drift * dt + params.sigma * dW[t, i]

                # Jump part - Poisson process
                num_jumps = rng.poisson(params.jump_intensity * dt)

                if num_jumps > 0:
                    # Generate jump sizes (log-normal)
                    jump_magnitudes = rng.normal(
                        params.jump_mean, params.jump_volatility, num_jumps
                    )
                    total_jump = np.sum(jump_magnitudes)
                    log_S += total_jump

                    # Record jump information
                    jump_times[i].append((t + 1) * dt)
                    jump_sizes[i].append(total_jump)

                log_prices[t + 1, i] = log_S
                prices[t + 1, i] = np.exp(log_S)

        return MertonResult(prices, log_prices, jump_times, jump_sizes)

    def get_initial_value_names(self) -> list[str]:
        """Return names of initial value parameters for Merton process."""
        return [StandardInitialValueNames.ASSET_PRICE]

    def set_initial_values(self, **kwargs) -> None:
        """Set initial values on the Merton process."""
        if StandardInitialValueNames.ASSET_PRICE in kwargs:
            self.S_0 = kwargs[StandardInitialValueNames.ASSET_PRICE]

    def get_initial_values(self) -> dict[str, Any]:
        """Get current initial values."""
        return {StandardInitialValueNames.ASSET_PRICE: self.S_0}

    def _get_params_for_process(self, i: int) -> MertonParameters:
        """Get parameters for process i."""
        if isinstance(self.parameters, tuple):
            return self.parameters[i]
        return self.parameters

    def get_parameters(self) -> dict:
        """Get process parameters."""
        return {
            "process_type": "MertonJumpDiffusion",
            "parameters": self.parameters,
            "n_processes": self.n_processes,
            "correlation": self.correlation,
        }


def estimate_merton_parameters(
    price_data: np.ndarray, dt: float = 1 / 252, method: str = "method_of_moments"
) -> MertonParameters:
    """
    Estimate Merton Jump Diffusion parameters from price data.

    Args:
        price_data: Price series (1D array)
        dt: Time increment
        method: Estimation method ("method_of_moments" or "maximum_likelihood")

    Returns:
        Estimated MertonParameters
    """
    if price_data.ndim != 1:
        raise ValueError("Data must be 1D array for single process estimation")

    log_returns = np.diff(np.log(price_data))

    if method == "method_of_moments":
        # Method of moments estimation
        mean_return = np.mean(log_returns) / dt
        var_return = np.var(log_returns, ddof=1) / dt
        skew_return = float(
            np.mean(
                ((log_returns - np.mean(log_returns)) / np.std(log_returns, ddof=1))
                ** 3
            )
        )
        kurt_return = float(
            np.mean(
                ((log_returns - np.mean(log_returns)) / np.std(log_returns, ddof=1))
                ** 4
            )
        )

        # Initial estimates assuming some jumps
        sigma_est = np.sqrt(var_return * 0.8)  # Assume 80% of variance from diffusion
        jump_intensity_est = max(0.1, (kurt_return - 3) / 10)  # Rough heuristic
        jump_vol_est = 0.1  # Initial guess
        jump_mean_est = skew_return * 0.1  # Rough heuristic

        # Use linear regression to refine estimates if we have enough data
        if len(log_returns) > 50:
            # Create features for regression: [constant, lagged_return, return_squared]
            X = np.column_stack(
                [np.ones(len(log_returns) - 1), log_returns[:-1], log_returns[:-1] ** 2]
            )
            y = log_returns[1:]

            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, y)

            # Adjust estimates based on regression residuals
            residuals = y - reg.predict(X)
            residual_var = np.var(residuals, ddof=1)

            # Refine sigma estimate
            sigma_est = min(sigma_est, np.sqrt(residual_var / dt))

        # Adjust drift for jump compensation
        jump_compensation = np.exp(jump_mean_est + 0.5 * jump_vol_est**2) - 1
        mu_est = mean_return + jump_intensity_est * jump_compensation

        return MertonParameters(
            mu=float(mu_est),
            sigma=float(sigma_est),
            jump_intensity=float(jump_intensity_est),
            jump_mean=float(jump_mean_est),
            jump_volatility=float(jump_vol_est),
        )
    else:
        raise ValueError(f"Estimation method '{method}' not implemented")


def create_simple_merton(
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float = 0.0,
    jump_volatility: float = 0.1,
) -> MertonJumpDiffusion:
    """
    Create a simple Merton Jump Diffusion model.

    Args:
        mu: Drift coefficient
        sigma: Diffusion volatility
        jump_intensity: Average jumps per unit time
        jump_mean: Mean of log jump sizes
        jump_volatility: Volatility of log jump sizes

    Returns:
        MertonJumpDiffusion instance
    """
    params = MertonParameters(
        mu=mu,
        sigma=sigma,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_volatility=jump_volatility,
    )

    return MertonJumpDiffusion(params, n_processes=1)
