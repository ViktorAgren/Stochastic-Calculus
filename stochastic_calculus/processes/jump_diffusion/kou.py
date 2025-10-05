"""Kou double exponential jump diffusion model implementation."""

from typing import Optional, Union, NamedTuple
from dataclasses import dataclass
import numpy as np

from ...core.utils import validate_positive
from ..brownian.utils import generate_correlated_brownian


@dataclass
class KouParameters:
    """Parameters for Kou double exponential jump diffusion model."""

    mu: float  # Drift coefficient
    sigma: float  # Diffusion volatility
    jump_intensity: float  # λ - average number of jumps per unit time
    p_up: float  # Probability of upward jump
    eta_up: float  # Rate parameter for upward jumps (exponential)
    eta_down: float  # Rate parameter for downward jumps (exponential)

    def __post_init__(self) -> None:
        """Validate parameters."""
        validate_positive(self.sigma, "diffusion volatility sigma")
        validate_positive(self.jump_intensity, "jump intensity lambda")
        validate_positive(self.eta_up, "upward jump rate eta_up")
        validate_positive(self.eta_down, "downward jump rate eta_down")

        if not (0 <= self.p_up <= 1):
            raise ValueError(f"Jump probability p_up must be in [0,1], got {self.p_up}")


class KouResult(NamedTuple):
    """Container for Kou jump diffusion simulation results."""

    prices: np.ndarray  # Asset prices S(t)
    log_prices: np.ndarray  # Log prices ln(S(t))
    jump_times: list  # List of jump times for each path
    jump_sizes: list  # List of jump sizes for each path
    jump_directions: list  # List of jump directions (1=up, -1=down) for each path


class KouJumpDiffusion:
    """
    Kou double exponential jump diffusion model.

    Solves the SDE:
    dS = μS dt + σS dW + S dJ

    Where J is a compound Poisson process with double exponential jumps:
    - Jump intensity λ
    - Upward jumps: Y ~ Exponential(η₊) with probability p
    - Downward jumps: Y ~ -Exponential(η₋) with probability (1-p)
    """

    def __init__(
        self,
        parameters: Union[KouParameters, tuple[KouParameters, ...]],
        n_processes: Optional[int] = None,
        correlation: Optional[float] = None,
    ) -> None:
        """
        Initialize Kou Jump Diffusion model.

        Args:
            parameters: Single KouParameters or tuple for multiple assets
            n_processes: Number of processes (ignored if parameters is tuple)
            correlation: Correlation between diffusion processes
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
        dt: float = 1 / 252,
        S_0: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ) -> KouResult:
        """
        Simulate Kou Jump Diffusion paths.

        Args:
            n_steps: Number of time steps
            dt: Time increment (default: daily = 1/252)
            S_0: Initial asset prices (default: 100.0)
            random_state: Random seed

        Returns:
            KouResult with prices, log prices, and jump information
        """
        rng = np.random.default_rng(random_state)

        if S_0 is None:
            S_0 = 100.0

        if np.isscalar(S_0):
            S_0 = np.full(self.n_processes, S_0)
        else:
            S_0 = np.asarray(S_0)

        prices = np.zeros((n_steps + 1, self.n_processes))
        log_prices = np.zeros((n_steps + 1, self.n_processes))
        jump_times = [[] for _ in range(self.n_processes)]
        jump_sizes = [[] for _ in range(self.n_processes)]
        jump_directions = [[] for _ in range(self.n_processes)]

        prices[0] = S_0
        log_prices[0] = np.log(S_0)

        # Generate correlated Brownian motions for diffusion part
        dW = generate_correlated_brownian(
            n_steps, self.n_processes, self.correlation, dt, random_state
        )

        # Simulate each asset
        for i in range(self.n_processes):
            params = self._get_params_for_process(i)

            # Jump compensation for risk-neutral pricing
            # E[e^Y - 1] for double exponential jumps
            jump_compensation = (
                params.p_up * params.eta_up / (params.eta_up - 1)
                + (1 - params.p_up) * params.eta_down / (params.eta_down + 1)
                - 1
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
                    # Generate jump sizes using double exponential distribution
                    total_jump = 0.0
                    for _ in range(num_jumps):
                        # Determine jump direction
                        if rng.random() < params.p_up:
                            # Upward jump: exponential distribution
                            jump_size = rng.exponential(1 / params.eta_up)
                            direction = 1
                        else:
                            # Downward jump: negative exponential distribution
                            jump_size = -rng.exponential(1 / params.eta_down)
                            direction = -1

                        total_jump += jump_size
                        jump_directions[i].append(direction)

                    log_S += total_jump

                    # Record jump information
                    jump_times[i].append((t + 1) * dt)
                    jump_sizes[i].append(total_jump)

                log_prices[t + 1, i] = log_S
                prices[t + 1, i] = np.exp(log_S)

        return KouResult(prices, log_prices, jump_times, jump_sizes, jump_directions)

    def _get_params_for_process(self, i: int) -> KouParameters:
        """Get parameters for process i."""
        if isinstance(self.parameters, tuple):
            return self.parameters[i]
        return self.parameters

    def get_parameters(self) -> dict:
        """Get process parameters."""
        return {
            "process_type": "KouJumpDiffusion",
            "parameters": self.parameters,
            "n_processes": self.n_processes,
            "correlation": self.correlation,
        }


def estimate_kou_parameters(
    price_data: np.ndarray, dt: float = 1 / 252
) -> KouParameters:
    """
    Estimate Kou Jump Diffusion parameters using method of moments.

    Args:
        price_data: Price series (1D array)
        dt: Time increment

    Returns:
        Estimated KouParameters
    """
    if price_data.ndim != 1:
        raise ValueError("Data must be 1D array for single process estimation")

    log_returns = np.diff(np.log(price_data))

    # Basic statistics
    mean_return = np.mean(log_returns) / dt
    var_return = np.var(log_returns, ddof=1) / dt
    skew_return = float(
        np.mean(
            ((log_returns - np.mean(log_returns)) / np.std(log_returns, ddof=1)) ** 3
        )
    )
    kurt_return = float(
        np.mean(
            ((log_returns - np.mean(log_returns)) / np.std(log_returns, ddof=1)) ** 4
        )
    )

    # Estimate diffusion volatility (assume 70% of variance from diffusion)
    sigma_est = np.sqrt(var_return * 0.7)

    # Estimate jump parameters using moments
    # Higher moments indicate jump presence
    jump_intensity_est = max(0.05, (kurt_return - 3) / 15)

    # Use skewness to estimate jump asymmetry
    if skew_return > 0:
        p_up_est = 0.6  # More upward jumps
    else:
        p_up_est = 0.4  # More downward jumps

    # Estimate jump rates (rough heuristics)
    eta_up_est = max(5.0, 20.0 / (1 + abs(skew_return)))
    eta_down_est = max(5.0, 20.0 / (1 + abs(skew_return)))

    # Adjust drift for jump compensation
    jump_compensation = (
        p_up_est * eta_up_est / (eta_up_est - 1)
        + (1 - p_up_est) * eta_down_est / (eta_down_est + 1)
        - 1
    )
    mu_est = mean_return + jump_intensity_est * jump_compensation

    return KouParameters(
        mu=float(mu_est),
        sigma=float(sigma_est),
        jump_intensity=float(jump_intensity_est),
        p_up=float(p_up_est),
        eta_up=float(eta_up_est),
        eta_down=float(eta_down_est),
    )


def create_simple_kou(
    mu: float,
    sigma: float,
    jump_intensity: float,
    p_up: float = 0.5,
    eta_up: float = 10.0,
    eta_down: float = 10.0,
) -> KouJumpDiffusion:
    """
    Create a simple Kou Jump Diffusion model.

    Args:
        mu: Drift coefficient
        sigma: Diffusion volatility
        jump_intensity: Average jumps per unit time
        p_up: Probability of upward jump
        eta_up: Rate parameter for upward jumps
        eta_down: Rate parameter for downward jumps

    Returns:
        KouJumpDiffusion instance
    """
    params = KouParameters(
        mu=mu,
        sigma=sigma,
        jump_intensity=jump_intensity,
        p_up=p_up,
        eta_up=eta_up,
        eta_down=eta_down,
    )

    return KouJumpDiffusion(params, n_processes=1)
