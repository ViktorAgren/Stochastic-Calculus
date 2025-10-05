"""Bates model implementation (Heston with jumps)."""

from typing import Optional, Union, NamedTuple
from dataclasses import dataclass
import numpy as np

from ...core.utils import validate_positive
from ..brownian.utils import validate_correlation
from ..stochastic_vol.heston import HestonParameters


@dataclass
class BatesParameters:
    """Parameters for Bates model (Heston + jumps)."""

    heston_params: HestonParameters
    price_jump_intensity: float  # λ_S - price jump intensity
    price_jump_mean: float  # μ_S - mean of log price jumps
    price_jump_volatility: float  # σ_S - volatility of log price jumps
    vol_jump_intensity: float  # λ_v - volatility jump intensity
    vol_jump_mean: float  # μ_v - mean of volatility jumps
    vol_jump_volatility: float  # σ_v - volatility of volatility jumps
    jump_correlation: float  # ρ_J - correlation between price and vol jumps

    def __post_init__(self) -> None:
        """Validate parameters."""
        validate_positive(self.price_jump_intensity, "price jump intensity λ_S")
        validate_positive(self.price_jump_volatility, "price jump volatility σ_S")
        validate_positive(self.vol_jump_intensity, "volatility jump intensity λ_v")
        validate_positive(self.vol_jump_mean, "volatility jump mean μ_v")
        validate_positive(self.vol_jump_volatility, "volatility jump volatility σ_v")
        validate_correlation(self.jump_correlation)


class BatesResult(NamedTuple):
    """Container for Bates model simulation results."""

    prices: np.ndarray  # Asset prices S(t)
    volatilities: np.ndarray  # Volatility processes v(t)
    log_prices: np.ndarray  # Log prices ln(S(t))
    price_jump_times: list  # List of price jump times for each path
    price_jump_sizes: list  # List of price jump sizes for each path
    vol_jump_times: list  # List of volatility jump times for each path
    vol_jump_sizes: list  # List of volatility jump sizes for each path


class BatesModel:
    """
    Bates model - Heston stochastic volatility with correlated jumps.

    Solves the system:
    dS = μS dt + √v S dW₁ + S dJ_S
    dv = κ(θ - v)dt + σ√v dW₂ + dJ_v

    Where J_S and J_v are correlated compound Poisson processes.
    """

    def __init__(
        self,
        parameters: Union[BatesParameters, tuple[BatesParameters, ...]],
        n_processes: Optional[int] = None,
        inter_asset_correlation: Optional[float] = None,
    ) -> None:
        """
        Initialize Bates model.

        Args:
            parameters: Single BatesParameters or tuple for multiple assets
            n_processes: Number of assets (ignored if parameters is tuple)
            inter_asset_correlation: Correlation between different assets
        """
        self.inter_asset_correlation = inter_asset_correlation

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
        v_0: Optional[Union[float, np.ndarray]] = None,
        random_state: Optional[int] = None,
    ) -> BatesResult:
        """
        Simulate Bates model paths.

        Args:
            n_steps: Number of time steps
            dt: Time increment (default: daily = 1/252)
            S_0: Initial asset prices (default: 100.0)
            v_0: Initial volatilities (default: theta)
            random_state: Random seed

        Returns:
            BatesResult with prices, volatilities, log prices, and jump information
        """
        rng = np.random.default_rng(random_state)

        # Set initial conditions
        if S_0 is None:
            S_0 = 100.0
        if v_0 is None:
            if isinstance(self.parameters, tuple):
                v_0 = np.array([p.heston_params.theta for p in self.parameters])
            else:
                v_0 = self.parameters.heston_params.theta

        # Ensure arrays
        S_0 = np.atleast_1d(S_0)
        v_0 = np.atleast_1d(v_0)

        if len(S_0) == 1 and self.n_processes > 1:
            S_0 = np.full(self.n_processes, S_0[0])
        if len(v_0) == 1 and self.n_processes > 1:
            v_0 = np.full(self.n_processes, v_0[0])

        # Initialize result arrays
        prices = np.zeros((n_steps + 1, self.n_processes))
        volatilities = np.zeros((n_steps + 1, self.n_processes))
        log_prices = np.zeros((n_steps + 1, self.n_processes))

        # Jump tracking
        price_jump_times = [[] for _ in range(self.n_processes)]
        price_jump_sizes = [[] for _ in range(self.n_processes)]
        vol_jump_times = [[] for _ in range(self.n_processes)]
        vol_jump_sizes = [[] for _ in range(self.n_processes)]

        prices[0] = S_0
        volatilities[0] = v_0
        log_prices[0] = np.log(S_0)

        # Generate correlated Brownian motions for diffusion parts
        if self.n_processes == 1:
            # Single asset case
            params = self._get_params_for_process(0)
            heston_params = params.heston_params

            dW1 = rng.normal(0, np.sqrt(dt), n_steps)
            dW2_indep = rng.normal(0, np.sqrt(dt), n_steps)
            dW2 = (
                heston_params.rho * dW1 + np.sqrt(1 - heston_params.rho**2) * dW2_indep
            )

            S_path, v_path, pjt, pjs, vjt, vjs = self._simulate_single_bates(
                n_steps, dt, params, S_0[0], v_0[0], dW1, dW2, rng
            )

            prices[:, 0] = S_path
            volatilities[:, 0] = v_path
            log_prices[:, 0] = np.log(S_path)
            price_jump_times[0] = pjt
            price_jump_sizes[0] = pjs
            vol_jump_times[0] = vjt
            vol_jump_sizes[0] = vjs
        else:
            # Multi-asset case
            from ...core.utils import generate_correlated_brownian

            # Generate correlated price Brownian motions
            if random_state is not None:
                temp_seed = rng.integers(0, 2**31)
            else:
                temp_seed = None

            dW1_matrix = generate_correlated_brownian(
                n_steps, self.n_processes, self.inter_asset_correlation, dt, temp_seed
            )

            # Simulate each asset
            for i in range(self.n_processes):
                params = self._get_params_for_process(i)
                heston_params = params.heston_params

                # Get price Brownian motion for this asset
                dW1 = dW1_matrix[:, i]

                # Generate correlated vol Brownian motion
                dW2_indep = rng.normal(0, np.sqrt(dt), n_steps)
                dW2 = (
                    heston_params.rho * dW1
                    + np.sqrt(1 - heston_params.rho**2) * dW2_indep
                )

                # Simulate this asset
                S_path, v_path, pjt, pjs, vjt, vjs = self._simulate_single_bates(
                    n_steps, dt, params, S_0[i], v_0[i], dW1, dW2, rng
                )

                prices[:, i] = S_path
                volatilities[:, i] = v_path
                log_prices[:, i] = np.log(S_path)
                price_jump_times[i] = pjt
                price_jump_sizes[i] = pjs
                vol_jump_times[i] = vjt
                vol_jump_sizes[i] = vjs

        return BatesResult(
            prices,
            volatilities,
            log_prices,
            price_jump_times,
            price_jump_sizes,
            vol_jump_times,
            vol_jump_sizes,
        )

    def _get_params_for_process(self, i: int) -> BatesParameters:
        """Get parameters for process i."""
        if isinstance(self.parameters, tuple):
            return self.parameters[i]
        return self.parameters

    def _simulate_single_bates(
        self,
        n_steps: int,
        dt: float,
        params: BatesParameters,
        S_0: float,
        v_0: float,
        dW1: np.ndarray,
        dW2: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, list, list, list, list]:
        """Simulate single Bates asset."""
        heston_params = params.heston_params

        S = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)
        price_jump_times = []
        price_jump_sizes = []
        vol_jump_times = []
        vol_jump_sizes = []

        S[0] = S_0
        v[0] = v_0

        # Price jump compensation
        price_jump_compensation = (
            np.exp(params.price_jump_mean + 0.5 * params.price_jump_volatility**2) - 1
        )
        adjusted_drift = (
            heston_params.mu - params.price_jump_intensity * price_jump_compensation
        )

        for t in range(n_steps):
            # Apply absorption for volatility
            v_t = max(v[t], 0.0)
            sqrt_v_t = np.sqrt(v_t)

            # Diffusion parts (same as Heston)
            dv_diffusion = (
                heston_params.kappa * (heston_params.theta - v_t) * dt
                + heston_params.sigma * sqrt_v_t * dW2[t]
            )

            dS_diffusion = adjusted_drift * S[t] * dt + sqrt_v_t * S[t] * dW1[t]

            # Jump parts
            dS_jump = 0.0
            dv_jump = 0.0

            # Price jumps
            num_price_jumps = rng.poisson(params.price_jump_intensity * dt)
            if num_price_jumps > 0:
                price_jump_magnitudes = rng.normal(
                    params.price_jump_mean,
                    params.price_jump_volatility,
                    num_price_jumps,
                )
                total_price_jump = np.sum(price_jump_magnitudes)
                dS_jump = S[t] * (np.exp(total_price_jump) - 1)

                price_jump_times.append((t + 1) * dt)
                price_jump_sizes.append(total_price_jump)

            # Volatility jumps
            num_vol_jumps = rng.poisson(params.vol_jump_intensity * dt)
            if num_vol_jumps > 0:
                # Generate correlated volatility jumps
                if num_price_jumps > 0 and len(price_jump_magnitudes) > 0:
                    # Use jump correlation if both price and vol jumps occur
                    corr_component = params.jump_correlation * price_jump_magnitudes[0]
                    uncorr_component = np.sqrt(
                        1 - params.jump_correlation**2
                    ) * rng.normal(params.vol_jump_mean, params.vol_jump_volatility)
                    vol_jump_magnitude = corr_component + uncorr_component
                else:
                    vol_jump_magnitude = rng.normal(
                        params.vol_jump_mean, params.vol_jump_volatility
                    )

                dv_jump = vol_jump_magnitude

                vol_jump_times.append((t + 1) * dt)
                vol_jump_sizes.append(vol_jump_magnitude)

            # Update processes
            v[t + 1] = max(v[t] + dv_diffusion + dv_jump, 0.0)
            S[t + 1] = max(S[t] + dS_diffusion + dS_jump, 1e-8)

        return S, v, price_jump_times, price_jump_sizes, vol_jump_times, vol_jump_sizes

    def get_parameters(self) -> dict:
        """Get process parameters."""
        return {
            "process_type": "BatesModel",
            "parameters": self.parameters,
            "n_processes": self.n_processes,
            "inter_asset_correlation": self.inter_asset_correlation,
        }


def estimate_bates_parameters(
    price_data: np.ndarray, volatility_proxy: np.ndarray, dt: float = 1 / 252
) -> BatesParameters:
    """
    Estimate Bates parameters using method of moments.

    Args:
        price_data: Asset price time series (1D array)
        volatility_proxy: Volatility proxy (e.g., realized volatility)
        dt: Time increment

    Returns:
        Estimated BatesParameters
    """
    if price_data.ndim != 1 or volatility_proxy.ndim != 1:
        raise ValueError("Data must be 1D arrays for single asset estimation")

    # First estimate Heston parameters as base
    from ..stochastic_vol.heston import estimate_heston_parameters

    heston_params = estimate_heston_parameters(price_data, volatility_proxy, dt)

    # Estimate jump parameters from return moments
    log_returns = np.diff(np.log(price_data))
    vol_changes = np.diff(volatility_proxy**2) / dt

    # Higher moments suggest jump presence
    return_skew = float(
        np.mean(
            ((log_returns - np.mean(log_returns)) / np.std(log_returns, ddof=1)) ** 3
        )
    )
    return_kurt = float(
        np.mean(
            ((log_returns - np.mean(log_returns)) / np.std(log_returns, ddof=1)) ** 4
        )
    )

    # Simple heuristic estimates for jump parameters
    price_jump_intensity = max(0.05, (return_kurt - 3) / 20)
    price_jump_volatility = max(0.05, abs(return_skew) * 0.2)
    price_jump_mean = return_skew * 0.1

    vol_jump_intensity = max(0.05, np.std(vol_changes) * 10)
    vol_jump_mean = max(0.01, np.mean(vol_changes) * 2)
    vol_jump_volatility = max(0.01, np.std(vol_changes) * 0.5)

    # Estimate jump correlation from data
    if len(log_returns) == len(vol_changes):
        jump_correlation = np.corrcoef(log_returns, vol_changes)[0, 1]
        jump_correlation = np.clip(jump_correlation, -0.99, 0.99)
    else:
        jump_correlation = -0.3  # Default negative correlation

    return BatesParameters(
        heston_params=heston_params,
        price_jump_intensity=float(price_jump_intensity),
        price_jump_mean=float(price_jump_mean),
        price_jump_volatility=float(price_jump_volatility),
        vol_jump_intensity=float(vol_jump_intensity),
        vol_jump_mean=float(vol_jump_mean),
        vol_jump_volatility=float(vol_jump_volatility),
        jump_correlation=float(jump_correlation),
    )


def create_simple_bates(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    mu: float,
    price_jump_intensity: float,
    price_jump_mean: float = 0.0,
    price_jump_volatility: float = 0.1,
    vol_jump_intensity: float = 0.1,
    vol_jump_mean: float = 0.01,
    vol_jump_volatility: float = 0.05,
    jump_correlation: float = -0.3,
) -> BatesModel:
    """
    Create a simple Bates model.

    Returns:
        BatesModel instance
    """
    heston_params = HestonParameters(kappa, theta, sigma, rho, mu)

    bates_params = BatesParameters(
        heston_params=heston_params,
        price_jump_intensity=price_jump_intensity,
        price_jump_mean=price_jump_mean,
        price_jump_volatility=price_jump_volatility,
        vol_jump_intensity=vol_jump_intensity,
        vol_jump_mean=vol_jump_mean,
        vol_jump_volatility=vol_jump_volatility,
        jump_correlation=jump_correlation,
    )

    return BatesModel(bates_params, n_processes=1)
