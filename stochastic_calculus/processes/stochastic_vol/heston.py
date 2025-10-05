"""Heston stochastic volatility model implementation."""

from typing import Optional, Union, NamedTuple, Any
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression

from ...core.utils import validate_positive
from ...core.protocols import StochasticProcess
from ..brownian.utils import validate_correlation
from ..brownian.standard import BrownianMotion


@dataclass
class HestonParameters:
    """Parameters for Heston stochastic volatility model."""

    kappa: float  # Volatility mean reversion speed
    theta: float  # Long-term volatility level
    sigma: float  # Volatility of volatility
    rho: float  # Price-volatility correlation
    mu: float  # Drift of underlying asset

    def __post_init__(self) -> None:
        """Validate parameters."""
        validate_positive(self.kappa, "mean reversion speed kappa")
        validate_positive(self.theta, "long-term volatility theta")
        validate_positive(self.sigma, "volatility of volatility sigma")
        validate_correlation(self.rho)

        # Feller condition: 2κθ ≥ σ²
        if 2 * self.kappa * self.theta < self.sigma**2:
            raise ValueError(
                f"Feller condition violated: 2κθ ({2*self.kappa*self.theta:.4f}) < σ² ({self.sigma**2:.4f})"
            )


class HestonResult(NamedTuple):
    """Container for Heston simulation results."""

    prices: np.ndarray  # Asset prices S(t)
    volatilities: np.ndarray  # Volatility processes v(t)
    log_prices: np.ndarray  # Log prices for convenience


class HestonProcess(StochasticProcess):
    """
    Heston stochastic volatility model.

    Solves the system:
    dS = μ S dt + √v S dW₁
    dv = κ(θ - v)dt + σ√v dW₂

    with correlation ρ between dW₁ and dW₂.
    """

    def __init__(
        self,
        parameters: Union[HestonParameters, tuple[HestonParameters, ...]],
        n_processes: Optional[int] = None,
        inter_asset_correlation: Optional[float] = None,
        S_0: Optional[Union[float, np.ndarray]] = None,
        v_0: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        """
        Initialize Heston model.

        Args:
            parameters: Single HestonParameters or tuple for multiple assets
            n_processes: Number of assets (ignored if parameters is tuple)
            inter_asset_correlation: Correlation between different assets
            S_0: Initial asset prices (default: 100.0)
            v_0: Initial volatilities (default: theta values)
        """
        self.inter_asset_correlation = inter_asset_correlation
        self.S_0 = S_0
        self.v_0 = v_0

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
    ) -> HestonResult:
        """
        Simulate Heston model using Euler-Maruyama scheme.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed

        Returns:
            HestonResult with prices, volatilities, and log prices
        """

        # Set initial conditions
        S_0 = self.S_0
        v_0 = self.v_0

        if S_0 is None:
            S_0 = 100.0
        if v_0 is None:
            if isinstance(self.parameters, tuple):
                v_0 = np.array([p.theta for p in self.parameters])
            else:
                v_0 = self.parameters.theta

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

        prices[0] = S_0
        volatilities[0] = v_0
        log_prices[0] = np.log(S_0)

        # Generate correlated Brownian motions using BrownianMotion composition
        if self.n_processes == 1:
            # Single asset case: Generate correlated W1, W2 for price and volatility
            params = self._get_params_for_process(0)

            # Create BrownianMotion for correlated price and volatility processes
            brownian_2d = BrownianMotion(n_processes=2, correlation=params.rho)
            brownian_result = brownian_2d.simulate(n_steps, dt, random_state)

            dW1 = brownian_result.increments[:, 0]  # Price Brownian motion
            dW2 = brownian_result.increments[:, 1]  # Volatility Brownian motion

            S_path, v_path = self._simulate_single_heston(
                n_steps, dt, params, S_0[0], v_0[0], dW1, dW2
            )

            prices[:, 0] = S_path
            volatilities[:, 0] = v_path
            log_prices[:, 0] = np.log(S_path)
        else:
            # Multi-asset case: Generate inter-asset correlated price processes
            # plus leverage-correlated volatility processes

            # Generate inter-asset correlated price Brownian motions
            price_brownian = BrownianMotion(
                n_processes=self.n_processes, correlation=self.inter_asset_correlation
            )
            price_result = price_brownian.simulate(n_steps, dt, random_state)
            dW1_matrix = price_result.increments

            # For each asset, generate leverage-correlated volatility Brownian motion
            for i in range(self.n_processes):
                params = self._get_params_for_process(i)

                # Get price Brownian motion for this asset
                dW1 = dW1_matrix[:, i]

                # Generate independent volatility Brownian motion
                vol_brownian = BrownianMotion(n_processes=1)
                vol_result = vol_brownian.simulate(n_steps, dt, random_state)
                dW2_indep = vol_result.increments[:, 0]

                # Apply leverage correlation
                dW2 = params.rho * dW1 + np.sqrt(1 - params.rho**2) * dW2_indep

                # Simulate this asset
                S_path, v_path = self._simulate_single_heston(
                    n_steps, dt, params, S_0[i], v_0[i], dW1, dW2
                )

                prices[:, i] = S_path
                volatilities[:, i] = v_path
                log_prices[:, i] = np.log(S_path)

        return HestonResult(prices, volatilities, log_prices)


    def _get_params_for_process(self, i: int) -> HestonParameters:
        """Get parameters for process i."""
        if isinstance(self.parameters, tuple):
            return self.parameters[i]
        return self.parameters

    def _simulate_single_heston(
        self,
        n_steps: int,
        dt: float,
        params: HestonParameters,
        S_0: float,
        v_0: float,
        dW1: np.ndarray,
        dW2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate single Heston asset using Euler-Maruyama."""
        S = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)

        S[0] = S_0
        v[0] = v_0

        for t in range(n_steps):
            # Apply absorption for volatility
            v_t = max(v[t], 0.0)
            sqrt_v_t = np.sqrt(v_t)

            # Volatility process (CIR-type)
            dv = (
                params.kappa * (params.theta - v_t) * dt
                + params.sigma * sqrt_v_t * dW2[t]
            )
            v[t + 1] = max(v[t] + dv, 0.0)

            # Price process
            dS = params.mu * S[t] * dt + sqrt_v_t * S[t] * dW1[t]
            S[t + 1] = S[t] + dS

            # Ensure prices stay positive
            S[t + 1] = max(S[t + 1], 1e-8)

        return S, v

    def get_initial_value_names(self) -> list[str]:
        """Return names of initial value parameters for Heston process."""
        return [StandardInitialValueNames.ASSET_PRICE, StandardInitialValueNames.VOLATILITY]

    def set_initial_values(self, **kwargs) -> None:
        """Set initial values on the Heston process."""
        if StandardInitialValueNames.ASSET_PRICE in kwargs:
            self.S_0 = kwargs[StandardInitialValueNames.ASSET_PRICE]
        if StandardInitialValueNames.VOLATILITY in kwargs:
            self.v_0 = kwargs[StandardInitialValueNames.VOLATILITY]

    def get_initial_values(self) -> dict[str, Any]:
        """Get current initial values."""
        return {
            StandardInitialValueNames.ASSET_PRICE: self.S_0,
            StandardInitialValueNames.VOLATILITY: self.v_0,
        }

    def get_parameters(self) -> dict:
        """Get process parameters."""
        return {
            "process_type": "HestonProcess",
            "parameters": self.parameters,
            "n_processes": self.n_processes,
            "inter_asset_correlation": self.inter_asset_correlation,
        }


def estimate_heston_parameters(
    price_data: np.ndarray, volatility_proxy: np.ndarray, dt: float = 1 / 252
) -> HestonParameters:
    """
    Estimate Heston parameters using method of moments.

    Args:
        price_data: Asset price time series (1D array)
        volatility_proxy: Volatility proxy (e.g., realized volatility)
        dt: Time increment

    Returns:
        Estimated HestonParameters
    """
    if price_data.ndim != 1 or volatility_proxy.ndim != 1:
        raise ValueError("Data must be 1D arrays for single asset estimation")

    # Estimate drift from price data
    log_returns = np.diff(np.log(price_data))
    mu = np.mean(log_returns) / dt

    # Estimate volatility process parameters
    v_data = volatility_proxy**2  # Convert to variance
    dv = np.diff(v_data) / dt
    v_lag = v_data[:-1]

    # Linear regression: dv = a + b*v + noise
    X = np.column_stack([np.ones(len(v_lag)), v_lag])
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, dv)

    a_coef = reg.coef_[0]  # κθ
    b_coef = reg.coef_[1]  # -κ

    kappa = -b_coef
    theta = a_coef / kappa if kappa > 0 else np.mean(v_data)

    # Estimate sigma from residuals
    residuals = dv - reg.predict(X)
    sigma = np.sqrt(dt) * np.std(residuals)

    # Estimate correlation
    price_returns = np.diff(price_data) / price_data[:-1]
    vol_changes = np.diff(volatility_proxy)

    if len(price_returns) == len(vol_changes):
        rho = np.corrcoef(price_returns, vol_changes)[0, 1]
        rho = np.clip(rho, -0.99, 0.99)
    else:
        rho = -0.5  # Default negative correlation

    # Ensure valid parameters
    kappa = max(kappa, 0.001)
    theta = max(theta, 0.001)
    sigma = max(sigma, 0.001)

    # Ensure Feller condition
    if 2 * kappa * theta < sigma**2:
        kappa = (sigma**2) / (2 * theta) + 0.001

    return HestonParameters(kappa=kappa, theta=theta, sigma=sigma, rho=rho, mu=mu)


def create_simple_heston(
    kappa: float, theta: float, sigma: float, rho: float, mu: float
) -> HestonProcess:
    """
    Convenience function to create simple Heston model.

    Returns:
        HestonProcess instance
    """
    params = HestonParameters(kappa, theta, sigma, rho, mu)
    return HestonProcess(params, n_processes=1)
