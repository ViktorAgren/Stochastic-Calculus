"""Reusable protocol-compliant components for Brownian motion processes."""

from typing import Union, Optional
import numpy as np

from ...core.protocols import Drift, Sigma, InitialValue
from ...core.utils import validate_positive


class ConstantDrift:
    """Constant drift component for Brownian motion processes."""
    
    def __init__(self, n_steps: int, mu: Union[float, tuple[float, ...]]) -> None:
        """
        Initialize constant drift component.
        
        Args:
            n_steps: Number of time steps
            mu: Drift parameter(s)
        """
        self.n_steps = n_steps
        self._mu = np.atleast_1d(mu)
        self._n_processes = len(self._mu)
        
    @property
    def sample_size(self) -> int:
        return self.n_steps
        
    @property 
    def n_processes(self) -> int:
        return self._n_processes
        
    def get_drift(self, random_state: Optional[int] = None) -> np.ndarray:
        """Get constant drift matrix."""
        return np.tile(self._mu, (self.n_steps, 1))


class ConstantVolatility:
    """Constant volatility component for Brownian motion processes."""
    
    def __init__(self, n_steps: int, sigma: Union[float, tuple[float, ...]]) -> None:
        """
        Initialize constant volatility component.
        
        Args:
            n_steps: Number of time steps
            sigma: Volatility parameter(s)
        """
        self.n_steps = n_steps
        self._sigma = np.atleast_1d(sigma)
        self._n_processes = len(self._sigma)
        
        # Validate all volatilities are positive
        for s in self._sigma:
            validate_positive(s, "volatility")
        
    @property
    def sample_size(self) -> int:
        return self.n_steps
        
    @property
    def n_processes(self) -> int:
        return self._n_processes
        
    def get_volatility(self, random_state: Optional[int] = None) -> np.ndarray:
        """Get constant volatility matrix."""
        return np.tile(self._sigma, (self.n_steps, 1))


class FixedInitialPrices:
    """Fixed initial prices component for financial processes."""
    
    def __init__(self, S_0: Union[float, tuple[float, ...]]) -> None:
        """
        Initialize fixed initial prices component.
        
        Args:
            S_0: Initial price(s)
        """
        self._S0 = np.atleast_1d(S_0)
        self._n_processes = len(self._S0)
        
        # Validate all initial prices are positive
        for s in self._S0:
            validate_positive(s, "initial price")
        
    @property
    def n_processes(self) -> int:
        return self._n_processes
        
    def get_initial_values(self, random_state: Optional[int] = None) -> np.ndarray:
        """Get fixed initial values."""
        return self._S0.copy()


def create_gbm_with_components(
    mu: Union[float, tuple[float, ...]],
    sigma: Union[float, tuple[float, ...]],
    S_0: Union[float, tuple[float, ...]],
    n_steps: int,
):
    """
    Create GeometricBrownianMotion using proper protocol-compliant components.
    
    This demonstrates the correct way to build processes using composable components.
    
    Args:
        mu: Drift parameter(s)
        sigma: Volatility parameter(s)
        S_0: Initial price(s)
        n_steps: Number of time steps
        
    Returns:
        GeometricBrownianMotion instance built with proper components
    """
    from .geometric import GeometricBrownianMotion
    
    # Create proper protocol-compliant components
    drift = ConstantDrift(n_steps, mu)
    volatility = ConstantVolatility(n_steps, sigma)
    initial = FixedInitialPrices(S_0)
    
    return GeometricBrownianMotion(drift, volatility, initial)