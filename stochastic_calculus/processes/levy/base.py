"""Base classes and utilities for Lévy processes."""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
from scipy import stats
from scipy.special import kv

from ..brownian.standard import BrownianMotion


class LevyProcess(ABC):
    """Abstract base class for Lévy processes."""

    @abstractmethod
    def characteristic_function(
        self, u: Union[float, np.ndarray], t: float = 1.0
    ) -> Union[complex, np.ndarray]:
        """
        Characteristic function of the Lévy process at time t.

        Args:
            u: Argument(s) for characteristic function
            t: Time parameter

        Returns:
            Characteristic function value(s)
        """
        pass

    @abstractmethod
    def levy_measure_density(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Lévy measure density function.

        Args:
            x: Point(s) to evaluate density

        Returns:
            Density value(s)
        """
        pass

    def simulate_subordinator(
        self, n_steps: int, dt: float, random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate the subordinator (time change) for time-changed processes.

        Args:
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed

        Returns:
            Subordinator increments
        """
        # Default implementation returns deterministic time
        return np.full(n_steps, dt)


class TimeChangedBrownianMotion:
    """Utility class for time-changed Brownian motion processes."""

    @staticmethod
    def simulate_time_changed_bm(
        n_steps: int,
        subordinator: np.ndarray,
        drift: float = 0.0,
        volatility: float = 1.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate time-changed Brownian motion using BrownianMotion composition.

        Args:
            n_steps: Number of steps
            subordinator: Time change increments
            drift: Drift parameter
            volatility: Volatility parameter
            random_state: Random seed

        Returns:
            Time-changed Brownian motion increments
        """
        # Use BrownianMotion composition for standard Brownian increments
        brownian = BrownianMotion(n_processes=1)
        # Use dt=1 because we'll scale by subordinator
        brownian_result = brownian.simulate(n_steps, dt=1.0, random_state=random_state)
        bm_increments = brownian_result.increments[:, 0]

        # Apply time change and scaling
        return drift * subordinator + volatility * bm_increments * np.sqrt(subordinator)


class LevyJumpGenerator:
    """Utility class for generating jumps from Lévy measures."""

    @staticmethod
    def compound_poisson_simulation(
        lambda_rate: float,
        jump_distribution,
        n_steps: int,
        dt: float,
        random_state: Optional[int] = None,
    ) -> tuple[np.ndarray, list]:
        """
        Simulate compound Poisson process.

        Args:
            lambda_rate: Jump intensity
            jump_distribution: Jump size distribution (scipy.stats object)
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed

        Returns:
            Tuple of (increments, jump_times)
        """
        rng = np.random.default_rng(random_state)

        increments = np.zeros(n_steps)
        jump_times = []

        for t in range(n_steps):
            # Number of jumps in this interval
            num_jumps = rng.poisson(lambda_rate * dt)

            if num_jumps > 0:
                # Generate jump sizes
                jumps = jump_distribution.rvs(size=num_jumps, random_state=rng)
                increments[t] = np.sum(jumps)
                jump_times.append((t + 1) * dt)

        return increments, jump_times

    @staticmethod
    def shot_noise_representation(
        levy_measure_density,
        truncation_level: float,
        n_steps: int,
        dt: float,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate Lévy process using shot noise representation for small jumps.

        Args:
            levy_measure_density: Lévy measure density function
            truncation_level: Truncation level for small jumps
            n_steps: Number of time steps
            dt: Time increment
            random_state: Random seed

        Returns:
            Process increments
        """
        # This is a simplified implementation
        # In practice, this would use more sophisticated algorithms
        rng = np.random.default_rng(random_state)
        return rng.normal(0, np.sqrt(dt), n_steps)


def bessel_k(nu: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Modified Bessel function of the second kind.

    Args:
        nu: Order parameter
        x: Argument(s)

    Returns:
        Bessel function value(s)
    """
    return kv(nu, x)


def generalized_inverse_gaussian_rvs(
    p: float, a: float, b: float, size: int = 1, random_state: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Generate random variates from Generalized Inverse Gaussian distribution.

    Args:
        p: Shape parameter
        a: Scale parameter
        b: Scale parameter
        size: Number of samples
        random_state: Random seed

    Returns:
        Random samples
    """
    # Simplified implementation using acceptance-rejection
    # In practice, would use more efficient algorithms
    rng = np.random.default_rng(random_state)

    # For now, use a gamma approximation
    # This should be replaced with proper GIG sampling
    beta = np.sqrt(a / b)

    if size == 1:
        return stats.gamma.rvs(a=abs(p), scale=1 / beta, random_state=rng)
    else:
        return stats.gamma.rvs(a=abs(p), scale=1 / beta, size=size, random_state=rng)
